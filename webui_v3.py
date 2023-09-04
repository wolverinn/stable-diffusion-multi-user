from __future__ import annotations

import os
import sys
import time
import importlib
import signal
import re
import warnings
import json
from threading import Thread
from typing import Iterable

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from packaging import version

import logging

logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

from modules import paths, timer, import_hook, errors, devices  # noqa: F401

startup_timer = timer.startup_timer

import torch
import pytorch_lightning   # noqa: F401 # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")


startup_timer.record("import torch")

import gradio
startup_timer.record("import gradio")

import ldm.modules.encoders.modules  # noqa: F401
startup_timer.record("import ldm")

from modules import extra_networks
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, queue_lock  # noqa: F401

# Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

from modules import shared, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks, config_states
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.scripts
import modules.sd_hijack
import modules.sd_hijack_optimizations
import modules.sd_models
import modules.sd_vae
import modules.sd_unet
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

startup_timer.record("other imports")


if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None


def fix_asyncio_event_loop_policy():
    """
        The default `asyncio` event loop policy only automatically creates
        event loops in the main threads. Other threads must create event
        loops explicitly or `asyncio.get_event_loop` (and therefore
        `.IOLoop.current`) will fail. Installing this policy allows event
        loops to be created automatically on any thread, matching the
        behavior of Tornado versions prior to 5.0 (or 5.0 on Python 2).
    """

    import asyncio

    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        # "Any thread" and "selector" should be orthogonal, but there's not a clean
        # interface for composing policies so pick the right base.
        _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy  # type: ignore
    else:
        _BasePolicy = asyncio.DefaultEventLoopPolicy

    class AnyThreadEventLoopPolicy(_BasePolicy):  # type: ignore
        """Event loop policy that allows loop creation on any thread.
        Usage::

            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        """

        def get_event_loop(self) -> asyncio.AbstractEventLoop:
            try:
                return super().get_event_loop()
            except (RuntimeError, AssertionError):
                # This was an AssertionError in python 3.4.2 (which ships with debian jessie)
                # and changed to a RuntimeError in 3.4.3.
                # "There is no current event loop in thread %r"
                loop = self.new_event_loop()
                self.set_event_loop(loop)
                return loop

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


def check_versions():
    if shared.cmd_opts.skip_version_check:
        return

    expected_torch_version = "2.0.0"

    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        errors.print_error_explanation(f"""
You are running torch {torch.__version__}.
The program is tested to work with torch {expected_torch_version}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

    expected_xformers_version = "0.0.20"
    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            errors.print_error_explanation(f"""
You are running xformers {xformers.__version__}.
The program is tested to work with xformers {expected_xformers_version}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())


def restore_config_state_file():
    config_state_file = shared.opts.restore_config_state_file
    if config_state_file == "":
        return

    shared.opts.restore_config_state_file = ""
    shared.opts.save(shared.config_filename)

    if os.path.isfile(config_state_file):
        print(f"*** About to restore extension state from file: {config_state_file}")
        with open(config_state_file, "r", encoding="utf-8") as f:
            config_state = json.load(f)
            config_states.restore_extension_config(config_state)
        startup_timer.record("restore extension config")
    elif config_state_file:
        print(f"!!! Config state backup not found: {config_state_file}")


def validate_tls_options():
    if not (cmd_opts.tls_keyfile and cmd_opts.tls_certfile):
        return

    try:
        if not os.path.exists(cmd_opts.tls_keyfile):
            print("Invalid path to TLS keyfile given")
        if not os.path.exists(cmd_opts.tls_certfile):
            print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
    except TypeError:
        cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
        print("TLS setup invalid, running webui without TLS")
    else:
        print("Running with TLS")
    startup_timer.record("TLS")


def get_gradio_auth_creds() -> Iterable[tuple[str, ...]]:
    """
    Convert the gradio_auth and gradio_auth_path commandline arguments into
    an iterable of (username, password) tuples.
    """
    def process_credential_line(s) -> tuple[str, ...] | None:
        s = s.strip()
        if not s:
            return None
        return tuple(s.split(':', 1))

    if cmd_opts.gradio_auth:
        for cred in cmd_opts.gradio_auth.split(','):
            cred = process_credential_line(cred)
            if cred:
                yield cred

    if cmd_opts.gradio_auth_path:
        with open(cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                for cred in line.strip().split(','):
                    cred = process_credential_line(cred)
                    if cred:
                        yield cred


def configure_sigint_handler():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    if not os.environ.get("COVERAGE_RUN"):
        # Don't install the immediate-quit handler when running under coverage,
        # as then the coverage report won't be generated.
        signal.signal(signal.SIGINT, sigint_handler)


def configure_opts_onchange():
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()), call=False)
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    shared.opts.onchange("gradio_theme", shared.reload_gradio_theme)
    shared.opts.onchange("cross_attention_optimization", wrap_queued_call(lambda: modules.sd_hijack.model_hijack.redo_hijack(shared.sd_model)), call=False)
    startup_timer.record("opts onchange")


def initialize():
    print("start init")
    fix_asyncio_event_loop_policy()
    validate_tls_options()
    configure_sigint_handler()
    # check_versions()
    modelloader.cleanup_models()
    configure_opts_onchange()

    modules.sd_models.setup_model()
    startup_timer.record("setup SD model")

    codeformer.setup_model(cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    initialize_rest(reload_script_modules=False)


def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    sd_samplers.set_samplers()
    extensions.list_extensions()
    startup_timer.record("list extensions")

    restore_config_state_file()

    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modules.sd_models.list_models()
    startup_timer.record("list SD models")

    localization.list_localizations(cmd_opts.localizations_dir)

    with startup_timer.subcategory("load scripts"):
        modules.scripts.load_scripts()

    if reload_script_modules:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    modules.sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")
    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    modules.script_callbacks.on_list_optimizers(modules.sd_hijack_optimizations.list_optimizers)
    modules.sd_hijack.list_optimizers()
    startup_timer.record("scripts list_optimizers")

    modules.sd_unet.list_unets()
    startup_timer.record("scripts list_unets")

    def load_model():
        """
        Accesses shared.sd_model property to load model.
        After it's available, if it has been loaded before this access by some extension,
        its optimization may be None because the list of optimizaers has neet been filled
        by that time, so we apply optimization again.
        """

        shared.sd_model  # noqa: B018

        if modules.sd_hijack.current_optimizer is None:
            modules.sd_hijack.apply_optimizations()

    Thread(target=load_model).start()

    Thread(target=devices.first_time_calculation).start()

    shared.reload_hypernetworks()
    startup_timer.record("reload hypernetworks")

    # ui_extra_networks.initialize()
    # ui_extra_networks.register_default_pages()

    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    startup_timer.record("initialize extra networks")


def setup_middleware(app):
    app.middleware_stack = None  # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    configure_cors_middleware(app)
    app.build_middleware_stack()  # rebuild middleware stack on-the-fly


def configure_cors_middleware(app):
    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_credentials": True,
    }
    if cmd_opts.cors_allow_origins:
        cors_options["allow_origins"] = cmd_opts.cors_allow_origins.split(',')
    if cmd_opts.cors_allow_origins_regex:
        cors_options["allow_origin_regex"] = cmd_opts.cors_allow_origins_regex
    app.add_middleware(CORSMiddleware, **cors_options)


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def api_only():
    initialize()

    app = FastAPI()
    setup_middleware(app)
    api = create_api(app)

    modules.script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)


def stop_route(request):
    shared.state.server_command = "stop"
    return Response("Stopping.")

## manual API
from modules.api.api import script_name_to_index, validate_sampler_name, encode_pil_to_base64, decode_base64_to_image
import gradio as gr

def get_script(script_name, script_runner):
    if script_name is None or script_name == "":
        return None, None
    script_idx = script_name_to_index(script_name, script_runner.scripts)
    return script_runner.scripts[script_idx]
def init_default_script_args(script_runner):
    #find max idx from the scripts in runner and generate a none array to init script_args
    last_arg_index = 1
    for script in script_runner.scripts:
        if last_arg_index < script.args_to:
            last_arg_index = script.args_to
    # None everywhere except position 0 to initialize script args
    script_args = [None]*last_arg_index
    script_args[0] = 0
    # get default values
    with gr.Blocks(): # will throw errors calling ui function without this
        for script in script_runner.scripts:
            if script.ui(script.is_img2img):
                ui_default_values = []
                for elem in script.ui(script.is_img2img):
                    ui_default_values.append(elem.value)
                script_args[script.args_from:script.args_to] = ui_default_values
    return script_args
def get_selectable_script(script_name, script_runner):
    if script_name is None or script_name == "":
        return None, None

    script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
    script = script_runner.selectable_scripts[script_idx]
    return script, script_idx
def init_script_args(request, default_script_args, selectable_scripts, selectable_idx, script_runner):
    script_args = default_script_args.copy()
    # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
    if selectable_scripts:
        script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request.script_args
        script_args[0] = selectable_idx + 1
    # Now check for always on scripts
    if request.alwayson_scripts:
        for alwayson_script_name in request.alwayson_scripts.keys():
            alwayson_script = get_script(alwayson_script_name, script_runner)
            if alwayson_script is None:
                raise HTTPException(status_code=422, detail=f"always on script {alwayson_script_name} not found")
            # Selectable script in always on script param check
            if alwayson_script.alwayson is False:
                raise HTTPException(status_code=422, detail="Cannot have a selectable script in the always on scripts params")
            # always on script with no arg should always run so you don't really need to add them to the requests
            if "args" in request.alwayson_scripts[alwayson_script_name]:
                # min between arg length in scriptrunner and arg length in the request
                for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(request.alwayson_scripts[alwayson_script_name]["args"]))):
                    script_args[alwayson_script.args_from + idx] = request.alwayson_scripts[alwayson_script_name]["args"][idx]
    return script_args

from modules.api import models
from modules.api.models import PydanticModelGenerator, StableDiffusionTxt2ImgProcessingAPI, StableDiffusionImg2ImgProcessingAPI
from modules import scripts, ui
from modules.processing import process_images, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
def text2imgapi(txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
    script_runner = scripts.scripts_txt2img
    if not script_runner.scripts:
        script_runner.initialize_scripts(False)
        ui.create_ui()
    default_script_arg_txt2img = []
    if not default_script_arg_txt2img:
        default_script_arg_txt2img = init_default_script_args(script_runner)
    selectable_scripts, selectable_script_idx = get_selectable_script(txt2imgreq.script_name, script_runner)
    populate = txt2imgreq.copy(update={  # Override __init__ params
        "sampler_name": validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
        "do_not_save_samples": not txt2imgreq.save_images,
        "do_not_save_grid": not txt2imgreq.save_images,
    })
    if populate.sampler_name:
        populate.sampler_index = None  # prevent a warning later on
    args = vars(populate)
    args.pop('script_name', None)
    args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
    args.pop('alwayson_scripts', None)
    script_args = init_script_args(txt2imgreq, default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)
    send_images = args.pop('send_images', True)
    args.pop('save_images', None)
    p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
    p.scripts = script_runner
    # p.outpath_grids = opts.outdir_txt2img_grids
    # p.outpath_samples = opts.outdir_txt2img_samples
    shared.state.begin()
    if selectable_scripts is not None:
        p.script_args = script_args
        processed = scripts.scripts_txt2img.run(p, *p.script_args) # Need to pass args as list here
    else:
        p.script_args = tuple(script_args) # Need to pass args as tuple here
        processed = process_images(p)
    shared.state.end()
    b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
    return models.TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())
def img2imgapi(img2imgreq: models.StableDiffusionImg2ImgProcessingAPI):
    init_images = img2imgreq.init_images
    if init_images is None:
        return
    mask = img2imgreq.mask
    if mask:
        mask = decode_base64_to_image(mask)
    script_runner = scripts.scripts_img2img
    if not script_runner.scripts:
        script_runner.initialize_scripts(True)
        ui.create_ui()
    default_script_arg_img2img = []
    if not default_script_arg_img2img:
        default_script_arg_img2img = init_default_script_args(script_runner)
    selectable_scripts, selectable_script_idx = get_selectable_script(img2imgreq.script_name, script_runner)
    populate = img2imgreq.copy(update={  # Override __init__ params
        "sampler_name": validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
        "do_not_save_samples": not img2imgreq.save_images,
        "do_not_save_grid": not img2imgreq.save_images,
        "mask": mask,
    })
    if populate.sampler_name:
        populate.sampler_index = None  # prevent a warning later on
    args = vars(populate)
    args.pop('include_init_images', None)  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
    args.pop('script_name', None)
    args.pop('script_args', None)  # will refeed them to the pipeline directly after initializing them
    args.pop('alwayson_scripts', None)
    script_args = init_script_args(img2imgreq, default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)
    send_images = args.pop('send_images', True)
    args.pop('save_images', None)
    p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)
    p.init_images = [decode_base64_to_image(x) for x in init_images]
    p.scripts = script_runner
    # p.outpath_grids = opts.outdir_img2img_grids
    # p.outpath_samples = opts.outdir_img2img_samples
    shared.state.begin()
    if selectable_scripts is not None:
        p.script_args = script_args
        processed = scripts.scripts_img2img.run(p, *p.script_args) # Need to pass args as list here
    else:
        p.script_args = tuple(script_args) # Need to pass args as tuple here
        processed = process_images(p)
    shared.state.end()
    b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
    if not img2imgreq.include_init_images:
        img2imgreq.init_images = None
        img2imgreq.mask = None
    return models.ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=processed.js())


import cv2
import base64
from PIL import Image
from io import BytesIO
def manual_api():
    initialize()
    modules.script_callbacks.before_ui_callback()

    # Read Image in RGB order
    img = cv2.imread('111.jpg')

    # Encode into PNG and send to ControlNet
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')

    # A1111 payload
    payload = {
        "prompt": 'beautiful girl, silver dress, gentle, smiling, sunny',
        "negative_prompt": "",
        "batch_size": 1,
        "steps": 20,
        "cfg_scale": 7,
        "width": 512,
        "height": 768,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": encoded_image,
                        "module": "canny",
                        "model": "control_v11p_sd15_canny [d14c016b]",
                    }
                ]
            }
        }
    }
    # req
    req = StableDiffusionTxt2ImgProcessingAPI(**payload)

    # generate
    resp = text2imgapi(req)
    with open("cnres.txt", 'w') as f:
        json.dump(vars(resp), f)
        # f.write(str(vars(resp)))

def manual_img2img():
    initialize()
    modules.script_callbacks.before_ui_callback()

    # Read Image in RGB order
    img = Image.open('rem111.png')

    # Encode into PNG and send to ControlNet
    encoded_image = base64.b64encode(get_img_bytes(img)).decode('utf-8')

    # A1111 payload
    payload = {
        "init_images": [encoded_image],
        "prompt": '8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, on the beach',
        "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
        "batch_size": 1,
        "steps": 20,
        "cfg_scale": 7,
        "width": 800,
        "height": 800,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": encoded_image,
                        "module": "canny",
                        "model": "control_v11p_sd15_canny [d14c016b]",
                    }
                ]
            }
        }
    }
    # req
    req = StableDiffusionImg2ImgProcessingAPI(**payload)

    # generate
    resp = img2imgapi(req)
    with open("cnres.txt", 'w') as f:
        json.dump(vars(resp), f)

def get_img_bytes(image):
    bytes_data = None
    with BytesIO() as output_bytes:
        image.save(output_bytes, format="PNG")
        bytes_data = output_bytes.getvalue()
    return bytes_data

def get_resp_struct():
    data = {}
    with open("cnres.txt", 'r') as f:
        data = json.load(f)
    for k in data.keys():
        print(k)
    # print(data["parameters"])
    print(data["info"])
    i = 1
    for img in data["images"]:
        # 只取第一张
        print("img length b64: ", len(img))
        img_data = base64.b64decode(img)
        with open("{}.jpg".format(i), 'wb') as f:
            f.write(img_data)
        i += 1
    # for k in data["parameters"].keys():
    #     print(k)

if __name__ == "__main__":
    # manual_api()
    manual_img2img()
    get_resp_struct()
