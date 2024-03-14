# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
# replace background with controlnet: https://replicate.com/wolverinn/realistic-background

from cog import BasePredictor, BaseModel, Input, Path
from typing import Optional, List, Dict

import os
import requests
import uuid
import base64
import warnings
from io import BytesIO
import json
from PIL import Image
os.environ["IGNORE_CMD_ARGS_ERRORS"] = "true"
from modules.shared_cmd_options import cmd_opts
from modules import initialize_util, shared, timer
# from webui import startup_timer
import importlib
import sys
import warnings
from threading import Thread

from modules.timer import startup_timer
import modules.ui

class Output(BaseModel):
    payload: Dict
    images: List[Path]


def initialize_helper_v1(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    from modules.shared_cmd_options import cmd_opts

    from modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    from modules import extensions
    extensions.list_extensions()
    startup_timer.record("list extensions")

    initialize_util.restore_config_state_file()
    startup_timer.record("restore config state file")

    from modules import shared, upscaler, scripts
    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        scripts.load_scripts()
        return

    from modules import sd_models
    sd_models.list_models()
    startup_timer.record("list SD models")

    from modules import localization
    localization.list_localizations(cmd_opts.localizations_dir)
    startup_timer.record("list localizations")

    with startup_timer.subcategory("load scripts"):
        scripts.load_scripts()

    if reload_script_modules:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    from modules import modelloader
    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    from modules import sd_vae
    sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    from modules import textual_inversion
    textual_inversion.textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    from modules import script_callbacks, sd_hijack_optimizations, sd_hijack
    script_callbacks.on_list_optimizers(sd_hijack_optimizations.list_optimizers)
    sd_hijack.list_optimizers()
    startup_timer.record("scripts list_optimizers")

    from modules import sd_unet
    sd_unet.list_unets()
    startup_timer.record("scripts list_unets")

    def load_model():
        """
        Accesses shared.sd_model property to load model.
        After it's available, if it has been loaded before this access by some extension,
        its optimization may be None because the list of optimizaers has neet been filled
        by that time, so we apply optimization again.
        """
        from modules import devices
        devices.torch_npu_set_device()

        shared.sd_model  # noqa: B018

        if sd_hijack.current_optimizer is None:
            sd_hijack.apply_optimizations()

        devices.first_time_calculation()
    if not shared.cmd_opts.skip_load_model_at_start:
        Thread(target=load_model).start()

    from modules import shared_items
    shared_items.reload_hypernetworks()
    startup_timer.record("reload hypernetworks")

    from modules import ui_extra_networks
    ui_extra_networks.initialize()
    ui_extra_networks.register_default_pages()

    from modules import extra_networks
    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    startup_timer.record("initialize extra networks")

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # sd webui
        initialize_util.fix_torch_version()
        initialize_util.fix_asyncio_event_loop_policy()
        initialize_util.validate_tls_options()
        initialize_util.configure_sigint_handler()
        initialize_util.configure_opts_onchange()

        from modules import sd_models
        sd_models.setup_model()
        startup_timer.record("setup SD model")

        from modules.shared_cmd_options import cmd_opts

        from modules import codeformer_model
        warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
        codeformer_model.setup_model(cmd_opts.codeformer_models_path)
        startup_timer.record("setup codeformer")

        from modules import gfpgan_model
        gfpgan_model.setup_model(cmd_opts.gfpgan_models_path)
        startup_timer.record("setup gfpgan")

        initialize_helper_v1(reload_script_modules=False)
        from modules import script_callbacks
        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = modules.ui.create_ui()
        startup_timer.record("create ui")
        app, local_url, share_url = shared.demo.launch(
            share=cmd_opts.share,
            server_name=initialize_util.gradio_server_name(),
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            ssl_verify=cmd_opts.disable_tls_verify,
            debug=cmd_opts.gradio_debug,
            auth=None,
            inbrowser=False,
            prevent_thread_lock=True,
            allowed_paths=cmd_opts.gradio_allowed_path,
            app_kwargs={
                "docs_url": "/docs",
                "redoc_url": "/redoc",
            },
            root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
        )
        startup_timer.record("gradio launch")
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
        initialize_util.setup_middleware(app)
        from modules.api.api import Api
        from modules.call_queue import queue_lock
        api = Api(app, queue_lock)
        with startup_timer.subcategory("app_started_callback"):
            script_callbacks.app_started_callback(shared.demo, app)
        timer.startup_record = startup_timer.dump()
        print(f"Startup time: {startup_timer.summary()}.")

    def predict(
        self,
        image: Path = Input(description="Image"),
        prompt: str = Input(description="prompt en", default="RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"),
        negative_prompt: str = Input(description="negative prompt", default="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"),
        sampler_name: str = Input(description="sampler name", default="DPM++ SDE Karras", choices=['DPM++ 2M Karras', 'DPM++ SDE Karras', 'DPM++ 2M SDE Exponential', 'DPM++ 2M SDE Karras', 'Euler a', 'Euler', 'LMS', 'Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a', 'DPM++ 2M', 'DPM++ SDE', 'DPM++ 2M SDE', 'DPM++ 2M SDE Heun', 'DPM++ 2M SDE Heun Karras', 'DPM++ 2M SDE Heun Exponential', 'DPM++ 3M SDE', 'DPM++ 3M SDE Karras', 'DPM++ 3M SDE Exponential', 'DPM fast', 'DPM adaptive', 'LMS Karras', 'DPM2 Karras', 'DPM2 a Karras', 'DPM++ 2S a Karras']),
        steps: int = Input(description="steps", default=20),
        cfg_scale: int = Input(description="cfg scale", default=7),
        denoising_strength: float = Input(description="denoising strength", default=0.75),
        seed: int = Input(description="seed", default=-1),
    ) -> Output:
        """Run a single prediction on the model"""
        img_data = Image.open(image)
        input_encoded = base64.b64encode(get_img_bytes(img_data)).decode('utf-8')
        # A1111 payload
        payload = {
            "init_images": [
                input_encoded
            ],
            "resize_mode": 0,
            "denoising_strength": denoising_strength,
            # "mask": encoded_image,
            # "mask_blur": 2,
            # "inpainting_fill": 2,
            # "inpaint_full_res": True,
            # "inpaint_full_res_padding": only_masked_padding_pixels,
            # "inpainting_mask_invert": 1,
            # "initial_noise_multiplier": 0,
            "prompt": prompt,
            "seed": seed,
            "sampler_name": sampler_name,
            "batch_size": 1,
            "n_iter": 1,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": img_data.size[0],
            "height": img_data.size[1],
            # "restore_faces": false,
            # "tiling": false,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            "negative_prompt": negative_prompt,
            "alwayson_scripts": {},
        }
        # generate
        resp = requests.post("http://127.0.0.1:7860/sdapi/v1/img2img", json=payload)
        resp_payload = resp.json()

        cnres_img = input_encoded
        if "images" in resp_payload.keys() and len(resp_payload['images']) > 0:
            cnres_img = resp_payload['images'][0]
        gen_bytes = BytesIO(base64.b64decode(cnres_img))
        gen_data = Image.open(gen_bytes)
        filename = "{}.png".format(uuid.uuid1())
        gen_data.save(fp=filename, format="PNG")
        # output dict
        resp_payload.pop('images', None)
        resp_payload.pop('parameters', None)
        output_dict = Output(payload=resp_payload, image=Path(filename))
        return output_dict

def get_img_bytes(image):
    bytes_data = None
    with BytesIO() as output_bytes:
        image.save(output_bytes, format="PNG")
        bytes_data = output_bytes.getvalue()
    return bytes_data
