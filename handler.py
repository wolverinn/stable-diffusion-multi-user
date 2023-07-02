import runpod
import os
import time

import os
import sys
import signal
import re
from typing import Dict, List, Any
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.gzip import GZipMiddleware
from packaging import version

import logging
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

from modules import errors
from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call

import torch

# Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

from modules import shared, devices, ui_tempdir, extensions
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.sd_vae
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork
from modules import extra_networks, extra_networks_hypernet
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
from modules.api.api import encode_pil_to_base64


def initialize():
    # check_versions()

    extensions.list_extensions()
    # localization.list_localizations(cmd_opts.localizations_dir)

    # if cmd_opts.ui_debug_mode:
    #     shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
    #     modules.scripts.load_scripts()
    #     return

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)

    modelloader.list_builtin_upscalers()
    modules.scripts.load_scripts()
    modelloader.load_upscalers()

    modules.sd_vae.refresh_vae_list()

    # modules.textual_inversion.textual_inversion.list_textual_inversion_templates()

    try:
        modules.sd_models.load_model()
    except Exception as e:
        errors.display(e, "loading stable diffusion model")
        print("", file=sys.stderr)
        print("Stable diffusion model failed to load, exiting", file=sys.stderr)
        exit(1)

    shared.opts.data["sd_model_checkpoint"] = shared.sd_model.sd_checkpoint_info.title

    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()))
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)

    # shared.reload_hypernetworks()

    # ui_extra_networks.intialize()
    # ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
    # ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
    # ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

    extra_networks.initialize()
    extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())
    modules.script_callbacks.before_ui_callback()

## load your model(s) into vram here
initialize()

def handler(event):
    # do the things
    input_data = event["input"]

    args = {
        "do_not_save_samples": True,
        "do_not_save_grid": True,
        "outpath_samples": "./output",
        "prompt": "lora:koreanDollLikeness_v15:0.66, best quality, ultra high res, (photorealistic:1.4), 1girl, beige sweater, black choker, smile, laughing, bare shoulders, solo focus, ((full body), (brown hair:1), looking at viewer",
        "negative_prompt": "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, (ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.331), blurry, 3hands,4fingers,3arms, bad anatomy, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts,poorly drawn face,mutation,deformed",
        "sampler_name": "DPM++ SDE Karras",
        "steps": 20, # 25
        "cfg_scale": 8,
        "width": 512,
        "height": 768,
        "seed": -1,
    }
    if input_data.get("prompt", "") != "":
        prompt = input_data["prompt"]
        args["prompt"] = prompt
    if input_data.get("negative_prompt", "") != "":
        args["negative_prompt"] = input_data["negative_prompt"]
    if input_data.get("sampler_name", "") != "":
        args["sampler_name"] = input_data["sampler_name"]
    if input_data.get("steps", 0) > 0:
        args["steps"] = input_data["steps"]
    if input_data.get("cfg_scale", 0) > 0:
        args["cfg_scale"] = input_data["cfg_scale"]
    if input_data.get("width", 0) > 0:
        args["width"] = input_data["width"]
    if input_data.get("height", 0) > 0:
        args["height"] = input_data["height"]
    if input_data.get("seed", 0) > 0:
        args["seed"] = input_data["seed"]
    p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
    processed = process_images(p)
    single_image_b64 = encode_pil_to_base64(processed.images[0]).decode('utf-8')
    return {
        "img_data": single_image_b64,
        "parameters": processed.images[0].info.get('parameters', ""),
    }


runpod.serverless.start({
    "handler": handler
})