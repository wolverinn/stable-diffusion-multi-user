# inference handler for huggingface
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

    # if cmd_opts.tls_keyfile is not None and cmd_opts.tls_keyfile is not None:

    #     try:
    #         if not os.path.exists(cmd_opts.tls_keyfile):
    #             print("Invalid path to TLS keyfile given")
    #         if not os.path.exists(cmd_opts.tls_certfile):
    #             print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
    #     except TypeError:
    #         cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
    #         print("TLS setup invalid, running webui without TLS")
    #     else:
    #         print("Running with TLS")

    # make the program just exit at ctrl+c without waiting for anything
    # def sigint_handler(sig, frame):
    #     print(f'Interrupted with signal {sig} in {frame}')
    #     os._exit(0)

    # signal.signal(signal.SIGINT, sigint_handler)

