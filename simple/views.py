from django.shortcuts import render
from django.http import HttpResponse,JsonResponse,HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
import time
import os
import json
from webui_v3 import initialize, text2imgapi, img2imgapi
import threading

from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.api.models import StableDiffusionTxt2ImgProcessingAPI, StableDiffusionImg2ImgProcessingAPI
from modules.api.api import encode_pil_to_base64_str
import modules
from io import BytesIO
import base64
from modules import shared, sd_models
from PIL import Image

# Create your views here.

os.environ["IGNORE_CMD_ARGS_ERRORS"] = "true"
initialize()
queue_lock = threading.Lock() # 一台机器只能同时处理一个请求，所以这个lock是全局的

# class _txt2img_request(BaseModel):
#     prompt: str

# class _DefaultOutputData(BaseModel):
#     img_data: str
#     parameters: str

# ***************** stable web *****************

def homepage(request):
    return render(request, "home.html")

@csrf_exempt
def txt2img_v2(request):
    raw_req = request.body.decode('utf-8')
    req_json = json.loads(raw_req)
    
    sd_req = StableDiffusionTxt2ImgProcessingAPI(**req_json)
    resp = text2imgapi(sd_req)

    return JsonResponse(vars(resp))

@csrf_exempt
def img2img_v2(request):
    raw_req = request.body.decode('utf-8')
    req_json = json.loads(raw_req)
    
    sd_req = StableDiffusionImg2ImgProcessingAPI(**req_json)
    resp = img2imgapi(sd_req)

    return JsonResponse(vars(resp))

@csrf_exempt
def progress_v2(request):
    raw_req = request.body.decode('utf-8')
    req_json = json.loads(raw_req)
    if shared.state.job_count == 0:
        return JsonResponse({"progress": 0, "eta": 0})

    # avoid dividing zero
    progress = 0.01

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    time_since_start = time.time() - shared.state.time_start
    eta = (time_since_start/progress)
    eta_relative = eta-time_since_start

    progress = min(progress, 1)
    return JsonResponse({"progress": progress, "eta": eta_relative})

@csrf_exempt
def interrupt_v2(request):
    raw_req = request.body.decode('utf-8')
    req_json = json.loads(raw_req)
    shared.state.interrupt()
    return JsonResponse({"msg": "success"})


@csrf_exempt
def txt2img(request):
    raw_req = request.body.decode('utf-8')
    req_json = json.loads(raw_req)
    args = {
        # "outpath_samples": "C:\\Users\\wolvz\\Desktop",
        "prompt": "lora:koreanDollLikeness_v15:0.66, best quality, ultra high res, (photorealistic:1.4), 1girl, beige sweater, black choker, smile, laughing, bare shoulders, solo focus, ((full body), (brown hair:1), looking at viewer",
        "negative_prompt": "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, (ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.331), blurry, 3hands,4fingers,3arms, bad anatomy, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts,poorly drawn face,mutation,deformed",
        "sampler_name": "DPM++ SDE Karras",
        "steps": 20, # 25
        "cfg_scale": 8,
        "width": 512,
        "height": 768,
        "seed": -1,
        "do_not_save_samples": True,
        "do_not_save_grid": True,
        "restore_faces": False, # 面部修复
        "n_iter": 1, # 生成批次
    }
    if req_json.get("prompt", "") != "":
        args["prompt"] = req_json["prompt"]
    if req_json.get("negative_prompt", "") != "":
        args["negative_prompt"] = req_json["negative_prompt"]
    if req_json.get("sampler_name", "") != "":
        args["sampler_name"] = req_json["sampler_name"]
    if req_json.get("steps", 0) > 0:
        args["steps"] = req_json["steps"]
    if req_json.get("cfg_scale", 0) > 0:
        args["cfg_scale"] = req_json["cfg_scale"]
    if req_json.get("width", 0) > 0:
        args["width"] = req_json["width"]
    if req_json.get("height", 0) > 0:
        args["height"] = req_json["height"]
    if req_json.get("seed", 0) > 0:
        args["seed"] = req_json["seed"]
    if req_json.get("restore_faces", 0) > 0:
        args["restore_faces"] = True
    if req_json.get("n_iter", 0) > 0:
        args["n_iter"] = req_json["n_iter"]
    print("get request: ", args)

    # some parameters
    task_id = req_json.get("task_id", "")
    if len(task_id) <= 0:
        return JsonResponse({"err": "task required"})
    model_name = req_json.get("model", "")

    with queue_lock:
        if len(model_name) > 0:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(dir_path, "../models/Stable-diffusion", model_name)
            checkpoint_info = sd_models.CheckpointInfo(filename)
            sd_models.reload_model_weights(info=checkpoint_info)

        p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
        shared.state.begin()
        shared.state.task_id = task_id
        processed = process_images(p)
        shared.state.end()
        # single_image_b64 = encode_pil_to_base64(processed.images[0]).decode('utf-8')
        b64images = list(map(encode_pil_to_base64_str, processed.images))

    return JsonResponse({
            # "img_data": single_image_b64,
            "images": b64images,
            "parameters": processed.js(),
        })

@csrf_exempt
def img2img(request):
    raw_req = request.body.decode('utf-8')
    req_json = json.loads(raw_req)
    args = {
        # "outpath_samples": "C:\\Users\\wolvz\\Desktop",
        "prompt": "lora:koreanDollLikeness_v15:0.66, best quality, ultra high res, (photorealistic:1.4), 1girl, beige sweater, black choker, smile, laughing, bare shoulders, solo focus, ((full body), (brown hair:1), looking at viewer",
        "negative_prompt": "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, (ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.331), blurry, 3hands,4fingers,3arms, bad anatomy, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts,poorly drawn face,mutation,deformed",
        "sampler_name": "DPM++ SDE Karras",
        "steps": 20, # 25
        "cfg_scale": 8,
        "width": 512,
        "height": 768,
        "seed": -1,
        "do_not_save_samples": True,
        "do_not_save_grid": True,
        "restore_faces": False, # 面部修复
        "n_iter": 1, # 生成批次
    }
    if req_json.get("prompt", "") != "":
        args["prompt"] = req_json["prompt"]
    if req_json.get("negative_prompt", "") != "":
        args["negative_prompt"] = req_json["negative_prompt"]
    if req_json.get("sampler_name", "") != "":
        args["sampler_name"] = req_json["sampler_name"]
    if req_json.get("steps", 0) > 0:
        args["steps"] = req_json["steps"]
    if req_json.get("cfg_scale", 0) > 0:
        args["cfg_scale"] = req_json["cfg_scale"]
    if req_json.get("width", 0) > 0:
        args["width"] = req_json["width"]
    if req_json.get("height", 0) > 0:
        args["height"] = req_json["height"]
    if req_json.get("seed", 0) > 0:
        args["seed"] = req_json["seed"]
    if req_json.get("restore_faces", 0) > 0:
        args["restore_faces"] = True
    if req_json.get("n_iter", 0) > 0:
        args["n_iter"] = req_json["n_iter"]
    print("get request: ", args)

    # img2img params
    resize_mode = req_json.get("resize_mode", 0)
    if resize_mode < 0 or resize_mode > 3:
        # 0-just resize，1-crop and resize，2-resize and fill, 3-just resize(latent upscale)
        resize_mode = 0
    denoising_strength = req_json.get("denoising_strength", 0.75)
    init_b64_images = req_json.get("init_images", [])
    if len(init_b64_images) <= 0:
        return JsonResponse({"err": "image required"})
    init_images = []
    for img_b64 in init_b64_images:
        img_bytes = base64.b64decode(img_b64)
        img_io = BytesIO(img_bytes)
        img = Image.open(img_io)
        init_images.append(img)

    # some parameters
    task_id = req_json.get("task_id", "")
    if len(task_id) <= 0:
        return JsonResponse({"err": "task required"})
    model_name = req_json.get("model", "")

    with queue_lock:
        if len(model_name) > 0:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(dir_path, "../models/Stable-diffusion", model_name)
            checkpoint_info = sd_models.CheckpointInfo(filename)
            sd_models.reload_model_weights(info=checkpoint_info)

        p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, init_images=init_images, resize_mode=resize_mode, denoising_strength=denoising_strength, **args)
        shared.state.begin()
        shared.state.task_id = task_id
        processed = process_images(p)
        shared.state.end()
        # single_image_b64 = encode_pil_to_base64(processed.images[0]).decode('utf-8')
        b64images = list(map(encode_pil_to_base64_str, processed.images))

    return JsonResponse({
            # "img_data": single_image_b64,
            "images": b64images,
            "parameters": processed.js(),
        })

@csrf_exempt
def progress(request):
    raw_req = request.body.decode('utf-8')
    req_json = json.loads(raw_req)
    task_id = req_json.get("task_id", "")
    if len(task_id) <= 0:
        return JsonResponse({"err": "invalid task"})
    if shared.state.job_count == 0:
        return JsonResponse({"progress": 0, "eta": 0})

    # check task_id
    if shared.state.task_id != task_id:
        return JsonResponse({"progress": 0, "eta": 0})
    
    progress = 0.01 # avoid dividing zero
    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    time_since_start = time.time() - shared.state.time_start
    eta = (time_since_start/progress)
    eta_relative = eta-time_since_start

    progress = min(progress, 1)
    return JsonResponse({"progress": progress, "eta": eta_relative})

@csrf_exempt
def interrupt(request):
    raw_req = request.body.decode('utf-8')
    req_json = json.loads(raw_req)
    task_id = req_json.get("task_id", 0)
    if len(task_id) <= 0:
        return JsonResponse({"err": "invalid task"})
    if shared.state.task_id != task_id:
        return JsonResponse({"msg": "no match task"})
    shared.state.interrupt()
    return JsonResponse({"msg": "success"})

@csrf_exempt
def list_models(request):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, "../models/Stable-diffusion")
    model_list = []
    for file in os.listdir(model_path):
        if file.split('.')[-1] == "txt":
            continue
        model_list.append(file)
    return JsonResponse({"models": model_list})

