from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
import time
import os
import json
import threading
import requests
import sys
import replicate

# load balancing views
# 假设有10台GPU机器，因为用户的task是在某一台机器上生成的，这10台机器之间并没有做分布式同步
# 所以一旦用户请求到了某一台GPU机器，后续获取进度/停止生成等操作都需要访问这台特定的GPU机器
# 所以在session中记录访问的GPU机器ip，直到请求结束后一段时间再释放

# ================ IMPORTANT: fill the configs below ======================
ip_list = {
    "1.1.1.1": {},
    "2.2.2.2": {},
}
REPLICATE_API_TOKEN = ""
# =========================================================================

KEEP_ALIVE_SECONDS = 60
MAX_TIMEOUT_SECONDS = 600
MAX_REQUESTS_PER_MACHINE = 10

ERR_MAX_REQUEST = "max requests reached, try again later"
PROMPT_MAX_LEN = 600

session_key_lock = threading.Lock()
GLOBAL_SESSION_KEY = 1

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def choose_machine() -> str:
    chosen_ip = ""
    min_cnt = 1000000
    for ip, session_info_list in ip_list.items():
        valid_cnt = 0
        need_del_session_keys = []
        for session_key, session_info in session_info_list.items():
            if get_map_default(session_info, "lb_expire") <= int(time.time()):
                need_del_session_keys.append(session_key)
            else:
                valid_cnt += 1
        for session_key in need_del_session_keys:
            del ip_list[ip][session_key] # expired, need to be deleted
        if valid_cnt <= min_cnt and valid_cnt <= MAX_REQUESTS_PER_MACHINE:
            chosen_ip = ip
            min_cnt = valid_cnt
    return chosen_ip

def use_machine(request, ip):
    if ip not in ip_list.keys():
        return
    session_key = 0
    global GLOBAL_SESSION_KEY
    with session_key_lock:
        if GLOBAL_SESSION_KEY >= sys.maxsize-1:
            GLOBAL_SESSION_KEY = 0
        GLOBAL_SESSION_KEY += 1
        session_key = GLOBAL_SESSION_KEY
    expire = int(time.time()) + MAX_TIMEOUT_SECONDS
    request.session["lb_expire"] = expire
    request.session["ip"] = ip
    request.session["session_key"] = session_key
    ip_list[ip][session_key] = {
        "lb_expire": expire,
    }
    print("ip list: ", ip_list)

def get_ip_for_session(request) -> str:
    expire_time = request.session.get("lb_expire", 0)
    ip = request.session.get("ip", "")
    if expire_time <= int(time.time()) or len(ip) <= 0:
        ip = choose_machine()
        if len(ip) > 0:
            use_machine(request, ip)
    return ip


def routing(request, api_path: str) -> JsonResponse:
    if len(api_path) <= 0:
        return JsonResponse({"err": "api path empty"})
    ip = get_ip_for_session(request)
    if len(ip) <= 0:
        return JsonResponse({"err": ERR_MAX_REQUEST})
    # logic here
    raw_req = request.body.decode('utf-8')
    print("formating request: {}".format(raw_req))
    resp = requests.post("http://{}/{}".format(ip, api_path), data=raw_req)
    if resp.status_code != 200:
        print("routing err, path: {}, resp: {}".format(api_path, resp))
        return JsonResponse({"err": "internal error occurred"})
    resp_json = json.loads(resp.text)

    # finally keep-alive here
    expire = int(time.time()) + KEEP_ALIVE_SECONDS
    request.session["lb_expire"] = expire
    session_key = request.session.get("session_key", 0)
    if ip in ip_list.keys() and session_key > 0 and session_key in ip_list[ip].keys():
        ip_list[ip][session_key] = {
        "lb_expire": expire,
    }
    return JsonResponse(resp_json)

# routing views

def demo_page(request):
    return render(request, "multi_demo.html")

@csrf_exempt
def txt2img(request):
    API_PATH = "txt2img/"
    return routing(request, API_PATH)

@csrf_exempt
def txt2img_with_fallback(request):
    # returns image uri in "img_data"
    API_PATH = "txt2img/"
    resp = routing(request, API_PATH)
    uri = ""
    if resp.get("err", "") == ERR_MAX_REQUEST:
        # fallback to replicate, but lacks:
        # 1.progress check; 2.choose model(todo)
        prompt = request.POST.get("prompt")
        if len(prompt) > PROMPT_MAX_LEN:
            prompt = prompt[:PROMPT_MAX_LEN]
        uri = replicate_chill_watcher_generate(prompt)
    else:
        resp_json = json.loads(resp.content)
        uri = 'data:image/png;base64,' + resp_json["images"][0]
    # print("get fallback txt2img uri: ", uri)
    return JsonResponse({"img_data": uri})

@csrf_exempt
def img2img(request):
    API_PATH = "img2img/"
    return routing(request, API_PATH)

@csrf_exempt
def progress(request):
    API_PATH = "progress/"
    return routing(request, API_PATH)

@csrf_exempt
def interrupt(request):
    API_PATH = "interrupt/"
    return routing(request, API_PATH)

@csrf_exempt
def list_models(request):
    API_PATH = "list_models/"
    return routing(request, API_PATH)
    

def get_map_default(int_map, key: str) -> int:
    if int_map.__contains__(key):
        return int_map[key]
    else:
        return 0


# this is just a demo for txt2img_fallback, replace with your own replicate model
def replicate_chill_watcher_generate(prompt: str) -> str:
    start = time.time()
    output = replicate.run(
        "wolverinn/chill_watcher:53d24c51f11d93e26f88cc53a00b5c392e5eb62272e07c46152af66a14e27cae",
        input={"prompt": prompt}
    )
    end = time.time()
    print("chill watcher replicate cost(milli seconds): {}".format(end*1e3-start*1e3))
    print("chill resp: ", output)
    return output

