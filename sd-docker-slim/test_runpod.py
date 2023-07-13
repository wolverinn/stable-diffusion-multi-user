import requests
import time
import json
import base64

PROMPT_CHILL_DEFAULT = '''
(8k, RAW photo, best quality, masterpiece:1.2), (realistic, photo-realistic:1.37),<lora:koreanDollLikeness_v10:0.5> <lora:stLouisLuxuriousWheels_v1:1>,st. louis (luxurious wheels) ,1girl,(Kpop idol), (aegyo sal:1),hair ornament, portrait, necklace,cute, night, professional lighting, photon mapping, radiosity, physically-based rendering, thighhighs, smile, pose, silver hair,sheer sleeveless white shirt, white skirt, white pantyhose,cat ear,room, iso 950, HDR+, white balance
'''
endpoint = "k8t98pqb2q4ny4"
RUNPOD_TOKEN = ""

def chill_watcher_generate(prompt: str):
    url = "https://api.runpod.ai/v2/{}/run".format(endpoint)
    form_data = {
        "input": {
            "api": "txt2img",
            "prompt": prompt,
            "model": ""
        }
    }
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer "+RUNPOD_TOKEN,
    }
    resp = requests.post(url, headers=header, json=form_data)
    if resp.status_code != 200:
        print("chill watcher error: ", resp.content)
        return ""
    resp_data = json.loads(resp.content)
    print(resp_data)
    gen_id = resp_data["id"]
    while True:
        status = get_status(gen_id)
        status_json = json.loads(status)
        if status_json.get("executionTime", "") != "":
            print("job finished, cost(milli seconds): ", status_json["executionTime"])
        if status_json.get("output", "") != "":
            if status_json["output"].get("err", "") != "" or status_json["output"].get("msg", "") != "":
                print("get error: ", status_json["output"])
                return ""
            print("generate params: ", status_json["output"]["parameters"])
            img_b64 = status_json["output"]["img_data"]
            img_data = base64.b64decode(img_b64)
            with open("1.png", 'wb') as f:
                f.write(img_data)
            return ""
        else:
            print(status)
        time.sleep(1.5)

def get_status(id: str):
    url = "https://api.runpod.ai/v2/{}/status/".format(endpoint) + id
    header = {
        "Authorization": "Bearer "+RUNPOD_TOKEN,
    }
    resp = requests.post(url, headers=header)
    if resp.status_code != 200:
        print("chill watcher error: ", resp.content)
        return ""
    return resp.content

def get_models():
    url = "https://api.runpod.ai/v2/{}/runsync".format(endpoint)
    form_data = {
        "input": {
            "api": "list_models",
        }
    }
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer "+RUNPOD_TOKEN,
    }
    resp = requests.post(url, headers=header, json=form_data)
    if resp.status_code != 200:
        print("chill watcher error: ", resp.content)
        return ""
    print(resp.content)

# 测试文生图
chill_watcher_generate(PROMPT_CHILL_DEFAULT)

# 测试获取可用模型
get_models()
