# stable-diffusion docker slim

# Features

- deploy on runpod serverless
- autoscaling with highly customized scaling strategy
- text2img, img2img, list models API
- upload models at any time, takes effect immediately
- if you want ControlNet&other extensions & up-to-date sd-webui features & full webui API support & torch-2.2, I've packed a docker image ready to use, checkout my article [here](https://mp.weixin.qq.com/s/qA3Mehdbi9lrszdmc3-0Yw)

# Deploy Steps

## Build and upload docker
Note: you can also skip this step by using the docker image I've already packed: `wolverinn/sd_multi_demo:v101`

- get a Ubuntu machine with GPU, download the project files
- install docker on ubuntu: https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository
- cd to the directory with `Dockerfile`
    - to make your custom docker, modify `handler.py` for uploading outputs & supporting more API ...
- `docker build -t sd-multi .`
- `docker login`
- docker tag and then docker push

## Create a storage on runpod
- upload model files to storage volume. under `/workspace` you should upload the whole `models` directory. The directory structure should look like this:

```
/workspace
    - /models
        - /VAE
        - /Lora
        - /Stable-diffusion
        ...
```

## Deploy on runpod serverless
check the official guide: https://docs.runpod.io/docs/template-creation

1. create a template, and in "container image", use your own docker image or my demo image: `wolverinn/sd_multi_demo:v101`
2. create a serverless endpoint. In `advanced-select network volume`, use your storage created before. In `container configuration-select template`, use the template created before.

# Test your API
check out `test_runpod.py`

## API definition
checkout `idl.yaml`