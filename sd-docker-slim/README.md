# stable-diffusion docker slim

# Features

- deploy on runpod serverless
- autoscaling with highly customized scaling strategy
- text2img, img2img, list models
- upload models at any time, takes effect immediately
- if you want ControlNet&other extensions&up-to-date sd-webui API&torch-2.0, checkout my article [here](https://mp.weixin.qq.com/s/qA3Mehdbi9lrszdmc3-0Yw)

# Deploy Steps

## Build and upload docker
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

# Test your API
check out `test_runpod.py`

## API definition
checkout `idl.yaml`