# Stable Diffusion Multi-user
> stable diffusion multi-user django server code with multi-GPU load balancing 

# Features

- a django server that provides stable-diffusion http API, including:
    - txt2img, img2img(todo)
    - check generating progress
    - interrupt generating
    - list available models
    - ...
- supports civitai models and lora, etc.
- supports multi-user queuing
- supports multi-user separately changing models, and won't affect each other
- provides downstream load-balancing server code that automatically do load-balancing among available GPU servers, and ensure that user requests are sent to the same server within one generation cycle

You can build your own UI, community features, account login&payment, etc. based on these functions!

# Project directory structure

The project can be roughly divided into two parts: django server code, and stable-diffusion-webui code that we use to initialize and run models. And I'll mainly explain the django server part.

In the main project directory:

- `modules/`: stable-diffusion-webui modules
- `sd_multi/`: the django project name
    - `urls.py`: server API path configuration
- `simple/`: the main django code
    - `views.py`: main API processing logic
    - `lb_views.py`: load-balancing API
- `requirements.txt`: stable diffusion pip requirements
- `setup.sh`: run it with options to setup the server environment
- `gen_http_conf.py`: called in `setup.sh` to setup the apache configuration

# Deploy on a GPU server

# Deploy the load-balancing server