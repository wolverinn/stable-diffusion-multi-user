- `/txt2img/`: try the txt2img with stable diffusion
```
// demo request
task_id: required string,
model: optional string, // change model with this param
prompt: optional string,
negative_prompt: optional string,
sampler_name: optional string,
steps: optional int, // default=20
cfg_scale: optional int, // default=8
width: optional int, // default=512
height: optional int, // default=768
seed: optional int // default=-1
restore_faces: optional int // default=0
n_iter: optional int // default = 1
// ...
// modify views.py for more optional parameters

// response
images: list<string>, // image base64 data list
parameters: string
```

- `/img2img`: stable diffusion img2img
```
// demo request
task_id: required string,
model: optional string, // change model with this param
prompt: optional string,
negative_prompt: optional string,
sampler_name: optional string,
steps: optional int, // default=20
cfg_scale: optional int, // default=8
width: optional int, // default=512
height: optional int, // default=768
seed: optional int // default=-1
restore_faces: optional int // default=0
n_iter: optional int // default = 1
resize_mode: optional int // default=0
denoising_strength: optional double // default=0.75
init_images: optional list<base64 image data>
// ...
// modify views.py for more optional parameters

// response
images: list<string>, // image base64 data list
parameters: string
```

- `/progress/`: get the generation progress
```
// request
task_id: required string

// response
progress: float, // progress percentage
eta: float, // eta seconds
```

- `/interrupt/`: terminate an unfinished generation
```
// request
task_id: required string
```

- `/list_models/`: list available models
```
// response
models: list<string>
```