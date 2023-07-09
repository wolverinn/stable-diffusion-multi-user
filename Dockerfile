FROM python:3.10-slim

# install docker on ubuntu: https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository
# build image: docker build -t sd-multi .
# docker image ls
# dockerhub create repo
# login on ubuntu: docker login
# tag: docker tag sd-multi wolverinn/sd_multi_demo:v1
# push: docker push wolverinn/sd_multi_demo:v1

WORKDIR /

COPY requirements.txt /
# COPY torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl .

RUN apt-get update && apt-get install -y libgl1-mesa-glx  && \
    apt-get install -y libglib2.0-0 && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install runpod==0.9.12 && \
    pip3 install -r requirements.txt && \
    # pip3 install torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl && \
    pip3 install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    # apt-get remove -y git && \
    # rm -rf torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl && \
    pip3 cache purge

COPY . /

# COPY . /

# ENTRYPOINT ["/docker_entrypoint.sh"]

CMD [ "python", "-u", "/handler.py" ]
