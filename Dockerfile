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

RUN apt-get update && apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 && \
    pip3 install runpod 
    
RUN apt-get update && apt-get install -y git && \
    pip3 install -r requirements.txt && \
    pip3 install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

COPY configs* /
COPY extensions-builtin* /
COPY modules* /
COPY repositories* /
COPY scripts* /
COPY handler.py /
COPY config.json /

COPY docker_entrypoint.sh /
RUN chmod +x docker_entrypoint.sh

# COPY . /

ENTRYPOINT ["/docker_entrypoint.sh"]

CMD [ "python", "-u", "/handler.py" ]