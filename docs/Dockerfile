FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    FORCE_CUDA="1" \
    DEBIAN_FRONTEND=noninteractive \
    LANG="C.UTF-8"

WORKDIR /root

# own config
COPY myconfig myconfig
RUN mv myconfig/condarc .condarc \
        && mv myconfig/pydistutils.cfg .pydistutils.cfg \
        && mv myconfig/Shanghai /etc/localtime \
        && mkdir -p .ssh \
        && mv myconfig/trans-docker.pub .ssh/ \
        && mv myconfig/sshconfig .ssh/config \
        && rm -r myconfig

# (Optional, use Mirror to speed up downloads)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list

# Install the required packages
RUN apt-get update && apt-get install -y build-essential apt-utils ca-certificates vim git wget ssh curl bzip2 tree htop\
    && apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 bmon iotop g++\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm /etc/apt/sources.list.d/cuda.list

# ssh login
RUN cd /etc/ssh && echo "PermitRootLogin yes" >> sshd_config

# set ssh key
RUN cat .ssh/trans-docker.pub >> .ssh/authorized_keys \
        && echo "/etc/init.d/ssh start" >> ~/.bashrc

# Install Miniconda
RUN wget -q -O Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        && /bin/bash Miniconda.sh -b -p .miniconda \
        && rm Miniconda.sh
ENV PATH /root/.miniconda/bin:$PATH
RUN /bin/bash -c "source .miniconda/bin/activate" && conda init

# Pip set
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple \
        && pip config set install.trusted-host mirrors.aliyun.com

# Install torch
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

ENTRYPOINT bash
