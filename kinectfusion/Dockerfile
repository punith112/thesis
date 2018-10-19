ARG DOCKER_BASE_IMAGE=nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

FROM $DOCKER_BASE_IMAGE

MAINTAINER Sashidhar Kanuboddi <sashi2295@gmail.com>

WORKDIR /root

RUN apt-get update && \
    apt-get -y install \
    libopenni2-0 libopenni2-dev openni2-utils \
    git \
    vim \
    cmake \
    software-properties-common # to enable the add-apt-repos command

RUN apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE && \
    add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u && \
    apt-get update && \
    apt-get -y install \
    librealsense2-dev librealsense2-dbg librealsense2-utils

COPY installOpenCV.sh /root

RUN chmod +x installOpenCV.sh && \
    ./installOpenCV.sh

RUN git clone https://github.com/eigenteam/eigen-git-mirror.git && \
    cd eigen-git-mirror && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install && \
    cd ../..

RUN git clone --recursive https://github.com/chrdiller/KinectFusionApp.git && \
    cd KinectFusionApp && \
    cmake . && \
    make

RUN echo "PS1='🐳  \[\033[1;36m\]\h \[\033[1;34m\]\W\[\033[0;35m\] \[\033[1;36m\]# \[\033[0m\]'" >> /root/.bashrc # https://wiki.ubuntuusers.de/Bash/Prompt/
