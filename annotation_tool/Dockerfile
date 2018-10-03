ARG DOCKER_BASE_IMAGE=ubuntu:14.04

FROM $DOCKER_BASE_IMAGE

MAINTAINER Daniel Hofer <hoferda@hs-weingarten.de>

RUN apt-get update && \
    apt-get -y install \
    git \
    vim

SHELL ["/bin/bash", "-c"]

# Install ros indigo desktop full
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116 && \
    apt-get update && \
    apt-get -y install ros-indigo-desktop-full && \
    apt-get -y install ros-indigo-pcl-ros && \
    apt-get -y install ros-indigo-pcl-conversions && \
    apt-get -y install ros-indigo-pcl-msgs && \
    apt-get -y install libvtk5.8-qt4 && \
    apt-get -y install libvtk5-qt4-dev

RUN mkdir -p /root/catkin_ws/src/prj-iki-robotics/

WORKDIR /root/catkin_ws/src/prj-iki-robotics/
RUN git clone https://github.com/strands-project/annotation_tool_kth.git

RUN cd ~/catkin_ws && . /opt/ros/indigo/setup.bash && catkin_make && source devel/setup.bash

WORKDIR /root/catkin_ws/src/prj-iki-robotics

CMD ["~/catkin_ws/devel/lib/annotation_tool/annotation_tool"]
