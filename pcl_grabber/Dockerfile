ARG DOCKER_BASE_IMAGE=141.69.58.19:5000/iki/ros-docker-base:kinetic

FROM $DOCKER_BASE_IMAGE

ARG ROS_DISTRO=kinetic

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get clean && \
    apt-get update && apt-get -y install \
    ros-$ROS_DISTRO-openni2-launch \
    ros-$ROS_DISTRO-rviz
