# Docker for ur3 repo
# ros-melodic-base, gazebo9, gps, python libraries 
# Python 3 version

ARG UBUNTU_VERSION=18.04

FROM tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer Cristian Beltran "beltran@hlab.sys.es.osaka-u.ac.jp"

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
#### Use an official ROS runtime as a parent image
# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO melodic
RUN apt-get update && apt-get install -y \
    ros-melodic-ros-core=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt-get update && apt-get install -y \
    ros-melodic-ros-base=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# install universal robot ros packages
RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-gazebo-ros-pkgs \
    ros-$ROS_DISTRO-spacenav-node \
    ros-$ROS_DISTRO-rqt-common-plugins \
    # install catkin
    ros-$ROS_DISTRO-catkin \
    python-catkin-tools \
    # Install Numpy Boost
    libboost-dev \
    libboost-python-dev \
    libboost-system-dev \
    libboost-all-dev \
    libatlas-base-dev \
    libprotobuf-dev \
    protobuf-compiler \
    # python dependencies
    python-setuptools \
    python3-tk \
    python3-numpy \
    # utils
    locate \
    aptitude \
    vim htop tmux \
    curl wget \
    tk \
    spacenavd \
    && rm -rf /var/lib/apt/lists/*

### Gazebo ###
# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys D2486D2DD83DB69272AFE98867170598AF249743

# setup sources.list
RUN . /etc/os-release \
    && echo "deb http://packages.osrfoundation.org/gazebo/$ID-stable `lsb_release -sc` main" > /etc/apt/sources.list.d/gazebo-latest.list

ENV DEBIAN_FRONTEND noninteractive

# install gazebo packages
RUN apt-get update && apt-get install -q -y \
    binutils \
    mesa-utils \
    module-init-tools \
    x-window-system \
    gazebo9 \
    libgazebo9-dev \
    && rm -rf /var/lib/apt/lists/*

ADD ./docker/orocos_kinematics_dynamics.tar.xz /root/

# Install SIP 4.19.8
ADD ./docker/sip-4.19.8.tar.gz /root/

RUN /bin/bash -c "cd ~/ \
               && cd sip-4.19.8 \
               && python configure.py \
               && make -j4 && make install"

# Install PyKDL
RUN apt update && apt -y install libeigen3-dev && rm -rf /var/lib/apt/lists/*
RUN /bin/bash -c "cd ~/orocos_kinematics_dynamics/orocos_kdl \
               && mkdir build && cd build \
               && cmake -DCMAKE_BUILD_TYPE=Release .. \
               && make -j4 && make install"

RUN /bin/bash -c "cd ~/orocos_kinematics_dynamics/python_orocos_kdl \
               && mkdir build && cd build \
               && cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DPYTHON_VERSION=3 .. \
               && make -j4"

COPY ./docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

### Ros Workspace ###
# Set up the workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
               && mkdir -p ~/ros_ws/src \
               && cd ~/ros_ws/ \
               && catkin init"

# Installing repo required for homework
RUN /bin/bash -c "cd ~/ros_ws/src \
               && git clone https://bitbucket.org/traclabs/trac_ik.git \
               && git clone https://github.com/cambel/ur3.git ros_ur3 \
               && git clone https://github.com/cambel/ur_openai_gym.git \
               && git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git universal_robots_ros_driver \
               && git clone -b calibration_devel https://github.com/fmauch/universal_robot.git universal_robot"

# Updating ROSDEP and installing dependencies
RUN cd ~/ros_ws \
    && rosinstall ~/ros_ws/src /opt/ros/$ROS_DISTRO src/ros_ur3/dependencies.rosinstall \
    && apt-get update \
    && rosdep fix-permissions \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -y \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONIOENCODING UTF-8
RUN /bin/bash -c "cd ~/ros_ws/src/gazebo_ros_link_attacher \
                  && git checkout melodic-devel \
                  && cd ~/ros_ws/src/robotiq \
                  && git fetch origin && git checkout melodic-devel"

## Python libraries ##
RUN python -m pip install pip --upgrade && \
    pip install matplotlib==2.2.3 spicy protobuf pyyaml pyquaternion rospkg \
    lxml tqdm catkin-pkg empy PyVirtualDisplay defusedxml gym psutil pyprind

# Compiling ros workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
               && cd ~/ros_ws/ \
               && rm -rf build \
               && catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so -DPYTHON_VERSION=3"

################################################
# Custom python libs
################################################
RUN apt-get update && apt-get -y upgrade && rm -rf /var/lib/apt/lists/*

RUN pip install Cython cpprb

# ur_ikfast
RUN /bin/bash -c "cd ~/ \
               && mkdir pylibs && cd pylibs \
               && git clone https://github.com/cambel/ur_ikfast.git \
               && cd ur_ikfast && pip install -e ."

# fix for sip
RUN rm /usr/lib/python3/dist-packages/sip.cpython-36m-x86_64-linux-gnu.so

RUN apt-get update && apt-get install -y \
    ros-melodic-moveit \
    ros-melodic-moveit-runtime \
    ros-melodic-moveit-python \
    ros-melodic-moveit-resources \
    && rm -rf /var/lib/apt/lists/*

RUN pip install Cython cpprb

# fix problem with tf2rl
RUN apt-get -qq update && \
    apt-get -y --force-yes install \
    software-properties-common

# tf2rl
RUN /bin/bash -c "cd ~/pylibs \
    && git clone -b dev https://github.com/cambel/tf2rl.git \
    && cd tf2rl && pip install -e ."

# Workaround for python3 + tf2
RUN pip install --extra-index-url https://rospypi.github.io/simple/ tf2_ros --ignore-installed
RUN pip install --upgrade pip

RUN mkdir -p ~/.gazebo

RUN pip install netifaces
RUN pip install gym[atari]==0.15.7

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# setup environment
EXPOSE 11345

RUN echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc
RUN echo 'source ~/ros_ws/devel/setup.bash' >> ~/.bashrc

RUN echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
RUN echo 'export PYTHONPATH=/root/orocos_kinematics_dynamics/python_orocos_kdl/build:$PYTHONPATH' >> ~/.bashrc
RUN echo 'export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/ros_ws/src/ros-universal-robots/ur3_gazebo/models/' >> ~/.bashrc
RUN echo "alias ur3e_dual_sim='roslaunch ur3_gazebo ur3e_dual_gripper.launch pause:=0'" >> ~/.bashrc
RUN echo "alias ur3e_sim='roslaunch ur3_gazebo ur3e_cubes.launch'" >> ~/.bashrc
RUN echo 'function ur3e_robot { roslaunch ur_robot_driver ur3e_bringup.launch robot_ip:="$1" kinematics_config:=/root/dev/tools/my_robot_calibration.yaml; }'  >> ~/.bashrc
RUN echo 'function ur3_robot { roslaunch ur_robot_driver ur3_bringup.launch robot_ip:="$1" kinematics_config:=/root/dev/tools/my_robot_calibration.yaml; }'  >> ~/.bashrc
