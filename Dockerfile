# FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
# FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
# FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN ln -sv /usr/bin/python3 /usr/bin/python

# Install conda
RUN cd /opt \
    # && mkdir conda \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Create environment
RUN conda create -n pytorch3d -y python=3.9 \
    && conda install -n pytorch3d -y pytorch=1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# RUN conda create -n pytorch3d python=3.9 \
#     && conda run -n pytorch3d pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN conda install -n pytorch3d -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -n pytorch3d -c bottler nvidiacub
RUN conda install -n pytorch3d pytorch3d==0.7.4 -c pytorch3d
# RUN conda run -n pytorch3d pip install pytorch3d==0.7.2 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html
# ENV FORCE_CUDA=1
# ENV TORCH_CUDA_ARCH_LIST="7.5+PTX"
# ENV CUB_HOME="/opt/conda/pkgs/nvidiacub-1.10.0-0/"
# RUN conda run -n pytorch3d pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"

# RUN python -m pip install \
#     pytorch3d==0.7.4

RUN conda run -n pytorch3d pip install \
    scikit-image==0.22.0 \
    matplotlib==3.8.1 \
    imageio==2.32.0 \
    opencv-python==4.4.0.46 \
    # opencv-python==4.8.1.78 \
    trimesh==4.0.3 \
    open3d-cpu==0.17.0 \
    kornia==0.7.0

ENV PYTHONPATH="${PYTHONPATH}:/code/"

# Add ROS support
# add the keys
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN wget -q -O - https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# install ros
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
	ros-noetic-tf \
    ros-noetic-catkin \
    ros-noetic-vision-msgs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

# install python dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-rosdep \
    python3-catkin-tools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN conda run -n pytorch3d pip install rospkg
RUN conda run -n pytorch3d pip install 'git+https://github.com/eric-wieser/ros_numpy'

RUN sudo rosdep init
RUN rosdep update
RUN mkdir -p /root/catkin_build_ws/src
RUN /bin/bash -c  '. /opt/ros/noetic/setup.bash; cd /root/catkin_build_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so; catkin build'

# clone and build message and service definitions
COPY src/tracebot_msgs /root/catkin_build_ws/src/tracebot_msgs
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd ~/catkin_build_ws; catkin build'
# source the catkin workspace
RUN echo "source ~/catkin_build_ws/devel/setup.bash" >> ~/.bashrc

COPY ros_entrypoint_noetic.sh /
RUN chmod +x /ros_entrypoint_noetic.sh

RUN /bin/bash -c 'conda init'

ENTRYPOINT ["/ros_entrypoint_noetic.sh"]
# docker run -it --rm --gpus all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /home/jbweibel/code/inverse_rendering/negar-layegh-inverse-rendering:/home -v "/home/jbweibel/dataset/Tracebot/Tracebot_Negar_2022_08_04":"/home/jbweibel/dataset/Tracebot/Tracebot_Negar_2022_08_04" diffrend bash
# cd /home && conda run -n pytorch3d python src/main.py