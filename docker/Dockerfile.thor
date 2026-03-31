# ZED + ROS 2 Humble + SplaTAM in one container
ARG JP_MAJOR=6
ARG JP_MINOR=0
ARG JP_PATCH=0
ARG ZED_SDK_MAJOR=4
ARG ZED_SDK_MINOR=2
ARG ROS_DISTRO=humble

### BUILD IMAGE ###
FROM stereolabs/zed:${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}-devel-jetson-jp${JP_MAJOR}.${JP_MINOR}.${JP_PATCH} AS build

ARG JP_MAJOR
ARG JP_MINOR
ARG JP_PATCH
ARG ZED_SDK_MAJOR
ARG ZED_SDK_MINOR
ARG ROS_DISTRO

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility
ENV ROS_DISTRO=$ROS_DISTRO

# ROS install
RUN apt update && apt install -y curl && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list && \
    apt update && apt upgrade -y && \
    apt install -y --no-install-recommends ros-$ROS_DISTRO-ros-core && \
    rm -rf /var/lib/apt/lists/*

# ZED wrapper build deps
RUN apt-get update || true && \
    apt-get install --no-install-recommends -y \
        ros-dev-tools \
        ros-${ROS_DISTRO}-ros-base \
        ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-image-transport-plugins \
        ros-${ROS_DISTRO}-diagnostic-updater \
        ros-${ROS_DISTRO}-xacro \
        build-essential \
        python3-colcon-mixin \
        python3-flake8-docstrings \
        python3-pip \
        python3-pytest-cov && \
    pip3 install argcomplete numpy empy lark && \
    rm -rf /var/lib/apt/lists/*

# Fetch wrapper and resolve deps
WORKDIR /ros2_ws
RUN apt-get update && \
    git clone -b humble-v4.2.5 --recursive https://github.com/stereolabs/zed-ros2-wrapper.git src/zed-ros2-wrapper && \
    cd src/zed-ros2-wrapper && \
    git submodule update --init --recursive && \
    source /opt/ros/$ROS_DISTRO/setup.bash && \
    (rosdep init || true) && \
    rosdep update --rosdistro $ROS_DISTRO && \
    rosdep install --from-paths /ros2_ws/src -y -i && \
    rm -rf /var/lib/apt/lists/*

# Build wrapper
WORKDIR /ros2_ws
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)

### RUNTIME IMAGE ###
FROM stereolabs/zed:${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}-runtime-jetson-jp${JP_MAJOR}.${JP_MINOR}.${JP_PATCH} AS runtime

ARG JP_MAJOR
ARG JP_MINOR
ARG JP_PATCH
ARG ZED_SDK_MAJOR
ARG ZED_SDK_MINOR
ARG ROS_DISTRO

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility
ENV ROS_DISTRO=$ROS_DISTRO

# ROS install
RUN apt update && apt install -y curl && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list && \
    apt update && apt upgrade -y && \
    apt install -y --no-install-recommends ros-$ROS_DISTRO-ros-core && \
    rm -rf /var/lib/apt/lists/*

# ZED runtime deps + SplaTAM native deps
RUN apt-get update || true && \
    apt-get install --no-install-recommends -y \
        ros-dev-tools \
        ros-${ROS_DISTRO}-ros-base \
        ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-image-transport-plugins \
        ros-${ROS_DISTRO}-diagnostic-updater \
        ros-${ROS_DISTRO}-xacro \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-zed-msgs \
        build-essential \
        curl \
        git \
        cmake \
        pkg-config \
        ninja-build \
        python3-colcon-mixin \
        python3-flake8-docstrings \
        python3-pip \
        python3-pytest-cov \
        python3-venv \
        python3-dev \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libopenblas-dev && \
    pip3 install argcomplete numpy empy lark && \
    rm -rf /var/lib/apt/lists/*

# Copy built wrapper workspace
COPY --from=build /ros2_ws /ros2_ws

# Source workspace for interactive shells
RUN test -f "/ros2_ws/install/setup.bash" && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /etc/bash.bashrc

# Version file
RUN echo $(cat /ros2_ws/src/zed-ros2-wrapper/zed_wrapper/package.xml | grep '<version>' | sed -r 's/.*<version>([0-9]+.[0-9]+.[0-9]+)<\/version>/\1/g') >> /version.txt

# SplaTAM runtime
# SplaTAM runtime
WORKDIR /SplaTAM
COPY . /SplaTAM

ENV PIP_NO_BUILD_ISOLATION=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

# Set this explicitly because CUDA extensions are being built during docker build
# For Jetson Orin use 8.7. If you are truly on Thor later, this will need to change with the base image.
ENV TORCH_CUDA_ARCH_LIST=8.7

ARG TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip uninstall -y torch torchvision torchaudio numpy || true && \
    python3 -m pip install  numpy==1.26.4 && \
    python3 -m pip install  --force-reinstall  "${TORCH_INSTALL}" && \
    python3 -m pip install  \
        typing_extensions \
        filelock \
        sympy \
        networkx \
        jinja2 \
        fsspec

# cuSPARSELt is required for newer Jetson PyTorch wheels
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    wget -q https://raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh -O /tmp/install_cusparselt.sh && \
    export CUDA_VERSION=12.1 && \
    bash /tmp/install_cusparselt.sh && \
    rm -f /tmp/install_cusparselt.sh && \
    rm -rf /var/lib/apt/lists/*

# packages that should not replace torch/numpy
RUN python3 -m pip install --no-cache-dir \
    opencv-python \
    open3d==0.16.0 \
    tqdm==4.65.0 \
    Pillow \
    imageio \
    matplotlib \
    kornia \
    natsort \
    pyyaml \
    wandb \
    plyfile==0.8.1

RUN python3 -m pip install --no-cache-dir --no-deps \
    lpips \
    torchmetrics \
    pytorch-msssim

RUN python3 -m pip install --no-cache-dir \
    lightning-utilities \
    packaging \
    "pyparsing>=3.1,<4"

RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    numpy==1.26.4



RUN python3 -m pip install  --force-reinstall \
    "numpy==1.26.4" \
    "pyparsing>=3.1,<4"

#RUN python3 -m pip install  --force-reinstall "pyparsing>=3.1,<4"
# force numpy back to 1.x after all installs
#RUN python3 -m pip install  --force-reinstall  numpy==1.26.4
RUN python3 -m pip install --force-reinstall "pyparsing>=3.1,<4"
RUN python3 -m pip install  --force-reinstall  numpy==1.26.4

RUN python3 - <<'PY'
import torch, numpy
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("numpy:", numpy.__version__)
assert torch.version.cuda == "12.2", f"Expected 12.2, got {torch.version.cuda}"
assert numpy.__version__.startswith("1.26"), f"Expected numpy 1.26.x, got {numpy.__version__}"
PY

RUN python3 -m pip install --no-build-isolation \
    /SplaTAM/third_party/diff-gaussian-rasterization

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
COPY docker/healthcheck.sh /healthcheck.sh
HEALTHCHECK --interval=2s --timeout=1s --start-period=20s --retries=1 \
    CMD ["/healthcheck.sh"]

ARG TARGETARCH
ARG HUSARNET_DDS_RELEASE="v1.3.5"
ENV HUSARNET_DDS_DEBUG=FALSE
RUN curl -L https://github.com/husarnet/husarnet-dds/releases/download/${HUSARNET_DDS_RELEASE}/husarnet-dds-linux-${TARGETARCH} -o /usr/bin/husarnet-dds && \
    chmod +x /usr/bin/husarnet-dds

WORKDIR /SplaTAM
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]