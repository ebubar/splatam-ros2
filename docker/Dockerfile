FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git ca-certificates curl wget unzip \
    cmake ninja-build build-essential pkg-config \
    gcc-11 g++-11 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install ROS2 Humble
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-ros-core \
      ros-humble-geometry-msgs \
      ros-humble-sensor-msgs \
      ros-humble-std-msgs \
      ros-humble-image-transport \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Source ROS2 setup in bashrc for interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10

ENV CC=gcc-11
ENV CXX=g++-11

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

ENV PIP_NO_BUILD_ISOLATION=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /SplaTAM

# Copy repository into the image so vendored packages (third_party/) are available
COPY . /SplaTAM

RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN python -m pip install --no-cache-dir \
  "numpy<2" \
  opencv-python \
  open3d==0.16.0 \
  tqdm==4.65.0 Pillow imageio matplotlib kornia natsort pyyaml wandb lpips torchmetrics \
  cyclonedds pytorch-msssim plyfile==0.8.1

ARG TORCH_CUDA_ARCH_LIST=8.9
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN python -m pip install --no-cache-dir --no-build-isolation \
  ./third_party/diff-gaussian-rasterization

RUN python -c "import torch, numpy as np, cv2, diff_gaussian_rasterization; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('numpy', np.__version__); print('cv2', cv2.__version__); print('rasterizer ok')"
