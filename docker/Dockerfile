FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    build-essential \
    cmake \
    pkg-config \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel

# -------------------------------------------------------------------
# Install Cyclone DDS first so python cyclonedds can build on aarch64
# -------------------------------------------------------------------
RUN git clone --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cyclonedds && \
    mkdir -p /tmp/cyclonedds/build && \
    cd /tmp/cyclonedds/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds && \
    cmake --build . --target install -j"$(nproc)" && \
    rm -rf /tmp/cyclonedds

ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV CMAKE_PREFIX_PATH=/opt/cyclonedds:${CMAKE_PREFIX_PATH:-}
ENV LD_LIBRARY_PATH=/opt/cyclonedds/lib:${LD_LIBRARY_PATH:-}

# Work in repo root inside container
WORKDIR /SplaTAM

# Copy repo into image before any local path installs
COPY . /SplaTAM

# Python packages
RUN python -m pip install --no-cache-dir \
    "numpy<2" \
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
    lpips \
    torchmetrics \
    cyclonedds==0.10.5 \
    pytorch-msssim \
    plyfile==0.8.1

# Install local rasterizer package
#RUN python -m pip install --no-cache-dir --no-build-isolation \
#    /SplaTAM/third_party/diff-gaussian-rasterization
RUN python -m pip install --no-cache-dir --no-build-isolation \
  ./third_party/diff-gaussian-rasterization

CMD ["/bin/bash"]
