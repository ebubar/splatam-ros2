# SplaTAM ROS2 realtime-gsplat image for Jetson Thor (Blackwell, sm_110 / CC 11.0).
#
# Builds BOTH rasterizer backends against the NGC torch in the base image:
#   * gsplat (Apache-2.0)  -> default engine (render_backend="gsplat")
#   * diff-gaussian-rasterization (INRIA, research-only) -> "cuda" fallback,
#     kept because it is the guaranteed-buildable backend if a gsplat aarch64
#     source build is ever problematic on a given JetPack.
#
# There is no prebuilt aarch64 gsplat wheel, so gsplat is compiled from source
# (--no-build-isolation, MAX_JOBS=1 to bound RAM) with the CUDA dev toolchain
# from the NGC base. An AOT `import gsplat` at build time avoids a first-frame
# JIT stall at runtime (important for realtime).
FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake pkg-config ninja-build \
    python3-dev python3-venv curl ca-certificates ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    locales gnupg2 lsb-release software-properties-common && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /SplaTAM
COPY . /SplaTAM

RUN python -m pip install --no-cache-dir \
      "numpy<2" opencv-python tqdm==4.65.0 Pillow imageio matplotlib \
      kornia natsort pyyaml wandb lpips torchmetrics pytorch-msssim plyfile==0.8.1

# Jetson Thor Blackwell = compute capability 11.0.
ENV TORCH_CUDA_ARCH_LIST="11.0"
ENV MAX_JOBS=1

# Default gsplat engine (Apache-2.0) = REQUIRED build, source + AOT import check.
RUN python -m pip install --no-cache-dir --no-build-isolation gsplat==1.4.0 && \
    python -c "import gsplat; print('gsplat', gsplat.__version__, 'import OK on Thor')"

# Optional "cuda" fallback backend, from the VENDORED copy (not the SSH submodule).
# Non-fatal: guaranteed-buildable fallback if a gsplat aarch64 build ever regresses.
RUN python -m pip install --no-cache-dir --no-build-isolation \
    /SplaTAM/third_party/diff-gaussian-rasterization || \
    echo "WARN: optional cuda fallback build failed; gsplat backend still available."

COPY . /SplaTAM

RUN chmod +x /SplaTAM/docker/splatam_entrypoint.sh

ENTRYPOINT ["/SplaTAM/docker/splatam_entrypoint.sh"]
CMD ["/bin/bash"]
