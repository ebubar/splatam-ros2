# SplaTAM Docker (reproducible GPU setup)

This repo includes a **clean, reproducible** Docker workflow for running SplaTAM with:
- CUDA 12.1 base image
- Python 3.10 in a dedicated venv at `/opt/venv`
- PyTorch CUDA wheels (cu121)
- `diff-gaussian-rasterization-w-depth` built inside the image
- `numpy<2` (SplaTAM uses `np.unicode_` which was removed in NumPy 2.x)
- OpenCV + Open3D for visualization scripts

## Prereqs on host
- NVIDIA driver + working `nvidia-smi`
- Docker + NVIDIA Container Toolkit installed (`docker run --gpus all ... nvidia-smi` works)
- Optional for visualization: X server available and `$DISPLAY` set

## Files
- `docker/Dockerfile` — builds the runtime image
- `docker/run.sh` — runs the container with GPU + optional X forwarding

## Build
From repo root:
```bash
docker build -t splatam-clean:cu121 -f docker/Dockerfile .

## For a clean rebuild
docker build --no-cache -t splatam-clean:cu121 -f docker/Dockerfile .

## CUDA arch note

Docker builds typically run without a visible GPU, so CUDA arch auto-detection can fail for CUDA extensions.
This image pins:

TORCH_CUDA_ARCH_LIST=8.9 (Ada / RTX 4000 Ada)

Override if needed:

docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="8.6" \
  -t splatam-clean:cu121 -f docker/Dockerfile .

Run

From repo root:

./docker/run.sh


This mounts the repo into the container at /SplaTAM.

Sanity checks (inside container)
python -c "import torch; print('cuda avail:', torch.cuda.is_available(), 'torch cuda:', torch.version.cuda)"
python -c "import diff_gaussian_rasterization; print('rasterizer ok')"
python -c "import cv2; print('cv2 ok')"
python -c "import open3d as o3d; print('open3d', o3d.__version__)"

Container GPU diagnostic script
--------------------------------
We've added a small helper script `scripts/check_gpu.sh` you can run inside the container to print useful GPU diagnostics (nvidia-smi, torch CUDA availability, /dev nodes, nvcc if present).

Usage (after launching the container):

```bash
# inside the running container
bash scripts/check_gpu.sh
```

Host-side quick diagnostics
---------------------------
If the container cannot see the GPU, run these on the host to narrow the problem:

```bash
# modern Docker + NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# legacy runtime (nvidia-docker2)
docker run --rm --runtime=nvidia nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# check device nodes
ls -l /dev | grep nvidia || true

# docker info (look for runtimes and GPU support)
docker info
```

Dataset download scripts

The image includes wget and unzip, so the repo scripts should work:

bash bash_scripts/download_tum.sh
bash bash_scripts/download_replica.sh