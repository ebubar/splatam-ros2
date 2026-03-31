#!/usr/bin/env bash
set -euo pipefail

xhost +local:root >/dev/null 2>&1 || true

# Image names
IMG="splatam-clean:cu121"
TEST_IMG="nvidia/cuda:12.1.1-base-ubuntu22.04"

echo "Checking Docker GPU support..."

# Try modern Docker --gpus flag
if docker run --rm --gpus all ${TEST_IMG} nvidia-smi >/dev/null 2>&1; then
  GPU_RUN_ARGS=("--gpus" "all")
  echo "Using: --gpus all"
elif docker run --rm --runtime=nvidia ${TEST_IMG} nvidia-smi >/dev/null 2>&1; then
  GPU_RUN_ARGS=("--runtime" "nvidia")
  echo "Using legacy: --runtime=nvidia"
else
  # Try device mapping if /dev/nvidia* exists on the host
  if compgen -G "/dev/nvidia*" >/dev/null; then
    echo "Falling back to mapping /dev/nvidia* devices into the container"
    GPU_RUN_ARGS=()
    for dev in /dev/nvidia*; do
      GPU_RUN_ARGS+=("--device" "$dev")
    done
  else
    cat <<EOF
ERROR: No GPU device accessible to Docker.

Possible causes and fixes:
- The host is missing NVIDIA drivers or the NVIDIA Container Toolkit.
  Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- On newer Docker, enable the NVIDIA runtime and ensure `docker run --gpus all` works.
- On older setups, install the legacy `nvidia-docker2` and use `--runtime=nvidia`.

Quick diagnostics you can run on the host:
  docker run --rm --gpus all ${TEST_IMG} nvidia-smi
  docker run --rm --runtime=nvidia ${TEST_IMG} nvidia-smi
  ls /dev | grep nvidia || true

After fixing the host-side configuration, re-run this script.
EOF
    exit 1
  fi
fi

echo "Launching container ${IMG}..."

#docker run --rm -it \
#  "${GPU_RUN_ARGS[@]}" \
#  --ipc=host \
#  --shm-size=16g \
#  --network host \
#  -e DISPLAY=$DISPLAY \
#  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#  -v "$(pwd)":/SplaTAM \
#  -w /SplaTAM \
#  ${IMG} bash
docker run --rm -it \
  "${GPU_RUN_ARGS[@]}" \
  --ipc=host \
  --shm-size=16g \
  --network host \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/SplaTAM \
  -w /SplaTAM \
  "$IMG" bash
