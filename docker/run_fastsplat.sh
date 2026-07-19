#!/bin/bash
# Launch the fastsplat GPU container with the repo bind-mounted at /workspace.
set -euo pipefail

IMAGE="${IMAGE:-fastsplat:latest}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pick the best available GPU flag.
GPU_ARGS=""
if docker info 2>/dev/null | grep -qi "nvidia"; then
    GPU_ARGS="--gpus all"
elif command -v nvidia-container-runtime >/dev/null 2>&1; then
    GPU_ARGS="--runtime nvidia"
else
    echo "[warn] no NVIDIA docker runtime detected; running without GPU"
fi

exec docker run --rm -it \
    ${GPU_ARGS} \
    --ipc=host --shm-size=16g \
    -v "${REPO_DIR}":/workspace \
    -w /workspace \
    "${IMAGE}" \
    "${@:-bash}"
