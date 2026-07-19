#!/bin/bash
# Entrypoint for the fastsplat GPU image.
#
# The NGC PyTorch base ships HPC-X (UCX/UCC) libs that can shadow system libs
# and break some imports; put them first on the loader path (same fix the
# SplaTAM entrypoint uses). Then exec whatever command was passed.
set -e

export LD_LIBRARY_PATH="/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH:-}"

# Cache HuggingFace model downloads on the mounted volume if available.
if [ -d "/workspace" ]; then
    export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
fi

exec "$@"
