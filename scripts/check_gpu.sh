#!/usr/bin/env bash
set -euo pipefail

echo "Running in-container GPU diagnostic checks"

echo "1) nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found in container PATH"
fi

echo
echo "2) Python Torch CUDA availability"
python - <<'PY'
import sys
try:
    import torch
    print('torch', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    print('torch cuda version:', torch.version.cuda)
except Exception as e:
    print('Python GPU check failed:', e)
    sys.exit(2)
PY

echo
echo "3) List /dev for nvidia devices"
ls -l /dev | grep nvidia || true

echo
echo "4) Check for CUDA toolkit (nvcc)"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo "nvcc not found (CUDA toolkit not installed in container)"
fi

echo
echo "Diagnostic complete. If CUDA isn't available, verify host-side NVIDIA drivers and container runtime." 
