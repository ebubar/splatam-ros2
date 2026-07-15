#!/bin/bash
# Robust installer for the realtime gsplat pipeline.
#
# Installs, in order:
#   1. pure-python deps (requirements.txt) — never fails on a CUDA build
#   2. gsplat (Apache-2.0, the DEFAULT engine) — arch-aware source build
#   3. (optional) the INRIA diff-gaussian-rasterization "cuda" fallback backend,
#      built from the VENDORED copy in third_party/ (no SSH submodule, no network)
#
# You do NOT need the cuda fallback for the default gsplat path.
#
# Usage:
#   bash bash_scripts/install.bash                     # core deps + gsplat
#   bash bash_scripts/install.bash --with-cuda-fallback  # also the optional fallback
#   GSPLAT_VERSION=1.4.0 bash bash_scripts/install.bash  # pin a gsplat version
#   TORCH_CUDA_ARCH_LIST=8.9 bash bash_scripts/install.bash  # override arch autodetect
set -uo pipefail

info='\033[96m'; ok='\033[92m'; warn='\033[93m'; err='\033[91m'; END='\033[0m'
say()  { printf "${1}%b${END}\n" "$2"; }
die()  { say "$err" "[ERROR] $1"; exit 1; }

WITH_FALLBACK=0
for arg in "$@"; do
    case "$arg" in
        --with-cuda-fallback) WITH_FALLBACK=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) die "unknown argument: $arg (see --help)" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || die "$PY not found. Activate your conda/venv first."

# --- 1. Verify torch + CUDA -------------------------------------------------
say "$info" "[1/4] Checking PyTorch + CUDA..."
if ! "$PY" -c "import torch" 2>/dev/null; then
    die "PyTorch is not installed. Install a CUDA build first, e.g.:
    pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
  (gsplat needs a modern torch/CUDA — the repo's legacy torch 1.12 will NOT build gsplat.)"
fi
"$PY" - <<'PYEOF' || die "PyTorch cannot see a CUDA GPU. gsplat requires an NVIDIA GPU + CUDA build of torch."
import sys, torch
print(f"    torch {torch.__version__}  cuda {torch.version.cuda}  available={torch.cuda.is_available()}")
sys.exit(0 if torch.cuda.is_available() else 1)
PYEOF

# --- 2. Core pure-python deps ----------------------------------------------
say "$info" "[2/4] Installing core Python deps (requirements.txt)..."
"$PY" -m pip install -r requirements.txt || die "core dependency install failed."
say "$ok" "    core deps OK"

# --- 3. gsplat (default engine) --------------------------------------------
say "$info" "[3/4] Installing gsplat (Apache-2.0 default engine)..."
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    ARCH="$("$PY" -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || true)"
    if [ -n "$ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="$ARCH"
        say "$info" "    autodetected GPU arch -> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    else
        say "$warn" "    could not autodetect arch; letting gsplat choose (set TORCH_CUDA_ARCH_LIST to override)"
    fi
else
    say "$info" "    using TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi

if "$PY" -c "import gsplat" 2>/dev/null; then
    say "$ok" "    gsplat already importable; skipping (set --force via reinstall to rebuild)"
else
    GSPLAT_VERSION="${GSPLAT_VERSION:-1.4.0}"
    say "$info" "    building gsplat==$GSPLAT_VERSION (first build can take several minutes)..."
    if ! "$PY" -m pip install --no-build-isolation "gsplat==$GSPLAT_VERSION"; then
        say "$warn" "    --no-build-isolation build failed; retrying with build isolation..."
        "$PY" -m pip install "gsplat==$GSPLAT_VERSION" || die "gsplat install failed.
  Check: nvcc on PATH (nvcc --version), CUDA dev headers present, and
  TORCH_CUDA_ARCH_LIST matches your GPU. On x86 a prebuilt wheel may exist:
    pip install gsplat --index-url https://docs.gsplat.studio/whl/pt23cu121"
    fi
fi
"$PY" -c "import gsplat; print('    gsplat', gsplat.__version__, 'import OK')" || die "gsplat installed but fails to import."
say "$ok" "    gsplat OK"

# --- 4. Optional INRIA cuda fallback (vendored copy) -----------------------
if [ "$WITH_FALLBACK" -eq 1 ]; then
    say "$info" "[4/4] Installing optional cuda fallback (third_party/diff-gaussian-rasterization)..."
    VENDORED="$REPO_ROOT/third_party/diff-gaussian-rasterization"
    if [ ! -f "$VENDORED/setup.py" ]; then
        say "$warn" "    vendored rasterizer not found at $VENDORED; skipping."
    elif "$PY" -m pip install --no-build-isolation "$VENDORED"; then
        say "$ok" "    cuda fallback OK"
    else
        say "$warn" "    cuda fallback build failed — the default gsplat engine still works; continuing."
    fi
else
    say "$info" "[4/4] Skipping optional cuda fallback (run with --with-cuda-fallback to include it)."
fi

# --- numpy guard ------------------------------------------------------------
# torch/gsplat installs can pull numpy 2.x, which breaks ROS cv_bridge (built
# against numpy 1.x). Re-assert numpy<2 as the very last dependency action.
say "$info" "Enforcing numpy<2 (required by ROS cv_bridge)..."
if ! "$PY" -c "import numpy,sys; sys.exit(0 if numpy.__version__[0]=='1' else 1)" 2>/dev/null; then
    say "$warn" "    numpy>=2 detected; downgrading to numpy<2..."
    "$PY" -m pip install "numpy<2" || say "$warn" "    numpy downgrade failed — cv_bridge may crash; pin manually: pip install 'numpy<2'"
fi
"$PY" -c "import numpy; print('    numpy', numpy.__version__)"

# --- Self-test --------------------------------------------------------------
say "$info" "Running backend self-test..."
"$PY" scripts/tools/render_backend_selftest.py || say "$warn" "self-test reported issues (see above)."

say "$ok" ""
say "$ok" "=================================================================="
say "$ok" " Install complete."
say "$ok" " Next:  python3 scripts/tools/preflight.py        # startup checks"
say "$ok" "        python3 scripts/zed2i_gsplat_live.py --config configs/zed2i/zed2i_gsplat_desktop.py"
say "$ok" "=================================================================="
