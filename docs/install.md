# Installing from scratch (bare metal)

This is the full setup guide for a machine where nothing is installed yet. It gets
the **realtime gsplat pipeline** (`scripts/zed2i_gsplat_live.py`) running directly
on the machine — the recommended path before touching Docker.

> **The one thing that trips everyone up:** ROS 2 ships `rclpy`, `cv_bridge`, and
> `message_filters` as **apt packages compiled against the system Python** for your
> ROS distro. They will **not** import under a fresh conda env or a plain venv with
> its own Python — `cv_bridge` especially, because it's a C++ extension linked to
> the system Python's ABI. The reliable fix (used below) is a **venv created from
> the system Python with `--system-site-packages`**, so it inherits the system ROS
> packages while still letting you `pip install` torch/gsplat on top. This mirrors
> the pattern in the repo's own Docker image.

---

## 0. Pick your target

Everything below is the same except the Python version and where torch comes from.

| Target | OS | ROS 2 | System Python | torch install | gsplat arch |
|---|---|---|---|---|---|
| **x86 + Humble** | Ubuntu 22.04 | Humble | **3.10** | `pip … --index-url .../whl/cu121` | 8.6 / 8.9 / 9.0 |
| **x86 + Jazzy** | Ubuntu 24.04 | Jazzy | **3.12** | `pip … --index-url .../whl/cu121` | 8.6 / 8.9 / 9.0 |
| **Jetson Thor** | JetPack (Ubuntu 24.04) | Jazzy | **3.12** | JetPack/NGC Jetson wheel (**not** the x86 index) | **11.0** |

GPU compute capability: desktop Ampere `8.6`, Ada `8.9`, Hopper `9.0`; Jetson Thor
(Blackwell) `11.0`. `TORCH_CUDA_ARCH_LIST` uses these; the installer autodetects it.

---

## 1. System prerequisites (apt)

### 1.1 NVIDIA driver + CUDA
```bash
nvidia-smi        # driver + GPU visible?
nvcc --version    # CUDA toolkit present (needed to build gsplat from source)
```
On desktop, install the NVIDIA driver + CUDA toolkit if `nvcc` is missing. On Jetson
these come with JetPack.

### 1.2 ROS 2 + the ZED-facing ROS packages
Install ROS 2 for your distro, then the two packages the node imports:
```bash
# Ubuntu 22.04 / Humble:
sudo apt install ros-humble-ros-base ros-humble-cv-bridge ros-humble-message-filters
# Ubuntu 24.04 / Jazzy (and Jetson Thor):
sudo apt install ros-jazzy-ros-base ros-jazzy-cv-bridge ros-jazzy-message-filters
```
(Full ROS install instructions: <https://docs.ros.org>.)

### 1.3 Python venv tooling + build deps
```bash
# Humble (Python 3.10):
sudo apt install python3.10-venv python3.10-dev build-essential git
# Jazzy / Thor (Python 3.12):
sudo apt install python3.12-venv python3.12-dev build-essential git
```

---

## 2. Configure the Python virtual environment (the important part)

### 2.1 Why not a plain conda env
A conda env brings its own Python (often a different minor version) and its own
`numpy`. ROS's `cv_bridge` is a compiled extension built against the **system**
Python and **system numpy**; loaded under a mismatched interpreter it fails with
`ImportError`/ABI errors. So for this project we use a **system-Python venv**, not
conda. (If you truly cannot apt-install ROS, the alternative is RoboStack — ROS from
conda-forge — but that's a separate ROS install and out of scope here.)

### 2.2 Create the venv (matching your ROS Python)
Use the **system Python that matches your ROS distro**, and pass
`--system-site-packages` so the venv can see system packages:
```bash
# Humble (Python 3.10):
python3.10 -m venv --system-site-packages ~/venvs/splatam
# Jazzy / Jetson Thor (Python 3.12):
python3.12 -m venv --system-site-packages ~/venvs/splatam
```
Or use the helper, which picks the right Python for `$ROS_DISTRO` and prints the
next steps:
```bash
source /opt/ros/<distro>/setup.bash          # so the helper can detect the distro
bash bash_scripts/make_venv.bash             # creates ~/venvs/splatam
```

### 2.3 Sourcing order — do this in **every** shell
Order matters. Source ROS first (puts ROS's Python packages on `PYTHONPATH`), then
activate the venv (same Python version as ROS):
```bash
source /opt/ros/<distro>/setup.bash          # humble | jazzy
source ~/venvs/splatam/bin/activate
```
Tip: if `conda` auto-activates a `base` env, it can shadow the venv's Python — run
`conda deactivate` first. Add the two `source` lines to a small `env.sh` you can
`source` each session.

### 2.4 Verify ROS is visible in the venv (before anything else)
```bash
python --version                                     # must match your ROS Python (3.10 or 3.12)
python -c "import rclpy, cv_bridge, message_filters; print('ROS OK')"
```
If `cv_bridge` fails here, fix it now — see [Troubleshooting](#5-troubleshooting).
Almost always it's a Python-version mismatch, ROS not sourced, or the venv was made
without `--system-site-packages`.

### 2.5 The numpy pitfall
ROS `cv_bridge` (esp. on Humble) is built against **numpy 1.x**. Installing numpy 2
into the venv breaks it with ABI errors. The repo pins `numpy==1.26.4`
(`venv_requirements.txt`); keep it, and if a later `pip install` pulls numpy 2, pin
it back: `pip install "numpy==1.26.4"`.

---

## 3. Install PyTorch (per target)

**gsplat needs a modern torch (2.x) + CUDA 12.x.** Do NOT use the legacy torch 1.12.

```bash
# x86 desktop (Humble or Jazzy):
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

**Jetson Thor:** the x86 wheel above will not work. Install the JetPack/NGC
Jetson-specific torch wheel that matches your JetPack CUDA (see NVIDIA's "PyTorch for
Jetson" / NGC container notes). torch may already be present in a JetPack ML meta-package.

Verify CUDA is visible:
```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
```

---

## 4. Install the pipeline + gsplat

`requirements.txt` is **pure-python** (no CUDA extensions), and the installer adds
gsplat with the correct GPU arch. You do **not** need `diff-gaussian-rasterization`
for the default gsplat engine.
```bash
bash bash_scripts/install.bash                     # core deps + gsplat (autodetects arch)
# Jetson Thor (or to override autodetect):
TORCH_CUDA_ARCH_LIST=11.0 bash bash_scripts/install.bash
# also build the optional INRIA "cuda" fallback backend (from the vendored copy):
bash bash_scripts/install.bash --with-cuda-fallback
```
The installer verifies torch+CUDA, warms the gsplat JIT with an `import` check, and
runs the backend self-test at the end.

---

## 5. Verify the whole setup

```bash
python scripts/tools/preflight.py                  # torch/CUDA, engine, ROS, config — PASS/FAIL + fixes
python scripts/tools/render_backend_selftest.py    # renders known Gaussians through both backends
```
`preflight.py` should be all-PASS. The self-test should print `RESULT: PASS`.

You're ready to run — continue with **[docs/running_locally.md](running_locally.md)**
(feed a rosbag or live ZED, staged bring-up).

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `import cv_bridge` fails / ABI error | venv Python doesn't match the ROS distro Python (Humble=3.10, Jazzy=3.12), or ROS not sourced, or venv missing `--system-site-packages`. Recreate the venv from the right `python3.X` and `source /opt/ros/<distro>/setup.bash` first. |
| `import rclpy` fails | `source /opt/ros/<distro>/setup.bash` in this shell; ensure `ros-<distro>-ros-base` is installed. |
| numpy ABI / "compiled against a different numpy" | a dep pulled numpy 2. Pin back: `pip install "numpy==1.26.4"`. |
| `python` is the wrong version after activating | conda `base` is shadowing the venv — `conda deactivate`, then re-`source` the venv. |
| `pip install -r requirements.txt` fails on `diff-gaussian-rasterization` | old checkout — it's no longer in requirements. Pull latest; CUDA builds are handled by `install.bash`. |
| `import gsplat` fails / build error | torch/CUDA too old (need 2.x/CU12), or wrong `TORCH_CUDA_ARCH_LIST`. Re-run `TORCH_CUDA_ARCH_LIST=<arch> bash bash_scripts/install.bash`, or use `--with-cuda-fallback` and set `render_backend="cuda"`. |
| Jetson: `torch.cuda.is_available()` is False | you installed the x86 wheel. Install the JetPack/NGC Jetson torch wheel. |
