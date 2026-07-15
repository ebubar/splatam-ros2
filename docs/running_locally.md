# Running the pipeline on the machine (bare metal)

Get the pipeline working directly on the PC/Jetson first, *then* containerize.
Debugging the splatting pipeline is much easier without Docker in the loop — you
can edit code, restart in seconds, and see errors without rebuilding an image.

This runbook targets the **realtime gsplat node** (`scripts/zed2i_gsplat_live.py`)
but every step also works for the original CUDA node (`scripts/zed2i_splat_live.py`)
— just drop the gsplat-specific parts.

---

## 0. What you need

- An NVIDIA GPU (gsplat is CUDA-only). Desktop (x86) or Jetson Thor (aarch64).
- ROS2 installed on the machine that runs the node (Humble on the x86 PC in this
  repo's setup; Jazzy on Thor).
- A data source: **a recorded ROS2 bag** (recommended for first bring-up — no
  camera, deterministic) *or* a live ZED2i publishing over ROS2.

> The camera side (ZED SDK + `zed_wrapper`) only needs to exist on whatever
> machine publishes the ZED topics. For bag replay you don't need the ZED SDK at
> all on the splatting machine.

---

## 1. One-time install

**Full from-scratch setup (ROS + the Python virtual environment + torch + gsplat)
lives in [docs/install.md](install.md).** Do that first if this machine isn't set
up yet — the venv/ROS interaction (`cv_bridge` needs a `--system-site-packages`
venv on the ROS-matching Python) is the main thing to get right.

Quick path once ROS 2 + a CUDA build of torch 2.x are in place:
```bash
source /opt/ros/<distro>/setup.bash && source ~/venvs/splatam/bin/activate
bash bash_scripts/install.bash          # core deps + gsplat (autodetects GPU arch)
python scripts/tools/preflight.py       # torch/CUDA, engine, ROS, config checklist
```

### 1.5 (Live camera only) ZED SDK + zed_wrapper
Skip for bag replay. On the publishing machine install the ZED SDK and build the
ZED ROS2 wrapper, then launch it (see §4B).

---

## 2. Verify the install before touching ROS

```bash
python scripts/tools/preflight.py            # checklist: torch/CUDA, engine, ROS, config
python scripts/tools/render_backend_selftest.py
```
`preflight.py` prints PASS/FAIL for each requirement (torch+CUDA, a render engine,
ROS python, config loads) with the fix for anything missing. The self-test then
renders known Gaussians through **both** backends and diffs RGB/depth/alpha —
expect `RESULT: PASS`. If it SKIPs, an engine isn't importable yet; it's far
cheaper to catch here than over ROS.

---

## 3. Point the config at your setup

Pick a hardware profile: `configs/zed2i/zed2i_gsplat_desktop.py` (x86 GPU) or
`configs/zed2i/zed2i_gsplat_thor.py` (Jetson Thor — lower res, tighter Gaussian
cap, FPS throttle + checkpoints). Both are thin overrides of the base
`configs/zed2i/zed2i_gsplat_live.py`; edit that base (or your profile) to set:
- `render_backend`  — `"gsplat"` (default) or `"cuda"` (fallback / A-B).
- `num_frames`      — how many frames to process before saving.
- `ros.transport`   — `"raw"` (default) or `"compressed"` (limited-bandwidth links).
- `ros.min/max_depth_m`, `data.desired_image_width/height` — match your capture.
- topic names under `ros=dict(...)` — match `ros2 topic list` (see §5.1).

---

## 4. Get data flowing (pick one)

### 4A. ROS2 bag replay (recommended for first bring-up)
Two terminals on the PC, same `ROS_DOMAIN_ID`.

Terminal A — start the node (it waits for frames):
```bash
conda activate splatam
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
python3 scripts/zed2i_gsplat_live.py --config configs/zed2i/zed2i_gsplat_live.py
```
Terminal B — play the bag:
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
ros2 bag info  <bag_dir>        # confirm it has the RGB/depth/odom topics
ros2 bag play  <bag_dir>
```

### 4B. Live ZED2i
On the camera machine (e.g. Orin), same domain id:
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i pos_tracking_mode:=GEN_3
```
Then run the node exactly as in Terminal A above.

### Automated capture+replay
`bash_scripts/main.bash <run_name> <orin_ip> <local_ip> <seconds>` records on the
Orin, copies the bag back, and replays it into the pipeline. To drive the gsplat
node with it, run it with `NODE=gsplat` (the pipeline scripts honor that):
```bash
NODE=gsplat bash bash_scripts/main.bash zed_test <orin_ip> <pc_ip> 60
```

---

## 5. Staged bring-up (debug ladder)

Bring it up one layer at a time so a failure points at one thing.

**5.1 Confirm the topics arrive.** With the bag playing / camera live:
```bash
ros2 topic list | grep zed
ros2 topic hz /zed/zed_node/rgb/color/rect/image
ros2 topic hz /zed/zed_node/depth/depth_registered
ros2 topic echo --once /zed/zed_node/odom
```
No topics ⇒ `ROS_DOMAIN_ID` mismatch or DDS/Zenoh not bridged. Fix this before anything else.

**5.2 Validate the pipeline on the CUDA backend, synchronous.** Isolate ROS +
poses + mapping from gsplat: set `render_backend="cuda"`, `async_mapping=False`.
A clean run here means the data path works and any later problem is gsplat- or
threading-specific.

**5.3 Switch to gsplat.** Set `render_backend="gsplat"`, keep `async_mapping=False`.
Compare the exported result and per-frame log to 5.2. A crash here is a gsplat
API/convention issue — rerun `render_backend_selftest.py`.

**5.4 Turn on async mapping.** Set `async_mapping=True`. Watch the per-frame log:
```
Frame k/N | FPS=.. | gaussians=.. | recv=.. drop=..
```
`drop` climbing means the mapper can't keep up — lower `mapping.num_iters` or
raise `ros.process_every_n`.

**5.5 Stress the transport / poses.** If the link is bandwidth-limited, set
`ros.transport="compressed"`. If poses look unreliable, `tracking.mode="auto"`
(default) auto-refines only when odom is discontinuous/stale.

---

## 6. Visualize the output

```bash
python3 scripts/export_ply.py     configs/zed2i/zed2i_gsplat_live.py
python3 viz_scripts/final_recon.py configs/zed2i/zed2i_gsplat_live.py     # Open3D viewer
```
Or the all-in-one launcher (run node → export → view):
```bash
bash bash_scripts/zed2i_gsplat_live.bash
```
Inspect capture geometry of the run:
```bash
python3 scripts/analyze_capture_pattern.py --params experiments/ZED2i_Captures/zed2i_gsplat_demo/SplaTAM_ZED2i_gsplat/params.npz
```

---

## 7. Config knobs cheat-sheet (debugging)

| Symptom | Try |
|---|---|
| Isolate gsplat from the rest | `render_backend="cuda"` |
| Make runs reproducible / A-B | `async_mapping=False`, fixed `seed` |
| Mapper falling behind (`drop` rising) | lower `mapping.num_iters`, raise `ros.process_every_n` |
| Limited network bandwidth | `ros.transport="compressed"` |
| Smearing that looks like drift | keep `tracking.mode="auto"`; check §8 pose-desync row |
| First frame hangs | warm gsplat JIT (`import gsplat` at build/startup) |

---

## 8. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| Node prints "Waiting for RGB CameraInfo..." forever | camera_info topic not arriving; check `ros.rgb_info_topic` and §5.1 |
| No frames processed at all | `ROS_DOMAIN_ID` mismatch, or 2-way RGB+depth sync never matches — widen `ros.sync_slop` |
| `pip install -r requirements.txt` fails on `diff-gaussian-rasterization` | You're on an old checkout — it's no longer in requirements. Pull latest; `requirements.txt` is pure-python and gsplat/the fallback install via `bash bash_scripts/install.bash`. You do NOT need that rasterizer for the gsplat path. |
| `import gsplat` fails / build error | torch/CUDA too old (use torch 2.x/CU121), or wrong `TORCH_CUDA_ARCH_LIST`; re-run `bash bash_scripts/install.bash` with `TORCH_CUDA_ARCH_LIST=<your-arch>`, or fall back to `render_backend="cuda"` (`install.bash --with-cuda-fallback`) |
| `cv_bridge` import crashes after importing torch | ROS/torch library clash — `source` ROS **before** launching python; if needed prepend HPC-X/ROS libs to `LD_LIBRARY_PATH` |
| CUDA out of memory | lower `data.desired_image_*`, `mapping_window_size`, or `mapping.num_iters` |
| Splat looks smeared / "drifty" | pose↔image desync — prefer `transport="compressed"` to reduce lateness, keep the timestamp/SLERP pose association (default), tighten `ros.sync_slop`; verify odom actually publishes (`ros2 topic hz /zed/zed_node/odom`) |
| First-frame long pause then runs | gsplat JIT-compiling kernels; warm with `python -c "import gsplat"` after install |

---

## 9. Once it's stable → Docker

The same pipeline runs in `docker/demo/compose.yml` (CUDA-enabled `splatam`
container). Select the gsplat node with `NODE=gsplat docker compose up`. See
`docs/gsplat_realtime.md`. Do this only after §5 passes bare metal — then any
remaining issue is environment/packaging, not the pipeline.
