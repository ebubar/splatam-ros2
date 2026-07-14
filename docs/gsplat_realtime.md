# Near-realtime gsplat splatting on ZED poses

This pipeline replaces the slow, non-commercial CUDA rasterizer with **gsplat**
(Apache-2.0) and restructures the ZED2i live node for near-realtime operation
while keeping the existing ROS2 + ZED-pose front end. It answers the "traditional
VGGT/MapAnything + Gsplat" request with a lighter, defense-license-clean design.

## Why this instead of VGGT/MapAnything + a new engine

The ZED2i already provides camera poses (SDK positional tracking) and neural
depth, so a COLMAP-style SfM front-end mostly re-derives what we already have.
The real problems in the old path were: a slow rasterizer, wasted per-frame pose
tracking, synchronous mapping that blocked ingestion, and **network-induced
pose↔image desync** (not bad odometry).

### Licensing (for potential defense work)

| Component | License | Defense-OK |
|---|---|---|
| gsplat (nerfstudio) | Apache-2.0 | ✅ default engine |
| INRIA diff-gaussian-rasterization (old engine) | research/non-commercial | ❌ fallback only |
| ZED SDK positional tracking | Stereolabs (your sensor SDK) | ✅ primary poses |
| MapAnything `facebook/map-anything-apache` | Apache-2.0 | ✅ optional SfM |
| MapAnything `facebook/map-anything` | CC-BY-NC-4.0 | ❌ |
| VGGT (code + `VGGT-1B-Commercial`) | commercial **excl. military** | ❌ disqualified |

**Bottom line:** gsplat + ZED poses is the rugged, license-clean path. If a
learned SfM front-end is required, use **MapAnything-Apache, not VGGT**.

## What changed

- **Engine seam** (`utils/render_backend.py`): `get_loss` / `add_new_gaussians`
  now render through a backend switch — `render_backend="gsplat"` (default) or
  `"cuda"` (unchanged fallback). Offline `rgbd_slam` is untouched.
- **Trusted poses + gated fallback**: no per-frame tracking by default; a short
  photometric refinement fires only when the pose looks unreliable
  (`tracking.mode="auto"`).
- **Network-robust ingestion** (`utils/pose_buffer.py`, `utils/pose_source.py`):
  RGB+depth are synced 2-way; odometry is buffered and interpolated (SLERP) at
  the *image* timestamp, so the pose always matches the pixels. Optional
  compressed transport (`ros.transport="compressed"`).
- **Async mapping**: the ROS callback only enqueues; a mapper thread does all
  CUDA work (`async_mapping=True`). Set `False` for a deterministic A/B.
- **Capture guidance** (`utils/capture_guidance.py`, off by default) +
  `scripts/analyze_capture_pattern.py` to guide/assess capture geometry.

## Run

Runs both locally and in Docker. gsplat needs an NVIDIA GPU (CUDA); the Docker
`splatam`/`thor` image provides it (NGC CUDA base + `runtime: nvidia`), and ROS
Jazzy is installed into that same container at entrypoint. The `zed_ros2`
container is ROS/ZED-only and has no gsplat.

```bash
# Local (system ROS + CUDA + torch):
bash bash_scripts/zed2i_gsplat_live.bash
# or: python3 scripts/zed2i_gsplat_live.py --config configs/zed2i/zed2i_gsplat_live.py

# Docker (compose): the pipeline is node-selectable and backward compatible.
#   default (unset NODE) -> original CUDA node
#   NODE=gsplat          -> realtime gsplat node
cd docker/demo && NODE=gsplat docker compose up
# equivalently inside the running splatam container:
#   NODE=gsplat bash bash_scripts/splat_pipeline_thor.bash
```

Key config switches (`configs/zed2i/zed2i_gsplat_live.py`):
`render_backend`, `pose_source`, `async_mapping`, `tracking.mode`,
`ros.transport`, `gsplat.render_mode`, `mapping.num_iters`.

## Build

`gsplat==1.4.0` is added to `requirements.txt` / `environment.yml`. Set
`TORCH_CUDA_ARCH_LIST` to your GPU before install (desktop `8.9`/`9.0`; Jetson
Thor `11.0`). Docker: `docker/Dockerfile.splatam` (build-arg `CUDA_ARCH`) and
`docker/Dockerfile.thor` build gsplat + the CUDA fallback with an AOT import check.

## Verify

1. `python3 scripts/tools/render_backend_selftest.py` — diffs cuda vs gsplat on
   known Gaussians (run on the GPU box). RGB/alpha should match.
2. Deterministic A/B: `async_mapping=False`, replay one rosbag with
   `render_backend=cuda` then `gsplat`; compare `export_ply.py` output + FPS.
3. Realtime: `async_mapping=True`; watch the per-frame `FPS` / `drop` log.
4. Pose A/B (PM): `pose_source=zed_odom` vs `mapanything`.
5. Capture geometry: `python3 scripts/analyze_capture_pattern.py --params <params.npz>`.

> Note: this repo checkout was developed/validated by review + byte-compile only;
> steps 1–5 require the GPU/ROS/ZED stack and should be run on the target machine.
