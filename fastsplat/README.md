# fastsplat — fast object-centric splats from sparse ZED images

A SplaTAM-free pipeline that turns a **sparse** set of ZED images into a good
**Gaussian splat of a single object** in **≤ 5 minutes**:

```
ZED ──(ROS2 / Zenoh transfer)──▶ ingest sparse keyframes
                               ─▶ MapAnything  (feed-forward SfM: poses + metric point cloud)
                               ─▶ object isolation  (auto 3D crop, or SAM2)
                               ─▶ gsplat  (time-boxed optimization)
                               ─▶ object_splat.ply
```

It keeps the existing ZED → ROS2 (Zenoh) transfer and drops SplaTAM. Instead of
SLAM, poses come from **MapAnything** (Apache-2.0, `facebook/map-anything-apache`)
which solves all cameras jointly, and the splat is optimized with **gsplat**.

## Why this replaces SplaTAM (and fixes the ghost splats)

The "ghost splats at different heights" you saw are a symptom of **ZED odometry
drift**: per-frame odom poses are locally fine but globally inconsistent, so the
same surface is deposited at slightly different world positions across frames.

MapAnything estimates **globally-consistent poses from the images themselves**,
so that inconsistency disappears. By default (`sfm.input_mode: rgb_intrinsics`)
we trust only the stable ZED **intrinsics** and let MapAnything re-estimate the
poses — we do **not** feed raw odom poses. If you want to experiment, the input
mode is switchable:

| `sfm.input_mode`         | images | ZED intrinsics | ZED depth | ZED odom poses |
|--------------------------|:------:|:--------------:|:---------:|:--------------:|
| `rgb`                    | ✅ | — | — | — |
| `rgb_intrinsics` (default) | ✅ | ✅ | — | — |
| `rgb_intrinsics_depth`   | ✅ | ✅ | ✅ | — |
| `full_priors`            | ✅ | ✅ | ✅ | ✅ (drift risk) |

## Pipeline stages

| Stage | Module | Output (`<workdir>/<run_name>/…`) |
|-------|--------|-----------------------------------|
| 1. ingest | `fastsplat.ingest` | `capture/` — images (+ intrinsics/depth/odom) |
| 2. SfM | `fastsplat.sfm` (MapAnything) | `sfm/` — `cameras.json`, `points.ply` |
| 3. object | `fastsplat.object` | `object/` — cropped `points.ply`, masks, `crop.json` |
| 4. splat | `fastsplat.splat` (gsplat) | `splat/object_splat.ply` |

**Object focus** (`object.method`):
- `auto_crop` (default) — no extra model: strip far background, remove outliers,
  DBSCAN-keep the dominant near-camera cluster, crop a padded box.
- `sam2` — segment the object per image with SAM2 (center-click prompt); keep
  only points inside the masks and supervise gsplat on object pixels. Requires
  installing SAM2 and a checkpoint (see `object.sam2` in the config).

**Time budget** — gsplat stops at `splat.max_steps` **or** `splat.time_budget_s`
(default 240 s), whichever comes first, to keep the whole run under 5 minutes.

## Install / build

```bash
# GPU compute image (MapAnything + gsplat); no ROS inside.
docker build -t fastsplat:latest -f docker/Dockerfile.fastsplat .
./docker/run_fastsplat.sh
```

Or bare-metal (needs torch + CUDA already installed):

```bash
pip install -r fastsplat/requirements.txt
```

## Run

**A) From a folder of images** (simplest — decoupled from ROS):

```bash
python -m fastsplat.run_pipeline \
    --config configs/fastsplat/fastsplat.py \
    --image-dir data/mug/images
```

**B) Live from the ZED** (two containers sharing a volume):

```bash
# 1) bring up the ZED -> ROS2 transfer you already have
bash bash_scripts/zenoh/start.bash <orin_ip> <local_ip>

# 2) in the ROS2/ZED container: capture sparse keyframes to a shared folder
bash bash_scripts/fastsplat_capture.bash configs/fastsplat/fastsplat.py /data/capture

# 3) in the fastsplat GPU container: SfM -> object -> splat on that folder
python -m fastsplat.run_pipeline --config configs/fastsplat/fastsplat.py \
    --stage sfm,object,splat
#   (point ingest.image_dir / the capture dir at /data/capture, or set
#    ingest.source: ros2 to capture directly if the GPU image also has rclpy)
```

Re-run just one stage while tuning (each stage reads the previous stage's dir):

```bash
python -m fastsplat.run_pipeline --config configs/fastsplat/fastsplat.py --stage object,splat
```

## Output

`splat/object_splat.ply` is a standard 3DGS `.ply` — open it in
[SuperSplat](https://playcanvas.com/supersplat/editor), PolyCam, or any 3DGS
viewer.

## Tuning cheatsheet

| Symptom | Config knob |
|---------|-------------|
| Background leaking into the splat | `object.auto_crop.depth_percentile` (lower the upper bound), or switch to `object.method: sam2` |
| Object partially cropped away | raise `object.auto_crop.depth_percentile` upper bound, raise `dbscan_eps`, or `keep_largest_cluster: false` |
| Too slow | lower `splat.max_steps` / `splat.time_budget_s`, `sfm.max_points`, `ingest.max_frames` |
| Blurry / under-fit | raise `splat.max_steps`, enable `splat.sh_degree: 3` for view-dependent color |
| Floaters around object | raise `splat.strategy.prune_opa`, tighten `object.auto_crop.outlier_std` |

> **Note:** MapAnything and gsplat are third-party research code with evolving
> APIs. The wrappers in `fastsplat/sfm` and `fastsplat/splat` cite the exact API
> they were written against; if you install a different version and an argument
> name changed, adjust it there.
