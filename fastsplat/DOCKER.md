# fastsplat — Docker install & run

Step-by-step for running the fastsplat pipeline (MapAnything SfM → object
isolation → gsplat) in Docker.

> **Where does this run?** The fastsplat **compute** image (MapAnything + gsplat)
> is built for an **x86_64 machine with an NVIDIA dGPU** — the same kind of box
> you already run the SplaTAM image on. The ZED stays on the Orin/Thor and just
> streams frames over the existing ROS2/Zenoh transfer. Do **not** build this
> image on the Jetson itself.

---

## 0. Prerequisites (one time, on the GPU machine)

- NVIDIA driver installed (`nvidia-smi` works)
- Docker (`docker --version`)
- NVIDIA Container Toolkit, so Docker can see the GPU:

```bash
# quick check — this must print your GPU table:
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If that fails, install the toolkit:
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

---

## 1. Get the code

```bash
git clone https://github.com/ebubar/splatam-ros2.git
cd splatam-ros2
git checkout claude/fast-splat-sparse-images-o2jjsy
```

---

## 2. Build the fastsplat image

```bash
docker build -t fastsplat:latest -f docker/Dockerfile.fastsplat .
```

This installs gsplat and MapAnything (Apache checkpoint) on top of the NGC
PyTorch base. **First build is slow** (gsplat compiles a CUDA extension) — 10–20
min is normal. It only happens once.

> If your GPU arch isn't in the default list, pass it explicitly (Ampere `8.6`,
> Ada `8.9`, Hopper `9.0`):
> ```bash
> docker build -t fastsplat:latest -f docker/Dockerfile.fastsplat \
>   --build-arg TORCH_CUDA_ARCH_LIST="8.9" .
> ```

---

## 3. Run — Path A: from a folder of images (simplest, recommended first)

Put a sparse set of images of your object (20–40 photos from different angles) in
the repo, e.g. `data/mug/images/`. Then:

```bash
# open a shell in the GPU container (repo is mounted at /workspace)
./docker/run_fastsplat.sh

# inside the container:
python -m fastsplat.run_pipeline \
    --config configs/fastsplat/fastsplat.py \
    --image-dir data/mug/images
```

Or as a one-shot (no interactive shell):

```bash
./docker/run_fastsplat.sh python -m fastsplat.run_pipeline \
    --config configs/fastsplat/fastsplat.py \
    --image-dir data/mug/images
```

**Result:** `experiments/FastSplat/object_splat/splat/object_splat.ply`
(a standard 3DGS `.ply` — drag it into <https://playcanvas.com/supersplat/editor>).

Intermediate outputs you can inspect:
- `.../sfm/points.ply` — full fused MapAnything cloud (sanity-check poses/geometry)
- `.../object/points.ply` — the isolated object cloud (what gsplat will splat)

---

## 4. Run — Path B: live from the ZED (uses your existing ROS2/Zenoh transfer)

Two containers share a folder: the **ROS2/ZED** container captures sparse
keyframes; the **fastsplat** container splats them.

**4a. Bring up the ZED → ROS2 transfer** (as you do today), so the ZED topics are
visible on the GPU machine:

```bash
bash bash_scripts/zenoh/start.bash <orin_ip> <local_ip>
```

**4b. Capture sparse keyframes** — run this inside the ROS2/ZED container
(`docker/Dockerfile.ros2_zed`), which has `rclpy` + `cv_bridge`. It selects
keyframes by camera motion and writes them into the repo:

```bash
# inside the ROS2/ZED container, at the repo root:
bash bash_scripts/fastsplat_capture.bash \
    configs/fastsplat/fastsplat.py \
    data/zed_capture
# move the object / camera around; it stops at ingest.max_frames (default 40)
```

> Already have a rosbag instead of a live camera? Replay it into the same
> capture node — in the ROS container: `ros2 bag play <bag>` in one shell and the
> `fastsplat_capture.bash` command above in another.

**4c. Splat it** — in the fastsplat GPU container, run only SfM → object → splat
on the captured folder:

```bash
./docker/run_fastsplat.sh python -m fastsplat.run_pipeline \
    --config configs/fastsplat/fastsplat.py \
    --image-dir data/zed_capture \
    --stage sfm,object,splat
```

(`--image-dir` here points at the capture folder; `--stage sfm,object,splat`
skips re-ingesting.)

---

## 5. Iterate quickly

Each stage writes to its own dir and the next stage reads it, so re-run just the
part you're tuning without redoing SfM:

```bash
# re-run only object isolation + splatting after editing the config
./docker/run_fastsplat.sh python -m fastsplat.run_pipeline \
    --config configs/fastsplat/fastsplat.py --stage object,splat
```

Common knobs (in `configs/fastsplat/fastsplat.py`):

| Want | Change |
|------|--------|
| Cleaner object cutout | `object: method: "sam2"` (needs SAM2 install — see below) |
| Less background | lower the upper bound of `object.auto_crop.depth_percentile` |
| Faster run | lower `splat.max_steps` / `splat.time_budget_s`, `sfm.max_points` |
| Sharper result | raise `splat.max_steps`; set `splat.sh_degree: 3` |
| Try trusting ZED depth/poses | `sfm.input_mode: "rgb_intrinsics_depth"` or `"full_priors"` |

---

## 6. (Optional) enable SAM2 object segmentation

`object.method: "sam2"` needs SAM2 + a checkpoint. Uncomment the `sam2` line in
`fastsplat/requirements.txt`, rebuild the image, then download a checkpoint into
the repo (e.g. `checkpoints/sam2.1_hiera_small.pt`) and point
`object.sam2.checkpoint` / `model_cfg` at it.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit not installed/configured (step 0). |
| gsplat build errors during `docker build` | set `--build-arg TORCH_CUDA_ARCH_LIST="<your arch>"`. |
| MapAnything download is slow every run | it's cached under `/workspace/.hf_cache` (mounted), so only the first run downloads. |
| `no posed frames` at the splat stage | the SfM stage didn't produce poses — check `.../sfm/points.ply` and `cameras.json`. |
| Object got cropped away | loosen `object.auto_crop` (raise `depth_percentile` upper bound / `dbscan_eps`) or set `object.method: "none"` to splat the whole scene first. |
