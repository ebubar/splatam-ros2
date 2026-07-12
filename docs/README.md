# ZED2i → ROS2 → SplaTAM (Live Gaussian Splatting)

This setup streams live RGB-D + odometry from a **ZED2i camera** (running on an NVIDIA Orin or Thor) over a ROS2 network and performs **online Gaussian Splatting (SplaTAM)** on a CUDA-capable PC, producing a splat in near-realtime.

## System Architecture

```
ZED2i Camera
      │
      ▼
Orin / Thor (ROS2 + zed_wrapper node)
      │   DDS / ROS2 network (shared ROS_DOMAIN_ID)
      ▼
PC (SplaTAM)  scripts/zed2i_splat_live.py
      │
      ▼
Gaussian params.npz  →  export_ply.py  →  splat.ply
      │
      ▼
viz_scripts/final_recon.py  (Open3D viewer)
```

## Stack Startup — Quick Reference

Full live stack, in start order (all machines on the same network with the
same `ROS_DOMAIN_ID` — the bash scripts default to `77`):

**1. Camera host (Orin/Thor) — ZED node**

```bash
ssh <user>@<orin_ip>
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=77
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i pos_tracking_mode:=GEN_3
```

**2. (Optional, recommended) RTAB-Map alongside — enables the offline
quality re-splat after capture** (see "Offline Quality Path" below):

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/zed/zed_node/rgb/color/rect/image \
  depth_topic:=/zed/zed_node/depth/depth_registered \
  camera_info_topic:=/zed/zed_node/rgb/color/rect/image/camera_info \
  odom_topic:=/zed/zed_node/odom visual_odometry:=false \
  frame_id:=zed_camera_link approx_sync:=true
```

**3. SplaTAM host (CUDA PC) — live splatting**

```bash
conda activate splatam_v2 && export ROS_DOMAIN_ID=77
./bash_scripts/zed2i_live.bash      # SLAM -> PLY export -> viewer
```

**4. Control machine (any machine on the network) — watch the splat build**

```bash
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=77
ros2 run rqt_image_view rqt_image_view /splatam/live_render
```

**5. After capture (optional) — high-quality offline re-splat** using the
loop-closed RTAB-Map poses:

```bash
# Export poses first: rtabmap-databaseViewer -> File -> Export poses -> TUM
bash_scripts/offline_refine.bash <bag_dir> poses.txt my_scene
```

No hardware? Replay a converted dataset instead of steps 1–2:
`python3 scripts/dataset_player.py --dataset <scene_dir> --rate 5` with the
node on `configs/zed2i/zed2i_replay_test.py` (see "Testing Without a Camera").

## Requirements

**Hardware**
* ZED2i camera
* NVIDIA Orin or Thor (runs the ZED ROS2 wrapper node)
* PC with CUDA GPU (runs SplaTAM) — or run everything on the edge device via Docker (see `bash_scripts/main_thor.bash`)

**Software**
* ROS2 Humble + ZED ROS2 wrapper (on the camera host)
* Python 3.10, PyTorch + CUDA, diff-gaussian-rasterization (vendored in `third_party/`)
* Open3D, cv_bridge, message_filters (on the SplaTAM host)

## ROS Topics Used

The SplaTAM node subscribes to (configured in `configs/zed2i/zed2i_splat_live.py`):

```
/zed/zed_node/rgb/color/rect/image                 (RGB, bgra8)
/zed/zed_node/rgb/color/rect/image/camera_info
/zed/zed_node/depth/depth_registered               (depth, 32FC1, meters)
/zed/zed_node/depth/depth_registered/camera_info
/zed/zed_node/odom                                 (ZED positional tracking)
```

RGB and depth are time-synchronized; camera poses are kept in a buffer and
**interpolated to each image's exact timestamp** (matching the ZED SDK's
query-pose-at-frame-time behavior — pairing images with unsynchronized odom
messages caused layer-offset ghosting). The pose source is configurable:
`ros.pose_source="pose"` uses `/zed/zed_node/pose` (positional tracking with
loop corrections — recommended for small-scene captures), `"odom"` uses raw
VIO odometry. SplaTAM tracking then refines the interpolated pose.

## Configuration

Everything is driven by one config file: `configs/zed2i/zed2i_splat_live.py`.

Key settings:

| Setting | Meaning |
| --- | --- |
| `num_frames` | Frames to process before saving and exiting |
| `full_res_width/height` | Working resolution (default 640×360) |
| `ros.process_every_n` | Process every Nth synced frame (throttle) |
| `ros.min_depth_m` / `max_depth_m` | Depth clipping range |
| `map_every`, `keyframe_every` | Mapping / keyframe cadence |
| `tracking.num_iters`, `mapping.num_iters` | Optimization iterations per frame |

See `docs/TUNING.md` for how these affect splat quality.

## Running the Pipeline

### Option A: Live camera

**Terminal A — camera host (Orin/Thor):**

```bash
ssh <user>@<orin_ip>
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=<domain_id>
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i pos_tracking_mode:=GEN_3
```

**Terminal B — SplaTAM PC:**

```bash
conda activate splatam_v2       # or your environment
export ROS_DOMAIN_ID=<domain_id>
./bash_scripts/zed2i_live.bash
```

`zed2i_live.bash` runs the full pipeline: live SplaTAM → PLY export → Open3D viewer.
To run just the SLAM node:

```bash
python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_splat_live.py
```

Both hosts must be on the same network with the same `ROS_DOMAIN_ID`
(the bash scripts default to `77`).

### Option B: Recorded ROS2 bag

Two terminals on the PC, same `ROS_DOMAIN_ID`:

```bash
# Terminal A
python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_splat_live.py

# Terminal B
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=<domain_id>
ros2 bag play <bag_name>
```

SplaTAM can start before or after playback — it waits for synced frames.
Verify a bag has the required topics with `ros2 bag info <bag_path>`.

### Option C: Fully automated (record on Orin, run on PC)

```bash
bash bash_scripts/main.bash <run_name> <orin_ip> <pc_ip> <duration_sec>
# e.g.
bash bash_scripts/main.bash zed2i_walk 10.131.7.87 10.131.7.185 60
```

This launches the ZED node on the Orin, records a bag for `<duration_sec>`,
copies it to the PC, then plays it back into a freshly started SplaTAM.

For the all-in-one edge (Thor) Docker workflow, see `bash_scripts/main_thor.bash`
and `docker/`.

## Offline Quality Path (RTAB-Map loop-closed poses)

The live splat uses ZED VIO poses, which drift. For a much cleaner splat,
record a bag, get globally consistent poses from RTAB-Map (loop closure +
pose-graph optimization), and re-run SplaTAM offline with those poses fixed:

```
bag ──> RTAB-Map ──> optimized poses (TUM)
 │                        │
 └──> rtabmap2dataset.py ─┴──> dataset ──> splatam.py (use_gt_poses=True)
```

### 1. Run RTAB-Map

Either live during capture (it's light enough to run alongside), or on the
bag afterwards. It consumes the same ZED topics plus the ZED odometry:

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/zed/zed_node/rgb/color/rect/image \
  depth_topic:=/zed/zed_node/depth/depth_registered \
  camera_info_topic:=/zed/zed_node/rgb/color/rect/image/camera_info \
  odom_topic:=/zed/zed_node/odom \
  visual_odometry:=false \
  frame_id:=zed_camera_link \
  approx_sync:=true
```

`visual_odometry:=false` keeps the ZED's VIO as the odometry source;
RTAB-Map adds the pose graph and loop closures on top. Raise
`Rtabmap/DetectionRate` (default 1 Hz) for denser output poses.

### 2. Export optimized poses (TUM format)

```bash
rtabmap-databaseViewer ~/.ros/rtabmap.db
# File -> Export poses -> TUM format -> save as poses.txt
```

### 3. Convert + splat

```bash
bash_scripts/offline_refine.bash <bag_dir> poses.txt [scene_name]
```

This converts the bag + poses into a NeRFCapture-format dataset
(`scripts/rtabmap2dataset.py`), runs offline SplaTAM with poses fixed
(`configs/zed2i/zed2i_offline_rtabmap.py`), exports the PLY, and opens the
viewer. Output lands in
`experiments/ZED2i_Captures/<scene_name>/SplaTAM_ZED2i_Offline/`.

Notes:

* The converter needs a sourced ROS2 environment (`rosbag2_py`).
* Poses exported with `frame_id:=zed_camera_link` are handled by default;
  pass `--pose-frame optical` to `rtabmap2dataset.py` if your poses are
  already in the left optical frame.
* One frame is extracted per optimized graph node; with the default 1 Hz
  detection rate a 2-minute bag yields ~120 frames.

## Watching the Splat Build in Realtime

The live node renders the current reconstruction from the current camera pose
every frame and publishes it as a ROS image on `/splatam/live_render`
(config: `viz.publish_live_render`, `viz.render_every`). Watch it from **any
machine on the ROS network** (e.g. the control machine):

```bash
ros2 run rqt_image_view rqt_image_view /splatam/live_render
```

Set `viz.render_save_every=N` to also save every Nth render under the run
directory for later inspection.

## Testing Without a Camera (dataset replay)

`scripts/dataset_player.py` replays any converted dataset (rgb/, depth/,
transforms.json) as ZED-style topics, including odometry, so the full live
pipeline can be exercised on a dev machine:

```bash
# Terminal A
python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_replay_test.py
# Terminal B
python3 scripts/dataset_player.py --dataset experiments/ZED2i_Captures/<scene> --rate 5
```

## Handling ZED Frame Drops / Slow Input Over ROS

Symptoms: the live node logs frames far apart in time, tracking degrades,
and the splat gets ghosty. Diagnose *after the fact* with
`scripts/capture_report.py` — big translation/rotation steps between
processed frames are the fingerprint of drops or too-fast motion.

Mitigations, in order of effectiveness:

1. **Record a bag on the camera host and process offline** (the
   `main.bash` / offline-quality path). Bag recording on the Orin drops far
   fewer frames than shipping images over WiFi/DDS to a busy SLAM process,
   and offline replay can run slower than realtime — zero drops.
2. **Lower the published resolution/rate at the source** (ZED wrapper
   params, e.g. `pub_resolution: CUSTOM`, `pub_frame_rate: 15`): fewer/smaller
   messages beat compressed transports for latency.
3. **Match the processing rate honestly**: raise `ros.process_every_n` so
   SplaTAM works on a steady cadence instead of draining the sync queue in
   bursts. A regular 3 Hz beats an erratic 4–10 Hz.
4. **DDS tuning on lossy networks**: prefer wired links for the camera host;
   consider the zenoh bridge (`docs/zenoh_setup.txt`, `bash_scripts/zenoh/`)
   over WiFi.
5. **Slow the robot** — especially turns. The capture report's rotation-step
   metric is the one to watch; >6°/processed-frame costs quality fast.

## Object-of-Interest High-Res Splat (automated ROI)

Goal: robot walkthrough gives a low-res scene splat; an object detector (or
you) picks out items of interest; those get re-splatted at high quality from
only the frames that observed them. One command:

```bash
# text prompt (open-vocabulary, needs: pip install transformers)
bash_scripts/object_splat.bash experiments/ZED2i_Captures/<scene> --prompt "a keyboard"

# or point at a rectangle on a frame (no extra deps)
bash_scripts/object_splat.bash experiments/ZED2i_Captures/<scene> --rect <frame> <x> <y> <w> <h>
```

Set `OBJECT_SPLAT_MASK=1` to additionally mask depth outside the ROI in the
subset frames — the entire Gaussian budget then goes to the object itself
(object-only splat, no background; experimental).

Under the hood: `scripts/detect_roi.py` finds the object in the RGB frames
(OWL-ViT for prompts) and back-projects it through the depth into a 3D box
(`roi_<label>.json`); `scripts/roi_refine.py --roi-json` builds the frame
subset; SplaTAM re-splats it at 200 mapping iterations and exports the PLY.
Each stage can also be run separately — see the script docstrings. Heavier
scene-graph pipelines (e.g. ConceptGraphs) can drop in by writing the same
`roi_<label>.json` format: `{"box": [xmin, xmax, ymin, ymax, zmin, zmax]}`
in the splat's first-camera coordinate frame.

## Region-of-Interest Refinement (SuperSplat as the GUI)

To get a high-quality splat of one region (e.g. an object of interest):

1. Open the full `splat.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor).
2. Box-select the region, invert selection, delete, and export the remaining
   splats as a PLY. Don't move/rotate the splat before exporting.
3. Build a dataset subset of only the frames that observed that region:

```bash
python3 scripts/roi_refine.py --roi-ply roi.ply \
    --dataset experiments/ZED2i_Captures/<scene> --out <scene>_roi
```

4. Re-splat just those frames with a big quality budget:

```bash
SPLATAM_SCENE=<scene>_roi SPLATAM_MAPPING_ITERS=200 SPLATAM_RUN_NAME=roi_refined \
    python3 scripts/splatam.py configs/zed2i/zed2i_offline_rtabmap.py
```

`--box xmin xmax ymin ymax zmin zmax` works instead of `--roi-ply` if you
already know the bounds (splat world frame = first-camera coordinates).

## Output

```
experiments/ZED2i_Captures/zed2i_ros2_demo/SplaTAM_ZED2i_ROS2/
  ├── params.npz                 # Gaussian parameters + trajectory
  ├── splat.ply                  # after export_ply.py
  └── depth_color_debug/         # per-frame colorized depth PNGs
```

View / export manually:

```bash
python3 viz_scripts/final_recon.py configs/zed2i/zed2i_splat_live.py
python3 scripts/export_ply.py    configs/zed2i/zed2i_splat_live.py
```

PLY splats can also be viewed in [SuperSplat](https://playcanvas.com/supersplat/editor).

## Troubleshooting

* **No frames arriving:** check `ROS_DOMAIN_ID` matches on both hosts, and that
  the topics above are visible with `ros2 topic list`. Zenoh users: see
  `docs/zenoh_setup.txt` and `bash_scripts/zenoh/`.
* **"Waiting for RGB CameraInfo":** the camera_info topic isn't being received;
  verify the info topic name in the config matches the ZED wrapper's output.
* **Frames drop / node lags:** raise `ros.process_every_n` or lower the working
  resolution in the config.
