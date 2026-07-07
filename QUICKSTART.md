# Quickstart

Three ways in, depending on what you have. Environment setup (once) is in
[docs/LAPTOP_QUICKSTART.md §1](docs/LAPTOP_QUICKSTART.md) — conda env,
torch/CUDA, the vendored rasterizer. All commands run from the repo root with
that env active (`conda activate splatam`).

---

## A. No camera, no data — demo on a public dataset (~30 min total)

```bash
source /opt/ros/humble/setup.bash
bash_scripts/get_tum_test_data.bash        # download + build test dataset (once)

# Scene splat (the "robot walkthrough" pass)
export SPLATAM_SCENE=tum_fr1_desk_v2 SPLATAM_IMAGE_WIDTH=640 SPLATAM_IMAGE_HEIGHT=480
export SPLATAM_RUN_NAME=scene SPLATAM_MAPPING_ITERS=120
python3 scripts/splatam.py configs/zed2i/zed2i_offline_rtabmap.py
python3 scripts/export_ply.py configs/zed2i/zed2i_offline_rtabmap.py

# Object-of-interest high-res splat (the book, ~3 min)
OBJECT_SPLAT_MASK=1 bash_scripts/object_splat.bash \
    experiments/ZED2i_Captures/tum_fr1_desk_v2 --rect 60 450 115 185 175
```

View any `splat.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor).
Full walkthrough + troubleshooting: [docs/LAPTOP_QUICKSTART.md](docs/LAPTOP_QUICKSTART.md).

---

## B. You have a robot bag (recorded per [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md))

```bash
source /opt/ros/humble/setup.bash

# 1. Bag -> dataset, poses from the ZED's own odometry (no RTAB-Map needed).
#    --pose-stride: odom rate / desired fps (e.g. 30 Hz odom -> stride 6 ≈ 5 fps)
python3 scripts/rtabmap2dataset.py \
    --bag /data/zed_bags/<bag_dir> \
    --poses-from-odom --pose-stride 6 \
    --output experiments/ZED2i_Captures/robot_run1 --overwrite

# 2. Was the capture any good? (motion, blur, depth coverage)
python3 scripts/capture_report.py --dataset experiments/ZED2i_Captures/robot_run1

# 3. Scene splat  (set W/H to the bag's camera resolution)
export SPLATAM_SCENE=robot_run1 SPLATAM_IMAGE_WIDTH=1280 SPLATAM_IMAGE_HEIGHT=720
export SPLATAM_RUN_NAME=scene SPLATAM_MAPPING_ITERS=60
python3 scripts/splatam.py configs/zed2i/zed2i_offline_rtabmap.py
python3 scripts/export_ply.py configs/zed2i/zed2i_offline_rtabmap.py

# 4. Object-of-interest splat, by text prompt (1-5 min depending on frames/iters)
OBJECT_SPLAT_MASK=1 bash_scripts/object_splat.bash \
    experiments/ZED2i_Captures/robot_run1 --prompt "a chair"
```

Quality upgrade later: run RTAB-Map over the same bag for loop-closed poses
and use `--poses poses.txt` instead of `--poses-from-odom`
(docs/README.md → "Offline Quality Path").

Tip: at 1280×720 SplaTAM is ~4× slower than 640×360; for a first look use
`SPLATAM_IMAGE_WIDTH=640 SPLATAM_IMAGE_HEIGHT=360`.

---

## C. Live camera on the network — realtime streaming splat

Start order (all machines: same network, `export ROS_DOMAIN_ID=77`):

```bash
# Robot (Orin/Thor)
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i pos_tracking_mode:=GEN_3

# SplaTAM PC
./bash_scripts/zed2i_live.bash

# Any machine: watch the splat build in realtime
ros2 run rqt_image_view rqt_image_view /splatam/live_render
```

Details, rosbag replay mode, and frame-drop help: [docs/README.md](docs/README.md).

---

| I want to... | Go to |
| --- | --- |
| Give the field team recording instructions | [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md) |
| Tune splat quality | [docs/TUNING.md](docs/TUNING.md) |
| Understand the architecture / all workflows | [docs/README.md](docs/README.md) |
| Set up the environment from scratch | [docs/LAPTOP_QUICKSTART.md](docs/LAPTOP_QUICKSTART.md) |
