# ZED2i → ROS2 → SplaTAM (Live Gaussian Splatting)

## Overview

This setup streams live RGB-D data from a **ZED2i camera** running on an NVIDIA Orin and performs **online Gaussian Splatting (SplaTAM)** on a separate PC.

### System Architecture

```
ZED2i Camera
      │
      ▼
Orin (ROS2 + ZED node)
      │   DDS / ROS2 network
      ▼
PC (SplaTAM)
  scripts/zed2i_demo.py
      │
      ▼
Gaussian Splat (params.npz)
      │
      ▼
viz_scripts/final_recon.py
```

---

## Hardware & Software

### Hardware

* ZED2i camera
* NVIDIA Orin (runs ZED ROS2 node)
* PC with CUDA GPU (runs SplaTAM)

### Software

* ROS2 Humble
* ZED ROS2 wrapper
* Python 3.10
* PyTorch + CUDA
* Open3D
* cv_bridge
* message_filters

---

## ROS2 Network Setup

Make sure both the Orin and PC use the same ROS domain:

```bash
export ROS_DOMAIN_ID=77
```

Both devices must be on the same network.

---

## ROS Topics Used

The following topics are used by the pipeline:

### RGB

```
/zed/zed_node/rgb/color/rect/image
/zed/zed_node/rgb/color/rect/image/camera_info
```

### Depth (aligned)

```
/zed/zed_node/depth/depth_registered
/zed/zed_node/depth/depth_registered/camera_info
```

### Optional Odometry

```
/zed/zed_node/odom
```

---

## Expected Image Encodings

Observed from the ZED2i ROS node:

```
RGB encoding:   bgra8
Depth encoding: 32FC1
```

* `bgra8` → converted to RGB inside the script
* `32FC1` → depth already in meters

---

## Project Structure

```
configs/
 └── zed2i/
      └── online_ros2.py

scripts/
 └── zed2i_demo.py

viz_scripts/
 └── final_recon.py

experiments/
 └── ZED2i_Captures/
```

---

## Configuration File

Location:

```
configs/zed2i/online_ros2.py
```

### Core Runtime Settings

```python
primary_device = "cuda:0"
seed = 0

num_frames = 6
```

---

### ROS Topic Settings

```python
ros=dict(
    rgb_topic="/zed/zed_node/rgb/color/rect/image",
    rgb_info_topic="/zed/zed_node/rgb/color/rect/image/camera_info",

    depth_topic="/zed/zed_node/depth/depth_registered",
    depth_info_topic="/zed/zed_node/depth/depth_registered/camera_info",

    use_odom=True,
    odom_topic="/zed/zed_node/odom",

    rgb_encoding="bgra8",
    depth_encoding="32FC1",

    rgb_is_bgr=True,
    depth_unit_scale_m=1.0,
)
```

---

### Image Resolution

```python
data=dict(
    desired_image_width=640,
    desired_image_height=360,

    densification_image_width=320,
    densification_image_height=180,
)
```

---

### Visualization Config

```python
viz=dict(
    viz_w=640,
    viz_h=360,
    viz_near=0.01,
    viz_far=50.0,

    view_scale=1.0,
    render_mode="rgb",
    show_sil=False,
    visualize_cams=True,
    offset_first_viz_cam=False,
)
```

---

## Script: `scripts/zed2i_demo.py`

### Purpose

This script runs the live online pipeline:

* Subscribes to synchronized ROS2 RGB + depth streams
* Converts and prepares tensors for SplaTAM
* Performs tracking and mapping
* Builds Gaussian splats online
* Saves the final reconstruction

### Synchronization

Uses:

* `message_filters.ApproximateTimeSynchronizer`

Inputs:

* RGB image
* Depth image
* RGB camera info
* Depth camera info

---

## Running the Pipeline

### 1. Start ZED Node (Orin)

Launch the ZED ROS2 node on the Orin as normal.

---

### 2. On the PC

Activate environment:

```bash
conda activate splatam_v2
export ROS_DOMAIN_ID=77
```

---

### 3. Run SplaTAM Online

```bash
python3 scripts/zed2i_demo.py \
  --config configs/zed2i/online_ros2.py
```

Expected console output:

```
Ready. Waiting for synced frames...
Frame 1/6 | rgb_enc=bgra8 depth_enc=32fc1
...
Saved SplaTAM output to:
experiments/ZED2i_Captures/zed2i_ros2_demo/SplaTAM_ZED2i_ROS2
```

---

## Output

Generated reconstruction:

```
experiments/ZED2i_Captures/
 └── zed2i_ros2_demo/
      └── SplaTAM_ZED2i_ROS2/
           └── params.npz
```

---

## Visualization

Run the final reconstruction viewer:

```bash
python3 viz_scripts/final_recon.py \
  configs/zed2i/online_ros2.py
```

This launches the Open3D viewer for interactive inspection.

---

## Current Status

✔ Live ZED2i ROS2 streaming
✔ RGB + depth synchronization
✔ Online SplaTAM processing
✔ Gaussian splat output saved
✔ Open3D visualization working

---

## Pipeline Summary

You now have a working live system:

```
ZED2i → ROS2 → Online SplaTAM → Gaussian Splat → Visualization
```
