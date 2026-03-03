# ZED2i → ROS2 → SplaTAM (Live Gaussian Splatting)

---

## Overview

This setup streams live RGB-D data from a **ZED2i camera** running on an NVIDIA Orin and performs **online Gaussian Splatting (SplaTAM)** on a separate PC.

---

## System Architecture

```
ZED2i Camera
      │
      ▼
Orin (ROS2 + ZED node)
      │   DDS / ROS2 network
      ▼
PC (SplaTAM)
  scripts/zed2i_splat_live.py
      │
      ▼
Live Gaussian Splat (real-time)
      │
      ▼
Gaussian params.npz
      │
      ▼
viz_scripts/final_recon.py
```

---

# Hardware & Software

## Hardware

* ZED2i camera
* NVIDIA Orin (runs ZED ROS2 node)
* PC with CUDA GPU (runs SplaTAM)

## Software

* ROS2 Humble
* ZED ROS2 wrapper
* Python 3.10
* PyTorch + CUDA
* diff-gaussian-rasterization
* Open3D
* cv_bridge
* message_filters

---

# ROS2 Network Setup

Both Orin and PC must use the same domain:

```bash
export ROS_DOMAIN_ID=77
```

Both devices must be on the same network.

---

# ROS Topics Used

## RGB

```
/zed/zed_node/rgb/color/rect/image
/zed/zed_node/rgb/color/rect/image/camera_info
```

## Depth (Aligned)

```
/zed/zed_node/depth/depth_registered
/zed/zed_node/depth/depth_registered/camera_info
```

---

# Expected Encodings

Observed from ZED2i:

```
RGB encoding:   bgra8
Depth encoding: 32FC1
```

* `bgra8` → converted internally to RGB
* `32FC1` → already in meters

---

# Project Structure

```
configs/
 └── zed2i/
      └── zed2i_splat_live.py

scripts/
 └── zed2i_splat_live.py

bash_scripts/
 └── zed_live.bash

viz_scripts/
 └── final_recon.py

experiments/
 └── ZED2i_Captures/
```

---

# Configuration File

Location:

```
configs/zed2i/zed2i_splat_live.py
```

## Core Settings

```python
primary_device = "cuda:0"
seed = 0
num_frames = 200
```

---

## ROS Settings

```python
ros=dict(
    rgb_topic="/zed/zed_node/rgb/color/rect/image",
    rgb_info_topic="/zed/zed_node/rgb/color/rect/image/camera_info",

    depth_topic="/zed/zed_node/depth/depth_registered",
    depth_info_topic="/zed/zed_node/depth/depth_registered/camera_info",

    depth_unit_scale_m=1.0,
)
```

---

# Live Rendering

Live rendering is controlled via CLI flags.

### Available Flags

| Flag                 | Description                   |
| -------------------- | ----------------------------- |
| `--live_cam`         | Show live camera feed         |
| `--live_depth`       | Show depth visualization      |
| `--live_splat`       | Show real-time Gaussian splat |
| `--live_max_fps <N>` | Limit rendering FPS           |

---

# Running the Pipeline

## 1. Start ZED on Orin

Launch the ZED ROS2 node normally.

---

## 2. On the PC

Activate environment:

```bash
conda activate splatam_v2
export ROS_DOMAIN_ID=77
```

---

## 3. Run SplaTAM

### Default (no live windows)

```bash
python3 scripts/zed2i_splat_live.py \
  --config configs/zed2i/zed2i_splat_live.py
```

---

### Live Camera Only

```bash
python3 scripts/zed2i_splat_live.py \
  --config configs/zed2i/zed2i_splat_live.py \
  --live_cam
```

---

### Live Gaussian Splat (Real Renderer)

```bash
python3 scripts/zed2i_splat_live.py \
  --config configs/zed2i/zed2i_splat_live.py \
  --live_splat
```

---

### Camera + Splat

```bash
python3 scripts/zed2i_splat_live.py \
  --config configs/zed2i/zed2i_splat_live.py \
  --live_cam --live_splat
```

---

### Camera + Depth + Splat

```bash
python3 scripts/zed2i_splat_live.py \
  --config configs/zed2i/zed2i_splat_live.py \
  --live_cam --live_depth --live_splat
```

---

# Quick Launch (Recommended)

Use the bash script:

```
bash_scripts/zed_live.bash
```

### Make executable

```bash
chmod +x bash_scripts/zed_live.bash
```

### Run

```bash
./bash_scripts/zed_live.bash
```

---

### With Extra Flags

```bash
./bash_scripts/zed_live.bash --live_cam
```

```bash
./bash_scripts/zed_live.bash --live_cam --live_depth
```

---

The bash script automatically:

* Loads config
* Enables `--live_splat`
* Launches final reconstruction viewer when complete

---

# Expected Output

Console:

```
Ready. Waiting for synced frames...
Frame 1/200 | rgb_enc=bgra8 depth_enc=32fc1
...
Saved SplaTAM output to:
experiments/ZED2i_Captures/...
```

---

# Output Files

```
experiments/ZED2i_Captures/
 └── SplaTAM_ZED2i/
      └── params.npz
```

---

# Visualization

After run completes:

```bash
python3 viz_scripts/final_recon.py \
  configs/zed2i/zed2i_splat_live.py
```

This launches the Open3D viewer.

---

# Current Status

✔ Live ZED2i ROS2 streaming
✔ RGB + depth synchronization
✔ Online tracking + mapping
✔ Real-time Gaussian splat rendering
✔ Live preview windows
✔ params.npz output saved
✔ Open3D viewer working

---

# Pipeline Summary

```
ZED2i → ROS2 → Online SplaTAM → Real-Time Gaussian Splat → Save → View
```
