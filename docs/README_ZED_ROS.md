# SplaTAM + ZED ROS2 Streaming (Protocol + Runbook)

This repo support **live RGB-D streaming from a ZED camera over ROS2** and feed frames into **SplaTAM** for tracking + Gaussian splatting mapping.

It adds:

* `datasets/zed_ros_dataset.py` — a dataset adapter that reads RGB + Depth + CameraInfo from ROS topics and returns frames in the same format as the existing datasets.
* `configs/zed/zed_ros_stream.py` — a config that runs SplaTAM using the `zed_ros` dataset.

---

## 1) What’s happening at runtime (high-level)

When you run:

```bash
PYTHONNOUSERSITE=1 python3 scripts/splatam_zed_zed.py configs/zed/zed_ros_stream.py
````

the following pipeline runs:

1. **Config loads**

   * `scripts/splatam_zed.py` loads your config python file.
   * Sets seed, picks GPU (`primary_device="cuda:0"`), creates output dirs.

2. **Dataset is created**

   * `get_dataset()` sees `dataset_name="zed_ros"` and constructs `ZedRosDataset(...)`.
   * `ZedRosDataset`:

     * initializes ROS2 (`rclpy.init(...)` only once)
     * subscribes to RGB topic, depth topic, and camera_info
     * synchronizes messages (typically with `message_filters`)
     * blocks until it has a synchronized RGB-D + CameraInfo triple

3. **SplaTAM initialization (Frame 0)**

   * `initialize_first_timestep()` pulls `dataset[0]`:

     * `color` (H,W,3) uint8
     * `depth` (H,W,1) float32-ish (meters or mm converted)
     * `intrinsics` (4x4 or 3x3 embedded in 4x4)
     * `pose` (4x4) identity / relative pose (for live, often identity at t=0)
   * Builds initial point cloud from depth + intrinsics
   * Initializes Gaussians and camera trajectory variables.

4. **Per-frame loop**

   * **Tracking step**: optimize camera pose for the current frame.
   * **Mapping step**: optimize Gaussians with keyframe selection and densification/pruning logic.

5. **Outputs**

   * Parameters, eval artifacts, and saved results go into:

     ```
     ./zedTest/zed_Captures/zed_stream/SplaTAM_zed_ros/
     ```

---

## 2) Config parameters (what matters, what we use for ZED)

This run uses `configs/zed/zed_ros_stream.py`. The config is split into a few major blocks:

* **Top-level run controls** (device, output folders, how long to run)
* **SLAM scheduling** (how often mapping runs, how keyframes are chosen)
* **Dataset (`data`)** (what dataset adapter to use + image sizes)
* **Tracking** (camera pose optimization per frame)
* **Mapping** (Gaussian optimization / densification / pruning)
* **Live view** (optional OpenCV visualization)

### 2.1 Global run parameters

* `primary_device="cuda:0"`
  Uses the GPU. For ZED live mode, this is basically required unless you accept very slow runtime.

* `seed=0`
  Makes some randomized parts repeatable (mainly in mapping frame selection).

* `workdir`, `run_name`, `overwrite`
  Control output location and whether to overwrite an existing run directory.

* `num_frames=300`
  For live streaming this means: “process the next 300 synchronized frames from ROS, then stop.”
  If you want “run forever”, you’d typically set `num_frames=-1` **and** adjust the main loop / dataset length assumptions.

### 2.2 Scheduling (speed vs quality knobs)

* `map_every=10`
  Runs mapping every 10 frames.
  Higher = faster overall FPS, lower = more frequent map updates (slower).

* `keyframe_every=max(1, num_frames // 5)`
  With `num_frames=300`, this becomes `60` → a keyframe about every 60 frames.

* `mapping_window_size=32`
  Mapping window size. Larger = better stability/quality, but slower mapping.

* `report_global_progress_every=100`, `eval_every=1`
  Logging/evaluation cadence (mostly affects console noise and minor overhead).

### 2.3 Scene initialization parameters (Gaussian scale + stability)

* `scene_radius_depth_ratio=3`
  Estimates a scene radius from max depth. Impacts heuristics and densification behavior.

* `mean_sq_dist_method="projective"`
  Initializes Gaussian scale using projective geometry (fast + common).

* `gaussian_distribution="isotropic"`
  One scale per Gaussian (same in all directions).
  `anisotropic` gives 3 scales (more flexible but more parameters and sometimes less stable).

### 2.4 `data` block (ZED streaming specifics)

This is the most important section for live ZED.

```py
data=dict(
    dataset_name="zed_ros",
    basedir=".",              # required by framework
    sequence="zed_stream",    # required by framework
    start=0,
    end=-1,
    stride=1,
    num_frames=num_frames,
    desired_image_width=640,
    desired_image_height=360,
    densification_image_width=320,
    densification_image_height=180,

```

What matters:

* `dataset_name="zed_ros"` 
  Switch that makes `get_dataset()` construct `ZedRosDataset(...)`.

* `basedir="."`, `sequence="zed_stream"`  (framework-required)
  ROS streaming doesn’t use files, but the framework expects these fields to exist.

* `num_frames=num_frames` 
  Controls how many synchronized ROS frames are processed before exit.

* `desired_image_width/height=640x360` 
  Main tracking resolution. Higher res = more detail, lower FPS.

* `densification_image_width/height=320x180` 
  Lower-resolution stream for densification (performance-friendly).

### 2.5 Tracking block (pose-only optimization per frame)

Tracking = camera pose estimation for the current frame.

Key flags:

* `use_gt_poses=False` 
  We are not using ZED odometry / GT poses. Pose is estimated from RGB-D alignment.

* `forward_prop=False` 
  Disables constant-velocity pose prediction. Can reduce “pulsing/zooming” if the prediction overshoots.

* `num_iters=5` 
  Number of gradient steps per frame for pose optimization.

Learning rates (critical):

```py
lrs=dict(
  cam_unnorm_rots=0.001,
  cam_trans=0.004,
  # everything else = 0.0 during tracking
)
```

Interpretation:

* Tracking updates **only** camera rotation + translation.
* Gaussians are frozen (LR=0).

If you see “zooming in/out”:

* Try lowering `cam_trans` (e.g., `0.002` or `0.001`).
* Also verify depth units are meters (mm vs m mistakes cause “scale pumping”).

### 2.6 Mapping block (Gaussian optimization)

Mapping is slower and is why mapping FPS is much lower than tracking FPS.

* `num_iters=60` 
  Mapping iterations each time mapping is triggered. Lowering this increases total FPS the most.

* `add_new_gaussians=True` 
  Adds new Gaussians in regions not explained by the current rendering.

* `prune_gaussians=True` 
  Removes weak / low-opacity Gaussians to control growth.

* `use_gaussian_splatting_densification=False`
  Densification is disabled in this config (more stable + faster).

Mapping learning rates update the scene, not the camera:

```py
lrs=dict(
  means3D=0.0001,
  rgb_colors=0.0025,
  unnorm_rotations=0.001,
  logit_opacities=0.05,
  log_scales=0.001,
  cam_unnorm_rots=0.0,
  cam_trans=0.0,
)
```

### 2.7 Live view (visualization)

```py
live_view=dict(
    enabled=True,
    during_tracking=False,
    tracking_render_every=10,
    show_depth=False,
),
```

* `enabled=True` opens OpenCV windows
* `during_tracking=False` renders after pose is finalized (not every tracking iteration)
* `tracking_render_every=10` shows every 10 frames (less overhead)

---

## 3) ROS “protocol” (topics, message types, encodings)

### 3.1 Required ROS topics

The dataset expects three streams:

1. **RGB image**

* Topic (example): `/zed/zed_node/rgb/color/rect/image/compressed`
* Type: `sensor_msgs/msg/CompressedImage`

2. **Depth image**

* Preferred types:

  * `sensor_msgs/msg/Image` encoding `16UC1` (mm) or `32FC1` (meters)
  * or `sensor_msgs/msg/CompressedImage` using *compressedDepth* transport
* NOTE: `compressedDepth` only works for **single-channel** 16-bit or 32-bit depth.

3. **Camera intrinsics**

* Topic (example): `/zed/zed_node/rgb/color/rect/camera_info`
* Type: `sensor_msgs/msg/CameraInfo`

### 3.2 Encoding expectations

#### RGB

* CompressedImage typically `jpeg` or `png`
* Decoded to BGR/RGB and returned as RGB uint8.

#### Depth (IMPORTANT)

If you see an error like:

> Compression requires single-channel 32bits-floating point or 16bit raw images (input format is: bgra8)

…it means your “depth” is actually a 4-channel color image. Correct depth must be:

* `32FC1` (float32 single channel) OR
* `16UC1` (uint16 single channel)

### 3.3 Synchronization expectations

Frames must be time-aligned:

* RGB, depth, camera_info should have close timestamps
* `ApproximateTimeSynchronizer` is recommended

---

## 4) Environment and dependencies

### 4.1 Conda env / Python

```bash
conda activate splatam_v2
PYTHONNOUSERSITE=1 python3 scripts/splatam_zed_zed.py configs/zed/zed_ros_stream.py
```

### 4.2 ROS2

```bash
source /opt/ros/humble/setup.bash
```

---

## 5) How to run

### 5.1 Start the ZED ROS node (Terminal A)

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

Verify topics exist:

```bash
ros2 topic list | grep zed
```

### 5.2 Verify rates (Terminal B/C/D)

```bash
ros2 topic hz /zed/zed_node/rgb/color/rect/image/compressed
ros2 topic hz /zed/zed_node/rgb/color/rect/camera_info
ros2 topic hz <DEPTH_TOPIC>
```

### 5.3 Run SplaTAM on live stream (Terminal E)

```bash
cd ~/VEIL/SplaTAM
conda activate splatam_v2
PYTHONNOUSERSITE=1 python3 scripts/splatam_zed.py configs/zed/zed_ros_stream.py
```

---

## 6) Live visualization (“real-time splatting”)

Rendering is implemented in `scripts/splatam_zed.py` via a helper like `live_render(...)` that:

* transforms gaussians to the current frame
* renders with `GaussianRasterizer`
* shows output via `cv2.imshow`

Performance notes:

* Rendering can be a bottleneck.
* Start with `during_mapping=False` and increase `tracking_render_every` if needed.

---

## 7) Offline vs live capture modes

### 7.1 “Live but finite frames”

With `num_frames=300`: process 300 synchronized ROS frames then exit.

### 7.2 Fully continuous live

Set `num_frames=-1` and ensure the main loop/dataset supports indefinite streaming.

---

## 8) Output artifacts

Outputs are saved to:

```
{workdir}/{run_name}/
```

Example:

```
./zedTest/zed_Captures/zed_stream/SplaTAM_zed_ros/
```

---

## 9) Troubleshooting

### 9.1 `ModuleNotFoundError: No module named 'datasets'`

Run from repo root and ensure `_BASE_DIR` is inserted into `sys.path`.

### 9.2 `RuntimeError: Context.init() must only be called once`

Guard `rclpy.init()`:

```py
import rclpy
if not rclpy.ok():
    rclpy.init(args=None)
```

### 9.3 `Timed out waiting for ZED frames`

Check topic names, confirm depth is real depth, and loosen sync if needed.

---

## 10) Quick command cheat-sheet

```bash
ros2 topic list | grep zed
ros2 topic hz /zed/zed_node/rgb/color/rect/image/compressed
ros2 topic hz /zed/zed_node/rgb/color/rect/camera_info
ros2 topic echo <DEPTH_TOPIC> --once

cd ~/VEIL/SplaTAM
conda activate splatam_v2
PYTHONNOUSERSITE=1 python3 scripts/splatam_zed.py configs/zed/zed_ros_stream.py
```

---

## 11) Notes on accuracy

To improve trajectory / depth metrics:

* confirm depth units (meters vs mm)
* confirm intrinsics match the RGB stream actually used
* confirm depth is aligned/rectified to RGB
