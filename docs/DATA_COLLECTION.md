# ZED2i Data Collection — Field Team One-Pager

Goal: record ROS2 bags from the robot's ZED2i that we can turn into 3D
Gaussian splats. Everything below runs on the robot's Jetson (Orin/Thor).

## 1. Start the camera

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i pos_tracking_mode:=GEN_3
```

Verify topics are up (in a second terminal, same two `source`/`export` lines):

```bash
ros2 topic hz /zed/zed_node/depth/depth_registered   # expect ~10-30 Hz
```

> If topic names differ on your wrapper version (`ros2 topic list | grep zed`),
> note the actual names and tell us — everything else still works.

## 2. Record (THE command)

```bash
ros2 bag record -o zed_capture_$(date +%Y%m%d_%H%M%S) \
  --max-cache-size 1073741824 \
  /zed/zed_node/rgb/color/rect/image \
  /zed/zed_node/rgb/color/rect/image/camera_info \
  /zed/zed_node/depth/depth_registered \
  /zed/zed_node/depth/depth_registered/camera_info \
  /zed/zed_node/odom \
  /tf /tf_static
```

Stop with **Ctrl-C** (important — a killed recorder writes a broken bag).

**Disk budget:** at HD720/15 fps expect roughly **6–7 GB per minute**.
Keep captures to **60–120 seconds** each; check free space first (`df -h .`).
If space is tight, launch the camera with `grab_resolution:=VGA` (~2 GB/min).

## 3. How to move the robot (this matters as much as the command)

* **Walk slowly** — about 0.5 m/s, slower than feels natural.
* **Turn slowly** — fast rotation is the #1 quality killer.
* Keep subjects **1–4 m** from the camera.
* For any **object of interest**: arc around it slowly (half to
  three-quarters of a circle), keeping it in frame, ~1–2 m away.
* **End where you started**, re-seeing the start area (enables loop closure).
* Good lighting; avoid pointing at bare/reflective walls for long stretches.
* Do **2–3 separate captures** rather than one long one.

## 4. Verify before leaving the site

```bash
ros2 bag info zed_capture_*
```

Check: duration matches, all 7 topics present, image topics have
`duration × frame-rate`-ish message counts (not near-zero).

## 5. Hand off

Send the **whole bag directory** (`metadata.yaml` + `.db3` files), e.g.:

```bash
scp -r zed_capture_20260706_1030 user@<pc_ip>:/data/zed_bags/
```

Include a one-line note per bag: where, what objects of interest, anything
weird (drops, lighting, fast sections).
