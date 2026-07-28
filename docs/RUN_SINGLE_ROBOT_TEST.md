# Single-robot live test runbook

Goal of this first test: confirm the reworked live pipeline produces a clean
splat **on one machine** (ZED on the splatting laptop), the same baseline that
already worked — then we move to networked / multi-robot.

Two terminals on the same laptop. Use the **same `ROS_DOMAIN_ID`** in both.

---

## 0. Prerequisites (once)

The terminal that runs SplaTAM needs **both** ROS 2 and the SplaTAM Python env
importable together (`torch` *and* `rclpy`). If the venv was built with
`--system-site-packages` after sourcing ROS, source ROS first, then activate it:

```bash
source /opt/ros/humble/setup.bash        # ROS 2 (rclpy, cv_bridge, message_filters)
source /path/to/venv/splatam/bin/activate  # SplaTAM env (torch, opencv, plyfile)
python3 -c "import rclpy, torch, cv2; print('env OK', torch.cuda.is_available())"
```

That last line must print `env OK True`. If `rclpy` isn't found, the venv can't
see the system ROS packages — rebuild it with `--system-site-packages`.

---

## 1. Terminal A — launch the ZED node (with positional tracking)

The config seeds each pose from ZED odometry, so odometry **must** be published:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77                    # must match Terminal B

ros2 launch zed_wrapper zed_camera.launch.py \
    camera_model:=zed2i \
    pos_tracking_mode:=GEN_3               # enables /zed/zed_node/odom
```

Tip for lowest latency on the first test, request VGA @ 15 fps via an override
file (`ros_params_override_path:=...` with `general.grab_resolution: 'VGA'`,
`general.grab_frame_rate: 15`), as in `bash_scripts/zed_capture.bash`.

## 2. Verify the topics are actually flowing (Terminal A or a third shell)

```bash
export ROS_DOMAIN_ID=77
ros2 topic hz /zed/zed_node/rgb/image_rect_color
ros2 topic hz /zed/zed_node/depth/depth_registered
ros2 topic hz /zed/zed_node/odom
```

All three must report a steady rate (~15 Hz). If `odom` is silent, positional
tracking didn't start — fix that before continuing (or switch to the VIO-free
mode in the troubleshooting section).

---

## 3. Terminal B — run the pipeline

```bash
source /opt/ros/humble/setup.bash
source /path/to/venv/splatam/bin/activate
export ROS_DOMAIN_ID=77                    # must match Terminal A
cd /path/to/splatam-ros2

bash bash_scripts/zed2i_live.bash configs/zed2i/zed2i_local_direct.py
```

Use `configs/zed2i/zed2i_local_direct.py` here, not the base
`zed2i_splat_live.py` directly — the base config defaults
`use_compressed=True` for the networked/WiFi deployment
(docs/FULL_STACK_SETUP.md), which needs the
`compressed-image-transport`/`compressed-depth-image-transport` ROS plugins.
On a direct wired connection those plugins are usually not installed and
there's no need for compression anyway; the local config turns it off (and
raises `num_frames`/`map_every` for a more honest quality look than the
45-frame default). If your build *does* have those plugins and you want to
exercise the compressed path locally, pass the base config instead.

This runs three stages in order:

1. `scripts/zed2i_splat_live.py` — live SLAM, **45 frames** then saves & exits.
2. `scripts/export_ply.py` — writes `splat.ply`.
3. `viz_scripts/final_recon.py` — opens the interactive viewer (press `q`/ESC).

**During the run, move the camera slowly and smoothly** — small baseline between
frames. 45 frames is a short confirmation run by design.

---

## 4. What "good" looks like

Per-frame log lines:

```
Frame 12/45 | FPS=3.20 | dropped=0 | gaussians=142,318
```

- `dropped=0` on a single machine (the buffer only drops under backlog). A few
  early drops are fine; steady `dropped>0` on one machine means mapping is
  slower than the camera — expected, harmless, that's the point of the design.
- `gaussians` climbs, then roughly stabilizes as pruning kicks in.
- The final viewer shows a coherent scene (walls/objects sharp, not smeared or
  doubled).

## 5. Where the outputs land

`experiments/ZED2i_Captures/zed2i_ros2_demo/SplaTAM_ZED2i_ROS2/`

```
params.npz                 map + refined poses + frame_stamps
splat.ply                  the splat (open in SuperSplat / PolyCam)
traj_tum.txt               per-frame trajectory (TUM, camera-to-world)
traj_keyframes_tum.txt     per-keyframe trajectory
map_meta.json              run summary (pose mode, frame counts, conventions)
rtabmap_export/            TUM RGB-D dataset for rtabmap (rgb/ depth/ *.txt calibration.yaml)
```

Quick check the rtabmap export populated:

```bash
cd experiments/ZED2i_Captures/zed2i_ros2_demo/SplaTAM_ZED2i_ROS2/rtabmap_export
ls rgb | wc -l          # ~44 keyframe images
head -3 groundtruth.txt
cat map_meta.json ../map_meta.json 2>/dev/null | head
```

---

## 6. Troubleshooting / tuning knobs

Edit `configs/zed2i/zed2i_splat_live.py`:

- **Splat looks smeared / doubles / drifts as you move** — tracking is either
  over- or under-correcting the VIO seed.
  - Over-correcting (jitter, sudden jumps): lower `tracking.lrs.cam_trans`
    (try `0.001`) and `cam_unnorm_rots` (try `0.0002`).
  - Under-correcting (drifts like raw odometry): raise them (try `0.004` /
    `0.0008`) or add tracking iters (`tracking_iters = 60`).
- **You don't trust ZED odometry at all** — set `pose_init = "constant_velocity"`
  (or `use_odom = False`). SplaTAM then tracks with no VIO seed. Keep motion
  slow and smooth; this mode is more sensitive to fast motion.
- **Too slow / too fast** — `mapping_iters` (default 180) is the biggest cost;
  lower for speed. `map_every` (10) controls mapping cadence.
- **Longer map** — raise `num_frames` (it's a hard cap; pose arrays are
  preallocated to it). Ctrl-C also saves a partial map now.
- **Want per-frame depth PNGs to debug depth** — set `save_depth_debug = True`
  (slower; off for real-time).

## 7. No camera handy? Replay a bag instead

If you have a recorded bag with the same topics, run Terminal B exactly as
above, and in place of Terminal A:

```bash
export ROS_DOMAIN_ID=77
ros2 bag play /path/to/your_bag --clock
```

---

## What to report back

- A couple of the `Frame N/45 | FPS=.. | dropped=.. | gaussians=..` lines.
- Whether the final viewer looks coherent (a photo/screenshot is ideal).
- `ls rtabmap_export/rgb | wc -l` and the first few lines of
  `rtabmap_export/groundtruth.txt`.

That confirms the single-robot rework end-to-end, and we move on to the
Gaussian-merge step for multi-robot.
