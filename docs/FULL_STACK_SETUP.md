# Full-stack setup: Orin NX (ZED) → WiFi → SplaTAM (edge deployment)

Two machines, talking over ROS 2 DDS on WiFi:

- **Orin NX — sensor side (on the robot).** Runs the ZED camera +
  `zed-ros2-wrapper`, publishes **compressed** RGB / depth + odom onto the LAN.
- **Splatting machine — SLAM side (ground station).** x86_64 + NVIDIA GPU. Runs
  this repo's live SplaTAM, subscribing over WiFi.

This is an **edge deployment**: the Orin is on a moving robot, so the link is
wireless. The whole stack is tuned for that — compressed transport plus a
pipeline that always processes the freshest frame and drops the rest.

---

## 0. Why this works over WiFi

Raw image topics are ~210 Mbps at VGA@15 — a non-starter on WiFi. **Compressed
transport** (JPEG RGB + `compressedDepth` PNG) cuts that to **~25–30 Mbps at
VGA@15**, which sits comfortably inside real 5 GHz WiFi. Two design choices carry
the rest:

- **`use_compressed=True`** (default in `configs/zed2i/zed2i_splat_live.py`) —
  the SLAM node subscribes to the compressed topics and decodes them itself, in
  the worker thread, on only the freshest frame.
- **Decoupled, freshest-frame processing** — WiFi jitter and dropped frames don't
  stall SLAM; each frame logs how many it dropped, so you can read link health
  live. This is safe because the pose is anchored to ZED odom, not to
  frame-to-frame continuity.

Keep both machines on **5 GHz**. VGA@15 is the reliable default; you can push to
HD720 later if the link is strong (compressed HD720 ≈ 40–60 Mbps).

---

## Part A — Orin NX (sensor side)

Assumes JetPack is flashed (Ubuntu 22.04, CUDA, cuDNN).

### A1. ZED SDK
Install the **ZED SDK matching your exact JetPack version** from
<https://www.stereolabs.com/developers/release> (a mismatch is the #1 ZED-on-Jetson
failure). Verify: `/usr/local/zed/tools/ZED_Explorer`.

### A2. ROS 2 Humble + the compression plugins (critical)
```bash
sudo apt update
sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions python3-rosdep \
    ros-humble-compressed-image-transport \
    ros-humble-compressed-depth-image-transport
sudo rosdep init 2>/dev/null; rosdep update
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && source ~/.bashrc
```
Those two `*-transport` plugins are what make the `/compressed` and
`/compressedDepth` topics exist. **Without them there is no compressed stream**
and you're back to 210 Mbps. They compress on the *publisher* side, so they must
be on the **Orin**.

### A3. Build the ZED ROS 2 wrapper
Pin the tag to match the SDK version from A1 (topic names have changed
between wrapper releases — see the note in A6):
```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone --recurse-submodules -b humble-v4.2.5 https://github.com/stereolabs/zed-ros2-wrapper.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc && source ~/.bashrc
```
This wrapper also needs `zed_msgs` (a separate repo on some tags) and
`backward_ros` if colcon reports them missing:
```bash
cd ~/ros2_ws/src
git clone https://github.com/stereolabs/zed-ros2-interfaces.git   # zed_msgs, if not already a submodule
git clone https://github.com/pal-robotics/backward_ros.git        # if apt has no ros-humble-backward-ros
```

### A4. Network identity + WiFi-friendly DDS (do the matching parts on BOTH machines)
```bash
echo 'export ROS_DOMAIN_ID=77' >> ~/.bashrc          # SAME on both machines
# WiFi APs often drop DDS multicast, so use CycloneDDS with explicit peers:
sudo apt install -y ros-humble-rmw-cyclonedds-cpp
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
cat > ~/cyclonedds.xml <<'XML'
<CycloneDDS><Domain>
  <General><Interfaces><NetworkInterface name="wlan0"/></Interfaces></General>
  <Discovery><ParticipantIndex>auto</ParticipantIndex><Peers>
    <Peer address="ORIN_IP"/>
    <Peer address="SPLAT_IP"/>
  </Peers></Discovery>
</Domain></CycloneDDS>
XML
echo 'export CYCLONEDDS_URI=file://'$HOME'/cyclonedds.xml' >> ~/.bashrc
source ~/.bashrc
```
Put the real IPs in the file, set the correct WiFi interface name (`ip a` to
check — often `wlan0`), and use this same config (with the same two peers) on the
splatting machine. Give the Orin a stable IP (router DHCP reservation).

### A5. Launch the ZED node (positional tracking ON — SplaTAM seeds from odom)
Create `~/zed_override.yaml`:
```yaml
/**:
  ros__parameters:
    general:
      grab_resolution: 'VGA'       # reliable WiFi default; 'HD720' if the link is strong
      grab_frame_rate: 15
```
Launch:
```bash
ros2 launch zed_wrapper zed_camera.launch.py \
    camera_model:=zed2i \
    pos_tracking_mode:=GEN_3 \
    ros_params_override_path:=$HOME/zed_override.yaml
```

### A6. Confirm the COMPRESSED topics exist and tick (on the Orin)
```bash
ros2 topic hz /zed/zed_node/rgb/image_rect_color/compressed
ros2 topic hz /zed/zed_node/depth/depth_registered/compressedDepth
ros2 topic hz /zed/zed_node/odom
```
All three must report a steady rate. If the `/compressed*` topics are missing,
the transport plugins (A2) aren't installed.

---

## Part B — Splatting machine (SLAM side / ground station)

Target stack (known-good): **Ubuntu 22.04, Python 3.10, ROS 2 Humble,
torch 2.3.0+cu121, CUDA toolkit 12.1.**

### B1. Driver
```bash
nvidia-smi        # must work; install driver + reboot if not
```

### B2. ROS 2 Humble
```bash
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions \
    ros-humble-cv-bridge ros-humble-image-transport ros-humble-message-filters \
    ros-humble-rmw-cyclonedds-cpp
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && source ~/.bashrc
```
Then set the **same** `ROS_DOMAIN_ID=77`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
and `CYCLONEDDS_URI` (same peers file, correct local interface) as in A4.

### B3. CUDA toolkit 12.1 (provides nvcc to build the rasterizer)
```bash
DISTRO=ubuntu2204
wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-1   # NOT "cuda"
cat >> ~/.bashrc <<'EOF'
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
source ~/.bashrc && nvcc --version     # must print release 12.1
```

### B4. Clone this repo (working branch)
```bash
cd ~ && git clone https://github.com/ebubar/splatam-ros2.git
cd splatam-ros2 && git checkout claude/splatam-realtime-performance-lyjvzj
```

### B5. Python env — venv with system site-packages (so it sees ROS `rclpy`)
```bash
python3 -m venv --system-site-packages ~/venvs/splatam     # --system-site-packages REQUIRED
source ~/venvs/splatam/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt          # deps + builds the rasterizer
# if the rasterizer git build fails, build the vendored copy against the venv torch:
pip install --no-build-isolation ./third_party/diff-gaussian-rasterization/
```

### B6. Verify the import chain (the gate — do this before touching the network)
```bash
source /opt/ros/humble/setup.bash && source ~/venvs/splatam/bin/activate
python3 -c "import torch, diff_gaussian_rasterization, rclpy, cv2, cv_bridge, message_filters; \
from datasets.gradslam_datasets import load_dataset_config; \
print('STACK OK | cuda', torch.cuda.is_available())"     # -> STACK OK | cuda True
```

---

## Part C — Connect the two over WiFi

1. Same `ROS_DOMAIN_ID`, same `RMW_IMPLEMENTATION`, same CycloneDDS peers on both.
2. From the **splatting machine**, confirm you receive the compressed streams:
   ```bash
   ros2 topic list | grep zed
   ros2 topic hz /zed/zed_node/rgb/image_rect_color/compressed
   ros2 topic hz /zed/zed_node/depth/depth_registered/compressedDepth
   ros2 topic hz /zed/zed_node/odom
   ```
   All three ticking = you're ready. Empty list = DDS discovery isn't crossing
   (re-check the peers file and interface names in A4).
3. Clock skew between machines does **not** break frame sync — RGB, depth, and
   odom are all stamped on the Orin, and the synchronizer only compares those
   three to each other.

---

## Part D — Run it

On the **splatting machine**, repo root, ROS + venv sourced:
```bash
cd ~/splatam-ros2
bash bash_scripts/zed2i_live.bash
```
Startup log should say `Transport: compressed`. Move the ZED slowly.

**What good looks like:**
```
Transport: compressed
Frame 12/45 | FPS=3.10 | dropped=6 | gaussians=138,402
```
`dropped>0` is expected and healthy over WiFi (SLAM is slower than the camera; it
takes the newest frame). `gaussians` climbs; the viewer shows a coherent scene.

**Outputs:** `experiments/ZED2i_Captures/zed2i_ros2_demo/SplaTAM_ZED2i_ROS2/` —
`splat.ply`, `traj_tum.txt`, `map_meta.json`, and `rtabmap_export/`.

---

## Part E — Troubleshooting

**`/compressed` or `/compressedDepth` topics don't exist** — the transport
plugins aren't on the Orin. Install `ros-humble-compressed-image-transport` and
`ros-humble-compressed-depth-image-transport` (A2) and relaunch the ZED node.

**`ros2 topic list` on the splatting machine is empty** — DDS discovery isn't
crossing WiFi. Confirm both machines share `ROS_DOMAIN_ID`,
`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, and a `cyclonedds.xml` listing both real
IPs with the correct WiFi interface name. Ping each way to confirm basic
reachability first.

**Node logs `Waiting for RGB CameraInfo...` forever** — the camera-info topic
name is wrong for your ZED build. Find it and set `ros.rgb_info_topic`:
```bash
ros2 topic list | grep camera_info
```

**Frames arrive but FPS is very low / `dropped` enormous** — link is
saturated or lossy. Drop `grab_frame_rate` to 10 or 5, or lower JPEG quality on
the Orin (image_transport `.../compressed` `jpeg_quality` param, e.g. 60). VGA
before HD720.

**Depth looks wrong / decode errors in compressed mode** — the ZED's
`compressedDepth` format varies (16UC1 mm vs 32FC1 quantized). The decoder
handles both; if you see `Unsupported depth encoding`, note the `format` string
in the error and send it. As a quick test, set `use_compressed=False` on a wired
link to isolate transport from geometry.

**`No module named diff_gaussian_rasterization`** — rasterizer didn't build;
`nvcc` missing/mismatched. Redo B3, then B5's `--no-build-isolation` line.

**`No module named 'datasets.gradslam_datasets'`** — old checkout; this branch
fixes it (`git pull`).

**`import rclpy` fails in the venv** — venv wasn't `--system-site-packages`;
recreate it (B5).

**Splat smeared / drifts** — VIO-seed vs tracking balance; adjust
`tracking.lrs.cam_trans` / `cam_unnorm_rots` per `docs/RUN_SINGLE_ROBOT_TEST.md` §6.
```
