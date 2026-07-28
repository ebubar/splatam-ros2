# Full-stack setup: Orin NX (ZED) → home network → SplaTAM

Two machines:

- **Orin NX — sensor side.** Runs the ZED camera + `zed-ros2-wrapper`, publishes
  RGB / depth / odom onto the LAN.
- **Splatting machine — SLAM side.** x86_64 + NVIDIA GPU. Runs this repo's live
  SplaTAM, subscribing to the Orin's topics over the network.

They talk over ROS 2 DDS on your home network. Nothing but ROS topics crosses
the wire.

---

## 0. The one decision that determines success tonight: wired vs Wi-Fi

SplaTAM here subscribes to **raw** (uncompressed) image topics. At ZED **VGA
(672×376) @ 15 fps** that is roughly:

| stream | per frame | @15 fps |
|--------|-----------|---------|
| RGB (bgr8) | ~0.76 MB | ~91 Mbps |
| depth (32FC1) | ~1.0 MB | ~121 Mbps |
| **total** | | **~210 Mbps** |

- **Wired gigabit Ethernet (strongly recommended for the first working run):**
  handles this with headroom. Do this tonight. Everything below assumes it.
- **Wi-Fi:** 5 GHz real-world throughput (~200–400 Mbps) makes raw VGA@15
  marginal — it may work, but expect jitter. The pipeline tolerates jitter (it
  always processes the freshest frame and drops the rest), but saturated
  bandwidth means growing latency. **If you must use Wi-Fi, drop the ZED frame
  rate to ~5 fps** (see Part A) — raw VGA@5 is ~70 Mbps and fits comfortably.
  A compressed-transport option for full-rate Wi-Fi is in Part E.

---

## Part A — Orin NX (sensor side)

Assumes JetPack is already flashed (gives you Ubuntu 22.04, CUDA, cuDNN).

### A1. ZED SDK
Install the **ZED SDK for your exact JetPack version** from
<https://www.stereolabs.com/developers/release>. Match the versions — a
mismatched SDK/JetPack is the most common ZED-on-Jetson failure.
```bash
chmod +x ZED_SDK_Tegra_*.run && ./ZED_SDK_Tegra_*.run
# verify the camera:
/usr/local/zed/tools/ZED_Explorer
```

### A2. ROS 2 Humble
```bash
sudo apt update && sudo apt install -y software-properties-common curl
# add the ROS 2 apt repo per docs.ros.org (Humble, Ubuntu 22.04), then:
sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions python3-rosdep
sudo rosdep init 2>/dev/null; rosdep update
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && source ~/.bashrc
```

### A3. Build the ZED ROS 2 wrapper
```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone --recurse-submodules https://github.com/stereolabs/zed-ros2-wrapper.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc && source ~/.bashrc
```

### A4. Network identity (do this on BOTH machines, same value)
```bash
echo 'export ROS_DOMAIN_ID=77' >> ~/.bashrc && source ~/.bashrc
```
Give the Orin a stable IP (DHCP reservation on your router, or static). Note it
— e.g. `192.168.1.50`.

### A5. Launch the ZED node (positional tracking ON — SplaTAM seeds from odom)
```bash
ros2 launch zed_wrapper zed_camera.launch.py \
    camera_model:=zed2i \
    pos_tracking_mode:=GEN_3
```
**Lower the rate/resolution** via an override file (recommended for network).
Create `~/zed_override.yaml`:
```yaml
/**:
  ros__parameters:
    general:
      grab_resolution: 'VGA'      # smallest; 'HD720' if you're wired and want more
      grab_frame_rate: 15          # set to 5 for Wi-Fi
```
and launch with `ros_params_override_path:=$HOME/zed_override.yaml` added.

### A6. Confirm it's publishing (on the Orin)
```bash
ros2 topic hz /zed/zed_node/rgb/color/rect/image
ros2 topic hz /zed/zed_node/depth/depth_registered
ros2 topic hz /zed/zed_node/odom          # must be alive — needs pos_tracking
```

---

## Part B — Splatting machine (SLAM side)

Target stack (known-good this session): **Ubuntu 22.04, Python 3.10, ROS 2
Humble, torch 2.3.0+cu121, CUDA toolkit 12.1.** Keeping Python 3.10 + Humble
matches the Orin and lets the venv see system `rclpy`.

### B1. NVIDIA driver
```bash
nvidia-smi        # must work. If not, install the driver and reboot first.
```

### B2. ROS 2 Humble
```bash
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions \
    ros-humble-cv-bridge ros-humble-image-transport ros-humble-message-filters
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && source ~/.bashrc
echo 'export ROS_DOMAIN_ID=77' >> ~/.bashrc && source ~/.bashrc   # MATCH the Orin
```

### B3. CUDA toolkit 12.1 (provides nvcc to build the rasterizer)
pip-torch ships the CUDA *runtime* but not `nvcc`. Install the matching toolkit:
```bash
DISTRO=ubuntu2204
wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1     # NOT "cuda" (that pulls a driver)
cat >> ~/.bashrc <<'EOF'
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
source ~/.bashrc
nvcc --version        # must print release 12.1
```

### B4. Clone this repo (the working branch)
```bash
cd ~
git clone https://github.com/ebubar/splatam-ros2.git
cd splatam-ros2
git checkout claude/splatam-realtime-performance-lyjvzj
```

### B5. Python env — venv with system site-packages (so it sees ROS `rclpy`)
```bash
python3 -m venv --system-site-packages ~/venvs/splatam   # --system-site-packages is REQUIRED
source ~/venvs/splatam/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt        # installs deps AND builds the rasterizer from git
```
If the rasterizer line in `requirements.txt` fails to build, build the vendored
copy explicitly (same code, offline, forced to use the venv's torch):
```bash
pip install --no-build-isolation ./third_party/diff-gaussian-rasterization/
```

### B6. Verify the whole import chain (this is the gate)
```bash
source /opt/ros/humble/setup.bash
source ~/venvs/splatam/bin/activate
python3 -c "import torch, diff_gaussian_rasterization, rclpy, cv2, cv_bridge, message_filters; \
from datasets.gradslam_datasets import load_dataset_config; \
print('STACK OK | cuda', torch.cuda.is_available())"
```
Must print `STACK OK | cuda True`. (Run it from the repo root so `datasets`
resolves to the repo.)

---

## Part C — Connect the two over the network

1. **Same `ROS_DOMAIN_ID`** on both (77 above). Same subnet.
2. From the **splatting machine**, confirm you can see the Orin's topics:
   ```bash
   ros2 topic list | grep zed            # should list /zed/zed_node/...
   ros2 topic hz /zed/zed_node/rgb/color/rect/image   # should tick at the Orin's rate
   ```
   If the list is empty, DDS discovery isn't crossing the network — see Part E.
3. Clock skew between machines does **not** break frame sync: RGB, depth, and
   odom are all timestamped on the Orin, and the synchronizer only compares
   those three to each other. (NTP is still nice to have.)

---

## Part D — Run it

On the **splatting machine**, repo root, both ROS and the venv sourced:
```bash
cd ~/splatam-ros2
bash bash_scripts/zed2i_live.bash
```
This runs live SLAM (45 frames, then saves & exits), exports `splat.ply`, and
opens the viewer. **Move the ZED slowly and smoothly.**

**What good looks like** — log lines:
```
Frame 12/45 | FPS=3.20 | dropped=4 | gaussians=142,318
```
`dropped>0` is normal over a network (SLAM is slower than the camera; it takes
the freshest frame). `gaussians` climbs; the viewer shows a coherent scene.

**Outputs** land in
`experiments/ZED2i_Captures/zed2i_ros2_demo/SplaTAM_ZED2i_ROS2/`:
`splat.ply`, `traj_tum.txt`, `map_meta.json`, and `rtabmap_export/` (the
per-keyframe TUM dataset for multi-robot melding).

Tuning knobs (`configs/zed2i/zed2i_splat_live.py`) and the pose-seed modes are
documented in `docs/RUN_SINGLE_ROBOT_TEST.md` §6 and
`docs/REALTIME_ARCHITECTURE.md`.

---

## Part E — Troubleshooting

**`ros2 topic list` on the splatting machine doesn't show the Orin's topics.**
DDS discovery isn't crossing the LAN (often Wi-Fi APs blocking multicast). Fix by
forcing CycloneDDS with an explicit unicast peer on **both** machines:
```bash
sudo apt install -y ros-humble-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
cat > ~/cyclonedds.xml <<'XML'
<CycloneDDS><Domain><Discovery><Peers>
  <Peer address="ORIN_IP"/>
  <Peer address="SPLAT_IP"/>
</Peers></Discovery></Domain></CycloneDDS>
XML
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
```
Put both real IPs in the file, set these on both machines, relaunch everything.

**Frames arrive but the run stalls / `dropped` climbs forever with low FPS on
Wi-Fi.** Bandwidth-bound. Either go wired, or drop `grab_frame_rate` to 5 on the
Orin, or use compressed transport: on the splatting machine, relay the small
compressed topics to local raw topics with `ros2 run image_transport republish`,
remap the outputs to `/splat/rgb` and `/splat/depth`, and point the config's
`ros.rgb_topic` / `ros.depth_topic` at those. (This trades a code-free setup for
much lower bandwidth.)

**`No module named diff_gaussian_rasterization`** — the rasterizer didn't build;
`nvcc` was missing or mismatched. Redo B3, then B5's `--no-build-isolation` line.

**`No module named 'datasets.gradslam_datasets'`** — you're on an old checkout;
this branch already fixes it (the package `__init__.py` rename). `git pull`.

**`import rclpy` fails inside the venv** — the venv wasn't created with
`--system-site-packages`. Recreate it (B5).

**Build error "unsupported GNU version" (Ubuntu 24.04 / gcc 13)** — CUDA 12.1
wants gcc ≤ 12: `sudo apt install -y gcc-12 g++-12` then rebuild with
`CC=gcc-12 CXX=g++-12 pip install --no-build-isolation ./third_party/diff-gaussian-rasterization/`.

**Splat looks smeared / drifts** — the VIO-seed vs tracking balance; adjust
`tracking.lrs.cam_trans` / `cam_unnorm_rots` per `docs/RUN_SINGLE_ROBOT_TEST.md` §6.
