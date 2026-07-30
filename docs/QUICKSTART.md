# Quickstart: ZED2i → ROS2 → Live SplaTAM

One pipeline, one entry point: `bash_scripts/start.bash`.

- **`local`** — ZED plugged into this machine. Live browser viewer, runs until you Ctrl-C. This is the normal path — start here.
- **`networked`** — ZED + `zed-ros2-wrapper` on a separate machine (e.g. an Orin on a robot), this machine subscribes over the network and runs SplaTAM + the live viewer. A genuinely different physical setup (camera and GPU on two different computers), not just an option — use this only when the camera really is on another machine.

(A fast 45-frame smoke test instead of the live viewer: `bash_scripts/start.bash local configs/zed2i/zed2i_local_direct.py`.)

Architecture rationale (why frames are decoupled from processing, why pose is seeded-then-refined, output file formats) is in [REALTIME_ARCHITECTURE.md](REALTIME_ARCHITECTURE.md) — that's background reading, not a setup guide. This doc is the setup guide, and every version number in it is one we've actually run, not a generic "should work" — where we hit a real gap, it says so.

---

## 1. Install — splatting machine (the one running SplaTAM)

**This exact combination is validated (ran the full pipeline successfully tonight):** Ubuntu 22.04, ROS2 Humble, Python 3.10 in a conda env, torch 2.3.1+cu121, CUDA toolkit 12.1 (installed *via conda*, not system apt — simpler, and what we actually used).

```bash
# ROS2 + the packages this pipeline's Python code imports directly
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions \
    ros-humble-cv-bridge ros-humble-image-transport ros-humble-message-filters \
    ros-humble-rmw-cyclonedds-cpp
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && source ~/.bashrc

# This repo
cd ~ && git clone https://github.com/ebubar/splatam-ros2.git
cd splatam-ros2

# Conda env -- validated versions, not "latest"
conda create -n splatam python=3.10 -y
conda activate splatam
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y     # gives nvcc 12.1 inside the env, no system install
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt          # remaining deps
pip install viser                        # the live browser viewer (see §4)

# Rasterizer build: set TORCH_CUDA_ARCH_LIST to your GPU's compute capability
# (8.6 = RTX 30-series, 8.9 = RTX 40-series, 9.0 = H100 -- check yours if different)
TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation ./third_party/diff-gaussian-rasterization/
```

No `--system-site-packages` trick needed for conda: as long as you `source /opt/ros/humble/setup.bash` *before* `conda activate splatam` in any shell that's about to run this pipeline, `rclpy` resolves via `PYTHONPATH` (which the ROS setup script sets), independent of which Python environment is active. Order matters — source ROS first, every time.

**Gate — run this before touching a camera or network:**

```bash
source /opt/ros/humble/setup.bash && conda activate splatam
python3 -c "import torch, diff_gaussian_rasterization, rclpy, cv2, cv_bridge, message_filters, viser; \
print('STACK OK | cuda', torch.cuda.is_available())"
```

Must print `STACK OK | cuda True`. `bash_scripts/start.bash check` runs the core of this same check for you at any time.

---

## 2. Install — camera host (only for `networked` mode: Orin/robot side)

Skip this whole section for `local` mode — there the camera is on the splatting machine itself, so §2a (SDK only, no ROS wrapper needed on that same machine beyond §1) is all that applies.

### 2a. ZED SDK — validated version per platform

- **x86_64 Ubuntu 22.04 (this laptop, tonight): SDK 4.2.5.** This is the version we actually confirmed working end-to-end — camera opens, positional tracking succeeds, real captures produced. Installer: `ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run` from <https://www.stereolabs.com/developers/release>.
- **Jetson Orin NX (aarch64): no validated-working version yet.** The Orin we tested had **SDK 5.4.1** pre-installed, and positional tracking fails on it deterministically — reproduced at the raw SDK level with ROS entirely out of the picture, so it's a camera-firmware/SDK issue, not something our config or the wrapper caused. We have **not** yet confirmed whether an older SDK line (e.g. 4.2.5, if a Jetson/L4T build of it exists and matches your JetPack version) avoids this. Until that's tested, treat Jetson SDK version as **open**, not solved — don't assume 5.4.1 will track correctly on your camera; check `docs/QUICKSTART.md` Troubleshooting for the exact failure signature and the no-tracking workaround.

Verify after install: `/usr/local/zed/tools/ZED_Explorer` opens and shows a live image.

### 2b. ROS2 Humble (Orin)

```bash
sudo apt update
sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions python3-rosdep \
    ros-humble-rmw-cyclonedds-cpp
sudo rosdep init 2>/dev/null; rosdep update
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && source ~/.bashrc
```

**Compressed-transport packages (`ros-humble-compressed-image-transport`, `ros-humble-compressed-depth-image-transport`)** are only needed if they aren't already there — **check before installing anything**:

```bash
dpkg -l | grep -E "compressed-image-transport|compressed-depth-image-transport"
```

These make `.../compressed` and `.../compressedDepth` topics exist, which matter for WiFi (raw is ~210 Mbps, a non-starter over WiFi). On the Orin we actually used tonight, both were already present — installing them again is a no-op, but don't treat this as a step you're guaranteed to need; confirm first.

### 2c. Build the ZED ROS2 wrapper — pin the exact tag matching your SDK

Topic names differ across wrapper releases — we hit this directly: wrapper tag `humble-v4.2.5` (paired with SDK 4.2.5) publishes `.../rgb/image_rect_color`; a Jetson build at wrapper `v5.4.1` publishes `.../rgb/color/rect/image` for the *same camera model*. Pinning the tag to match your installed SDK version is what keeps the topic names in this repo's configs correct — it's not optional hygiene, the pipeline will hang on `Waiting for RGB CameraInfo...` if it's wrong.

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone --recurse-submodules -b humble-v4.2.5 https://github.com/stereolabs/zed-ros2-wrapper.git   # matches SDK 4.2.5; use the tag matching YOUR installed SDK
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc && source ~/.bashrc
```

If colcon reports `zed_msgs` or `backward_ros` missing:

```bash
cd ~/ros2_ws/src
git clone https://github.com/stereolabs/zed-ros2-interfaces.git   # zed_msgs, if not already a submodule
git clone https://github.com/pal-robotics/backward_ros.git        # if apt has no ros-humble-backward-ros
```

### 2d. Network identity + WiFi-friendly DDS (do the matching parts on BOTH machines)

```bash
echo 'export ROS_DOMAIN_ID=77' >> ~/.bashrc          # SAME number on both machines
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
cat > ~/cyclonedds.xml <<'XML'
<CycloneDDS><Domain>
  <General><Interfaces><NetworkInterface name="YOUR_WIFI_IFACE"/></Interfaces></General>
  <Discovery><ParticipantIndex>auto</ParticipantIndex><Peers>
    <Peer address="ORIN_IP"/>
    <Peer address="SPLATTING_MACHINE_IP"/>
  </Peers></Discovery>
</Domain></CycloneDDS>
XML
echo 'export CYCLONEDDS_URI=file://'$HOME'/cyclonedds.xml' >> ~/.bashrc
source ~/.bashrc
```

Find your real WiFi interface name with `ip a` — it is **not always `wlan0`**; USB WiFi adapters commonly show up as `wlx<mac>` (we hit exactly this on a real Orin). Put the real IPs in, and use the identical peers file (both IPs, correct local interface name for each machine) on both sides.

---

## 3. Bring it up

### 3a. `local` — camera on this machine (the normal path)

```bash
source /opt/ros/humble/setup.bash && conda activate splatam
export ROS_DOMAIN_ID=77
cd ~/splatam-ros2
bash_scripts/start.bash local
```

Launches the camera if it isn't already publishing, then runs SplaTAM with the live browser viewer (§4) until you Ctrl-C (graceful — saves whatever's captured). Per-frame log lines look like:

```
Frame 12/312 | FPS=3.20 | dropped=0 | gaussians=142,318
```

`dropped=0` on one machine is normal; a few early drops are fine. `gaussians` should climb steadily.

**Fast smoke test instead** (45 fixed frames, no live viewer, exits and opens a static viewer automatically — good for "did the install work" without walking around):

```bash
bash_scripts/start.bash local configs/zed2i/zed2i_local_direct.py
```

### 3b. `networked` — camera on a robot/Orin

**On the Orin** (§2 already installed): launch the camera manually (this script doesn't reach across machines to start it for you):

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i pos_tracking_mode:=GEN_3
```

**On the splatting machine:**

```bash
source /opt/ros/humble/setup.bash && conda activate splatam
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
cd ~/splatam-ros2
bash_scripts/start.bash networked
```

Verify the link first if anything seems off:

```bash
ros2 topic list | grep zed        # empty = DDS discovery isn't crossing the network
ros2 topic hz /zed/zed_node/<rgb-topic>/compressed   # exact name depends on wrapper version, see §5
```

---

## 4. The live viewer (browser, works for any mode)

`scripts/live_viewer.py`, enabled via `viz.live_viewer=True` in a config (on by default for both `local` and `networked`). It's a `viser` web server running inside the SplaTAM process — fully decoupled from the SLAM worker (it only reads state on its own timer), so it can never slow down tracking/mapping, and it works over browser HTTP with no OS window/GLFW/Wayland dependency.

Open **`http://localhost:8080`** (or the splatting machine's LAN IP, from any other device on the network) while a `local` or `networked` run is active. The page shows:

- **Orbitable point cloud** of the map as it builds, plus the current camera frustum and a trail of everywhere you've walked — the direct answer to "what have I actually covered."
- **Live tuning** sliders (`map_every`, tracking/mapping iterations, min/max depth) — these write straight into the running node's config, which is already re-read fresh every frame, so changes apply immediately with no restart.
- **Save snapshot** — exports a PLY right now without interrupting the capture.
- **True splat render** checkbox — an occasional real rasterized render (heavier than the point-cloud proxy) for a genuine quality check mid-walk.

---

## 5. Outputs

`experiments/ZED2i_Captures/<scene>/<run_name>/`:

| File | Contents |
|---|---|
| `params.npz` | Gaussian map + refined poses + `frame_stamps` + keyframe indices |
| `splat.ply` | Exported splat (open in [SuperSplat](https://playcanvas.com/supersplat/editor) or PolyCam) |
| `traj_tum.txt` / `traj_keyframes_tum.txt` | TUM-format trajectories, camera-to-world |
| `map_meta.json` | Frame counts, pose mode, world-frame conventions |
| `rtabmap_export/` | Per-keyframe TUM RGB-D dataset for rtabmap (multi-robot roadmap; see REALTIME_ARCHITECTURE.md) |

---

## 6. Troubleshooting

**`ros2` / import chain broken** — run `bash_scripts/start.bash check`; each import is tested in isolation, so the output names the specific broken piece (e.g. `[ERR] diff_gaussian_rasterization import failed -- ...`) rather than a generic "something's wrong."

**Moved this repo/env to a different machine** — this is the single most common source of "worked yesterday, broken today," and `check` output tells you which of these it is:

- **`rclpy` import fails** — almost always sourcing order. `source /opt/ros/humble/setup.bash` must happen *before* `conda activate splatam`, in *every new shell* — it's not persisted from a previous shell, and conda activating first can shadow the `PYTHONPATH` entries ROS's setup script adds. If you open a fresh terminal, source ROS first, always.
- **`diff_gaussian_rasterization` import fails, or built fine but crashes/produces garbage at render time** — the rasterizer is a compiled CUDA extension; a build on one GPU does not transfer to a different GPU model. Rebuild it on the new machine with `TORCH_CUDA_ARCH_LIST` set to *that* GPU's compute capability (8.6 = RTX 30-series, 8.9 = RTX 40-series, 9.0 = H100 — check yours with `nvidia-smi --query-gpu=compute_cap --format=csv` if unsure):
  ```bash
  pip uninstall -y diff-gaussian-rasterization
  TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation ./third_party/diff-gaussian-rasterization/
  ```
- **`CUDA not available to torch`, but `nvidia-smi` itself works fine** — usually a torch build mismatched to the installed CUDA toolkit (reinstall torch per §1's exact pinned command, don't take "latest").
- **`CUDA not available to torch`, AND `nvidia-smi` itself errors or hangs** — check for a driver/kernel-module mismatch first, *before* touching the Python env at all: `nvidia-smi` printing `Driver/library version mismatch` means the loaded kernel module and the installed driver package are different versions (common after an unattended background driver upgrade that didn't reload the kernel module). We hit this exact failure mid-session once — the fix was a reboot; a live fix without rebooting is possible (`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`, sometimes needing the other `nvidia*` modules too) but can fail if a display session is actively using them, so reboot is the reliable option.
- **Everything imports fine but the pipeline behaves differently than the old machine** (drift, different FPS, different tracking behavior) — check `ros.pose_init` / `ros.use_odom` and the `tracking.lrs.*` values actually loaded; these are config, not environment, and are easy to assume carried over when they didn't (e.g. testing against a different config file by habit).

**Topic names don't match what a config expects** — `zed-ros2-wrapper` renames topics across releases (we've seen both `.../rgb/image_rect_color` and `.../rgb/color/rect/image` for the *same* camera model, on different wrapper tags). Always verify against the wrapper actually running:
```bash
ros2 topic list | grep -E "rgb|depth|camera_info"
```
and set `ros.rgb_topic` / `ros.rgb_info_topic` / `ros.depth_info_topic` in your config to match. Pin the wrapper tag at build time (§2c) to keep this from drifting under you.

**Node logs `Waiting for RGB CameraInfo...` forever** — camera-info topic name mismatch, same fix as above.

**`ros2 topic list` on the splatting machine is empty (networked mode)** — DDS discovery isn't crossing the network. Confirm both machines share `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, and a `cyclonedds.xml` with both real IPs and the *correct* local interface name on each side (`ip a` — don't assume `wlan0`). Ping each direction first to confirm basic reachability.

**`Failed to find a free participant index for domain N`** — CycloneDDS resource exhaustion from orphaned processes. `ros2 launch` spawns `robot_state_publisher` and the component container as siblings under its own PID; killing only the container leaks the launch process + state publisher, and a handful of leaked launches exhausts the auto-participant pool. Kill the *whole* tree:
```bash
pkill -9 -f 'ros2 launch zed_wrapper'; pkill -9 -f robot_state_publisher; pkill -9 -f component_container_isolated
```
(`bash_scripts/start.bash` already does this correctly for cameras it launched itself.) Also check for a stale local camera process still bound to the same `ROS_DOMAIN_ID`/topics from an earlier session — `ros2 node list` showing duplicate node names is the tell.

**ZED node fails with a calibration/corrupted-file error at startup** — the SDK caches each camera's factory calibration locally (`/usr/local/zed/settings/SN<serial>.conf`) after fetching it once from Stereolabs; a corrupted or partial cache (interrupted download, or Stereolabs' own servers being briefly unreachable) causes this. If you have another machine that has successfully opened the *same* camera before, copy its cached file over — it's a small plain-text file, safe to transfer via `scp` or even paste as text:
```bash
scp /usr/local/zed/settings/SN<serial>.conf <user>@<other-machine>:/usr/local/zed/settings/
```

**`Pos. Tracking not started: FAILURE` / node crashes after 3 retries, regardless of `pos_tracking_mode` (GEN_1/2/3), `imu_fusion`, or `depth_stabilization`** — this points to a genuine SDK/camera firmware incompatibility, not a wrapper or config problem (confirmed by reproducing the identical failure with the ZED SDK's own native "positional tracking" sample, entirely bypassing ROS). The likely fix is a firmware update, which requires: (a) the *desktop* SDK's `ZED_Explorer` tool — firmware updates cannot be run from a Jetson, only a PC — and (b) reaching Stereolabs' servers, which is not guaranteed (their DNS has had real, multi-network outages). In the meantime, this pipeline doesn't strictly need ZED's own tracking: set `ros.pose_init = "constant_velocity"` (or `ros.use_odom = False`) in your config so SplaTAM's own dense tracking seeds and refines pose with no VIO input at all — keep motion slow and smooth, since this mode is more sensitive to fast motion.

**Splat looks smeared / doubled / drifts as you move** — tracking is over- or under-correcting the pose seed. Over-correcting (jitter, sudden jumps): lower `tracking.lrs.cam_trans` (try `0.001`) and `cam_unnorm_rots` (try `0.0002`). Under-correcting (drifts like raw odometry): raise them, or raise `tracking.num_iters`.

**Interactive Open3D viewer (`viz_scripts/final_recon.py`) crashes with a GLFW/GLEW/display error** — known Open3D incompatibility with native Wayland sessions (`echo $WAYLAND_DISPLAY` non-empty). `bash_scripts/start.bash` already falls back gracefully; open the exported `splat.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor) instead, or switch that one desktop session to Xorg at the login screen.

**Too slow / too fast** — `mapping.num_iters` is the dominant per-frame cost; `map_every` controls mapping cadence. Both are live-tunable from the browser viewer (§4) without restarting.
