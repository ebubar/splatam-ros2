# Quickstart: ZED2i → ROS2 → Live SplaTAM

One pipeline, one entry point: `bash_scripts/start.bash`.

- **`local`** — ZED plugged into this machine. Live browser viewer, runs until you Ctrl-C. This is the normal path — start here.
- **`networked`** — ZED + `zed-ros2-wrapper` on a separate machine (e.g. an Orin on a robot), this machine subscribes over the network and runs SplaTAM + the live viewer. A genuinely different physical setup (camera and GPU on two different computers), not just an option — use this only when the camera really is on another machine.

Architecture rationale (why frames are decoupled from processing, why pose is seeded-then-refined, output file formats) is in [REALTIME_ARCHITECTURE.md](REALTIME_ARCHITECTURE.md) — that's background reading, not a setup guide. This doc is the setup guide, and every version number and parameter in it is one we've actually run, not a generic "should work" — where we hit a real gap, it says so.

---

## Fast path — new machine, hardware already in hand

The non-negotiables, in order. Everything here is explained in detail further down; this is just the sequence that avoids re-discovering the same failures we already hit.

1. **Splatting machine**: install per §1 exactly (pinned versions) — `bash_scripts/start.bash check` must print all green before you touch a camera.
2. **Camera host** (skip if `local` mode, camera's on the splatting machine): install SDK + matching wrapper tag per §2.
3. **If the camera host has a USB WiFi dongle**, check `lsusb -t` and make sure the ZED is on a *different* USB port/hub path than the dongle *before* you launch anything — persistent `CORRUPTED FRAME` errors from a WiFi-adjacent USB3 camera is real interference we hit, not a fluke (§7).
4. **Launch the camera, then verify with `ros2 topic hz`, not `ros2 topic list`.** `topic list` shows advertised topics whether or not anything is actually publishing to them — we lost time to a topic that was listed but dead. `hz` proves data is flowing (§2d, §7).
5. **Confirm positional tracking is real** before trusting it: `ros2 topic hz /zed/zed_node/pose` should show a steady rate, and `ros2 topic echo /zed/zed_node/pose --once` should show nonzero-looking values once you've moved the camera at all.
6. **Physically orient the camera right-side-up before you start capture, not after.** The reconstruction's whole world frame is anchored to the camera's pose in frame 1 — not gravity — so whatever orientation the camera is in when you hit start is what "up" means for the entire run (§7).
7. **Walk at a normal pace and watch the log's `dropped=` numbers.** A `dropped=` spike past ~50 in a single gap is the warning sign of a stall long enough to break tracking outright (map splits into disconnected pieces) rather than just drifting — see §5 if you see this regularly.

---

## 1. Install — splatting machine (the one running SplaTAM)

**This exact combination is validated (ran the full pipeline successfully):** Ubuntu 22.04, ROS2 Humble, Python 3.10 in a conda env, torch 2.3.1+cu121, CUDA toolkit 12.1 (installed *via conda*, not system apt — simpler, and what we actually used).

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
bash_scripts/start.bash check
```

Each import is checked in isolation, so a failure names the specific broken piece (e.g. `[ERR] diff_gaussian_rasterization import failed -- ...`) and a fix hint, not a generic "something's wrong." Must end with `CUDA available`.

---

## 2. Install — camera host (only for `networked` mode: Orin/robot side)

Skip this whole section for `local` mode — there the camera is on the splatting machine itself, so §2a (SDK only, no ROS wrapper needed on that same machine beyond §1) is all that applies.

### 2a. ZED SDK — validated version per platform

- **x86_64 Ubuntu 22.04: SDK 4.2.5.** Confirmed working end-to-end — camera opens, positional tracking succeeds, real captures produced. Installer: `ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run` from <https://www.stereolabs.com/developers/release>.
- **Jetson Orin NX (aarch64): SDK 5.1.1 confirmed working; SDK 5.4.1 confirmed broken.** We've now tested two physical Orins: one with SDK 5.4.1 pre-installed, where positional tracking fails deterministically (reproduced at the raw SDK level, ROS entirely out of the picture — a camera-firmware/SDK incompatibility, not a wrapper or config problem). A second Orin with **SDK 5.1.1** (wrapper tag `humble-v5.1.0-14-g72ee77c`) tracked correctly — confirmed both with a minimal headless native-SDK test (`sl::Camera::enablePositionalTracking()` → `tracking_state=OK` every frame) and live through the full ROS/SplaTAM pipeline. If you're setting up a new Jetson and get to choose, prefer the 5.1.x line over 5.4.1. This still isn't an exhaustive matrix — if your Orin came with something else, verify with the headless-native-test approach before assuming the ROS wrapper "hanging" is a wrapper problem (see §7).

Verify after install: `/usr/local/zed/tools/ZED_Explorer` opens and shows a live image.

### 2b. USB placement (Jetson/Orin with a USB WiFi dongle)

If the camera host uses a USB WiFi dongle (common on Jetson dev boards without native WiFi), **check the USB topology before you rely on the camera being stable**:

```bash
lsusb -t
```

We hit persistent `CORRUPTED FRAME` errors on every single grab — not intermittent, every frame — that turned out to be RF interference between the ZED's USB3 SuperSpeed link and a 2.4GHz USB WiFi dongle sharing a nearby port on the same hub. USB topology still showed the camera correctly negotiating USB3 (`5000M`), so this isn't a "wrong port speed" issue — it's physical proximity. Fix: move the ZED to a different USB port/hub path than the dongle (a couple centimeters of separation was enough for us). If you have a shielded USB3 cable or the dongle supports 5GHz, those are lower-effort alternatives worth trying first.

### 2c. ROS2 Humble (Orin)

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

These make `.../compressed` and `.../compressedDepth` topics exist, which matter for WiFi (raw is ~210 Mbps, a non-starter over WiFi).

### 2d. Build the ZED ROS2 wrapper — pin the exact tag matching your SDK

Topic names differ across wrapper releases, and **not always the way you'd guess from version ordering** — we've seen wrapper `humble-v4.2.5` publish `.../rgb/image_rect_color`, while a *newer* wrapper build (`humble-v5.1.0-14-g72ee77c`) went back to `.../rgb/color/rect/image`. Worse: on that newer build, the OLD path (`.../rgb/image_rect_color/compressed`) was still *advertised* by `ros2 topic list` but never actually published anything — `ros2 topic hz` on it sat at zero the whole time. **Always confirm with `topic hz`, not just `topic list`, after building or swapping a wrapper version:**

```bash
ros2 topic list | grep -E "rgb|depth|camera_info"     # what's advertised
ros2 topic hz /zed/zed_node/<candidate topic>          # confirm it's actually live
```

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

### 2e. Network identity + WiFi-friendly DDS (do the matching parts on BOTH machines)

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

Find your real WiFi interface name with `ip a` — it is **not always `wlan0`**; USB WiFi adapters commonly show up as `wlx<mac>`. Put the real IPs in, and use the identical peers file (both IPs, correct local interface name for each machine) on both sides. If you swap in a *different* Orin later (different IP), update this file on both machines — it's the most common thing to forget after a hardware swap.

---

## 3. Bring it up

### 3a. `local` — camera on this machine (the normal path)

```bash
source /opt/ros/humble/setup.bash && conda activate splatam
export ROS_DOMAIN_ID=77
cd ~/splatam-ros2
bash_scripts/start.bash local
```

Launches the camera if it isn't already publishing, then runs SplaTAM with the live browser viewer (§4) until you Ctrl-C (graceful — saves whatever's captured).

**Fast smoke test instead** (45 fixed frames, no live viewer, exits and opens a static viewer automatically — good for "did the install work" without walking around):

```bash
bash_scripts/start.bash local configs/zed2i/zed2i_local_direct.py
```

### 3b. `networked` — camera on a robot/Orin

**On the Orin** (§2 already installed): launch the camera manually (this script doesn't reach across machines to start it for you). **Make sure the camera is physically right-side-up before this step** — the world orientation gets locked in from frame 1 (see Fast path step 6 and §7).

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i pos_tracking_mode:=GEN_3
```

(A `[WARNING] GEN 2 is deprecated; consider updating to GEN 3` in the log even though you asked for GEN_3 is a benign wrapper quirk we've seen consistently — not a sign anything is broken.)

**On the splatting machine:**

```bash
source /opt/ros/humble/setup.bash && conda activate splatam
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$HOME/cyclonedds.xml
cd ~/splatam-ros2
bash_scripts/start.bash networked
```

Verify the link first if anything seems off:

```bash
ros2 topic list | grep zed                            # empty = DDS discovery isn't crossing the network
ros2 topic hz /zed/zed_node/<rgb-topic>/compressed     # exact name depends on wrapper version -- see §2d
ros2 topic hz /zed/zed_node/pose                       # confirms tracking is actually producing data
```

Per-frame log lines look like:

```
Frame 12/312 | FPS=2.3 | dropped=4 | gaussians=142,318
```

Some `dropped=` every frame is normal — the pipeline always processes the newest available frame and drops stale ones rather than queueing (real-time by design). What to actually watch: `dropped=` spiking past ~50 in a single gap, and whether `gaussians` keeps climbing steadily or explodes/plateaus abnormally — see §5.

---

## 4. The live viewer (browser, works for any mode)

`scripts/live_viewer.py`, enabled via `viz.live_viewer=True` in a config (on by default for both `local` and `networked`). It's a `viser` web server running inside the SplaTAM process — fully decoupled from the SLAM worker (it only reads state on its own timer), so it can never slow down tracking/mapping, and it works over browser HTTP with no OS window/GLFW/Wayland dependency.

Open **`http://localhost:8080`** (or the splatting machine's LAN IP, from any other device on the network) while a `local` or `networked` run is active. The page shows:

- **Orbitable point cloud** of the map as it builds, plus the current camera frustum and a trail of everywhere you've walked — the direct answer to "what have I actually covered."
- **Live tuning** sliders (`map_every`, tracking/mapping iterations, min/max depth) — these write straight into the running node's config, which is already re-read fresh every frame, so changes apply immediately with no restart.
- **Save snapshot** — exports a PLY right now without interrupting the capture.
- **True splat render** checkbox — an occasional real rasterized render (heavier than the point-cloud proxy) for a genuine quality check mid-walk.

For a proper interactive look at a saved `splat.ply` outside the live session, drag it into [SuperSplat](https://playcanvas.com/supersplat/editor) — it runs client-side in-browser, no real upload/publish step. `viz_scripts/final_recon.py` (the built-in Open3D viewer) is known-broken under native Wayland (§7); SuperSplat is the reliable fallback.

---

## 5. Tuning density vs. stability

The defaults in `configs/zed2i/zed2i_splat_live.py` (`tracking_iters=40`, `mapping.num_iters=60`, `densify_downscale_factor=2.0`) and `configs/zed2i/zed2i_networked_live_view.py` (`map_every=4`) are a validated baseline reached by testing against `configs/iphone/online_demo.py` — SplaTAM's own known-good reference config — and several live walks on real hardware. If you're getting a noticeably sparse or noticeably unstable map, here's the actual tradeoff space, not guesswork:

- **`mapping.num_iters` (optimization iterations per mapping pass) and `densify_downscale_factor` (resolution new Gaussians are sampled at — higher number = lower resolution = fewer, cheaper candidate points) are what determine how expensive each mapping pass is.** The original defaults (`num_iters=180`, `densify_downscale_factor=1.0`, i.e. full resolution) caused a **runaway feedback loop**: each mapping pass added so many points that the *next* pass got more expensive, which caused bigger real-world motion gaps, which added even more points to catch up — gaussian count reached 975K and FPS collapsed to 0.08. Matching the iPhone reference's `num_iters=60` fixed this. `densify_downscale_factor=1.0` alone was still too heavy; `4.0` (also from the iPhone reference) fixed the runaway but was visibly too sparse (66K points over a 149s walk). `2.0` is the current stable middle ground.
- **Pushing `densify_downscale_factor` below 2.0 to recover more density is risky**: at `1.5` we got a **hard tracking break** — the map split into two entirely disconnected pieces, not just smearing/drift — because one mapping pass got expensive enough (worst-case 84 dropped frames in a single gap, vs. ~31-48 at safer settings) that real motion outran what tracking could re-localize against on the other side.
- **`map_every` (how often a mapping pass fires) is the safer lever for adding density.** Lowering it from 8 to 4 (more frequent passes, same per-pass cost) got noticeably more total points (296,910 vs. 193,165 over a comparable-length walk) with no hard break — worst-case dropped-frame gap was 45, still comfortably under whatever threshold caused the break at 84. It does NOT make individual passes cheaper (a common misconception) — it just distributes the same total density growth into more, individually similar-cost passes, which in practice tolerated better than one fewer-but-bigger-mapping-pass approach.
- **The warning sign to watch, live, in the per-frame log**: `dropped=` numbers climbing past ~50-80 in a single gap. That's the signal you're approaching the threshold where tracking can lose lock entirely (map splits) rather than just drifting gracefully.
- **This still isn't loop closure.** None of the above stops ordinary SLAM drift when you walk a large loop and come back near your start point — SplaTAM has no mechanism to retroactively move Gaussians it already placed. That's a separate, harder problem; see [REALTIME_ARCHITECTURE.md](REALTIME_ARCHITECTURE.md) and the `rtabmap_export/` output (§6 outputs table) for the intended path there.
- These values were validated in `networked` mode specifically (WiFi hop + cross-machine DDS latency). `local` mode has lower end-to-end latency and may tolerate a higher `map_every` (its config default is still `8`) — untested this session, worth re-checking if you're tuning a `local` setup.

---

## 6. Outputs

`experiments/ZED2i_Captures/<scene>/<run_name>/`:

| File | Contents |
|---|---|
| `params.npz` | Gaussian map + refined poses + `frame_stamps` + keyframe indices |
| `splat.ply` | Exported splat (open in [SuperSplat](https://playcanvas.com/supersplat/editor) or PolyCam) |
| `traj_tum.txt` / `traj_keyframes_tum.txt` | TUM-format trajectories, camera-to-world |
| `map_meta.json` | Frame counts, pose mode, world-frame conventions |
| `rtabmap_export/` | Per-keyframe TUM RGB-D dataset (rgb/, depth/, associations, calibration, groundtruth) — every run produces this, specifically staged for later multi-robot/multi-session rtabmap map melding (the loop-closure path noted in §5) |

---

## 7. Troubleshooting

**`ros2` / import chain broken** — run `bash_scripts/start.bash check`; each import is tested in isolation, so the output names the specific broken piece (e.g. `[ERR] diff_gaussian_rasterization import failed -- ...`) rather than a generic "something's wrong."

**Moved this repo/env to a different machine** — this is the single most common source of "worked yesterday, broken today," and `check` output tells you which of these it is:

- **`rclpy` import fails** — almost always sourcing order. `source /opt/ros/humble/setup.bash` must happen *before* `conda activate splatam`, in *every new shell* — it's not persisted from a previous shell, and conda activating first can shadow the `PYTHONPATH` entries ROS's setup script adds. If you open a fresh terminal, source ROS first, always.
- **`diff_gaussian_rasterization` import fails, or built fine but crashes/produces garbage at render time** — the rasterizer is a compiled CUDA extension; a build on one GPU does not transfer to a different GPU model. Rebuild it on the new machine with `TORCH_CUDA_ARCH_LIST` set to *that* GPU's compute capability (8.6 = RTX 30-series, 8.9 = RTX 40-series, 9.0 = H100 — check yours with `nvidia-smi --query-gpu=compute_cap --format=csv` if unsure):
  ```bash
  pip uninstall -y diff-gaussian-rasterization
  TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation ./third_party/diff-gaussian-rasterization/
  ```
- **`CUDA not available to torch`, but `nvidia-smi` itself works fine** — usually a torch build mismatched to the installed CUDA toolkit (reinstall torch per §1's exact pinned command, don't take "latest").
- **`CUDA not available to torch`, AND `nvidia-smi` itself errors or hangs** — check for a driver/kernel-module mismatch first, *before* touching the Python env at all: `nvidia-smi` printing `Driver/library version mismatch` means the loaded kernel module and the installed driver package are different versions (common after an unattended background driver upgrade that didn't reload the kernel module). A reboot is the reliable fix; a live fix without rebooting is possible (`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`, sometimes needing the other `nvidia*` modules too) but can fail if a display session is actively using them.
- **Everything imports fine but the pipeline behaves differently than the old machine** (drift, different FPS, different tracking behavior) — check `ros.pose_init` / `ros.use_odom` and the `tracking.lrs.*` values actually loaded; these are config, not environment, and are easy to assume carried over when they didn't (e.g. testing against a different config file by habit).

**ZED node logs `CORRUPTED FRAME` continuously (every grab, not intermittent)** — on a Jetson/Orin with a USB WiFi dongle, this is very likely RF interference between the ZED's USB3 SuperSpeed link and a nearby 2.4GHz USB WiFi device, not a driver or bandwidth problem (`lsusb -t` will still show correct `5000M` negotiation). Fix: move the camera to a different USB port/hub path, physically away from the dongle — see §2b.

**Topic names don't match what a config expects** — `zed-ros2-wrapper` renames topics across releases, not always predictably by version number (we've seen both `.../rgb/image_rect_color` and `.../rgb/color/rect/image` for the *same* camera model, and one build advertised the old path via `topic list` while it was actually dead). Always verify with `topic hz`, not just `topic list`:
```bash
ros2 topic list | grep -E "rgb|depth|camera_info"      # what's advertised
ros2 topic hz /zed/zed_node/<candidate topic>            # confirm it's actually publishing
```
and set `ros.rgb_topic` / `ros.rgb_info_topic` / `ros.depth_info_topic` (and their `*_compressed_topic` counterparts) in your config to match — see `configs/zed2i/zed2i_networked_live_view.py` for a worked example overriding the base config's topics for a specific wrapper build.

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

**`Pos. Tracking not started: FAILURE` / node crashes after 3 retries, regardless of `pos_tracking_mode` (GEN_1/2/3), `imu_fusion`, or `depth_stabilization`** — this points to a genuine SDK/camera firmware incompatibility on that specific Jetson+SDK combination, not a wrapper or config problem. **Isolate SDK from ROS before assuming either is broken**: write a minimal headless test using the native SDK directly (`sl::Camera::open()`, `enablePositionalTracking()`, loop `grab()` + `getPosition()`, print to console — no GL/window dependency at all, so it works over plain SSH) and check whether `tracking_state` comes back `OK`. We've now seen this both ways on Jetson: SDK 5.4.1 fails this test outright; SDK 5.1.1 passes it and tracks correctly (§2a). If the native test also fails, the likely fix is a firmware update via `ZED_Explorer` (desktop SDK only, not runnable from a Jetson) — which requires reaching Stereolabs' servers, not guaranteed (their DNS has had real, multi-network outages). In the meantime, this pipeline doesn't strictly need ZED's own tracking: set `ros.pose_init = "constant_velocity"` (or `ros.use_odom = False`) in your config so SplaTAM's own dense tracking seeds and refines pose with no VIO input at all — keep motion slow and smooth, since this mode is more sensitive to fast motion.

**ROS wrapper launch appears to hang** — before killing it, check what it's actually doing. We've mistaken a *healthy* process for a hang before: one launch attempt showed high CPU (expected — NEURAL depth mode + tracking is genuinely expensive on a Jetson) and looked stuck, but its log showed it had already reached `=== Starting Positional Tracking ===` with valid transforms; killing it was premature. Check the log tail and `ros2 topic hz` on the expected topics before assuming it's actually stuck versus just working hard.

**Reconstruction is upside-down or tilted relative to the real room** — not a bug: the world frame is anchored to the camera's pose in frame 1 (`first_frame_w2c = Identity` in `scripts/zed2i_splat_live.py`), not to gravity. The ZED's odometry already fuses IMU data, so it accurately reports whatever orientation the camera actually had at that first frame — if the camera was tilted or upside-down when you hit start, that's exactly what "up" means for the rest of that run. Fix: physically orient the camera correctly *before* starting capture, not after. (There is currently no gravity-alignment step; if this becomes a recurring problem, that would be the code fix — using the IMU-fused orientation to auto-level the world frame at startup instead of trusting frame 1 as-is.)

**Gaussian count grows into the hundreds of thousands and FPS keeps dropping over the course of a run (not just periodic dips, a persistent downward trend)** — a runaway mapping-cost feedback loop, not ordinary load. See §5 for the fix (`mapping.num_iters` / `densify_downscale_factor`); the signature is gaussian count and per-mapping-pass cost both climbing together with no plateau.

**Splat map splits into two (or more) entirely disconnected pieces** — not ordinary drift (which smears/doubles geometry near the same location); this is tracking losing lock entirely during one very expensive mapping stall and re-localizing somewhere wrong. Check the log for a `dropped=` spike (60+) right before the break. See §5 — back off whichever density lever you last increased, or spread the same density over more/cheaper mapping passes (`map_every`) instead of fewer/heavier ones.

**Splat looks smeared / doubled / drifts as you move (without a hard split)** — tracking is over- or under-correcting the pose seed. Over-correcting (jitter, sudden jumps): lower `tracking.lrs.cam_trans` (try `0.001`) and `cam_unnorm_rots` (try `0.0002`). Under-correcting (drifts like raw odometry): raise them, or raise `tracking.num_iters`. If it's specifically a seam where you walked a loop and came back near your start, that's expected SplaTAM behavior (no loop closure) — see §5's last point.

**Interactive Open3D viewer (`viz_scripts/final_recon.py`) crashes with a GLFW/GLEW/display error** — known Open3D incompatibility with native Wayland sessions (`echo $WAYLAND_DISPLAY` non-empty). `bash_scripts/start.bash` already falls back gracefully; open the exported `splat.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor) instead (§4), or switch that one desktop session to Xorg at the login screen.

**Camera host suddenly unreachable mid-session (SSH/ping fails, was working moments ago)** — check physical power and USB-WiFi-dongle seating before assuming a software problem. We hit exactly this from a loose power connector; the fix was a simple reboot and re-launch, nothing to debug in software. After a reboot, re-verify USB topology (§2b) — a physical port change made before the reboot should persist, but confirm with `lsusb -t` rather than assuming.

**Too slow / too fast** — `mapping.num_iters` is the dominant per-frame cost; `map_every` controls mapping cadence. Both are live-tunable from the browser viewer (§4) without restarting. See §5 for the full tradeoff space if you're tuning for density vs. stability specifically.
