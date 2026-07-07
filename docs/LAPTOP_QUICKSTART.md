# Laptop Quickstart — Full Pipeline, No Camera Needed

Run the entire robot-walkthrough → low-res splat → object-of-interest →
high-res object splat pipeline on a single machine with an NVIDIA GPU, using
the public TUM `fr1_desk` dataset. Every command below has been validated
end-to-end (RTX 3070 Ti laptop, Ubuntu 22.04, ROS2 Humble).

## 0. Prerequisites

* NVIDIA GPU (8 GB VRAM is plenty at 640×480) with a recent driver
* ROS2 Humble (`/opt/ros/humble`) — only needed for bag conversion and the
  live-replay demo; the offline splatting itself doesn't use ROS
* miniconda/anaconda, ~10 GB disk

## 1. Environment (once, ~15 min)

```bash
git clone -b cleanup/docs-and-quality git@github.com:ebubar/splatam-ros2.git
cd splatam-ros2

conda create -y -n splatam python=3.10
conda activate splatam
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
grep -v "^git+" requirements.txt > /tmp/reqs.txt && pip install -r /tmp/reqs.txt
TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation ./third_party/diff-gaussian-rasterization
pip install rosbags                      # ROS1->ROS2 bag conversion (pure python)
pip install "transformers==4.44.2"      # optional: text-prompt object detection
```

`TORCH_CUDA_ARCH_LIST`: 8.6 covers RTX 30-series; use 8.9 for 40-series
(or set both: `"8.6;8.9"`).

Sanity check:

```bash
python -c "import torch, diff_gaussian_rasterization; print(torch.cuda.is_available())"
```

## 2. Get the test dataset (once, ~5 min)

```bash
source /opt/ros/humble/setup.bash    # needed for bag reading
bash_scripts/get_tum_test_data.bash
```

Downloads the TUM fr1_desk bag (~330 MB), converts it to ROS2, and builds a
SplaTAM-ready dataset at `experiments/ZED2i_Captures/tum_fr1_desk_v2`
(124 RGB-D frames with ground-truth poses standing in for RTAB-Map output).

## 3. Assess the capture pattern

```bash
python3 scripts/capture_report.py --dataset experiments/ZED2i_Captures/tum_fr1_desk_v2
```

Reports inter-frame motion, blur, and depth coverage against known-good
splatting ranges — run this on your own robot captures to diagnose frame
drops and motion that's too fast.

## 4. Scene splat ("robot walkthrough" quality pass)

```bash
export SPLATAM_SCENE=tum_fr1_desk_v2 SPLATAM_IMAGE_WIDTH=640 SPLATAM_IMAGE_HEIGHT=480
export SPLATAM_RUN_NAME=scene SPLATAM_MAPPING_ITERS=120
python3 scripts/splatam.py configs/zed2i/zed2i_offline_rtabmap.py
python3 scripts/export_ply.py configs/zed2i/zed2i_offline_rtabmap.py
```

~13 min on a 3070 Ti laptop (60 iters ≈ 7 min). View
`experiments/ZED2i_Captures/tum_fr1_desk_v2/scene/splat.ply` in
[SuperSplat](https://playcanvas.com/supersplat/editor).

## 5. Object-of-interest high-res splat

By text prompt (needs the `transformers` extra; ~600 MB model download on
first run):

```bash
bash_scripts/object_splat.bash experiments/ZED2i_Captures/tum_fr1_desk_v2 --prompt "a book"
```

Or by pointing at a rectangle on a frame (no extras needed — the OpenCV book
on frame 60):

```bash
bash_scripts/object_splat.bash experiments/ZED2i_Captures/tum_fr1_desk_v2 --rect 60 450 115 185 175
```

This detects the object, back-projects it to a 3D box, keeps only the frames
that observed it, and re-splats at 200 mapping iterations. Output PLY path is
printed at the end.

## 6. (Optional) Live-pipeline demo with realtime render topic

Two terminals, both with ROS sourced + conda env active:

```bash
# A: the live SLAM node (same code path as the real ZED)
python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_replay_test.py
# B: replay the dataset as ZED topics
python3 scripts/dataset_player.py --dataset experiments/ZED2i_Captures/tum_fr1_desk_v2 --rate 4 --loop
# C (optional): watch the splat build in realtime
ros2 run rqt_image_view rqt_image_view /splatam/live_render
```

The node processes 40 frames, publishes live renders, saves
`experiments/ZED2i_Captures/replay_test/.../params.npz`, and exits.

## Troubleshooting

* **`AttributeError: _ARRAY_API not found` warning at node startup** — benign
  numpy 1.x/2.x compat warning from cv_bridge; the node continues fine.
* **Rasterizer build fails** — check `nvcc --version` shows 12.1 (comes from
  the conda cuda-toolkit install) and TORCH_CUDA_ARCH_LIST matches your GPU.
* **Bag reader "yaml-cpp: bad conversion"** — the metadata patch step in
  `get_tum_test_data.bash` didn't run; re-run the script.
* **OWL-ViT finds nothing** — lower `--min-score`, reword the prompt
  ("a red book" vs "book"), or fall back to `--rect` mode.
