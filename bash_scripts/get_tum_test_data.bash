#!/bin/bash
# Download and prepare the TUM fr1_desk test dataset so the whole offline
# pipeline can be exercised with no camera and no RTAB-Map install
# (the mocap ground truth doubles as the TUM-format pose file).
#
# Usage:  bash_scripts/get_tum_test_data.bash
# Needs:  a sourced ROS2 environment + the splatam python env (see
#         docs/LAPTOP_QUICKSTART.md), ~1 GB disk, internet.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

DEST=datasets/tum_test
mkdir -p "$DEST"

BAG_URL=https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.bag
GT_URL=https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk-groundtruth.txt

[ -f "$DEST/rgbd_dataset_freiburg1_desk.bag" ] || wget -q --show-progress -O "$DEST/rgbd_dataset_freiburg1_desk.bag" "$BAG_URL"
[ -f "$DEST/groundtruth.txt" ] || wget -q -O "$DEST/groundtruth.txt" "$GT_URL"

python3 -c "import rosbags" 2>/dev/null || pip install rosbags

if [ ! -d "$DEST/fr1_desk_ros2" ]; then
    echo "Converting ROS1 bag -> ROS2..."
    rosbags-convert --src "$DEST/rgbd_dataset_freiburg1_desk.bag" --dst "$DEST/fr1_desk_ros2"
    # rosbags writes metadata newer than ROS2 Humble can read; patch it down
    sed -i "s/offered_qos_profiles: \[\]/offered_qos_profiles: ''/; /type_description_hash:/d; /custom_data: null/d; s/  version: 9/  version: 5/" \
        "$DEST/fr1_desk_ros2/metadata.yaml"
fi

echo "Building SplaTAM dataset (with depth edge filtering)..."
python3 scripts/rtabmap2dataset.py \
    --bag "$DEST/fr1_desk_ros2" \
    --poses "$DEST/groundtruth.txt" \
    --output experiments/ZED2i_Captures/tum_fr1_desk_v2 \
    --rgb-topic /camera/rgb/image_color \
    --depth-topic /camera/depth/image \
    --info-topic /camera/rgb/camera_info \
    --pose-frame optical --pose-stride 16 \
    --intrinsics 517.3 516.5 318.6 255.3 \
    --min-depth 0.3 --max-depth 5.0 --edge-filter 0.04 \
    --overwrite

echo ""
echo "Done. Dataset: experiments/ZED2i_Captures/tum_fr1_desk_v2"
echo "Next steps: see docs/LAPTOP_QUICKSTART.md"
