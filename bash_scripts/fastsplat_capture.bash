#!/bin/bash
# Capture sparse ZED keyframes into a folder for fastsplat.
#
# Run this where rclpy + the ZED topics are available -- i.e. inside the
# docker/Dockerfile.ros2_zed container, after the Zenoh transfer is up
# (bash_scripts/zenoh/start.bash) so the ZED topics are visible locally.
# It writes images (+ intrinsics, depth, odom) to the capture dir; the GPU
# container then runs SfM -> object -> splat on that folder.
#
# Usage:
#   bash bash_scripts/fastsplat_capture.bash [config] [out_dir]
set -euo pipefail

CONFIG="${1:-configs/fastsplat/fastsplat.py}"
OUT_DIR="${2:-experiments/FastSplat/object_splat/capture}"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

set +u
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f /opt/ros/jazzy/setup.bash ] && source /opt/ros/jazzy/setup.bash
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "[fastsplat] capturing sparse keyframes -> $OUT_DIR (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
python3 -u -m fastsplat.ingest.ros2_capture --config "$CONFIG" --out "$OUT_DIR"
echo "[fastsplat] capture complete. Now run bash_scripts/fastsplat_pipeline.bash with ingest.source=folder pointing at $OUT_DIR"
