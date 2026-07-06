#!/bin/bash
# Live ZED2i -> ROS2 -> SplaTAM with operator viewer.
# Open the operator viewer from a laptop at http://<this-machine-ip>:8080/
set -e

CONFIG="${1:-configs/zed2i/zed2i_splat_live.py}"

python3 scripts/zed2i_splat_live.py --config "$CONFIG"
python3 viz_scripts/final_recon.py "$CONFIG"
