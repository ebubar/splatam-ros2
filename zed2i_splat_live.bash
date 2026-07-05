#!/bin/bash
# Convenience wrapper: run the live ZED2i SplaTAM pipeline, then view the result.
# Requires a ZED2i publishing on the ROS2 network (or a rosbag replay) — see docs/README.md.
# For the full pipeline (PLY export + viewer) use bash_scripts/zed2i_live.bash.
set -e

CONFIG="${1:-configs/zed2i/zed2i_splat_live.py}"

python3 scripts/zed2i_splat_live.py --config "$CONFIG"
python3 viz_scripts/final_recon.py "$CONFIG"
