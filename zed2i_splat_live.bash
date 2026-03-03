#!/bin/bash
# bash_scripts/zed2i_online_ros2.bash
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: bash_scripts/zed2i_online_ros2.bash <config_file>"
  exit 1
fi

python3 scripts/zed2i_ros2_demo.py --config "$1"
python3 viz_scripts/final_recon.py "$1"