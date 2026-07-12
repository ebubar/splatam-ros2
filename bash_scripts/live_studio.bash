#!/bin/bash
# Live splat tuning studio: camera + SplaTAM node + browser UI.
# Run this in YOUR terminal (or tmux) so it survives assistant/tooling
# sessions. Ctrl-C stops everything cleanly.
#
#   bash_scripts/live_studio.bash [config]
#
# Then open http://localhost:8080 — orbit the growing splat, tweak
# parameters live, Reset map between experiments, Save snapshot anytime.
# Note: no -u — ROS setup.bash references unbound variables internally
set -eo pipefail

CONFIG="${1:-configs/zed2i/zed2i_local_tune.py}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

source /opt/ros/humble/setup.bash
[ -f ~/zed_ws/install/setup.bash ] && source ~/zed_ws/install/setup.bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/splatam/lib:${LD_LIBRARY_PATH:-}

STARTED_CAMERA=0
cleanup() {
    if [ "$STARTED_CAMERA" = "1" ]; then
        pkill -INT -f "zed_camera.launch" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if ! ros2 topic list 2>/dev/null | grep -q "/zed/zed_node/rgb"; then
    echo "[live_studio] Starting ZED camera..."
    ros2 launch zed_wrapper zed_camera.launch.py \
        camera_model:=zed2i pos_tracking_mode:=GEN_3 \
        > /tmp/zed_camera.log 2>&1 &
    STARTED_CAMERA=1
    for i in $(seq 1 30); do
        ros2 topic list 2>/dev/null | grep -q "/zed/zed_node/rgb" && break
        sleep 2
    done
    ros2 topic list 2>/dev/null | grep -q "/zed/zed_node/rgb" || {
        echo "[live_studio] Camera failed to start; see /tmp/zed_camera.log"
        exit 1
    }
fi

echo "[live_studio] Camera up. Starting SplaTAM node + studio..."
echo "[live_studio] Open http://localhost:8080 in a browser."
python3 scripts/zed2i_splat_live.py --config "$CONFIG"
