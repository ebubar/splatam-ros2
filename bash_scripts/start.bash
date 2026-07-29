#!/bin/bash
# Single entry point for the ZED2i -> ROS2 -> SplaTAM realtime pipeline.
# See docs/QUICKSTART.md for the full install + concept guide; this script is
# just the "bring it up" step once your environment is ready.
#
# Usage:
#   bash_scripts/start.bash check                 # verify the local env, exit
#   bash_scripts/start.bash local                  # camera on THIS machine, live browser viewer, runs until Ctrl-C
#   bash_scripts/start.bash networked              # camera on a remote Orin, live browser viewer, runs until Ctrl-C
#   bash_scripts/start.bash <mode> <config_path>   # override the config any mode uses
#     (e.g. `local configs/zed2i/zed2i_local_direct.py` for a fast 45-frame
#     smoke test instead of the live viewer)
#
# "local" also launches the ZED camera node on this machine if it isn't
# already publishing, and cleans it up on exit. "networked" assumes the
# camera is already running on the Orin (docs/QUICKSTART.md "Orin setup") --
# it only starts the SplaTAM subscriber + viewer on this machine.
set -uo pipefail   # no -e: we want to control cleanup/exit paths explicitly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

# ---- logging -------------------------------------------------------------- #
info()    { printf '\033[96m[INFO] %b\033[0m\n' "$*"; }
success() { printf '\033[92m[OK]   %b\033[0m\n' "$*"; }
warn()    { printf '\033[93m[WARN] %b\033[0m\n' "$*"; }
error()   { printf '\033[91m[ERR]  %b\033[0m\n' "$*" >&2; }

MODE="${1:-}"
CONFIG_OVERRIDE="${2:-}"

usage() {
    # Print just the contiguous header-comment block (stops at the first
    # non-comment line), not every "# ---- section ----" divider in the file.
    awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"
    exit 1
}
[ -z "$MODE" ] && usage

# ---- env check ------------------------------------------------------------- #
check_env() {
    local ok=1
    if ! command -v ros2 >/dev/null 2>&1; then
        error "ros2 not found -- source /opt/ros/humble/setup.bash first"
        ok=0
    else
        success "ros2 CLI available"
    fi

    if python3 -c "import torch, diff_gaussian_rasterization, rclpy, cv2, cv_bridge, message_filters" 2>/dev/null; then
        success "python env OK (torch + rclpy + rasterizer importable together)"
    else
        error "python import chain broken -- see docs/QUICKSTART.md 'Splatting machine setup'"
        ok=0
    fi

    if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        success "CUDA available"
    else
        error "CUDA not available to torch -- check nvidia-smi / driver"
        ok=0
    fi

    if [ -z "${ROS_DOMAIN_ID:-}" ]; then
        warn "ROS_DOMAIN_ID is not set (both machines must match for networked mode)"
    else
        success "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    fi

    [ "$ok" = 1 ]
}

if [ "$MODE" = "check" ]; then
    check_env
    exit $?
fi

check_env || { error "Environment check failed; fix the above before continuing."; exit 1; }

# ---- resolve config ---------------------------------------------------------#
case "$MODE" in
    local)       DEFAULT_CONFIG="configs/zed2i/zed2i_local_live_view.py"; LAUNCH_CAMERA=1 ;;
    networked)   DEFAULT_CONFIG="configs/zed2i/zed2i_networked_live_view.py"; LAUNCH_CAMERA=0 ;;
    *) error "Unknown mode: $MODE"; usage ;;
esac
CONFIG="${CONFIG_OVERRIDE:-$DEFAULT_CONFIG}"

if [ ! -f "$CONFIG" ]; then
    error "Config not found: $CONFIG"
    exit 1
fi

# ---- local camera (only for local mode) ------------------------------------ #
CAMERA_LAUNCH_PID=""

cleanup() {
    if [ -n "$CAMERA_LAUNCH_PID" ] && kill -0 "$CAMERA_LAUNCH_PID" 2>/dev/null; then
        info "Stopping locally-launched ZED camera node..."
        # `ros2 launch` spawns robot_state_publisher and the component
        # container as direct children of its own PID. Killing only the
        # container (and not these siblings, and not the launch process
        # itself) leaks a `ros2 launch` + robot_state_publisher pair per
        # run, which eventually exhausts CycloneDDS's participant pool
        # ("Failed to find a free participant index") -- kill the whole
        # small tree here, not just the launch process.
        pkill -9 -P "$CAMERA_LAUNCH_PID" 2>/dev/null
        kill -9 "$CAMERA_LAUNCH_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

if [ "$LAUNCH_CAMERA" = 1 ]; then
    if ros2 topic list 2>/dev/null | grep -q "/zed/zed_node/rgb"; then
        info "ZED camera already publishing; using the existing node."
    else
        info "Starting ZED camera node on this machine..."
        ros2 launch zed_wrapper zed_camera.launch.py \
            camera_model:=zed2i pos_tracking_mode:=GEN_3 \
            > /tmp/zed_camera_start.log 2>&1 &
        CAMERA_LAUNCH_PID=$!
        info "Waiting for camera topics (log: /tmp/zed_camera_start.log)..."
        for _ in $(seq 1 30); do
            ros2 topic list 2>/dev/null | grep -q "/zed/zed_node/rgb" && break
            sleep 2
        done
        if ! ros2 topic list 2>/dev/null | grep -q "/zed/zed_node/rgb"; then
            error "Camera did not come up -- see /tmp/zed_camera_start.log"
            error "Common cause: corrupted calibration cache -- see docs/QUICKSTART.md Troubleshooting."
            exit 1
        fi
        success "Camera publishing."
    fi
else
    info "Networked mode: expecting the camera on the Orin already (see docs/QUICKSTART.md)."
    if ! ros2 topic list 2>/dev/null | grep -q "/zed/zed_node"; then
        warn "No /zed/zed_node topics visible yet on this domain. If the Orin"
        warn "isn't up yet, start it now; this will keep waiting."
    fi
fi

# ---- run SplaTAM ------------------------------------------------------------#
info "Config: $CONFIG"
info "Running live SplaTAM (Ctrl-C saves a partial map cleanly)..."
python3 -u scripts/zed2i_splat_live.py --config "$CONFIG"
RUN_STATUS=$?

if [ "$RUN_STATUS" -ne 0 ]; then
    warn "SplaTAM exited with status $RUN_STATUS (Ctrl-C is expected and fine)."
fi

# ---- export + view ----------------------------------------------------------#
info "Exporting PLY..."
python3 -u scripts/export_ply.py "$CONFIG" || warn "PLY export failed -- check params.npz exists."

info "Launching final reconstruction viewer (q/ESC to quit)..."
python3 -u viz_scripts/final_recon.py "$CONFIG" || warn "Viewer failed to open (e.g. Wayland/GLFW) -- open the exported splat.ply in https://playcanvas.com/supersplat/editor instead."

success "Done. Output under: experiments/ZED2i_Captures/<scene>/<run_name>/"
