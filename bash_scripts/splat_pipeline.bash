#!/bin/bash
set -euo pipefail

export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}


# ROS SETUP

set +u
source /opt/ros/humble/setup.bash
# source ~/splatam_ws/install/setup.bash
set -u


# COLORS

print_color() {
    local color=$1
    shift
    printf "${color}%b${ENDCOLOR}\n" "$*"
}

color() {
    local var_name=$1
    local color_code=$2
    printf -v "$var_name" "\033[%s" "$color_code"
}

color info "96m"     # cyan
color success "92m"  # green
color warning "93m"  # yellow
color danger "91m"   # red
ENDCOLOR='\033[0m'

print_color "$info" "[INFO] QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
print_color "$info" "[INFO] DISPLAY=${DISPLAY:-<empty>}"
print_color "$info" "[INFO] XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<unset>}"
print_color "$info" "[INFO] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"


# CONFIG
# Node/config selection (backward compatible). Default = original CUDA node.
# Set NODE=gsplat to run the realtime gsplat node (scripts/zed2i_gsplat_live.py).

NODE="${NODE:-splatam}"
if [ "$NODE" = "gsplat" ]; then
    LIVE_SCRIPT="scripts/zed2i_gsplat_live.py"
    CONFIG="${CONFIG:-configs/zed2i/zed2i_gsplat_live.py}"
    OUTPUT="experiments/ZED2i_Captures/zed2i_gsplat_demo/SplaTAM_ZED2i_gsplat/splat.ply"
else
    LIVE_SCRIPT="scripts/zed2i_splat_live.py"
    CONFIG="${CONFIG:-configs/zed2i/zed2i_splat_live.py}"
    OUTPUT="experiments/ZED2i_Captures/zed2i_ros2_demo/SplaTAM_ZED2i_ROS2/splat.ply"
fi

if [ ! -f "$CONFIG" ]; then
    print_color "$danger" "[ERROR] Config not found: $CONFIG"
    exit 1
fi

if [ ! -f "$LIVE_SCRIPT" ]; then
    print_color "$danger" "[ERROR] $LIVE_SCRIPT not found"
    exit 1
fi

if [ ! -f "scripts/export_ply.py" ]; then
    print_color "$danger" "[ERROR] scripts/export_ply.py not found"
    exit 1
fi

if [ ! -f "viz_scripts/final_recon.py" ]; then
    print_color "$danger" "[ERROR] viz_scripts/final_recon.py not found"
    exit 1
fi

print_color "$info" "[INFO] Running live SplaTAM... (NODE=$NODE)"
print_color "$info" "[INFO] Script: $LIVE_SCRIPT"
print_color "$info" "[INFO] Config: $CONFIG"
print_color "$info" "[INFO] Args: $*"

python3 -u "$LIVE_SCRIPT" \
    --config "$CONFIG" \
    "$@"

print_color "$success" "[SUCCESS] Live run complete."

print_color "$info" "[INFO] Exporting PLY..."
python3 -u scripts/export_ply.py "$CONFIG"
print_color "$success" "[SUCCESS] PLY export complete."

print_color "$info" "[INFO] Launching final reconstruction viewer..."
print_color "$warning" ""
print_color "$warning" "-----------------------------------------------"
print_color "$warning" " Final Reconstruction Viewer Running"
print_color "$warning" "-----------------------------------------------"
print_color "$warning" " Press:  q  or  Q  or  ESC  to quit the viewer"
print_color "$warning" "-----------------------------------------------"
print_color "$warning" ""

python3 -u viz_scripts/final_recon.py "$CONFIG"

print_color "$success" ""
print_color "$success" "==============================================="
print_color "$success" "      ALL PIPELINE STAGES COMPLETED"
print_color "$success" "==============================================="
print_color "$success" "1. ✔ $LIVE_SCRIPT"
print_color "$success" "2. ✔ export_ply.py"
print_color "$success" "3. ✔ final_recon.py"
print_color "$success" ""
print_color "$success" "Output PLY file:"
print_color "$success" "  $OUTPUT"
print_color "$success" ""