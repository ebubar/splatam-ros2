#!/bin/bash
set -euo pipefail

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export NO_VIEWER="${NO_VIEWER:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

set +u
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi
if [ -f /ros2_ws/install/local_setup.bash ]; then
    source /ros2_ws/install/local_setup.bash
fi
set -u

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

color info "96m"
color success "92m"
color warning "93m"
color danger "91m"
ENDCOLOR='\033[0m'

log_info() {
    print_color "$info" "[INFO] $*"
}

log_success() {
    print_color "$success" "[SUCCESS] $*"
}

log_warn() {
    print_color "$warning" "[WARN] $*"
}

log_error() {
    print_color "$danger" "[ERROR] $*"
}

command -v python3 >/dev/null 2>&1 || {
    log_error "python3 not found"
    exit 1
}

CONFIG="configs/zed2i/zed2i_splat_live.py"
OUTPUT="experiments/ZED2i_Captures/zed2i_ros2_demo/SplaTAM_ZED2i_ROS2/splat.ply"

[ -f "$CONFIG" ] || { log_error "Config not found: $CONFIG"; exit 1; }
[ -f "scripts/zed2i_splat_live.py" ] || { log_error "scripts/zed2i_splat_live.py not found"; exit 1; }
[ -f "scripts/export_ply.py" ] || { log_error "scripts/export_ply.py not found"; exit 1; }
[ -f "viz_scripts/final_recon.py" ] || { log_error "viz_scripts/final_recon.py not found"; exit 1; }

log_info "Repo root=$REPO_ROOT"
log_info "QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
log_info "DISPLAY=${DISPLAY:-<empty>}"
log_info "XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<unset>}"
log_info "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

log_info "Running live SplaTAM..."
log_info "Config: $CONFIG"
log_info "Args: $*"

python3 -u scripts/zed2i_splat_live.py \
    --config "$CONFIG" \
    "$@"

log_success "Live SplaTAM run complete."

log_info "Exporting PLY..."
python3 -u scripts/export_ply.py "$CONFIG"
log_success "PLY export complete."

if [ "${NO_VIEWER}" = "1" ]; then
    log_warn "Skipping final reconstruction viewer."
else
    log_info "Launching final reconstruction viewer..."
    print_color "$warning" ""
    print_color "$warning" "-----------------------------------------------"
    print_color "$warning" " Final Reconstruction Viewer Running"
    print_color "$warning" "-----------------------------------------------"
    print_color "$warning" " Press: q or Q or ESC to quit the viewer"
    print_color "$warning" "-----------------------------------------------"
    print_color "$warning" ""

    python3 -u viz_scripts/final_recon.py "$CONFIG"
fi

print_color "$success" ""
print_color "$success" "==============================================="
print_color "$success" "      ALL PIPELINE STAGES COMPLETED"
print_color "$success" "==============================================="
print_color "$success" "1. zed2i_splat_live.py"
print_color "$success" "2. export_ply.py"
if [ "${NO_VIEWER}" = "1" ]; then
    print_color "$success" "3. final_recon.py skipped"
else
    print_color "$success" "3. final_recon.py"
fi
print_color "$success" ""
print_color "$success" "Output PLY file:"
print_color "$success" "  $OUTPUT"
print_color "$success" ""