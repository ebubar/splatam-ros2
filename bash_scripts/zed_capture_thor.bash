#!/bin/bash
set -euo pipefail

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

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY=0
export CAMERA_MODEL="${CAMERA_MODEL:-zed2i}"
export LD_LIBRARY_PATH="/usr/local/zed/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/zed/tools:${PATH}"

if [ -f /opt/ros/jazzy/setup.bash ]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
fi

if [ -f /ros2_ws/install/setup.bash ]; then
    set +u
    source /ros2_ws/install/setup.bash
    set -u
fi

if [ -f /ros2_ws/install/local_setup.bash ]; then
    set +u
    source /ros2_ws/install/local_setup.bash
    set -u
fi

command -v ros2 >/dev/null 2>&1 || {
    log_error "ros2 not found"
    exit 1
}

[ -d /usr/local/zed ] || {
    log_error "Missing mounted ZED SDK at /usr/local/zed"
    exit 1
}

[ -f /usr/local/zed/lib/libsl_zed.so ] || {
    log_error "Missing ZED SDK library: /usr/local/zed/lib/libsl_zed.so"
    exit 1
}

ros2 pkg list | grep -qx zed_wrapper || {
    log_error "zed_wrapper package not found"
    exit 1
}

log_info "Launching ZED wrapper..."
log_info "Command: ros2 launch zed_wrapper zed_camera.launch.py camera_model:=$CAMERA_MODEL"

ros2 launch zed_wrapper zed_camera.launch.py \
    camera_model:="$CAMERA_MODEL" \
    > /tmp/zed_launch.log 2>&1 &
ZED_PID=$!

echo "$ZED_PID" > /tmp/zed_launch.pid

sleep 3
if ! kill -0 "$ZED_PID" >/dev/null 2>&1; then
    log_error "ZED launch process exited immediately."
    tail -n 200 /tmp/zed_launch.log || true
    exit 1
fi

log_info "Waiting for ZED startup..."
sleep 20

TOPIC_OK=0
for i in {1..60}; do
    if ros2 topic list | grep -q '^/zed/'; then
        TOPIC_OK=1
        log_success "ZED topics are active."
        break
    fi
    if ! kill -0 "$ZED_PID" >/dev/null 2>&1; then
        log_error "ZED launch process died while waiting for topics."
        tail -n 200 /tmp/zed_launch.log || true
        exit 1
    fi
    sleep 1
done

if [ "$TOPIC_OK" -ne 1 ]; then
    log_error "ZED topics did not become active."
    tail -n 200 /tmp/zed_launch.log || true
    exit 1
fi

log_success "ZED wrapper is running."
exit 0