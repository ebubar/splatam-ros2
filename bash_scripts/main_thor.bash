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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_NAME="${1:-thor_run}"

if [ "${THOR_INSIDE_CONTAINER:-0}" != "1" ]; then
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
    export CAMERA_MODEL="${CAMERA_MODEL:-zed2i}"
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
    export ROS_LOCALHOST_ONLY=0
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
    export NO_VIEWER="${NO_VIEWER:-0}"

    IMAGE_NAME="${IMAGE_NAME:-splatam-thor:latest}"
    CONTAINER_NAME="${CONTAINER_NAME:-splatam-thor}"
    DOCKERFILE_PATH="${DOCKERFILE_PATH:-$REPO_ROOT/docker/Dockerfile}"

    command -v docker >/dev/null 2>&1 || {
        log_error "docker not found"
        exit 1
    }

    [ -f "$DOCKERFILE_PATH" ] || {
        log_error "Missing Dockerfile: $DOCKERFILE_PATH"
        exit 1
    }

    log_info "Host mode"
    log_info "Repo root        = $REPO_ROOT"
    log_info "Run name         = $RUN_NAME"
    log_info "Camera model     = $CAMERA_MODEL"
    log_info "ROS_DOMAIN_ID    = $ROS_DOMAIN_ID"
    log_info "NO_VIEWER        = $NO_VIEWER"
    log_info "Image            = $IMAGE_NAME"
    log_info "Dockerfile       = $DOCKERFILE_PATH"

    xhost +si:localuser:root >/dev/null 2>&1 || true

    docker build \
        -f "$DOCKERFILE_PATH" \
        -t "$IMAGE_NAME" \
        "$REPO_ROOT"

    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

    docker run -it --rm \
        --name "$CONTAINER_NAME" \
        --runtime nvidia \
        --network host \
        --ipc host \
        --pid host \
        --privileged \
        -e THOR_INSIDE_CONTAINER=1 \
        -e DISPLAY="${DISPLAY:-:0}" \
        -e QT_QPA_PLATFORM="$QT_QPA_PLATFORM" \
        -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
        -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION" \
        -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY" \
        -e CAMERA_MODEL="$CAMERA_MODEL" \
        -e NO_VIEWER="$NO_VIEWER" \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e LD_LIBRARY_PATH="/usr/local/zed/lib:${LD_LIBRARY_PATH:-}" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /tmp:/tmp \
        -v /dev:/dev \
        -v /usr/local/zed:/usr/local/zed \
        -v "$REPO_ROOT":"$REPO_ROOT" \
        -w "$REPO_ROOT" \
        "$IMAGE_NAME" \
        bash "$0" "$RUN_NAME"

    exit 0
fi

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY=0
export CAMERA_MODEL="${CAMERA_MODEL:-zed2i}"
export NO_VIEWER="${NO_VIEWER:-0}"

cd "$REPO_ROOT"

if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

export LD_LIBRARY_PATH="/usr/local/zed/lib:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/zed/tools:${PATH}"

log_info "Container mode"
log_info "QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
log_info "DISPLAY=${DISPLAY:-<empty>}"
log_info "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
log_info "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
log_info "CAMERA_MODEL=$CAMERA_MODEL"
log_info "Repo root=$REPO_ROOT"

[ -d /usr/local/zed ] || {
    log_error "Missing mounted ZED SDK at /usr/local/zed"
    exit 1
}

[ -f /usr/local/zed/lib/libsl_zed.so ] || {
    log_error "Missing ZED SDK library: /usr/local/zed/lib/libsl_zed.so"
    exit 1
}

command -v ros2 >/dev/null 2>&1 || {
    log_error "ros2 not found in container"
    exit 1
}

[ -f "$REPO_ROOT/bash_scripts/zed_capture_thor.bash" ] || {
    log_error "Missing: $REPO_ROOT/bash_scripts/zed_capture_thor.bash"
    exit 1
}

[ -f "$REPO_ROOT/bash_scripts/splat_pipeline_thor.bash" ] || {
    log_error "Missing: $REPO_ROOT/bash_scripts/splat_pipeline_thor.bash"
    exit 1
}

cleanup() {
    log_warn "Main cleanup starting..."

    if [ -f /tmp/zed_launch.pid ]; then
        ZED_PID="$(cat /tmp/zed_launch.pid 2>/dev/null || true)"
        if [ -n "${ZED_PID:-}" ] && kill -0 "$ZED_PID" >/dev/null 2>&1; then
            log_warn "Stopping ZED wrapper PID $ZED_PID ..."
            kill -INT "$ZED_PID" >/dev/null 2>&1 || true
            sleep 2
            kill -TERM "$ZED_PID" >/dev/null 2>&1 || true
        fi
    fi

    log_warn "Main cleanup finished."
}

trap cleanup EXIT

log_info "1: Starting ZED capture pipeline..."
bash "$REPO_ROOT/bash_scripts/zed_capture_thor.bash"

log_success "ZED startup script completed."

log_info "2: Waiting 10 seconds before starting SplaTAM..."
sleep 10

log_info "3: Starting SplaTAM pipeline..."
bash "$REPO_ROOT/bash_scripts/splat_pipeline_thor.bash"

log_success "Main pipeline complete."