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
    print_color "$info" "[INFO] $*" >&2
}

log_success() {
    print_color "$success" "[SUCCESS] $*" >&2
}

log_warn() {
    print_color "$warning" "[WARN] $*" >&2
}

log_error() {
    print_color "$danger" "[ERROR] $*" >&2
}

# PARSE ARGS
RQT_FLAG=false
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rqt)
            RQT_FLAG=true
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"

if [ "$#" -lt 4 ]; then
    log_error "Usage:"
    log_error "  bash $0 <run_name> <orin_ip> <local_ip> <capture_seconds> [--rqt]"
    exit 1
fi

RUN_NAME="$1"
ORIN_HOST="$2"
LOCAL_HOST="$3"
CAPTURE_SECONDS="$4"

# config
ORIN_USER="nvidia"
ORIN_SSH="${ORIN_USER}@${ORIN_HOST}"

REMOTE_ROS_SETUP="/opt/ros/humble/setup.bash"
REMOTE_WS_SETUP="/home/nvidia/ros2_ws/install/setup.bash"
REMOTE_BASE_DIR="/home/nvidia/bags"

LOCAL_BASE_DIR="/home/teaming"

RUN_ID="${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
REMOTE_RUN_DIR="$REMOTE_BASE_DIR/$RUN_ID"
REMOTE_OVERRIDE_FILE="$REMOTE_RUN_DIR/zed_override.yaml"
REMOTE_BAG_PREFIX="$REMOTE_RUN_DIR/$RUN_NAME"

LOCAL_RUN_DIR="$LOCAL_BASE_DIR/$RUN_ID"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

check_orin_ssh() {
    log_info "Checking SSH login to ORIN (you may be prompted for password)..."
    if ssh -o ConnectTimeout=5 "$ORIN_SSH" "exit 0" >/dev/null 2>&1; then
        log_success "SSH authentication to ORIN succeeded."
    else
        log_error "SSH authentication to ORIN failed."
        exit 1
    fi
}

log_info "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
log_info "Run name: $RUN_NAME"
log_info "Run ID: $RUN_ID"
log_info "Duration: ${CAPTURE_SECONDS}s"
log_info "Orin: $ORIN_SSH"
log_info "Local host arg: $LOCAL_HOST"
log_info "Local machine: $(hostname)"
log_info "RQT enabled: $RQT_FLAG"

log_info "Preparing local directory..."
mkdir -p "$LOCAL_BASE_DIR"

check_orin_ssh

log_info "Starting remote capture workflow on Orin..."
log_info "Remote output will stream here."

if ssh "$ORIN_SSH" "
set -euo pipefail

mkdir -p '$REMOTE_RUN_DIR'

set +u
source '$REMOTE_ROS_SETUP'
source '$REMOTE_WS_SETUP'
set -u

export ROS_DOMAIN_ID='$ROS_DOMAIN_ID'

cat > '$REMOTE_OVERRIDE_FILE' <<'YAML'
/**:
  ros__parameters:
    general:
      grab_resolution: 'VGA'
      grab_frame_rate: 15
YAML

echo '[REMOTE] Launching ZED...'
ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zed2i \
  ros_params_override_path:='$REMOTE_OVERRIDE_FILE' \
  > '$REMOTE_RUN_DIR/zed_launch.log' 2>&1 &
ZED_PID=\$!

echo '[REMOTE] Waiting for ZED launch...'
sleep 12

echo '[REMOTE] Available ZED topics:'
ros2 topic list | grep zed || true

echo '[REMOTE] Waiting for image topic...'
for i in {1..30}; do
  if ros2 topic echo /zed/zed_node/rgb/color/rect/image/header --once >/dev/null 2>&1; then
    echo '[REMOTE] Topic active'
    break
  fi
  sleep 1
done

if [ '$RQT_FLAG' = 'true' ]; then
  echo '[REMOTE] Launching rqt_image_view...'
  export DISPLAY=:0
  ros2 run rqt_image_view rqt_image_view /zed/zed_node/rgb/color/rect/image \
    > '$REMOTE_RUN_DIR/rqt_image_view.log' 2>&1 &
  RQT_PID=\$!
fi

echo '[REMOTE] Recording for ${CAPTURE_SECONDS}s...'
timeout --signal=INT ${CAPTURE_SECONDS} \
  ros2 bag record -o '$REMOTE_BAG_PREFIX' \
  /zed/zed_node/rgb/color/rect/image \
  /zed/zed_node/rgb/color/rect/camera_info \
  /zed/zed_node/depth/depth_registered \
  /zed/zed_node/depth/depth_registered/camera_info \
  /zed/zed_node/pose \
  /zed/zed_node/odom \
  /zed/zed_node/imu/data \
  /tf \
  /tf_static \
  > '$REMOTE_RUN_DIR/zed_bag.log' 2>&1 || true

echo '[REMOTE] Stopping ZED...'
kill -INT \$ZED_PID || true
sleep 2
kill -TERM \$ZED_PID || true

if [ '$RQT_FLAG' = 'true' ]; then
  echo '[REMOTE] Stopping rqt_image_view...'
  kill \$RQT_PID || true
fi

sleep 1

echo '[REMOTE] Final run dir contents:'
ls -lah '$REMOTE_RUN_DIR'
find '$REMOTE_RUN_DIR' -maxdepth 2 -type f | sort

echo '[REMOTE] Capture complete.'
" >&2; then
    log_success "SSH commands on ORIN completed successfully."
    log_success "Remote capture workflow completed."
else
    log_error "Remote capture workflow failed."
    exit 1
fi

log_info "Copying full run directory from Orin to local machine..."

rm -rf "$LOCAL_RUN_DIR"

if scp -r "${ORIN_SSH}:${REMOTE_RUN_DIR}" "$LOCAL_BASE_DIR/" >&2; then
    log_success "SCP copy from ORIN succeeded."
else
    log_error "SCP copy from ORIN failed."
    exit 1
fi

if [ ! -d "$LOCAL_RUN_DIR" ]; then
    log_error "Local run directory was not copied: $LOCAL_RUN_DIR"
    exit 1
fi

log_info "Local run directory contents:"
ls -lah "$LOCAL_RUN_DIR" >&2
find "$LOCAL_RUN_DIR" -maxdepth 2 -type f | sort >&2

log_success ""
log_success "==============================================="
log_success "             CAPTURE COMPLETE"
log_success "==============================================="
log_success "Saved locally:"
log_success "  $LOCAL_RUN_DIR"
log_success ""

echo "$LOCAL_RUN_DIR"