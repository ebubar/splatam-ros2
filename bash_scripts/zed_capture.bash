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


if [ "$#" -lt 4 ]; then
    print_color "$danger" "[ERROR] Usage:"
    print_color "$danger" "  bash $0 <run_name> <orin_ip> <teamings_ip> <capture_seconds>"
    exit 1
fi

RUN_NAME="$1"
ORIN_HOST="$2"
TEAMINGS_HOST="$3"   
CAPTURE_SECONDS="$4"


# CONFIG
ORIN_USER="nvidia"

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


print_color "$info" "[INFO] ROS_DOMAIN_ID=$ROS_DOMAIN_ID" >&2
print_color "$info" "[INFO] Run name: $RUN_NAME" >&2
print_color "$info" "[INFO] Run ID: $RUN_ID" >&2
print_color "$info" "[INFO] Duration: ${CAPTURE_SECONDS}s" >&2
print_color "$info" "[INFO] Orin: ${ORIN_USER}@${ORIN_HOST}" >&2
print_color "$info" "[INFO] Teamings host arg: $TEAMINGS_HOST" >&2
print_color "$info" "[INFO] Local machine: $(hostname)" >&2


print_color "$info" "[INFO] Preparing local directory..." >&2
mkdir -p "$LOCAL_BASE_DIR"


print_color "$info" "[INFO] Starting remote capture workflow on Orin..." >&2
print_color "$info" "[INFO] Remote output will stream here." >&2

ssh "${ORIN_USER}@${ORIN_HOST}" "
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

echo '[REMOTE] Final run dir contents:'
ls -lah '$REMOTE_RUN_DIR'
find '$REMOTE_RUN_DIR' -maxdepth 2 -type f | sort

echo '[REMOTE] Capture complete.'
" >&2

print_color "$info" "[INFO] Copying full run directory from Orin to local machine..." >&2

rm -rf "$LOCAL_RUN_DIR"
scp -r "${ORIN_USER}@${ORIN_HOST}:${REMOTE_RUN_DIR}" "$LOCAL_BASE_DIR/" >&2

if [ ! -d "$LOCAL_RUN_DIR" ]; then
    print_color "$danger" "[ERROR] Local run directory was not copied: $LOCAL_RUN_DIR" >&2
    exit 1
fi

print_color "$info" "[INFO] Local run directory contents:" >&2
ls -lah "$LOCAL_RUN_DIR" >&2
find "$LOCAL_RUN_DIR" -maxdepth 2 -type f | sort >&2


print_color "$success" "" >&2
print_color "$success" "===============================================" >&2
print_color "$success" "             CAPTURE COMPLETE " >&2
print_color "$success" "===============================================" >&2
print_color "$success" "Saved locally:" >&2
print_color "$success" "  $LOCAL_RUN_DIR" >&2
print_color "$success" "" >&2

echo "$LOCAL_RUN_DIR"