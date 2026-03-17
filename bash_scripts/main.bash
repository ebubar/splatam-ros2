#!/bin/bash
set -euo pipefail


# FORCE ROS DOMAIN
export ROS_DOMAIN_ID=77


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

color info "96m"
color success "92m"
color warning "93m"
color danger "91m"
ENDCOLOR='\033[0m'

# script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 4 ]; then
    print_color "$danger" "[ERROR] Usage:"
    print_color "$danger" "  bash $0 <run_name> <orin_ip> <teamings_ip> <capture_seconds>"
    exit 1
fi

RUN_NAME="$1"
ORIN_HOST="$2"
TEAMINGS_HOST="$3"
CAPTURE_SECONDS="$4"


# config
CAPTURE_SCRIPT="$SCRIPT_DIR/zed_capture.bash"
PIPELINE_SCRIPT="$SCRIPT_DIR/slat_pipeline.bash"

printf "%b" "$danger"
cat <<'EOF'
 .----------------.  .----------------.  .----------------.  .----------------.
| .--------------. || .--------------. || .--------------. || .--------------. |
| | ____   ____  | || |  _________   | || |     _____    | || |   _____      | |
| ||_  _| |_  _| | || | |_   ___  |  | || |    |_   _|   | || |  |_   _|     | |
| |  \ \   / /   | || |   | |_  \_|  | || |      | |     | || |    | |       | |
| |   \ \ / /    | || |   |  _|  _   | || |      | |     | || |    | |   _   | |
| |    \ ' /     | || |  _| |___/ |  | || |     _| |_    | || |   _| |__/ |  | |
| |     \_/      | || | |_________|  | || |    |_____|   | || |  |________|  | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'
EOF
printf "%b\n" "$ENDCOLOR"

print_color "$info" "[INFO] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
print_color "$info" "[INFO] Run name      : $RUN_NAME"
print_color "$info" "[INFO] Orin host     : $ORIN_HOST"
print_color "$info" "[INFO] Teamings host : $TEAMINGS_HOST"
print_color "$info" "[INFO] Capture time  : ${CAPTURE_SECONDS}s"


# capture from zed
print_color "$info" "[INFO] Step 1: Running ZED capture..."

CAPTURE_OUTPUT="$(
    bash "$CAPTURE_SCRIPT" \
        "$RUN_NAME" \
        "$ORIN_HOST" \
        "$TEAMINGS_HOST" \
        "$CAPTURE_SECONDS"
)"

RUN_DIR="$(printf "%s\n" "$CAPTURE_OUTPUT" | tail -n 1)"

if [ -z "$RUN_DIR" ]; then
    print_color "$danger" "[ERROR] Capture script did not return a run directory."
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    print_color "$danger" "[ERROR] Returned run directory does not exist: $RUN_DIR"
    exit 1
fi

print_color "$success" "[SUCCESS] Capture finished."
print_color "$info" "[INFO] Run directory: $RUN_DIR"


# saearch bag dir
BAG_DIR="$(find "$RUN_DIR" -type f -name metadata.yaml -printf '%h\n' | head -n 1 || true)"

if [ -z "$BAG_DIR" ]; then
    print_color "$danger" "[ERROR] Could not find rosbag directory."
    find "$RUN_DIR" -maxdepth 3
    exit 1
fi

print_color "$info" "[INFO] Bag directory: $BAG_DIR"
ls -lah "$BAG_DIR"


# start splatam
print_color "$info" "[INFO] Step 2: Starting SplaTAM in background..."

bash "$PIPELINE_SCRIPT" &
SPLATAM_PID=$!

print_color "$info" "[INFO] SplaTAM PID: $SPLATAM_PID"


print_color "$info" "[INFO] Waiting 5 seconds before bag playback..."
sleep 5


print_color "$info" "[INFO] ros2 bag play using ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

print_color "$info" "[INFO] Playing ros2 bag..."

ros2 bag play "$BAG_DIR" &
BAG_PID=$!


print_color "$info" "[INFO] Waiting for bag + SplaTAM..."

wait "$BAG_PID"
print_color "$success" "[SUCCESS] Bag playback finished."

wait "$SPLATAM_PID"

print_color "$success" ""
print_color "$success" "==============================================="
print_color "$success" "         FULL PIPELINE COMPLETED"
print_color "$success" "==============================================="
print_color "$success" "Run folder:"
print_color "$success" "  $RUN_DIR"
print_color "$success" ""