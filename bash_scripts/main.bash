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

# LOG HELPERS
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

# script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 4 ]; then
    log_error "Usage:"
    log_error "  bash $0 <run_name> <orin_ip> <local_ip> <capture_seconds>"
    exit 1
fi

RUN_NAME="$1"
ORIN_HOST="$2"
LOCAL_HOST="$3"
CAPTURE_SECONDS="$4"

# config
ZENOH_SCRIPT="$SCRIPT_DIR/zenoh.bash"
CAPTURE_SCRIPT="$SCRIPT_DIR/zed_capture.bash"
PIPELINE_SCRIPT="$SCRIPT_DIR/splat_pipeline.bash"

# state
ZENOH_STARTED=0
SPLATAM_PID=""
BAG_PID=""

cleanup() {
    log_warn "Cleanup starting..."

    if [ -n "${BAG_PID:-}" ]; then
        kill "$BAG_PID" >/dev/null 2>&1 || true
    fi

    if [ -n "${SPLATAM_PID:-}" ]; then
        kill "$SPLATAM_PID" >/dev/null 2>&1 || true
    fi

    if [ "$ZENOH_STARTED" -eq 1 ]; then
        log_warn "Stopping Zenoh..."
        bash "$ZENOH_SCRIPT" stop "$ORIN_HOST" "$LOCAL_HOST" "$ROS_DOMAIN_ID" || true
    fi

    log_warn "Cleanup finished."
}

trap cleanup EXIT

# ASCII banner
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

# info
log_info "ROS_DOMAIN_ID  = $ROS_DOMAIN_ID"
log_info "Run name       = $RUN_NAME"
log_info "ORIN host      = $ORIN_HOST"
log_info "Local host     = $LOCAL_HOST"
log_info "Capture time   = ${CAPTURE_SECONDS}s"

# sanity checks
[ -f "$ZENOH_SCRIPT" ]    || { log_error "Missing: $ZENOH_SCRIPT"; exit 1; }
[ -f "$CAPTURE_SCRIPT" ]  || { log_error "Missing: $CAPTURE_SCRIPT"; exit 1; }
[ -f "$PIPELINE_SCRIPT" ] || { log_error "Missing: $PIPELINE_SCRIPT"; exit 1; }

# --------------------------------------------------
# Step 0: Start Zenoh first
# --------------------------------------------------
log_info "Step 0: Starting Zenoh..."
log_info "SSH/auth checks are handled inside: $ZENOH_SCRIPT"

if bash "$ZENOH_SCRIPT" start "$ORIN_HOST" "$LOCAL_HOST" "$ROS_DOMAIN_ID"; then
    ZENOH_STARTED=1
    log_success "Zenoh is up."
else
    ZENOH_STARTED=1
    log_warn "Zenoh start returned a non-zero status."
    log_warn "Continuing to ZED capture anyway."
fi

# --------------------------------------------------
# Step 1: Capture
# --------------------------------------------------
log_info "Step 1: Running ZED capture..."
log_info "SSH/auth checks are handled inside: $CAPTURE_SCRIPT"

CAPTURE_OUTPUT="$(
    bash "$CAPTURE_SCRIPT" \
        "$RUN_NAME" \
        "$ORIN_HOST" \
        "$LOCAL_HOST" \
        "$CAPTURE_SECONDS"
)"

RUN_DIR="$(printf "%s\n" "$CAPTURE_OUTPUT" | tail -n 1)"

if [ -z "$RUN_DIR" ]; then
    log_error "Capture script did not return a run directory."
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    log_error "Returned run directory does not exist: $RUN_DIR"
    exit 1
fi

log_success "Capture finished."
log_info "Run directory: $RUN_DIR"

# --------------------------------------------------
# Find bag dir
# --------------------------------------------------
BAG_DIR="$(find "$RUN_DIR" -type f -name metadata.yaml -printf '%h\n' | head -n 1 || true)"

if [ -z "$BAG_DIR" ]; then
    log_error "Could not find rosbag directory."
    find "$RUN_DIR" -maxdepth 3
    exit 1
fi

log_info "Bag directory: $BAG_DIR"
ls -lah "$BAG_DIR"

# --------------------------------------------------
# Prompt for Zenoh action BEFORE SplaTAM
# --------------------------------------------------
echo
log_info "Zenoh action before starting SplaTAM:"
echo "  1) continue  -> keep Zenoh running"
echo "  2) stop      -> stop Zenoh"
echo "  3) delete    -> stop + remove Zenoh"
echo

read -r -p "Enter choice [1/2/3 or continue/stop/delete] (default: continue): " ZENOH_CHOICE
ZENOH_CHOICE="${ZENOH_CHOICE:-continue}"

case "$ZENOH_CHOICE" in
    1|continue)
        log_info "Keeping Zenoh running."
        ;;

    2|stop)
        log_warn "Stopping Zenoh..."
        bash "$ZENOH_SCRIPT" stop "$ORIN_HOST" "$LOCAL_HOST" "$ROS_DOMAIN_ID" || true
        ZENOH_STARTED=0
        log_success "Zenoh stopped."
        ;;

    3|delete)
        log_warn "Deleting Zenoh (full teardown)..."
        bash "$ZENOH_SCRIPT" stop "$ORIN_HOST" "$LOCAL_HOST" "$ROS_DOMAIN_ID" || true
        ZENOH_STARTED=0
        log_success "Zenoh deleted."
        ;;

    *)
        log_warn "Invalid choice. Defaulting to continue."
        log_info "Keeping Zenoh running."
        ;;
esac

# --------------------------------------------------
# Step 2: Start SplaTAM
# --------------------------------------------------
log_info "Step 2: Starting SplaTAM in background..."

bash "$PIPELINE_SCRIPT" &
SPLATAM_PID=$!

log_info "SplaTAM PID: $SPLATAM_PID"

log_info "Waiting 5 seconds before bag playback..."
sleep 5

# --------------------------------------------------
# Step 3: Play bag
# --------------------------------------------------
log_info "Step 3: ros2 bag play using ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
log_info "Playing ros2 bag..."

ros2 bag play "$BAG_DIR" &
BAG_PID=$!

# --------------------------------------------------
# Step 4: Wait for bag
# --------------------------------------------------
log_info "Waiting for rosbag playback..."
wait "$BAG_PID"
BAG_PID=""
log_success "Bag playback finished."

# --------------------------------------------------
# Step 5: Wait for SplaTAM
# --------------------------------------------------
log_info "Waiting for SplaTAM to fully finish..."
wait "$SPLATAM_PID"
SPLATAM_PID=""
log_success "SplaTAM fully finished."

# --------------------------------------------------
# DONE
# --------------------------------------------------
log_success ""
log_success "==============================================="
log_success "         FULL PIPELINE COMPLETED"
log_success "==============================================="
log_success "Run folder:"
log_success "  $RUN_DIR"
log_success ""