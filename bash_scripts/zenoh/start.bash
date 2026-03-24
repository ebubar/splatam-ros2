#!/bin/bash
set -euo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

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

log_info()    { print_color "$info" "[INFO] $*"; }
log_success() { print_color "$success" "[SUCCESS] $*"; }
log_warn()    { print_color "$warning" "[WARN] $*"; }
log_error()   { print_color "$danger" "[ERROR] $*"; }

if [ "$#" -lt 2 ]; then
    log_error "Usage:"
    log_error "  bash $0 <orin_ip> <local_ip>"
    exit 1
fi

ORIN_HOST="$1"
LOCAL_HOST="$2"

ORIN_USER="nvidia"
ORIN_SSH="${ORIN_USER}@${ORIN_HOST}"
ROUTER_PORT=7447

check_orin_ssh() {
    log_info "Checking SSH login to ORIN (you may be prompted for password)..."
    if ssh -o ConnectTimeout=5 "$ORIN_SSH" "exit 0" >/dev/null 2>&1; then
        log_success "SSH authentication to ORIN succeeded."
    else
        log_error "SSH authentication to ORIN failed."
        exit 1
    fi
}

check_remote_router() {
    ssh "$ORIN_SSH" "docker ps --format '{{.Names}}' | grep -qx zenoh-router" >/dev/null 2>&1
}

check_remote_port() {
    ssh "$ORIN_SSH" "ss -ltn | grep ':${ROUTER_PORT} ' >/dev/null 2>&1" >/dev/null 2>&1
}

check_remote_bridge() {
    ssh "$ORIN_SSH" "pgrep -af zenoh_bridge_dds >/dev/null 2>&1" >/dev/null 2>&1
}

check_local_bridge() {
    pgrep -af "zenoh_bridge_dds.*tcp/${ORIN_HOST}:${ROUTER_PORT}" >/dev/null 2>&1
}

check_router_reachable() {
    nc -vz "$ORIN_HOST" "$ROUTER_PORT" >/dev/null 2>&1
}

wait_for_remote_router() {
    local ok=0
    for _ in {1..20}; do
        if check_remote_router; then
            ok=1
            break
        fi
        sleep 1
    done

    if [ "$ok" -ne 1 ]; then
        log_error "Router did not appear in docker ps."
        ssh "$ORIN_SSH" "docker ps -a || true"
        ssh "$ORIN_SSH" "cat /tmp/zenoh_router.log || true"
        exit 1
    fi
}

wait_for_remote_port() {
    local ok=0
    for _ in {1..20}; do
        if check_remote_port; then
            ok=1
            break
        fi
        sleep 1
    done

    if [ "$ok" -ne 1 ]; then
        log_error "Router is up but port ${ROUTER_PORT} is not listening."
        ssh "$ORIN_SSH" "cat /tmp/zenoh_router.log || true"
        exit 1
    fi
}

log_info "Starting Zenoh..."
log_info "ORIN host      = $ORIN_HOST"
log_info "Local host     = $LOCAL_HOST"
log_info "ROS_DOMAIN_ID  = $ROS_DOMAIN_ID"
log_info "Router port    = $ROUTER_PORT"

log_info "Checking ORIN reachability..."
ping -c 2 "$ORIN_HOST" >/dev/null
log_success "Can reach ORIN"

check_orin_ssh

log_info "Ensuring router + remote bridge are launched on ORIN..."

if ssh "$ORIN_SSH" "
    set -e

    if ! docker ps --format '{{.Names}}' | grep -qx zenoh-router; then
        echo '[REMOTE] Starting zenoh router...'
        docker rm -f zenoh-router >/dev/null 2>&1 || true
        nohup docker run --rm --name zenoh-router --net=host eclipse/zenoh \
          --cfg='listen/endpoints:[\"tcp/0.0.0.0:${ROUTER_PORT}\"]' \
          --cfg='scouting/multicast/enabled:false' \
          > /tmp/zenoh_router.log 2>&1 < /dev/null &
    else
        echo '[REMOTE] Zenoh router already running.'
    fi

    if ! pgrep -af zenoh_bridge_dds >/dev/null 2>&1; then
        echo '[REMOTE] Starting zenoh_bridge_dds...'
        nohup bash -lc '
            set +u
            source /opt/ros/humble/setup.bash
            source /home/nvidia/ros2_ws/install/setup.bash
            set -u
            export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
            exec ros2 run zenoh_bridge_dds zenoh_bridge_dds -e tcp/127.0.0.1:${ROUTER_PORT}
        ' > /tmp/zenoh_bridge_orin.log 2>&1 < /dev/null &
    else
        echo '[REMOTE] zenoh_bridge_dds already running.'
    fi
" >/dev/null; then
    log_success "Remote Zenoh launch commands completed."
else
    log_error "Remote Zenoh launch commands failed."
    exit 1
fi

wait_for_remote_router
wait_for_remote_port
log_success "Router is running and port ${ROUTER_PORT} is listening."

log_info "Checking local -> ORIN router connectivity..."
if check_router_reachable; then
    log_success "Local can reach ORIN router."
else
    log_error "Local cannot reach ${ORIN_HOST}:${ROUTER_PORT}"
    exit 1
fi

if check_local_bridge; then
    log_warn "Local zenoh_bridge_dds already running. Skipping start."
else
    log_info "Starting local zenoh_bridge_dds..."
    pkill -f "zenoh_bridge_dds.*tcp/${ORIN_HOST}:${ROUTER_PORT}" >/dev/null 2>&1 || true

    nohup bash -lc "
        set +u
        source /opt/ros/humble/setup.bash
        set -u
        export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
        exec ros2 run zenoh_bridge_dds zenoh_bridge_dds -e tcp/${ORIN_HOST}:${ROUTER_PORT}
    " > /tmp/zenoh_bridge_local.log 2>&1 < /dev/null &
fi

sleep 2

if check_local_bridge; then
    log_success "Local zenoh_bridge_dds is running."
else
    log_warn "Local zenoh_bridge_dds was launched but not confirmed yet."
    log_warn "Check: /tmp/zenoh_bridge_local.log"
fi

if check_remote_bridge; then
    log_success "Remote zenoh_bridge_dds is running."
else
    log_warn "Remote zenoh_bridge_dds was launched but not confirmed yet."
    log_warn "Check on ORIN: /tmp/zenoh_bridge_orin.log"
fi

log_success "Zenoh startup complete."