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

log_info "Stopping Zenoh..."
log_info "ORIN host      = $ORIN_HOST"
log_info "Local host     = $LOCAL_HOST"

pkill -f "zenoh_bridge_dds.*tcp/${ORIN_HOST}:${ROUTER_PORT}" >/dev/null 2>&1 || true

log_info "Connecting to ORIN to stop router/bridge..."
if ssh "$ORIN_SSH" "
    pkill -f zenoh_bridge_dds >/dev/null 2>&1 || true
    docker rm -f zenoh-router >/dev/null 2>&1 || true
" >/dev/null 2>&1; then
    log_success "Remote Zenoh stop commands completed."
else
    log_warn "Remote Zenoh stop commands returned non-zero."
fi

log_success "Zenoh stopped."