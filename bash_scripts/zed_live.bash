#!/bin/bash
set -euo pipefail

export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}

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

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}

print_color "$info" "[INFO] QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
print_color "$info" "[INFO] DISPLAY=${DISPLAY:-<empty>}"
print_color "$info" "[INFO] XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<unset>}"
print_color "$info" "[INFO] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

#clear
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


#CONFIG=$1
CONFIG="configs/zed2i/zed_splat_live.py"

if [ ! -f "$CONFIG" ]; then
    print_color "$danger" "[ERROR] Config not found: $CONFIG"
    exit 1
fi
if [ ! -f "scripts/zed_splat_live.py" ]; then
    print_color "$danger" "[ERROR] scripts/zed_splat_live.py not found"
    exit 1
fi

print_color "$info" "[INFO] Running live SplaTAM..."
print_color "$info" "[INFO] Config: $CONFIG"
print_color "$info" "[INFO] Args: $*"

python3 -u scripts/zed_splat_live.py \
    --config "$CONFIG" \
    "$@"

print_color "$success" "[SUCCESS] Live run complete."

print_color "$info" "[INFO] Launching final reconstruction viewer..."

python3 -u viz_scripts/final_recon.py "$CONFIG"
