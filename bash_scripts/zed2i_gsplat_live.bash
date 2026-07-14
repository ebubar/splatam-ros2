#!/bin/bash
# Launch the near-realtime gsplat live node, then export + view the result.
# Mirrors bash_scripts/zed2i_live.bash but drives scripts/zed2i_gsplat_live.py
# with the gsplat backend + trusted ZED poses + async mapping.
set -euo pipefail

export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}

info='\033[96m'; success='\033[92m'; warning='\033[93m'; danger='\033[91m'; ENDCOLOR='\033[0m'
say() { printf "${1}%b${ENDCOLOR}\n" "$2"; }

CONFIG="${1:-configs/zed2i/zed2i_gsplat_live.py}"
OUTPUT="experiments/ZED2i_Captures/zed2i_gsplat_demo/SplaTAM_ZED2i_gsplat/splat.ply"

if [ ! -f "$CONFIG" ]; then
    say "$danger" "[ERROR] Config not found: $CONFIG"; exit 1
fi
if [ ! -f "scripts/zed2i_gsplat_live.py" ]; then
    say "$danger" "[ERROR] scripts/zed2i_gsplat_live.py not found"; exit 1
fi

say "$info" "[INFO] ROS_DOMAIN_ID=$ROS_DOMAIN_ID  QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
say "$info" "[INFO] Running realtime gsplat live node (config: $CONFIG)"

python3 -u scripts/zed2i_gsplat_live.py --config "$CONFIG"
say "$success" "[SUCCESS] Live run complete."

say "$info" "[INFO] Exporting PLY..."
python3 -u scripts/export_ply.py "$CONFIG"
say "$success" "[SUCCESS] PLY export complete."

say "$info" "[INFO] Launching final reconstruction viewer (q/ESC to quit)..."
python3 -u viz_scripts/final_recon.py "$CONFIG"

say "$success" "==============================================="
say "$success" "      GSPLAT PIPELINE COMPLETE ✔"
say "$success" "  Output PLY: $OUTPUT"
say "$success" "==============================================="
