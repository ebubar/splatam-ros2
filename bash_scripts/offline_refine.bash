#!/bin/bash
# Offline quality path: bag + RTAB-Map loop-closed poses -> refined splat.
#
#   bag  ──rtabmap2dataset.py──>  dataset  ──splatam.py──>  params.npz
#                                                │
#                                     export_ply.py + final_recon.py
#
# Usage:
#   bash_scripts/offline_refine.bash <bag_dir> <poses_tum.txt> [scene_name]
#
# Requires a sourced ROS2 environment (for reading the bag) and the SplaTAM
# python environment. Export poses from RTAB-Map in TUM format
# (rtabmap-databaseViewer: File -> Export poses -> TUM).
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <bag_dir> <poses_tum.txt> [scene_name]"
    exit 1
fi

BAG="$1"
POSES="$2"
SCENE="${3:-offline_rtabmap}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/zed2i/zed2i_offline_rtabmap.py"
DATASET_DIR="experiments/ZED2i_Captures/$SCENE"

export SPLATAM_SCENE="$SCENE"

echo "[1/4] Converting bag + poses -> dataset: $DATASET_DIR"
python3 scripts/rtabmap2dataset.py \
    --bag "$BAG" \
    --poses "$POSES" \
    --output "$DATASET_DIR" \
    --overwrite

echo "[2/4] Running offline SplaTAM (poses fixed)..."
python3 scripts/splatam.py "$CONFIG"

echo "[3/4] Exporting PLY..."
python3 scripts/export_ply.py "$CONFIG"

echo "[4/4] Launching viewer (q/ESC to quit)..."
python3 viz_scripts/final_recon.py "$CONFIG"

echo "Done. Output: $DATASET_DIR/SplaTAM_ZED2i_Offline/"
