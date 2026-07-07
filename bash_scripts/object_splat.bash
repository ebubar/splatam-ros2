#!/bin/bash
# Object-of-interest high-res splat: find an object in a dataset (by text
# prompt or manual rectangle), keep only the frames that observed it, and
# re-splat it with a large quality budget.
#
#   dataset ──detect_roi.py──> 3D box ──roi_refine.py──> frame subset
#                                              │
#                       splatam.py (high iters) + export_ply.py ──> object PLY
#
# Usage:
#   bash_scripts/object_splat.bash <dataset_dir> --prompt "a keyboard" [iters]
#   bash_scripts/object_splat.bash <dataset_dir> --rect <frame> <x> <y> <w> <h> [iters]
#
# Examples:
#   bash_scripts/object_splat.bash experiments/ZED2i_Captures/tum_fr1_desk_v2 --prompt "a book"
#   bash_scripts/object_splat.bash experiments/ZED2i_Captures/tum_fr1_desk_v2 --rect 60 450 115 185 175 300
#
# Prompt mode needs: pip install transformers (first run downloads ~600 MB).
set -euo pipefail

if [ "$#" -lt 3 ]; then
    grep "^#" "$0" | head -18
    exit 1
fi

DATASET="$1"
MODE="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

ITERS=200

if [ "$MODE" = "--prompt" ]; then
    PROMPT="$1"
    [ "$#" -ge 2 ] && ITERS="$2"
    LABEL=$(echo "$PROMPT" | tr ' ' '_' | tr -cd '[:alnum:]_')
    python3 scripts/detect_roi.py --dataset "$DATASET" --prompt "$PROMPT" --label "$LABEL"
elif [ "$MODE" = "--rect" ]; then
    FRAME="$1"; X="$2"; Y="$3"; W="$4"; H="$5"
    [ "$#" -ge 6 ] && ITERS="$6"
    LABEL="frame${FRAME}"
    python3 scripts/detect_roi.py --dataset "$DATASET" \
        --frame "$FRAME" --rect "$X" "$Y" "$W" "$H" --label "$LABEL"
else
    echo "Second arg must be --prompt or --rect"
    exit 1
fi

SCENE_OUT="$(basename "$DATASET")_${LABEL}"

# OBJECT_SPLAT_MASK=1: mask depth outside the ROI so the whole Gaussian
# budget goes to the object (experimental; sharper object, no background)
MASK_FLAG=""
[ "${OBJECT_SPLAT_MASK:-0}" = "1" ] && MASK_FLAG="--mask-outside"

python3 scripts/roi_refine.py \
    --roi-json "$DATASET/roi_${LABEL}.json" \
    --dataset "$DATASET" \
    --out "$SCENE_OUT" \
    $MASK_FLAG \
    --overwrite

# Match the source dataset resolution
read -r IMG_W IMG_H <<< "$(python3 -c "
import json
m = json.load(open('$DATASET/transforms.json'))
print(m['w'], m['h'])")"

export SPLATAM_SCENE="$SCENE_OUT"
export SPLATAM_RUN_NAME="object_splat"
export SPLATAM_MAPPING_ITERS="$ITERS"
export SPLATAM_IMAGE_WIDTH="$IMG_W"
export SPLATAM_IMAGE_HEIGHT="$IMG_H"

echo "[object_splat] Re-splatting '$SCENE_OUT' at $ITERS mapping iters..."
python3 scripts/splatam.py configs/zed2i/zed2i_offline_rtabmap.py
python3 scripts/export_ply.py configs/zed2i/zed2i_offline_rtabmap.py

OUT_DIR="$(dirname "$DATASET")/$SCENE_OUT/object_splat"
echo ""
echo "[object_splat] Done. Object splat:"
echo "  $OUT_DIR/splat.ply"
echo "Open it in SuperSplat: https://playcanvas.com/supersplat/editor"
