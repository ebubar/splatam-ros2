#!/bin/bash
# High-quality splat backend: full 3D Gaussian Splatting optimization
# (nerfstudio "splatfacto" / gsplat) on a dataset produced by
# scripts/rtabmap2dataset.py. Much higher photorealism than the online
# SplaTAM mapper; typical runtime 3-8 minutes for a small scene.
#
# Usage:
#   bash_scripts/hq_splat.bash <dataset_dir> [iterations]
#
#   bash_scripts/hq_splat.bash experiments/ZED2i_Captures/robot_run1 8000
#
# Requires: pip install nerfstudio   (in the splatam env)
set -euo pipefail

if [ "$#" -lt 1 ]; then
    grep "^#" "$0" | head -12
    exit 1
fi

DATASET="$1"
ITERS="${2:-8000}"
NAME="$(basename "$DATASET")_hq"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

# Seed the Gaussians from the dataset's real depth geometry (idempotent)
python3 scripts/make_seed_points.py --dataset "$DATASET"

echo "[hq_splat] Training splatfacto for $ITERS iterations..."
ns-train splatfacto \
    --data "$DATASET" \
    --output-dir experiments/hq_outputs \
    --experiment-name "$NAME" \
    --vis tensorboard \
    --max-num-iterations "$ITERS" \
    --pipeline.model.random-init False \
    nerfstudio-data --load-3D-points True

RUN_DIR=$(ls -td experiments/hq_outputs/"$NAME"/splatfacto/*/ | head -1)

echo "[hq_splat] Evaluating..."
ns-eval --load-config "$RUN_DIR/config.yml" \
    --output-path "$RUN_DIR/metrics.json" || true
[ -f "$RUN_DIR/metrics.json" ] && \
    python3 -c "import json; r=json.load(open('$RUN_DIR/metrics.json'))['results']; \
print('  PSNR %.2f | SSIM %.3f | LPIPS %.3f' % (r['psnr'], r['ssim'], r['lpips']))"

echo "[hq_splat] Exporting PLY..."
ns-export gaussian-splat --load-config "$RUN_DIR/config.yml" --output-dir "$RUN_DIR"

echo ""
echo "[hq_splat] Done. High-quality splat:"
ls "$RUN_DIR"/*.ply
echo "Open in SuperSplat: https://playcanvas.com/supersplat/editor"
