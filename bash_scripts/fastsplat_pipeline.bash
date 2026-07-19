#!/bin/bash
# Run the full fastsplat pipeline (SfM -> object isolation -> gsplat) on the
# GPU compute image. Assumes sparse keyframes already exist (either an image
# folder for `ingest.source: folder`, or a prior ROS2 capture).
#
# Usage:
#   bash bash_scripts/fastsplat_pipeline.bash [config] [--image-dir DIR] [--stage LIST]
#
# Examples:
#   bash bash_scripts/fastsplat_pipeline.bash
#   bash bash_scripts/fastsplat_pipeline.bash configs/fastsplat/fastsplat.py --image-dir data/mug/images
#   bash bash_scripts/fastsplat_pipeline.bash configs/fastsplat/fastsplat.py --stage object,splat
set -euo pipefail

CONFIG="${1:-configs/fastsplat/fastsplat.py}"
if [ "$#" -ge 1 ]; then shift; fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "[fastsplat] config: $CONFIG"
python3 -u -m fastsplat.run_pipeline --config "$CONFIG" "$@"
