"""End-to-end fastsplat pipeline: ingest -> SfM -> object -> splat.

    python -m fastsplat.run_pipeline --config configs/fastsplat/fastsplat.py

Run a subset with --stage (comma-separated), e.g. re-run only splatting::

    python -m fastsplat.run_pipeline --config <cfg> --stage object,splat

Each stage writes into <workdir>/<run_name>/<stage>/ and the next stage reads
from the previous stage's directory, so stages can be resumed independently.
"""

import argparse
import os
import shutil
import time

from .utils.config import load_config

STAGES = ["ingest", "sfm", "object", "splat"]


def _dirs(cfg):
    root = os.path.join(cfg["workdir"], cfg["run_name"])
    return {
        "root": root,
        "capture": os.path.join(root, "capture"),
        "sfm": os.path.join(root, "sfm"),
        "object": os.path.join(root, "object"),
        "splat": os.path.join(root, "splat"),
    }


def run(cfg, stages=None):
    """Run the pipeline programmatically. Returns the output .ply path (or None)."""
    stages = stages or STAGES
    d = _dirs(cfg)

    if cfg.get("overwrite") and "ingest" in stages and os.path.isdir(d["root"]):
        shutil.rmtree(d["root"])
    os.makedirs(d["root"], exist_ok=True)

    t0 = time.time()

    if "ingest" in stages:
        from .ingest import run_ingest
        os.makedirs(d["capture"], exist_ok=True)
        run_ingest(cfg, d["capture"])

    if "sfm" in stages:
        from .sfm import run_sfm
        run_sfm(cfg, d["capture"], d["sfm"])

    if "object" in stages:
        from .object import run_isolate
        run_isolate(cfg, d["sfm"], d["object"])

    if "splat" in stages:
        from .splat import run_splat
        run_splat(cfg, d["object"], d["splat"])

    print(f"\n[fastsplat] pipeline finished in {time.time() - t0:.1f}s")
    ply = os.path.join(d["splat"], cfg["export"]["ply_name"])
    if os.path.isfile(ply):
        print(f"[fastsplat] object splat: {ply}")
        return ply
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fastsplat/fastsplat.py")
    ap.add_argument("--stage", default="all",
                    help="comma-separated subset of: " + ",".join(STAGES) + " (or 'all')")
    ap.add_argument("--image-dir", default=None, help="override ingest.image_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.image_dir:
        cfg["ingest"]["image_dir"] = args.image_dir
        cfg["ingest"]["source"] = "folder"

    stages = STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    run(cfg, stages)


if __name__ == "__main__":
    main()
