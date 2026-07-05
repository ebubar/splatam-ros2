#!/usr/bin/env python3
"""
Region-of-interest refinement: given a region of a splat you care about,
build a dataset subset of only the frames that observed it, so SplaTAM can
re-splat just that region at a much higher quality budget.

Workflow (GUI = SuperSplat, https://playcanvas.com/supersplat/editor):
  1. Open the full splat.ply in SuperSplat.
  2. Box-select the region of interest, delete everything else
     (Invert Selection -> Delete), and export the remaining splats as a PLY.
     Do NOT move/rotate/scale the splat before exporting.
  3. Run:
       python3 scripts/roi_refine.py \
           --roi-ply roi.ply \
           --dataset experiments/ZED2i_Captures/<scene> \
           --out <scene>_roi
  4. Re-splat at high quality:
       SPLATAM_SCENE=<scene>_roi SPLATAM_MAPPING_ITERS=200 \
           python3 scripts/splatam.py configs/zed2i/zed2i_offline_rtabmap.py

Alternatively pass the box directly: --box xmin xmax ymin ymax zmin zmax
(in the splat's world frame = first-camera coordinates).
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

P_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--roi-ply", help="PLY cropped to the ROI (e.g. SuperSplat export)")
    src.add_argument("--box", type=float, nargs=6,
                     metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    p.add_argument("--dataset", required=True,
                   help="Source dataset dir (rgb/, depth/, transforms.json)")
    p.add_argument("--out", required=True,
                   help="Output scene name (created next to the source dataset)")
    p.add_argument("--margin", type=float, default=0.15,
                   help="Grow the ROI box by this many meters on all sides")
    p.add_argument("--min-visible", type=int, default=2,
                   help="Keep a frame if at least this many of the box's 9 "
                        "test points (8 corners + center) project into it")
    p.add_argument("--max-dist", type=float, default=6.0,
                   help="Discard frames whose camera is farther than this from the box center")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def box_from_ply(path):
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
    return pts.min(axis=0), pts.max(axis=0)


def frame_sees_box(c2w_rel, K, w, h, box_pts, max_dist, min_visible):
    w2c = np.linalg.inv(c2w_rel)
    cam_pos = c2w_rel[:3, 3]
    center = box_pts.mean(axis=0)
    if np.linalg.norm(cam_pos - center) > max_dist:
        return False
    pts_cam = (w2c[:3, :3] @ box_pts.T + w2c[:3, 3:4]).T
    visible = 0
    for pt in pts_cam:
        if pt[2] <= 0.05:
            continue
        u = K[0, 0] * pt[0] / pt[2] + K[0, 2]
        v = K[1, 1] * pt[1] / pt[2] + K[1, 2]
        if 0 <= u < w and 0 <= v < h:
            visible += 1
    return visible >= min_visible


def main():
    args = parse_args()

    if args.roi_ply:
        lo, hi = box_from_ply(args.roi_ply)
    else:
        b = args.box
        lo = np.array([b[0], b[2], b[4]])
        hi = np.array([b[1], b[3], b[5]])
    lo, hi = lo - args.margin, hi + args.margin
    print(f"ROI box (with {args.margin} m margin): min={np.round(lo, 2)} max={np.round(hi, 2)}")

    corners = np.array([[x, y, z]
                        for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    box_pts = np.vstack([corners, (lo + hi) / 2.0])

    dataset = Path(args.dataset)
    with open(dataset / "transforms.json") as f:
        manifest = json.load(f)

    K = np.array([
        [manifest["fl_x"], 0, manifest["cx"]],
        [0, manifest["fl_y"], manifest["cy"]],
        [0, 0, 1.0],
    ])
    w, h = int(manifest["w"]), int(manifest["h"])

    # Splat world frame = first-camera frame (the dataloader makes poses
    # relative to frame 0), so express every pose relative to frame 0 too.
    c2ws = [P_GL @ np.array(fr["transform_matrix"]) @ P_GL for fr in manifest["frames"]]
    c2w0_inv = np.linalg.inv(c2ws[0])
    kept = [
        fr for fr, c2w in zip(manifest["frames"], c2ws)
        if frame_sees_box(c2w0_inv @ c2w, K, w, h, box_pts,
                          args.max_dist, args.min_visible)
    ]

    print(f"Frames observing ROI: {len(kept)}/{len(manifest['frames'])}")
    if not kept:
        sys.exit("No frames see the ROI. Check the box coordinates "
                 "(splat world frame) or raise --max-dist / lower --min-visible.")

    out = dataset.parent / args.out
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    (out / "rgb").mkdir(parents=True)
    (out / "depth").mkdir(parents=True)

    for fr in kept:
        for key in ("file_path", "depth_path"):
            src, dst = dataset / fr[key], out / fr[key]
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy(src, dst)

    out_manifest = dict(manifest)
    out_manifest["frames"] = kept
    with open(out / "transforms.json", "w") as f:
        json.dump(out_manifest, f, indent=2)

    print(f"Wrote ROI dataset: {out}")
    print("Refine with:")
    print(f"  SPLATAM_SCENE={args.out} SPLATAM_MAPPING_ITERS=200 "
          f"SPLATAM_RUN_NAME=roi_refined \\\n"
          f"      python3 scripts/splatam.py configs/zed2i/zed2i_offline_rtabmap.py")


if __name__ == "__main__":
    main()
