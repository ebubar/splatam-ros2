#!/usr/bin/env python3
"""Plan a benchmark-quality capture for an object, and/or score existing coverage.

No hardware needed -- this is a planning tool. Give it an object size (or a box)
and it prints the ideal set of viewpoints derived from splat/NeRF benchmark
capture patterns (see docs/CAPTURE_PATTERNS.md). Optionally point it at a saved
``params.npz`` to report how well the poses you already captured cover that box
and where the remaining gaps are.

Examples:
    # Ideal plan for a ~0.6 m object, written to a JSON file:
    python3 scripts/plan_capture.py --radius 0.3 --out plan.json

    # Plan for an explicit world-frame box:
    python3 scripts/plan_capture.py --box -0.4 0 0.6 0.4 0.8 1.4

    # Coverage report for a box against an existing capture:
    python3 scripts/plan_capture.py --box -0.4 0 0.6 0.4 0.8 1.4 \
        --params experiments/.../params.npz
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livesplat.roi import RegionOfInterest
from livesplat import capture_advisor


def build_roi(args) -> RegionOfInterest:
    if args.box is not None:
        b = args.box
        return RegionOfInterest(min_xyz=tuple(b[:3]), max_xyz=tuple(b[3:]), label="object")
    center = np.array(args.center, dtype=float)
    if args.radius is not None:
        # radius is half the box diagonal -> half-extent along each axis
        h = float(args.radius) / np.sqrt(3.0)
        half = np.array([h, h, h])
    else:
        half = np.array(args.size, dtype=float) / 2.0
    return RegionOfInterest(min_xyz=tuple(center - half), max_xyz=tuple(center + half),
                            label="object")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--radius", type=float, help="object radius (half box-diagonal), metres")
    g.add_argument("--size", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="object bounding-box size in metres")
    ap.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    metavar=("X", "Y", "Z"), help="object centre in world frame")
    ap.add_argument("--box", type=float, nargs=6,
                    metavar=("minX", "minY", "minZ", "maxX", "maxY", "maxZ"),
                    help="explicit world-frame axis-aligned box (overrides size/center)")
    ap.add_argument("--elevations", type=float, nargs="+",
                    default=list(capture_advisor.DEFAULT_ELEVATIONS_DEG),
                    help="elevation rings in degrees")
    ap.add_argument("--az-step", type=float, default=capture_advisor.DEFAULT_AZIMUTH_STEP_DEG,
                    help="azimuth spacing in degrees")
    ap.add_argument("--params", type=str, default=None,
                    help="SplaTAM params.npz to score existing coverage of the box")
    ap.add_argument("--out", type=str, default=None, help="write the plan to this JSON file")
    args = ap.parse_args()

    if args.box is None and args.radius is None and args.size is None:
        args.radius = 0.3  # sensible default object (~0.6 m)

    roi = build_roi(args)
    plan = capture_advisor.ideal_capture_plan(
        roi, elevations_deg=tuple(args.elevations), azimuth_step_deg=args.az_step
    )

    lo, hi = roi.as_numpy()
    print(f"Object box: min={tuple(round(float(v),3) for v in lo)} "
          f"max={tuple(round(float(v),3) for v in hi)}  "
          f"radius={roi.radius():.3f} m")
    print(f"Ideal capture plan: {len(plan)} viewpoints "
          f"({len(args.elevations)} rings x ~{int(round(360/args.az_step))} az), "
          f"radius {capture_advisor.DEFAULT_RADIUS_SCALE:.1f}x object.")
    for vp in plan[: min(len(plan), 6)]:
        print(f"  az {vp.azimuth_deg:6.1f}  el {vp.elevation_deg:5.1f}  "
              f"pos {tuple(round(v,2) for v in vp.position)}")
    if len(plan) > 6:
        print(f"  ... (+{len(plan) - 6} more)")

    report = {"roi": roi.to_dict(), "plan": [vp.to_dict() for vp in plan]}

    if args.params:
        data = np.load(args.params, allow_pickle=True)
        if "gt_w2c_all_frames" not in data:
            print("\n[warn] params.npz has no 'gt_w2c_all_frames'; cannot score coverage.")
        else:
            w2c = np.asarray(data["gt_w2c_all_frames"])
            cov = capture_advisor.analyse_coverage(roi, w2c)
            report["coverage"] = cov
            print(f"\nExisting capture: {cov['n_views']} poses -> "
                  f"coverage {cov['coverage']*100:.0f}% of the view sphere.")
            if cov["hints"]:
                print("Next-best views to fill the biggest gaps:")
                for h in cov["hints"]:
                    print(f"  az {h['azimuth_deg']:6.1f}  el {h['elevation_deg']:5.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote plan to {args.out}")


if __name__ == "__main__":
    main()
