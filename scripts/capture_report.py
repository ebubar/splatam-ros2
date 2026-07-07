#!/usr/bin/env python3
"""
Analyze a dataset's capture pattern and score it against ranges that produce
good splats. Use it to diagnose a robot walkthrough (frame drops, motion too
fast, blur, bad depth) and to design capture trajectories.

    python3 scripts/capture_report.py --dataset experiments/ZED2i_Captures/<scene>

Metrics per consecutive frame pair / per frame:
  - translation step (m) and rotation step (deg): how far the camera moved
    between the frames SplaTAM actually gets. Big steps (fast motion or
    dropped frames) mean less overlap, worse tracking and mapping.
  - motion blur score (variance of Laplacian): low = blurry frame.
  - depth coverage: fraction of pixels with valid depth after filtering.

Target ranges are derived from captures that splat well (TUM fr1/desk-class
handheld and slower): they are heuristics, not hard limits.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

P_GL = np.diag([1.0, -1.0, -1.0, 1.0])
DEPTH_PNG_SCALE = 6553.5

# Heuristic "splats well" ranges
GOOD = {
    "trans_step_m": (0.005, 0.10),
    "rot_step_deg": (0.2, 6.0),
    "blur_var": (60.0, None),
    "depth_valid_frac": (0.55, None),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--sample-every", type=int, default=1,
                   help="Analyze every Nth frame (images are read from disk)")
    return p.parse_args()


def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def flag(value, lo, hi):
    if lo is not None and value < lo:
        return "LOW"
    if hi is not None and value > hi:
        return "HIGH"
    return "ok"


def main():
    args = parse_args()
    base = Path(args.dataset)
    with open(base / "transforms.json") as f:
        manifest = json.load(f)
    frames = manifest["frames"][:: args.sample_every]
    if len(frames) < 2:
        raise SystemExit("Need at least 2 frames")

    c2ws = [P_GL @ np.array(fr["transform_matrix"]) @ P_GL for fr in frames]

    trans_steps, rot_steps, blurs, depth_fracs = [], [], [], []
    for i, fr in enumerate(frames):
        if i > 0:
            rel = np.linalg.inv(c2ws[i - 1]) @ c2ws[i]
            trans_steps.append(float(np.linalg.norm(rel[:3, 3])))
            rot_steps.append(rot_angle_deg(rel[:3, :3]))
        gray = cv2.imread(str(base / fr["file_path"]), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            blurs.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        depth = cv2.imread(str(base / fr["depth_path"]), cv2.IMREAD_UNCHANGED)
        if depth is not None:
            depth_fracs.append(float((depth > 0).mean()))

    def stats(name, arr, lo, hi, unit):
        arr = np.array(arr)
        med, p90 = np.median(arr), np.percentile(arr, 90)
        worst = arr.max() if name != "blur_var" and name != "depth_valid_frac" else arr.min()
        f = flag(med, lo, hi)
        bounds = f"good: {lo if lo is not None else ''}..{hi if hi is not None else ''}{unit}"
        print(f"  {name:18s} median={med:8.3f}{unit}  p90={p90:8.3f}{unit} "
              f" worst={worst:8.3f}{unit}   [{f}] ({bounds})")
        return f

    print(f"Capture report: {base}  ({len(frames)} frames analyzed)")
    print()
    flags = {}
    flags["trans"] = stats("trans_step_m", trans_steps, *GOOD["trans_step_m"], " m")
    flags["rot"] = stats("rot_step_deg", rot_steps, *GOOD["rot_step_deg"], " deg")
    flags["blur"] = stats("blur_var", blurs, *GOOD["blur_var"], "")
    flags["depth"] = stats("depth_valid_frac", depth_fracs, *GOOD["depth_valid_frac"], "")

    big_steps = int(np.sum(np.array(trans_steps) > GOOD["trans_step_m"][1]))
    print(f"\n  frames with oversized translation steps (likely drops/fast motion): "
          f"{big_steps}/{len(trans_steps)}")

    print("\nGuidance:")
    if flags["trans"] == "HIGH" or big_steps > len(trans_steps) * 0.2:
        print("  * Camera moves too far between processed frames: slow the robot,")
        print("    raise the processed frame rate (lower ros.process_every_n), or fix")
        print("    frame drops (see 'Handling ZED frame drops' in docs/README.md).")
    if flags["rot"] == "HIGH":
        print("  * Rotation between frames is large: slow turns; rotating in place")
        print("    at high speed is the worst case for tracking and blur.")
    if flags["blur"] == "LOW":
        print("  * Frames are blurry: slow down, add light, or raise camera exposure")
        print("    priority. Blurry inputs put a hard ceiling on splat sharpness.")
    if flags["depth"] == "LOW":
        print("  * Depth coverage is poor: scene may be too far/too close for the")
        print("    sensor's range, or reflective/textureless. Check min/max depth.")
    if all(v == "ok" for v in flags.values()) and big_steps <= len(trans_steps) * 0.2:
        print("  * Capture pattern looks good for splatting.")


if __name__ == "__main__":
    main()
