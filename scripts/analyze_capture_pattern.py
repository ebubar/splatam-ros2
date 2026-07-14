#!/usr/bin/env python3
"""
Offline capture-pattern analyzer.

Given a SplaTAM output (params.npz containing gt_w2c_all_frames) or a raw poses
array, this characterizes the physical capture trajectory and compares it to the
geometry of the successful 3DGS benchmark captures:

  * translation baseline between consecutive frames (parallax, not pure rotation)
  * viewing-angle change between consecutive frames (~20-30 deg is healthy)
  * camera-to-scene-centroid distance consistency (fixed-radius orbits)
  * azimuth coverage around the scene (loop closure / gaps)

Use it to sanity-check a capture and to calibrate the thresholds in
utils/capture_guidance.py against a known-good capture.

Usage:
  python3 scripts/analyze_capture_pattern.py --params experiments/.../params.npz
  python3 scripts/analyze_capture_pattern.py --poses poses_w2c.npy --out report.png
"""

import argparse
import os

import numpy as np


def load_w2c(args):
    if args.params:
        data = np.load(args.params, allow_pickle=True)
        key = "gt_w2c_all_frames" if "gt_w2c_all_frames" in data else None
        if key is None:
            raise SystemExit(f"{args.params} has no 'gt_w2c_all_frames' (keys: {list(data.keys())})")
        return np.asarray(data[key], dtype=np.float64)
    if args.poses:
        arr = np.load(args.poses, allow_pickle=True)
        return np.asarray(arr, dtype=np.float64)
    raise SystemExit("Provide --params <params.npz> or --poses <NxMx4x4 .npy>")


def cam_centers(w2c):
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    # center = -R^T t  (batched)
    return -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)


def view_dirs(w2c):
    R = w2c[:, :3, :3]
    z = np.array([0.0, 0.0, 1.0])
    return np.einsum("nij,j->ni", np.transpose(R, (0, 2, 1)), z)


def analyze(w2c):
    n = w2c.shape[0]
    centers = cam_centers(w2c)
    dirs = view_dirs(w2c)
    centroid = np.median(centers, axis=0)

    baselines = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    cosang = np.clip(np.sum(dirs[1:] * dirs[:-1], axis=1), -1, 1)
    angles = np.rad2deg(np.arccos(cosang))
    radii = np.linalg.norm(centers - centroid, axis=1)
    az = np.rad2deg(np.arctan2(centers[:, 1] - centroid[1], centers[:, 0] - centroid[0]))

    stats = {
        "num_frames": n,
        "baseline_median_m": float(np.median(baselines)) if n > 1 else 0.0,
        "baseline_min_m": float(np.min(baselines)) if n > 1 else 0.0,
        "angle_median_deg": float(np.median(angles)) if n > 1 else 0.0,
        "radius_mean_m": float(np.mean(radii)),
        "radius_cov": float(np.std(radii) / (np.mean(radii) + 1e-9)),
        "azimuth_span_deg": float(az.max() - az.min()),
        "pure_rotation_frac": float(np.mean(baselines < 1e-3)) if n > 1 else 0.0,
    }
    return stats, centers, angles, baselines, radii, az, centroid


def verdict(stats):
    msgs = []
    if stats["pure_rotation_frac"] > 0.2:
        msgs.append("WARN: many near-pure-rotation frames -> weak parallax (bad for 3DGS).")
    if stats["angle_median_deg"] < 5:
        msgs.append("WARN: tiny inter-frame angle -> views too similar; orbit more.")
    if stats["angle_median_deg"] > 40:
        msgs.append("WARN: large inter-frame angle -> overlap likely <60%; slow down / add frames.")
    if stats["radius_cov"] > 0.4:
        msgs.append("WARN: inconsistent camera-to-scene distance -> keep a steadier radius.")
    if stats["azimuth_span_deg"] < 180:
        msgs.append("INFO: <180deg azimuth coverage -> partial orbit; consider closing the loop.")
    if not msgs:
        msgs.append("OK: capture geometry looks healthy (parallax present, steady radius, good coverage).")
    return msgs


def plot(out, centers, angles, baselines, radii, az, centroid):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(matplotlib unavailable, skipping plot: {exc})")
        return
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(centers[:, 0], centers[:, 1], "-o", ms=3)
    ax1.plot([centroid[0]], [centroid[1]], "r*", ms=12, label="centroid")
    ax1.set_title("Top-down trajectory (x-y)"); ax1.axis("equal"); ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(angles, bins=30); ax2.axvspan(20, 30, color="g", alpha=0.2, label="healthy 20-30 deg")
    ax2.set_title("Inter-frame view-angle change (deg)"); ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(baselines, bins=30); ax3.set_title("Inter-frame baseline (m)")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(radii, "-"); ax4.set_title("Camera-to-centroid distance per frame (m)")

    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default=None, help="params.npz with gt_w2c_all_frames")
    ap.add_argument("--poses", type=str, default=None, help="Nx4x4 world->cam .npy")
    ap.add_argument("--out", type=str, default="capture_pattern_report.png")
    args = ap.parse_args()

    w2c = load_w2c(args)
    if w2c.ndim != 3 or w2c.shape[-2:] != (4, 4):
        raise SystemExit(f"Expected [N,4,4] poses, got {w2c.shape}")

    stats, centers, angles, baselines, radii, az, centroid = analyze(w2c)
    print("== Capture pattern stats ==")
    for k, v in stats.items():
        print(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    print("== Verdict ==")
    for m in verdict(stats):
        print("  " + m)
    plot(args.out, centers, angles, baselines, radii, az, centroid)


if __name__ == "__main__":
    main()
