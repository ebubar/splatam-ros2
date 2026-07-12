#!/usr/bin/env python3
"""
Generate a seed point cloud for 3DGS training (nerfstudio splatfacto) by
back-projecting the dataset's depth maps, and register it in transforms.json
as "ply_file_path". Seeding from real geometry converges faster and cleaner
than splatfacto's random initialization.

    python3 scripts/make_seed_points.py --dataset experiments/ZED2i_Captures/<scene>
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

P_GL = np.diag([1.0, -1.0, -1.0, 1.0])
DEPTH_PNG_SCALE = 6553.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--every-frame", type=int, default=4)
    p.add_argument("--pixel-stride", type=int, default=8)
    p.add_argument("--max-points", type=int, default=400_000)
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.dataset)
    with open(base / "transforms.json") as f:
        manifest = json.load(f)

    fx, fy = manifest["fl_x"], manifest["fl_y"]
    cx, cy = manifest["cx"], manifest["cy"]

    pts_all, cols_all = [], []
    for fr in manifest["frames"][:: args.every_frame]:
        depth_png = cv2.imread(str(base / fr["depth_path"]), cv2.IMREAD_UNCHANGED)
        bgr = cv2.imread(str(base / fr["file_path"]))
        if depth_png is None or bgr is None:
            continue
        depth = depth_png.astype(np.float32) / DEPTH_PNG_SCALE

        s = args.pixel_stride
        d = depth[::s, ::s]
        c = bgr[::s, ::s]
        vs, us = np.mgrid[0 : depth.shape[0] : s, 0 : depth.shape[1] : s]
        valid = d > 0
        z = d[valid]
        u = us[valid].astype(np.float32)
        v = vs[valid].astype(np.float32)
        pts_cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)

        # transform_matrix is OpenGL c2w; work in CV convention
        c2w = P_GL @ np.array(fr["transform_matrix"]) @ P_GL
        pts_w = (c2w[:3, :3] @ pts_cam.T + c2w[:3, 3:4]).T
        pts_all.append(pts_w)
        cols_all.append(c[valid][:, ::-1])  # BGR -> RGB

    pts = np.vstack(pts_all)
    cols = np.vstack(cols_all)
    if len(pts) > args.max_points:
        sel = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[sel], cols[sel]

    # nerfstudio applies the same GL flip to the seed cloud as to the poses,
    # so store the points in the same (OpenGL-consistent) world as the json:
    # our poses are P @ c2w_cv @ P, i.e. world coords get flipped by P too.
    pts_gl = pts @ P_GL[:3, :3]  # symmetric diag flip

    from plyfile import PlyData, PlyElement
    vert = np.zeros(len(pts_gl), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vert["x"], vert["y"], vert["z"] = pts_gl.T
    vert["red"], vert["green"], vert["blue"] = cols.T
    ply_path = base / "seed_points.ply"
    PlyData([PlyElement.describe(vert, "vertex")]).write(str(ply_path))

    manifest["ply_file_path"] = "seed_points.ply"
    with open(base / "transforms.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(pts_gl):,} seed points to {ply_path} and registered "
          f"ply_file_path in transforms.json")


if __name__ == "__main__":
    main()
