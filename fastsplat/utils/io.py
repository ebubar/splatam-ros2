"""Dataset + point cloud + Gaussian .ply I/O.

Capture / SfM directory layout used across the pipeline::

    <dir>/
        images/frame_00000.png ...
        depth/frame_00000.npy        # optional, float32 metres
        cameras.json                 # camera intrinsics + optional c2w per frame
        points.npz                   # SfM fused cloud: xyz, rgb, conf
        points.ply                   # same cloud, viewable

``cameras.json`` schema::

    {
      "frames": [
        {
          "file_path": "images/frame_00000.png",
          "w": 640, "h": 360,
          "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]],   # optional
          "c2w": [[...4x4...]],                            # optional, OpenCV cam2world
          "depth_path": "depth/frame_00000.npy",          # optional
          "is_metric": true
        }, ...
      ]
    }
"""

import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# cameras.json
# ---------------------------------------------------------------------------
def save_cameras_json(path, frames, meta=None):
    payload = {"frames": frames}
    if meta:
        payload.update(meta)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_cameras_json(path):
    with open(path) as f:
        return json.load(f)


def _np(x):
    return None if x is None else np.asarray(x, dtype=np.float32)


def frame_intrinsics(frame):
    K = frame.get("intrinsics")
    return None if K is None else _np(K).reshape(3, 3)


def frame_c2w(frame):
    T = frame.get("c2w")
    return None if T is None else _np(T).reshape(4, 4)


# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------
def save_points_npz(path, xyz, rgb, conf=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = dict(xyz=np.asarray(xyz, np.float32), rgb=np.asarray(rgb, np.float32))
    if conf is not None:
        data["conf"] = np.asarray(conf, np.float32)
    np.savez(path, **data)


def load_points_npz(path):
    d = np.load(path)
    return d["xyz"], d["rgb"], (d["conf"] if "conf" in d else None)


def save_point_cloud_ply(path, xyz, rgb):
    """Write a plain colored point cloud (uint8 rgb) as binary PLY."""
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 + 1e-6 else 1.0), 0, 255).astype(np.uint8)
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


# ---------------------------------------------------------------------------
# Gaussian splat .ply  (INRIA / 3DGS convention, SuperSplat-compatible)
# ---------------------------------------------------------------------------
def save_gaussian_ply(path, means, scales_log, quats_wxyz, opacities_logit, f_dc, f_rest=None):
    """Write a 3DGS .ply.

    All values are written *raw* (log scales, logit opacities, SH coefficients)
    exactly as viewers expect -- they apply exp()/sigmoid() themselves. This
    matches ``scripts/export_ply.py`` in the SplaTAM pipeline.

    Args:
        means:            (N, 3)
        scales_log:       (N, 3) log-space scales
        quats_wxyz:       (N, 4) normalized quaternions, w,x,y,z order
        opacities_logit:  (N, 1) or (N,) logit opacities
        f_dc:             (N, 3) SH DC term (band 0)
        f_rest:           (N, 3*(K-1)) higher-order SH, flattened as gsplat stores
                          them (per-channel blocks), or None.
    """
    means = np.asarray(means, np.float32)
    scales_log = np.asarray(scales_log, np.float32)
    quats_wxyz = np.asarray(quats_wxyz, np.float32)
    opacities_logit = np.asarray(opacities_logit, np.float32).reshape(-1, 1)
    f_dc = np.asarray(f_dc, np.float32)
    n = means.shape[0]

    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fields += [(f"f_dc_{i}", "<f4") for i in range(3)]

    rest_cols = 0
    if f_rest is not None and np.asarray(f_rest).size > 0:
        f_rest = np.asarray(f_rest, np.float32).reshape(n, -1)
        rest_cols = f_rest.shape[1]
        fields += [(f"f_rest_{i}", "<f4") for i in range(rest_cols)]

    fields += [("opacity", "<f4")]
    fields += [(f"scale_{i}", "<f4") for i in range(3)]
    fields += [(f"rot_{i}", "<f4") for i in range(4)]

    arr = np.zeros(n, dtype=np.dtype(fields))
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    for i in range(3):
        arr[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(rest_cols):
        arr[f"f_rest_{i}"] = f_rest[:, i]
    arr["opacity"] = opacities_logit[:, 0]
    for i in range(3):
        arr[f"scale_{i}"] = scales_log[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = quats_wxyz[:, i]

    header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header_lines += [f"property float {name}" for name, _ in fields]
    header_lines += ["end_header", ""]
    header = "\n".join(header_lines)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
    return n
