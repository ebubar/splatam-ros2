#!/usr/bin/env python3
"""
Backend convention self-test.

Renders a handful of known Gaussians at one fixed pose/intrinsics through BOTH
rendering backends (cuda + gsplat) and diffs RGB / depth / alpha. This catches
the easy-to-get-wrong conversions before any ROS wiring:
  * quaternion order (wxyz)      * scales = exp(log_scales)
  * opacities = sigmoid, shape N * world-space means + w2c viewmat (gsplat)

Expected: RGB and coverage/alpha match closely. Depth may differ slightly in
low-alpha regions because gsplat "RGB+ED" returns expected (alpha-normalized)
depth while the CUDA path returns accumulated depth; compare where alpha ~ 1.

Runs only what is installed: if torch / a backend / CUDA is missing it reports
SKIP for that part rather than failing. Intended to be run on the target GPU box.

Usage:
  python3 scripts/tools/render_backend_selftest.py
"""

import os
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE_DIR)


def build_scene(device):
    import torch
    torch.manual_seed(0)
    N = 200
    # Gaussians in a plane ~2 m in front of the camera, spread in x/y.
    xy = (torch.rand(N, 2, device=device) - 0.5) * 1.5
    z = torch.full((N, 1), 2.0, device=device) + (torch.rand(N, 1, device=device) - 0.5) * 0.2
    means = torch.cat([xy, z], dim=1).float()
    params = {
        "means3D": torch.nn.Parameter(means),
        "rgb_colors": torch.nn.Parameter(torch.rand(N, 3, device=device).float()),
        "unnorm_rotations": torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(N, 1)),
        "logit_opacities": torch.nn.Parameter(torch.full((N, 1), 2.0, device=device)),  # ~0.88 opacity
        "log_scales": torch.nn.Parameter(torch.full((N, 1), -3.0, device=device)),      # small isotropic
        "cam_unnorm_rots": torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).T[None]),  # [1,4,1]
        "cam_trans": torch.nn.Parameter(torch.zeros(1, 3, 1, device=device)),
    }
    return params


def make_curr_data(device):
    import torch
    from utils.recon_helpers import setup_camera
    import numpy as np
    H, W = 240, 320
    fx = fy = 250.0
    K = np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1]], dtype=np.float32)
    w2c = torch.eye(4, device=device).float()
    cam = setup_camera(W, H, K, w2c.detach().cpu().numpy())
    curr_data = {
        "cam": cam,
        "w2c": w2c,
        "intrinsics": torch.tensor(K, device=device).float(),
        "im": torch.zeros(3, H, W, device=device),  # only shape (H,W) is read by gsplat
        "depth": torch.zeros(1, H, W, device=device),
        "id": 0,
    }
    return curr_data


def run_backend(name, params, curr_data, gsplat_cfg):
    from utils.render_backend import render_frame
    out = render_frame(params, 0, curr_data, {}, gaussians_grad=False,
                       camera_grad=False, backend=name, gsplat_cfg=gsplat_cfg)
    return out


def main():
    try:
        import torch
    except Exception as exc:
        print(f"SKIP: torch not available ({exc})")
        return
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available (both backends require a CUDA GPU).")
        return

    device = torch.device("cuda:0")
    params = build_scene(device)
    curr_data = make_curr_data(device)
    gsplat_cfg = {"render_mode": "RGB+ED", "packed": False, "near_plane": 0.01, "far_plane": 100.0}

    results = {}
    for name in ("cuda", "gsplat"):
        try:
            results[name] = run_backend(name, params, curr_data, gsplat_cfg)
            r = results[name]
            print(f"[{name}] im{tuple(r['im'].shape)} depth{tuple(r['depth'].shape)} "
                  f"sil{tuple(r['silhouette'].shape)} "
                  f"alpha_mean={float(r['silhouette'].mean()):.3f}")
        except Exception as exc:
            print(f"SKIP [{name}]: {exc}")

    if "cuda" in results and "gsplat" in results:
        a, b = results["cuda"], results["gsplat"]
        alpha_mask = (a["silhouette"] > 0.9) & (b["silhouette"] > 0.9)
        rgb_diff = (a["im"] - b["im"]).abs().mean().item()
        sil_diff = (a["silhouette"] - b["silhouette"]).abs().mean().item()
        if alpha_mask.any():
            d = (a["depth"][0] - b["depth"][0]).abs()[alpha_mask]
            depth_diff = float(d.mean())
        else:
            depth_diff = float("nan")
        print("== diffs (cuda vs gsplat) ==")
        print(f"  mean |RGB| diff        : {rgb_diff:.4f}  (expect small)")
        print(f"  mean |alpha| diff      : {sil_diff:.4f}  (expect small)")
        print(f"  mean |depth| diff@a>0.9: {depth_diff:.4f}  (expect small where alpha~1)")
        ok = rgb_diff < 0.1 and sil_diff < 0.1
        print("RESULT:", "PASS" if ok else "REVIEW (diffs larger than expected)")


if __name__ == "__main__":
    main()
