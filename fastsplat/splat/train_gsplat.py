"""Time-boxed object-centric Gaussian splatting with gsplat.

Initializes Gaussians from the isolated object point cloud and optimizes them
with the gsplat rasterizer under a hard wall-clock budget so the whole run stays
under the 5-minute target. Uses gsplat's DefaultStrategy for densify/prune.

gsplat API used (confirmed against nerfstudio-project/gsplat main):
    render_colors, render_alphas, info = rasterization(
        means, quats(wxyz), scales, opacities[...,N], colors, viewmats(world2cam),
        Ks, width, height, sh_degree=..., packed=..., render_mode="RGB", ...)
    strategy = DefaultStrategy(...); strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=...)
    strategy.step_pre_backward(params, optimizers, state, step, info)
    ... loss.backward() ...
    strategy.step_post_backward(params, optimizers, state, step, info, packed=...)
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.io import (
    frame_c2w,
    frame_intrinsics,
    load_cameras_json,
    load_points_npz,
    save_gaussian_ply,
)

C0 = 0.28209479177387814


def _rgb_to_sh(rgb):
    return (rgb - 0.5) / C0


def _load_view(obj_dir, fr, device):
    import cv2
    W, H = int(fr["w"]), int(fr["h"])
    img = cv2.imread(os.path.join(obj_dir, fr["file_path"]), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H))
    gt = torch.from_numpy(img).to(device).float() / 255.0        # (H,W,3)

    mask = None
    if fr.get("mask_path"):
        mp = os.path.join(obj_dir, fr["mask_path"])
        if os.path.isfile(mp):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m.shape[1] != W or m.shape[0] != H:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy((m > 127).astype(np.float32)).to(device)  # (H,W)

    c2w = torch.from_numpy(frame_c2w(fr)).to(device).float()
    viewmat = torch.linalg.inv(c2w)                              # world -> cam
    K = torch.from_numpy(frame_intrinsics(fr)).to(device).float()
    return dict(gt=gt, mask=mask, viewmat=viewmat, K=K, W=W, H=H)


def _gaussian_window(win, sigma, device):
    coords = torch.arange(win, device=device) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    return (g @ g.t()).unsqueeze(0).unsqueeze(0)                 # (1,1,win,win)


def _ssim(a, b, window):
    # a, b: (1,3,H,W). Per-channel SSIM with a small gaussian window.
    win = window.shape[-1]
    pad = win // 2
    w = window.expand(a.shape[1], 1, win, win)
    mu_a = F.conv2d(a, w, padding=pad, groups=a.shape[1])
    mu_b = F.conv2d(b, w, padding=pad, groups=a.shape[1])
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, w, padding=pad, groups=a.shape[1]) - mu_a2
    sb = F.conv2d(b * b, w, padding=pad, groups=a.shape[1]) - mu_b2
    sab = F.conv2d(a * b, w, padding=pad, groups=a.shape[1]) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return ssim.mean()


def _init_gaussians(cfg, xyz, rgb, device):
    scfg = cfg["splat"]
    n = xyz.shape[0]

    # Per-point scale from nearest-neighbor spacing (Open3D KD-tree in C++).
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        nn = np.asarray(pcd.compute_nearest_neighbor_distance())
        nn = np.clip(nn, 1e-4, None)
    except Exception:
        nn = np.full(n, float(np.linalg.norm(xyz.std(0)) / (n ** (1 / 3) + 1e-6)))
    scales = np.log(nn * float(scfg["init_scale"]))[:, None].repeat(3, axis=1)

    means = torch.tensor(xyz, dtype=torch.float32, device=device)
    scales = torch.tensor(scales, dtype=torch.float32, device=device)
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    opac = torch.logit(torch.full((n,), float(scfg["init_opacity"]), device=device))

    max_sh = int(scfg["sh_degree"])
    sh0 = torch.tensor(_rgb_to_sh(np.clip(rgb, 0, 1)), dtype=torch.float32, device=device)
    sh0 = sh0.unsqueeze(1)                                       # (N,1,3)
    n_rest = (max_sh + 1) ** 2 - 1
    shN = torch.zeros((n, n_rest, 3), device=device)

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opac),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }).to(device)
    return params


def _make_optimizers(params, cfg, scene_scale):
    lr = cfg["splat"]["lr"]
    # means lr scales with the scene extent (nerfstudio/gsplat convention).
    specs = {
        "means": lr["means"] * scene_scale,
        "scales": lr["scales"],
        "quats": lr["quats"],
        "opacities": lr["opacities"],
        "sh0": lr["sh0"],
        "shN": lr["shN"],
    }
    return {
        k: torch.optim.Adam([{"params": params[k], "lr": v, "name": k}], eps=1e-15)
        for k, v in specs.items()
    }


def _background(name, device):
    if name == "black":
        return torch.zeros(1, 3, device=device)
    if name == "random":
        return torch.rand(1, 3, device=device)
    return torch.ones(1, 3, device=device)                      # white


def run_splat(cfg, obj_dir, out_dir):
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    scfg = cfg["splat"]
    os.makedirs(out_dir, exist_ok=True)

    cams = load_cameras_json(os.path.join(obj_dir, "cameras.json"))
    frames = [fr for fr in cams["frames"] if frame_c2w(fr) is not None
              and frame_intrinsics(fr) is not None]
    if not frames:
        raise RuntimeError("splat: no posed frames (need c2w + intrinsics from SfM)")

    holdout = int(scfg.get("eval_holdout", 0))
    train_frames = frames[:-holdout] if holdout and len(frames) > holdout else frames
    views = [_load_view(obj_dir, fr, device) for fr in train_frames]

    xyz, rgb, _ = load_points_npz(os.path.join(obj_dir, "points.npz"))
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    # Scene scale from camera spread (fallback: point-cloud diagonal).
    centers = np.stack([frame_c2w(fr)[:3, 3] for fr in train_frames], 0)
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).mean()) or \
        float(np.linalg.norm(xyz.max(0) - xyz.min(0)) * 0.5)
    scene_scale = max(scene_scale, 1e-3)

    params = _init_gaussians(cfg, xyz, rgb, device)
    optimizers = _make_optimizers(params, cfg, scene_scale)

    strategy = DefaultStrategy(
        prune_opa=scfg["strategy"]["prune_opa"],
        grow_grad2d=scfg["strategy"]["grow_grad2d"],
        grow_scale3d=scfg["strategy"]["grow_scale3d"],
        refine_start_iter=scfg["strategy"]["refine_start_iter"],
        refine_stop_iter=scfg["strategy"]["refine_stop_iter"],
        refine_every=scfg["strategy"]["refine_every"],
        reset_every=scfg["strategy"]["reset_every"],
        absgrad=scfg["absgrad"],
        verbose=False,
    )
    use_densify = bool(scfg["densify"])
    if use_densify:
        strategy.check_sanity(params, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    bg = _background(scfg["background"], device)
    window = _gaussian_window(11, 1.5, device)
    ssim_lambda = float(scfg["ssim_lambda"])
    max_sh = int(scfg["sh_degree"])

    def render(view):
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N,K,3)
        rc, ra, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=view["viewmat"][None],          # (1,4,4)
            Ks=view["K"][None],                       # (1,3,3)
            width=view["W"],
            height=view["H"],
            sh_degree=max_sh,
            packed=scfg["packed"],
            near_plane=scfg["near_plane"],
            far_plane=scfg["far_plane"],
            render_mode="RGB",
            absgrad=scfg["absgrad"],
            rasterize_mode="antialiased",
            backgrounds=bg,
        )
        return rc, ra, info

    max_steps = int(scfg["max_steps"])
    budget = float(scfg["time_budget_s"])
    t0 = time.time()
    rng = np.random.default_rng(cfg["seed"])
    step = 0
    print(f"[splat] {params['means'].shape[0]:,} gaussians | "
          f"budget {budget:.0f}s / {max_steps} steps | densify={use_densify}")

    while step < max_steps:
        if time.time() - t0 > budget:
            print(f"[splat] time budget reached at step {step}")
            break
        view = views[int(rng.integers(len(views)))]
        rc, ra, info = render(view)                   # rc: (1,H,W,3)
        pred = rc[0]
        alpha = ra[0, ..., 0]                         # (H,W)

        gt = view["gt"]
        mask = view["mask"]
        if mask is not None:
            # Composite GT onto background outside the object, and encourage
            # rendered alpha to match the object silhouette.
            gt = gt * mask[..., None] + bg[0] * (1.0 - mask[..., None])

        if use_densify:
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        l1 = (pred - gt).abs().mean()
        s = _ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], window)
        loss = (1 - ssim_lambda) * l1 + ssim_lambda * (1 - s)
        if mask is not None:
            loss = loss + 0.5 * (alpha - mask).abs().mean()

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if use_densify:
            strategy.step_post_backward(
                params, optimizers, strategy_state, step, info, packed=scfg["packed"]
            )

        if step % int(scfg["log_every"]) == 0:
            print(f"[splat] step {step:5d} | loss {loss.item():.4f} | "
                  f"gaussians {params['means'].shape[0]:,} | "
                  f"{time.time() - t0:.1f}s")
        step += 1

    _export(cfg, params, obj_dir, out_dir)
    print(f"[splat] done in {time.time() - t0:.1f}s, {step} steps")
    return out_dir


def _export(cfg, params, obj_dir, out_dir):
    scfg = cfg["splat"]
    with torch.no_grad():
        means = params["means"]
        opac = torch.sigmoid(params["opacities"])
        keep = opac > scfg["strategy"]["prune_opa"]

        # Final spatial crop to the object box (if isolation produced one).
        crop_path = os.path.join(obj_dir, "crop.json")
        if os.path.isfile(crop_path):
            with open(crop_path) as f:
                box = json.load(f)
            mn = torch.tensor(box["aabb_min"], device=means.device)
            mx = torch.tensor(box["aabb_max"], device=means.device)
            inside = ((means >= mn) & (means <= mx)).all(dim=1)
            keep = keep & inside

        idx = torch.where(keep)[0]
        means = means[idx].cpu().numpy()
        scales_log = params["scales"][idx].cpu().numpy()
        quats = F.normalize(params["quats"][idx], dim=1).cpu().numpy()
        opac_logit = params["opacities"][idx].cpu().numpy()
        f_dc = params["sh0"][idx, 0].cpu().numpy()
        shN = params["shN"][idx]
        # INRIA f_rest layout: channels-major, i.e. transpose (K-1,3) -> (3,K-1).
        f_rest = shN.transpose(1, 2).reshape(shN.shape[0], -1).cpu().numpy() \
            if shN.shape[1] > 0 else None

    ply_path = os.path.join(out_dir, cfg["export"]["ply_name"])
    n = save_gaussian_ply(ply_path, means, scales_log, quats, opac_logit, f_dc, f_rest)
    print(f"[export] wrote {n:,} gaussians -> {ply_path}")
