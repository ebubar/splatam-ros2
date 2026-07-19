"""MapAnything feed-forward SfM.

Turns a sparse set of images (+ optional ZED priors) into globally-consistent
camera poses, intrinsics, and a fused metric point cloud -- in a single
feed-forward pass, no COLMAP.

Why this fixes the ZED "ghost splats at different heights":
    Those duplicated surfaces come from per-frame ZED odometry drift -- the
    poses are locally fine but globally inconsistent, so the same surface lands
    at slightly different world positions across frames. MapAnything solves all
    camera poses *jointly* from the images, producing one consistent geometry.
    By default we therefore do NOT feed raw odom poses (see ``input_mode``); we
    only optionally trust the ZED intrinsics, which are stable.

API reference (facebookresearch/map-anything, Apache-2.0 checkpoint):
    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained("facebook/map-anything-apache").to(device)
    preds = model.infer(views, memory_efficient_inference=..., use_amp=..., ...)
  Each view dict: {"img": (H,W,3) [0,255], "intrinsics": (3,3), "depth_z": (H,W),
                   "camera_poses": (4,4) OpenCV cam2world, "is_metric_scale": (1,)}
  Each prediction exposes: pts3d, depth_z, camera_poses (cam2world), intrinsics,
                           conf, mask, img_no_norm.
"""

import os

import numpy as np
import torch

from ..utils.io import (
    frame_c2w,
    frame_intrinsics,
    load_cameras_json,
    save_cameras_json,
    save_point_cloud_ply,
    save_points_npz,
)


def _load_rgb(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _squeeze_batch(x):
    """MapAnything returns per-view tensors with a leading batch dim; drop it."""
    t = x.detach().float().cpu() if torch.is_tensor(x) else torch.as_tensor(x)
    if t.ndim >= 1 and t.shape[0] == 1:
        t = t[0]
    return t.numpy()


def _view_field(view, *names):
    for n in names:
        if n in view and view[n] is not None:
            return view[n]
    return None


def _build_views(cfg, capture_dir, frames, device):
    """Assemble MapAnything input views from the capture, honoring input_mode."""
    mode = cfg["sfm"]["input_mode"]
    want_K = mode in ("rgb_intrinsics", "rgb_intrinsics_depth", "full_priors")
    want_depth = mode == "rgb_intrinsics_depth"
    want_pose = mode == "full_priors"

    views, rgbs = [], []
    for fr in frames:
        rgb = _load_rgb(os.path.join(capture_dir, fr["file_path"]))
        rgbs.append(rgb)
        view = {"img": torch.from_numpy(rgb.copy()).to(device).float()}

        K = frame_intrinsics(fr)
        if want_K and K is not None:
            view["intrinsics"] = torch.from_numpy(K).to(device).float()
        if want_depth and fr.get("depth_path"):
            depth = np.load(os.path.join(capture_dir, fr["depth_path"]))
            view["depth_z"] = torch.from_numpy(depth.astype(np.float32)).to(device)
            view["is_metric_scale"] = torch.tensor([bool(cfg["sfm"]["is_metric_scale"])])
        if want_pose:
            c2w = frame_c2w(fr)
            if c2w is not None:
                view["camera_poses"] = torch.from_numpy(c2w).to(device).float()
        views.append(view)
    return views, rgbs


def run_sfm(cfg, capture_dir, out_dir):
    from mapanything.models import MapAnything

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    scfg = cfg["sfm"]
    os.makedirs(out_dir, exist_ok=True)

    cams = load_cameras_json(os.path.join(capture_dir, "cameras.json"))
    frames = cams["frames"]
    if not frames:
        raise RuntimeError("SfM: no frames in capture")

    print(f"[sfm] MapAnything ({scfg['model']}) on {len(frames)} views, "
          f"input_mode={scfg['input_mode']}")
    model = MapAnything.from_pretrained(scfg["model"]).to(device).eval()

    views, rgbs = _build_views(cfg, capture_dir, frames, device)

    # Optional preprocessing (resize to network stride, normalize). Newer
    # releases do this inside infer(); call it when available for safety.
    try:
        from mapanything.utils.image import preprocess_inputs
        views = preprocess_inputs(views)
    except Exception:
        pass

    # Only pass kwargs the installed MapAnything.infer actually accepts, so a
    # version bump that renames/drops an option degrades gracefully.
    infer_kwargs = dict(
        memory_efficient_inference=scfg["memory_efficient_inference"],
        minibatch_size=scfg["minibatch_size"],
        use_amp=scfg["use_amp"],
        amp_dtype=scfg["amp_dtype"],
        apply_mask=scfg["apply_mask"],
        mask_edges=scfg["mask_edges"],
        apply_confidence_mask=scfg["apply_confidence_mask"],
        confidence_percentile=scfg["confidence_percentile"],
        use_multiview_confidence=scfg["use_multiview_confidence"],
    )
    try:
        import inspect
        accepted = set(inspect.signature(model.infer).parameters)
        infer_kwargs = {k: v for k, v in infer_kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        pass

    with torch.no_grad():
        preds = model.infer(views, **infer_kwargs)

    # preds is a per-view list (or a dict of stacked tensors); normalize to list.
    if isinstance(preds, dict):
        n = len(frames)
        preds = [{k: (v[i] if hasattr(v, "__getitem__") else v) for k, v in preds.items()}
                 for i in range(n)]

    all_xyz, all_rgb, all_conf = [], [], []
    out_frames = []
    conf_gate = float(scfg["confidence_percentile"])

    for i, (view_pred, fr, rgb) in enumerate(zip(preds, frames, rgbs)):
        pts3d = _squeeze_batch(_view_field(view_pred, "pts3d"))            # (H,W,3)
        c2w = _squeeze_batch(_view_field(view_pred, "camera_poses"))       # (4,4)
        K = _squeeze_batch(_view_field(view_pred, "intrinsics"))          # (3,3)
        conf = _view_field(view_pred, "conf")
        mask = _view_field(view_pred, "mask", "non_ambiguous_mask")
        color = _view_field(view_pred, "img_no_norm")

        H, W = pts3d.shape[0], pts3d.shape[1]
        pts = pts3d.reshape(-1, 3)

        if color is not None:
            col = _squeeze_batch(color).reshape(-1, 3)
            if col.max() > 1.5:
                col = col / 255.0
        else:
            import cv2
            col = cv2.resize(rgb, (W, H)).reshape(-1, 3).astype(np.float32) / 255.0

        valid = np.isfinite(pts).all(axis=1)
        if mask is not None:
            m = _squeeze_batch(mask).reshape(-1) > 0.5
            valid &= m
        if conf is not None:
            c = _squeeze_batch(conf).reshape(-1)
            if conf_gate > 0:
                thr = np.percentile(c[valid] if valid.any() else c, conf_gate)
                valid &= c >= thr
            all_conf.append(c[valid])
        all_xyz.append(pts[valid])
        all_rgb.append(col[valid])

        out_frames.append({
            "file_path": fr["file_path"],
            "w": int(W),
            "h": int(H),
            "intrinsics": K.tolist(),
            "c2w": c2w.tolist(),        # OpenCV cam2world
            "is_metric": True,
        })

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    conf = np.concatenate(all_conf, axis=0) if all_conf else None

    # Subsample the fused cloud to the point budget.
    max_pts = int(scfg["max_points"] or 0)
    if max_pts and xyz.shape[0] > max_pts:
        sel = np.random.default_rng(0).choice(xyz.shape[0], max_pts, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
        conf = conf[sel] if conf is not None else None

    save_cameras_json(os.path.join(out_dir, "cameras.json"), out_frames)
    save_points_npz(os.path.join(out_dir, "points.npz"), xyz, rgb, conf)
    save_point_cloud_ply(os.path.join(out_dir, "points.ply"), xyz, rgb)
    print(f"[sfm] fused cloud: {xyz.shape[0]:,} points -> {out_dir}/points.ply")
    return out_dir
