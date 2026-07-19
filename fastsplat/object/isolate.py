"""Object isolation: keep the target item, drop the surrounding scene.

Two config-switchable methods (``object.method``):

* ``auto_crop`` (default) -- no extra model. Strip far background by
  distance-from-camera band, remove statistical outliers, DBSCAN-cluster and
  keep the dominant near-camera cluster, then crop a padded box. Optionally
  reproject the kept points into each view to build silhouette masks for
  masked photometric supervision.

* ``sam2`` -- segment the object per image with SAM2 (a center-point prompt),
  keep only points that reproject inside the masks, and emit per-view alpha
  masks so gsplat only supervises object pixels.

Outputs (``out_dir``): ``points.npz`` / ``points.ply`` (cropped cloud),
``cameras.json`` (carrying an optional ``mask_path`` per frame), ``crop.json``
(the world-space AABB), and ``masks/`` if masks were produced.
"""

import json
import os

import numpy as np

from ..utils.io import (
    frame_c2w,
    frame_intrinsics,
    load_cameras_json,
    load_points_npz,
    save_cameras_json,
    save_point_cloud_ply,
    save_points_npz,
)


def _camera_centers(frames):
    centers = []
    for fr in frames:
        c2w = frame_c2w(fr)
        if c2w is not None:
            centers.append(c2w[:3, 3])
    if not centers:
        return None
    return np.stack(centers, axis=0)


def _project(points, c2w, K, W, H):
    """Project world points into a view. Returns pixel (u,v) and in-front mask."""
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ points.T + w2c[:3, 3:4]).T   # (N,3) OpenCV: z forward
    z = cam[:, 2]
    front = z > 1e-6
    uv = np.full((points.shape[0], 2), -1.0)
    zc = np.where(front, z, 1.0)
    uv[:, 0] = K[0, 0] * cam[:, 0] / zc + K[0, 2]
    uv[:, 1] = K[1, 1] * cam[:, 1] / zc + K[1, 2]
    inb = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, inb


def _silhouette_mask(points, c2w, K, W, H, dilate_px):
    import cv2
    uv, inb = _project(points, c2w, K, W, H)
    mask = np.zeros((H, W), np.uint8)
    pts = uv[inb].astype(np.int32)
    if pts.shape[0] >= 3:
        mask[pts[:, 1], pts[:, 0]] = 255
        k = max(1, int(dilate_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _auto_crop(cfg, xyz, rgb, frames):
    import open3d as o3d
    ac = cfg["object"]["auto_crop"]

    # 1) Strip far background using distance from the camera cloud centroid.
    centers = _camera_centers(frames)
    ref = centers.mean(axis=0) if centers is not None else xyz.mean(axis=0)
    d = np.linalg.norm(xyz - ref, axis=1)
    lo, hi = ac["depth_percentile"]
    keep = (d >= np.percentile(d, lo)) & (d <= np.percentile(d, hi))
    xyz, rgb = xyz[keep], rgb[keep]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1).astype(np.float64))

    # 2) Statistical outlier removal (kills sparse floaters / ghost fragments).
    if ac["outlier_nb"] > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(ac["outlier_nb"]), std_ratio=float(ac["outlier_std"])
        )

    # 3) DBSCAN; keep the dominant cluster (the object).
    if ac["keep_largest_cluster"]:
        labels = np.asarray(
            pcd.cluster_dbscan(eps=float(ac["dbscan_eps"]),
                               min_points=int(ac["dbscan_min_points"]))
        )
        if labels.size and labels.max() >= 0:
            counts = np.bincount(labels[labels >= 0])
            best = int(np.argmax(counts))
            sel = labels == best
            pcd = pcd.select_by_index(np.where(sel)[0])

    xyz = np.asarray(pcd.points, np.float32)
    rgb = np.asarray(pcd.colors, np.float32)

    # 4) Padded axis-aligned box.
    if xyz.shape[0] == 0:
        raise RuntimeError("auto_crop removed all points; loosen object.auto_crop")
    mn, mx = xyz.min(0), xyz.max(0)
    pad = ac["box_padding"] * (mx - mn)
    aabb = (mn - pad, mx + pad)
    return xyz, rgb, aabb


def _sam2_masks(cfg, capture_dir, frames, out_dir):
    import cv2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    s = cfg["object"]["sam2"]
    predictor = SAM2ImagePredictor(build_sam2(s["model_cfg"], s["checkpoint"]))
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    for fr in frames:
        img = cv2.cvtColor(
            cv2.imread(os.path.join(capture_dir, fr["file_path"])), cv2.COLOR_BGR2RGB
        )
        H, W = img.shape[:2]
        predictor.set_image(img)
        pt = np.array([[W / 2.0, H / 2.0]])   # center-click prompt
        masks, scores, _ = predictor.predict(
            point_coords=pt, point_labels=np.array([1]), multimask_output=True
        )
        mask = masks[int(np.argmax(scores))].astype(np.uint8) * 255
        if int(s["dilate_px"]) > 0:
            k = int(s["dilate_px"])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            mask = cv2.dilate(mask, kernel)
        # Store at the SfM frame resolution so it matches the intrinsics.
        fw, fh = int(fr.get("w", W)), int(fr.get("h", H))
        if mask.shape[1] != fw or mask.shape[0] != fh:
            mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        name = os.path.basename(fr["file_path"])
        mpath = os.path.join("masks", os.path.splitext(name)[0] + ".png")
        cv2.imwrite(os.path.join(out_dir, mpath), mask)
        fr["mask_path"] = mpath
    return frames


def _filter_by_masks(xyz, rgb, frames, out_dir):
    """Keep points that land inside the object mask in most views they appear in."""
    import cv2
    inside = np.zeros(xyz.shape[0], np.int32)
    seen = np.zeros(xyz.shape[0], np.int32)
    for fr in frames:
        c2w, K = frame_c2w(fr), frame_intrinsics(fr)
        if c2w is None or K is None or not fr.get("mask_path"):
            continue
        mask = cv2.imread(os.path.join(out_dir, fr["mask_path"]), cv2.IMREAD_GRAYSCALE)
        W, H = int(fr["w"]), int(fr["h"])
        if mask.shape[1] != W or mask.shape[0] != H:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        uv, inb = _project(xyz, c2w, K, W, H)
        idx = np.where(inb)[0]
        px = uv[idx].astype(np.int32)
        seen[idx] += 1
        hit = mask[px[:, 1], px[:, 0]] > 127
        inside[idx[hit]] += 1
    ok = (seen == 0) | (inside >= np.maximum(1, seen // 2))
    return xyz[ok], rgb[ok]


def run_isolate(cfg, sfm_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    method = cfg["object"]["method"]
    cams = load_cameras_json(os.path.join(sfm_dir, "cameras.json"))
    frames = cams["frames"]
    xyz, rgb, _ = load_points_npz(os.path.join(sfm_dir, "points.npz"))
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    aabb = None
    if method == "none":
        pass
    elif method == "auto_crop":
        xyz, rgb, aabb = _auto_crop(cfg, xyz, rgb, frames)
        if cfg["object"]["mask_supervision"]:
            masks_dir = os.path.join(out_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            dil = cfg["object"]["auto_crop"].get("dilate_px", 7)
            for fr in frames:
                c2w, K = frame_c2w(fr), frame_intrinsics(fr)
                if c2w is None or K is None:
                    continue
                m = _silhouette_mask(xyz, c2w, K, int(fr["w"]), int(fr["h"]), dil if dil else 7)
                name = os.path.basename(fr["file_path"])
                mpath = os.path.join("masks", os.path.splitext(name)[0] + ".png")
                import cv2
                cv2.imwrite(os.path.join(out_dir, mpath), m)
                fr["mask_path"] = mpath
    elif method == "sam2":
        frames = _sam2_masks(cfg, sfm_dir, frames, out_dir)
        xyz, rgb = _filter_by_masks(xyz, rgb, frames, out_dir)
        if xyz.shape[0]:
            mn, mx = xyz.min(0), xyz.max(0)
            pad = 0.05 * (mx - mn)
            aabb = (mn - pad, mx + pad)
    else:
        raise ValueError(f"Unknown object.method: {method!r}")

    save_points_npz(os.path.join(out_dir, "points.npz"), xyz, rgb)
    save_point_cloud_ply(os.path.join(out_dir, "points.ply"), xyz, rgb)
    save_cameras_json(os.path.join(out_dir, "cameras.json"), frames)
    if aabb is not None:
        with open(os.path.join(out_dir, "crop.json"), "w") as f:
            json.dump({"aabb_min": aabb[0].tolist(), "aabb_max": aabb[1].tolist()}, f)
    print(f"[object] method={method}: kept {xyz.shape[0]:,} points -> {out_dir}")
    return out_dir
