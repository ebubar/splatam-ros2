"""Ingest from an existing folder of images (no ROS needed).

If the source folder already contains a ``cameras.json`` (e.g. from a previous
ROS2 capture) it is carried through so intrinsics / depth / poses are preserved.
Otherwise a minimal ``cameras.json`` with just image paths + sizes is written and
MapAnything estimates intrinsics and poses.
"""

import glob
import os
import shutil

from ..utils.io import load_cameras_json, save_cameras_json


def _image_size(path):
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return int(img.shape[1]), int(img.shape[0])
    except Exception:
        pass
    return None, None


def prepare_from_folder(cfg, out_dir):
    ing = cfg["ingest"]
    src_dir = ing["image_dir"]
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"ingest.image_dir not found: {src_dir}")

    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Carry through an existing cameras.json if present.
    src_cams = os.path.join(src_dir, "cameras.json")
    if os.path.isfile(src_cams):
        cams = load_cameras_json(src_cams)
        frames = cams.get("frames", [])
        for fr in frames:
            src_img = os.path.join(src_dir, fr["file_path"])
            dst_name = os.path.basename(fr["file_path"])
            shutil.copy2(src_img, os.path.join(images_dir, dst_name))
            fr["file_path"] = os.path.join("images", dst_name)
            if fr.get("depth_path"):
                src_d = os.path.join(src_dir, fr["depth_path"])
                if os.path.isfile(src_d):
                    dst_d = os.path.join(out_dir, fr["depth_path"])
                    os.makedirs(os.path.dirname(dst_d), exist_ok=True)
                    shutil.copy2(src_d, dst_d)
        save_cameras_json(os.path.join(out_dir, "cameras.json"), frames, cams.get("meta"))
        return out_dir

    # Otherwise glob images and write a bare cameras.json.
    paths = sorted(glob.glob(os.path.join(src_dir, ing["image_glob"])))
    if not paths:
        raise FileNotFoundError(
            f"No images matching {ing['image_glob']!r} in {src_dir}"
        )
    max_frames = int(ing.get("max_frames") or 0)
    if max_frames and len(paths) > max_frames:
        # Uniformly subsample to the frame budget for sparse coverage.
        idx = [round(i * (len(paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
        paths = [paths[i] for i in sorted(set(idx))]

    frames = []
    for i, p in enumerate(paths):
        dst_name = f"frame_{i:05d}{os.path.splitext(p)[1].lower()}"
        shutil.copy2(p, os.path.join(images_dir, dst_name))
        w, h = _image_size(p)
        frames.append({"file_path": os.path.join("images", dst_name), "w": w, "h": h})

    save_cameras_json(os.path.join(out_dir, "cameras.json"), frames)
    print(f"[ingest] prepared {len(frames)} frames from {src_dir}")
    return out_dir
