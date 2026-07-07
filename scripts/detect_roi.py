#!/usr/bin/env python3
"""
Find a 3D region-of-interest box for an object in a dataset, for high-res
re-splatting with scripts/roi_refine.py.

Two modes:

1. Text prompt (open-vocabulary detection, ConceptGraph-style):
       python3 scripts/detect_roi.py --dataset <dir> --prompt "keyboard"
   Requires the optional detector extra:  pip install transformers
   (first run downloads the OWL-ViT model, ~600 MB).

2. Manual rectangle on a frame (no extra dependencies):
       python3 scripts/detect_roi.py --dataset <dir> --frame 60 --rect 300 180 200 120
   (--rect is x y width height in pixels on that frame's RGB image.)

Both modes back-project the selected pixels through the depth image into the
splat's world frame (first-camera coordinates) and write
<dataset>/roi_<label>.json plus print the equivalent roi_refine.py command.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

P_GL = np.diag([1.0, -1.0, -1.0, 1.0])
DEPTH_PNG_SCALE = 6553.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt", help="Open-vocabulary object description")
    mode.add_argument("--rect", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                      help="Manual 2D rectangle (requires --frame)")
    p.add_argument("--frame", type=int, default=None,
                   help="Frame index for --rect mode")
    p.add_argument("--every", type=int, default=5,
                   help="Prompt mode: run the detector on every Nth frame")
    p.add_argument("--min-score", type=float, default=0.15,
                   help="Prompt mode: minimum detection confidence")
    p.add_argument("--top-frames", type=int, default=5,
                   help="Prompt mode: fuse detections from this many best frames")
    p.add_argument("--label", default=None,
                   help="Name for the output roi json (default: from prompt/frame)")
    return p.parse_args()


def load_manifest(dataset):
    with open(Path(dataset) / "transforms.json") as f:
        return json.load(f)


def frame_c2w_rel(manifest, idx, c2w0_inv=None):
    c2w = P_GL @ np.array(manifest["frames"][idx]["transform_matrix"]) @ P_GL
    if c2w0_inv is None:
        c2w0 = P_GL @ np.array(manifest["frames"][0]["transform_matrix"]) @ P_GL
        c2w0_inv = np.linalg.inv(c2w0)
    return c2w0_inv @ c2w


def rect_to_world_points(dataset, manifest, frame_idx, rect):
    """Back-project the depth pixels inside a 2D rect into world coordinates
    (relative to frame 0, i.e. the splat's frame)."""
    fr = manifest["frames"][frame_idx]
    depth_png = cv2.imread(str(Path(dataset) / fr["depth_path"]), cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        sys.exit(f"Cannot read depth for frame {frame_idx}")
    depth = depth_png.astype(np.float32) / DEPTH_PNG_SCALE

    x, y, w, h = rect
    H, W = depth.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    patch = depth[y0:y1, x0:x1]
    vs, us = np.mgrid[y0:y1, x0:x1]
    z = patch.reshape(-1)
    us, vs = us.reshape(-1), vs.reshape(-1)
    valid = z > 0
    if valid.sum() < 20:
        return None
    z, us, vs = z[valid], us[valid], vs[valid]

    # Reject background pixels inside the rect: keep the dominant depth mode
    z_med = np.median(z)
    keep = np.abs(z - z_med) < max(0.3, 0.25 * z_med)
    z, us, vs = z[keep], us[keep], vs[keep]

    fx, fy = manifest["fl_x"], manifest["fl_y"]
    cx, cy = manifest["cx"], manifest["cy"]
    pts_cam = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=1)

    c2w = frame_c2w_rel(manifest, frame_idx)
    return (c2w[:3, :3] @ pts_cam.T + c2w[:3, 3:4]).T


def points_to_box(pts):
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    return lo, hi


def detect_prompt(dataset, manifest, prompt, every, min_score, top_frames):
    try:
        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
    except ImportError:
        sys.exit("Prompt mode needs the optional detector: pip install transformers\n"
                 "Or use manual mode: --frame <idx> --rect x y w h")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    model.eval()

    hits = []  # (score, frame_idx, [x, y, w, h])
    frames = manifest["frames"]
    for i in range(0, len(frames), every):
        bgr = cv2.imread(str(Path(dataset) / frames[i]["file_path"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = processor(text=[[prompt]], images=rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_size = torch.tensor([rgb.shape[:2]], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=min_score, target_sizes=target_size
        )[0]
        if len(results["scores"]) == 0:
            continue
        j = int(results["scores"].argmax())
        score = float(results["scores"][j])
        x0, y0, x1, y1 = [float(v) for v in results["boxes"][j]]
        hits.append((score, i, [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]))
        print(f"  frame {i}: '{prompt}' score={score:.2f}")

    if not hits:
        sys.exit(f"No detections for '{prompt}' (min score {min_score}). "
                 "Try lowering --min-score or a different wording.")

    hits.sort(reverse=True)
    all_pts = []
    for score, idx, rect in hits[:top_frames]:
        pts = rect_to_world_points(dataset, manifest, idx, rect)
        if pts is not None:
            all_pts.append(pts)
    if not all_pts:
        sys.exit("Detections had no valid depth to back-project.")
    return np.vstack(all_pts), hits[0][1]


def main():
    args = parse_args()
    manifest = load_manifest(args.dataset)

    if args.rect:
        if args.frame is None:
            sys.exit("--rect requires --frame <idx>")
        pts = rect_to_world_points(args.dataset, manifest, args.frame, args.rect)
        if pts is None:
            sys.exit("Not enough valid depth inside the rectangle.")
        label = args.label or f"frame{args.frame}"
        src = {"mode": "rect", "frame": args.frame, "rect": args.rect}
    else:
        print(f"Scanning for '{args.prompt}' (every {args.every}th frame)...")
        pts, best_frame = detect_prompt(
            args.dataset, manifest, args.prompt,
            args.every, args.min_score, args.top_frames,
        )
        label = args.label or args.prompt.replace(" ", "_")
        src = {"mode": "prompt", "prompt": args.prompt, "best_frame": best_frame}

    lo, hi = points_to_box(pts)
    box = [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]),
           float(lo[2]), float(hi[2])]

    out_path = Path(args.dataset) / f"roi_{label}.json"
    with open(out_path, "w") as f:
        json.dump({"box": box, "label": label, "source": src}, f, indent=2)

    size = np.round(hi - lo, 2)
    print(f"\nROI '{label}': size {size} m, box "
          f"x[{box[0]:.2f},{box[1]:.2f}] y[{box[2]:.2f},{box[3]:.2f}] z[{box[4]:.2f},{box[5]:.2f}]")
    print(f"Wrote {out_path}")
    print("\nNext:")
    print(f"  python3 scripts/roi_refine.py --roi-json {out_path} "
          f"--dataset {args.dataset} --out $(basename {args.dataset})_{label}")


if __name__ == "__main__":
    main()
