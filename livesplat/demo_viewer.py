#!/usr/bin/env python3
"""Offline operator-viewer demo -- exercise the full operator UX with no camera.

Runs the browser viewer against either a saved SplaTAM ``params.npz`` or a
synthetic scene, plus a simulated set of captured camera poses. You can orbit
the cloud, draw a region, hit "Refine", and watch live coverage % and
next-best-view hints update -- exactly what the operator sees in the field,
without a ZED, ROS2, or a GPU.

Usage:
    python3 -m livesplat.demo_viewer                       # synthetic scene
    python3 -m livesplat.demo_viewer --params path/to/params.npz
    python3 -m livesplat.demo_viewer --port 8080

Then open http://localhost:<port>/ in a browser.

This module is dependency-light (numpy + stdlib) so it runs anywhere.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from livesplat.bridge import LiveSplatBridge
from livesplat.viewer_server import OperatorViewerServer
from livesplat.roi import RegionOfInterest
from livesplat import capture_advisor
from utils import pose_utils as pu


def synthetic_scene(n=120_000):
    """A few coloured blobs on a ground plane -- something recognisable to box."""
    rng = np.random.default_rng(0)
    parts = []
    cols = []
    # ground plane
    g = rng.uniform([-3, -0.02, -3], [3, 0.02, 3], size=(n // 2, 3))
    parts.append(g)
    cols.append(np.tile([0.35, 0.35, 0.4], (g.shape[0], 1)))
    # three object blobs
    for c, color in [
        ([0.0, 0.4, 1.0], [0.9, 0.2, 0.2]),
        ([1.2, 0.3, 0.5], [0.2, 0.8, 0.3]),
        ([-1.0, 0.5, 1.5], [0.25, 0.5, 0.95]),
    ]:
        b = rng.normal(c, 0.18, size=(n // 6, 3))
        parts.append(b)
        cols.append(np.tile(color, (b.shape[0], 1)) + rng.normal(0, 0.03, (b.shape[0], 3)))
    xyz = np.concatenate(parts).astype(np.float32)
    rgb = np.clip(np.concatenate(cols), 0, 1).astype(np.float32)
    return xyz, rgb


def load_params(path):
    data = np.load(path, allow_pickle=True)
    xyz = np.asarray(data["means3D"], dtype=np.float32)
    rgb = np.clip(np.asarray(data["rgb_colors"], dtype=np.float32), 0, 1)
    return xyz, rgb


def simulated_capture_poses(center, radius=1.2, az_range=(0, 200), step=25, elevation=20.0):
    """A partial orbit around ``center`` -> deliberately leaves coverage gaps so
    the advisor has something to recommend."""
    w2c = []
    for az in range(int(az_range[0]), int(az_range[1]), int(step)):
        a = np.radians(az)
        e = np.radians(elevation)
        pos = center + radius * np.array([np.cos(e) * np.cos(a),
                                          np.sin(e),
                                          np.cos(e) * np.sin(a)])
        # look-at world->cam
        f = center - pos
        f /= np.linalg.norm(f)
        up = np.array([0.0, 1.0, 0.0])
        r = np.cross(up, f)
        r /= np.linalg.norm(r) + 1e-9
        u = np.cross(f, r)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([r, u, f], axis=1)
        c2w[:3, 3] = pos
        w2c.append(pu.invert_transform(c2w))
    return np.stack(w2c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default=None, help="path to a SplaTAM params.npz")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    if args.params:
        xyz, rgb = load_params(args.params)
        print(f"Loaded {xyz.shape[0]:,} Gaussians from {args.params}")
    else:
        xyz, rgb = synthetic_scene()
        print(f"Generated synthetic scene with {xyz.shape[0]:,} points")

    bridge = LiveSplatBridge(max_preview_points=150_000)
    server = OperatorViewerServer(bridge, host=args.host, port=args.port).start()
    bridge.publish_snapshot(xyz, rgb)
    print(f"\nOperator viewer running at:  http://localhost:{args.port}/")
    print("Box a region and hit Refine; coverage + hints update live. Ctrl-C to quit.\n")

    poses = None
    frame = 0
    try:
        while True:
            frame += 1
            # Service ROI requests: recompute coverage/hints for the boxed region.
            req = bridge.pop_refine_request()
            if req is not None:
                poses = simulated_capture_poses(req.roi.center())
                guidance = capture_advisor.analyse_coverage(req.roi, poses)
                bridge.publish_guidance(guidance)
                print(f"[demo] Refine #{req.request_id} on {req.roi.label}: "
                      f"coverage={guidance['coverage']*100:.0f}%  "
                      f"hints={len(guidance['hints'])}")
            bridge.publish_status(
                frame=frame, num_frames="∞",
                num_gaussians=int(xyz.shape[0]),
                preview_points=int(min(xyz.shape[0], bridge.max_preview_points)),
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down demo viewer.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
