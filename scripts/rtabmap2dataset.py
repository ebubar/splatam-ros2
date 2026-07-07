#!/usr/bin/env python3
"""
Convert a ZED2i ROS2 bag + RTAB-Map optimized poses into a NeRFCapture-format
dataset that scripts/splatam.py can load with dataset_name="nerfcapture".

Requires a sourced ROS2 environment (rosbag2_py, sensor_msgs).

Pose file: TUM format, one line per optimized graph node:
    timestamp x y z qx qy qz qw
Export from RTAB-Map with rtabmap-databaseViewer (File -> Export poses -> TUM)
or the /rtabmap/get_map graph.

For each optimized pose, the nearest RGB and depth frames in the bag (within
--slop seconds) are extracted. Output:
    <output>/rgb/<i>.png
    <output>/depth/<i>.png        (uint16, meters * 6553.5, max ~10 m)
    <output>/transforms.json      (NeRFCapture/NeRFStudio-style, OpenGL c2w)

Usage:
    python3 scripts/rtabmap2dataset.py \
        --bag <bag_dir> --poses <poses_tum.txt> \
        --output experiments/ZED2i_Captures/offline_rtabmap
"""

import argparse
import bisect
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Depth PNG scale matching the NeRFCapture dataloader
# (datasets/gradslam_datasets/nerfcapture.py hardcodes png_depth_scale=6553.5)
DEPTH_PNG_SCALE = 6553.5
DEPTH_PNG_MAX_M = 65535.0 / DEPTH_PNG_SCALE  # ~10 m

# CV (x right, y down, z forward) <-> OpenGL/NeRF (x right, y up, z back)
P_GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bag", required=True, help="Path to rosbag2 directory")
    poses = p.add_mutually_exclusive_group(required=True)
    poses.add_argument("--poses", help="RTAB-Map optimized poses, TUM format")
    poses.add_argument("--poses-from-odom", action="store_true",
                       help="Take poses from the bag's odometry topic instead "
                            "of a pose file (no RTAB-Map needed; poses will "
                            "have VIO drift). Combine with --pose-stride to "
                            "thin high-rate odometry (e.g. 30 Hz odom, "
                            "--pose-stride 6 -> 5 frames/sec).")
    p.add_argument("--odom-topic", default="/zed/zed_node/odom")
    p.add_argument("--output", required=True, help="Output dataset directory")
    p.add_argument("--rgb-topic", default="/zed/zed_node/rgb/color/rect/image")
    p.add_argument("--depth-topic", default="/zed/zed_node/depth/depth_registered")
    p.add_argument("--info-topic", default="/zed/zed_node/rgb/color/rect/image/camera_info")
    p.add_argument("--slop", type=float, default=0.05,
                   help="Max |frame - pose| timestamp difference in seconds")
    p.add_argument("--min-depth", type=float, default=0.3)
    p.add_argument("--max-depth", type=float, default=6.0)
    p.add_argument("--edge-filter", type=float, default=0.04,
                   help="Discard depth pixels at discontinuities: invalidate "
                        "where the local (5x5) depth spread exceeds this "
                        "fraction of the depth. Kills 'flying pixel' floaters. "
                        "0 disables.")
    p.add_argument("--pose-stride", type=int, default=1,
                   help="Keep every Nth pose. Use for dense pose files (e.g. "
                        "100 Hz mocap ground truth) so poses stay sparser than "
                        "the camera frame rate.")
    p.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                   default=None,
                   help="Override camera_info intrinsics (e.g. bags with "
                        "factory-default calibration)")
    p.add_argument("--pose-frame", choices=["zed_link", "optical"], default="zed_link",
                   help="Frame the TUM poses refer to. 'zed_link' (RTAB-Map "
                        "frame_id=zed_camera_link, default) applies the "
                        "link->left-optical extrinsic; 'optical' uses poses as-is.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def quat_to_rotmat(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def zed_link_to_left_optical():
    # Same extrinsic as scripts/zed2i_splat_live.py (from the ZED wrapper /tf_static)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    T[:3, 3] = [-0.01, 0.06, 0.015]
    return T


def load_tum_poses(path, pose_frame):
    T_link_opt = zed_link_to_left_optical() if pose_frame == "zed_link" else np.eye(4)
    times, mats = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = [float(x) for x in line.split()]
            if len(v) != 8:
                continue
            t, x, y, z, qx, qy, qz, qw = v
            c2w = np.eye(4)
            c2w[:3, :3] = quat_to_rotmat(qx, qy, qz, qw)
            c2w[:3, 3] = [x, y, z]
            times.append(t)
            mats.append(c2w @ T_link_opt)
    order = np.argsort(times)
    return [times[i] for i in order], [mats[i] for i in order]


def stride_poses(times, mats, stride):
    if stride <= 1:
        return times, mats
    return times[::stride], mats[::stride]


def load_odom_poses(bag_path, odom_topic, pose_frame):
    """First pass over the bag: collect camera poses from the odometry topic."""
    T_link_opt = zed_link_to_left_optical() if pose_frame == "zed_link" else np.eye(4)
    reader = open_bag(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if odom_topic not in type_map:
        sys.exit(f"Odom topic {odom_topic} not in bag. Available:\n  "
                 + "\n  ".join(sorted(type_map)))
    msg_type = get_message(type_map[odom_topic])

    times, mats = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != odom_topic:
            continue
        msg = deserialize_message(data, msg_type)
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        c2w = np.eye(4)
        c2w[:3, :3] = quat_to_rotmat(q.x, q.y, q.z, q.w)
        c2w[:3, 3] = [p.x, p.y, p.z]
        times.append(stamp_to_sec(msg.header))
        mats.append(c2w @ T_link_opt)
    order = np.argsort(times)
    return [times[i] for i in order], [mats[i] for i in order]


def nearest_pose_idx(pose_times, t):
    i = bisect.bisect_left(pose_times, t)
    best, best_dt = None, None
    for j in (i - 1, i):
        if 0 <= j < len(pose_times):
            dt = abs(pose_times[j] - t)
            if best_dt is None or dt < best_dt:
                best, best_dt = j, dt
    return best, best_dt


def image_msg_to_np(msg):
    enc = msg.encoding.lower()
    dtype, ch = {
        "bgra8": (np.uint8, 4), "rgba8": (np.uint8, 4),
        "bgr8": (np.uint8, 3), "rgb8": (np.uint8, 3),
        "mono8": (np.uint8, 1), "mono16": (np.uint16, 1),
        "16uc1": (np.uint16, 1), "32fc1": (np.float32, 1),
    }.get(enc, (None, None))
    if dtype is None:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    itemsize = np.dtype(dtype).itemsize
    row_elems = msg.step // itemsize
    img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_elems)
    img = img[:, : msg.width * ch]
    if ch > 1:
        img = img.reshape(msg.height, msg.width, ch)
    return img, enc


def to_bgr(img, enc):
    if enc == "bgra8":
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "mono8":
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img  # bgr8


def depth_to_meters(img, enc):
    if enc in ("16uc1", "mono16"):
        return img.astype(np.float32) / 1000.0
    return img.astype(np.float32)


def filter_depth_edges(depth_m, rel_thresh):
    """Zero out depth at discontinuities, where mixed foreground/background
    values back-project into free space as colorful floaters."""
    if rel_thresh <= 0:
        return depth_m
    valid = depth_m > 0
    d = depth_m.copy()
    d[~valid] = np.inf
    kernel = np.ones((5, 5), np.uint8)
    local_min = cv2.erode(d, kernel)
    d[~valid] = -np.inf
    local_max = cv2.dilate(d, kernel)
    spread = local_max - local_min
    edge = valid & (spread > rel_thresh * depth_m)
    out = depth_m.copy()
    out[edge] = 0.0
    return out


def stamp_to_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def open_bag(bag_path):
    storage_id = "sqlite3"
    meta = Path(bag_path) / "metadata.yaml"
    if meta.exists() and "mcap" in meta.read_text():
        storage_id = "mcap"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return reader


def main():
    args = parse_args()

    if args.poses_from_odom:
        pose_times, pose_mats = load_odom_poses(
            args.bag, args.odom_topic, args.pose_frame
        )
    else:
        pose_times, pose_mats = load_tum_poses(args.poses, args.pose_frame)
    pose_times, pose_mats = stride_poses(pose_times, pose_mats, args.pose_stride)
    if not pose_times:
        sys.exit("No poses found")
    print(f"Loaded {len(pose_times)} optimized poses "
          f"({pose_times[0]:.3f} .. {pose_times[-1]:.3f})")

    out = Path(args.output)
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    (out / "rgb").mkdir(parents=True)
    (out / "depth").mkdir(parents=True)

    reader = open_bag(args.bag)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for topic in (args.rgb_topic, args.depth_topic, args.info_topic):
        if topic not in type_map:
            sys.exit(f"Topic {topic} not in bag. Available:\n  "
                     + "\n  ".join(sorted(type_map)))
    msg_types = {t: get_message(type_map[t]) for t in
                 (args.rgb_topic, args.depth_topic, args.info_topic)}

    cam_info = None
    best_dt = {"rgb": {}, "depth": {}}  # pose_idx -> |dt| of best match written
    n_msgs = 0

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == args.info_topic:
            if cam_info is None:
                cam_info = deserialize_message(data, msg_types[topic])
            continue
        if topic not in (args.rgb_topic, args.depth_topic):
            continue
        kind = "rgb" if topic == args.rgb_topic else "depth"
        msg = deserialize_message(data, msg_types[topic])
        n_msgs += 1
        t = stamp_to_sec(msg.header)
        idx, dt = nearest_pose_idx(pose_times, t)
        if idx is None or dt > args.slop:
            continue
        prev = best_dt[kind].get(idx)
        if prev is not None and prev <= dt:
            continue
        img, enc = image_msg_to_np(msg)
        if kind == "rgb":
            cv2.imwrite(str(out / "rgb" / f"{idx}.png"), to_bgr(img, enc))
        else:
            depth_m = depth_to_meters(img, enc)
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            depth_m[(depth_m < args.min_depth) | (depth_m > args.max_depth)] = 0.0
            depth_m = filter_depth_edges(depth_m, args.edge_filter)
            depth_png = (np.clip(depth_m, 0.0, DEPTH_PNG_MAX_M)
                         * DEPTH_PNG_SCALE).astype(np.uint16)
            cv2.imwrite(str(out / "depth" / f"{idx}.png"), depth_png)
        best_dt[kind][idx] = dt

    if cam_info is None:
        sys.exit(f"No CameraInfo received on {args.info_topic}")

    # Keep only poses with both RGB and depth matched
    matched = sorted(set(best_dt["rgb"]) & set(best_dt["depth"]))
    for kind in ("rgb", "depth"):
        for idx in set(best_dt[kind]) - set(matched):
            (out / kind / f"{idx}.png").unlink(missing_ok=True)

    frames = []
    for idx in matched:
        c2w_gl = P_GL @ pose_mats[idx] @ P_GL
        frames.append({
            "file_path": f"rgb/{idx}.png",
            "depth_path": f"depth/{idx}.png",
            "transform_matrix": c2w_gl.tolist(),
            "timestamp": pose_times[idx],
        })

    K = np.array(cam_info.k).reshape(3, 3)
    fx, fy, cx, cy = (args.intrinsics if args.intrinsics
                      else (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    manifest = {
        "w": int(cam_info.width),
        "h": int(cam_info.height),
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "integer_depth_scale": 10.0 / 65535.0,
        "frames": frames,
    }
    with open(out / "transforms.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Scanned {n_msgs} image messages.")
    print(f"Wrote {len(frames)}/{len(pose_times)} frames to {out}")
    if len(frames) < len(pose_times) // 2:
        print("WARNING: fewer than half the poses matched a frame. "
              "Check topics, --slop, and that pose timestamps are ROS time.")
    print("Run SplaTAM with: python3 scripts/splatam.py "
          "configs/zed2i/zed2i_offline_rtabmap.py")


if __name__ == "__main__":
    main()
