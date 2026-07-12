#!/usr/bin/env python3
"""
Replay a NeRFCapture-format dataset (rgb/, depth/, transforms.json — e.g.
produced by scripts/rtabmap2dataset.py) as ZED-style ROS2 topics, so the live
SplaTAM node can be exercised without a camera:

    Terminal A: python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_replay_test.py
    Terminal B: python3 scripts/dataset_player.py --dataset experiments/ZED2i_Captures/<scene> --rate 5

Publishes RGB (bgr8), depth (32FC1, meters), CameraInfo, and Odometry with the
dataset's poses in the *optical* frame (use ros.odom_frame="optical" in the
node config).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry

DEPTH_PNG_SCALE = 6553.5
# CV <-> OpenGL camera convention flip (matches the nerfcapture dataloader)
P_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Dataset dir with transforms.json")
    p.add_argument("--rate", type=float, default=5.0, help="Frames per second")
    p.add_argument("--odom-rate", type=float, default=0.0,
                   help="Publish odometry on its own timer at this rate "
                        "(interpolated along the trajectory), with timestamps "
                        "independent of the image stamps — like a real ZED. "
                        "0 = publish odom together with each frame.")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--rgb-topic", default="/zed/zed_node/rgb/color/rect/image")
    p.add_argument("--rgb-info-topic", default="/zed/zed_node/rgb/color/rect/image/camera_info")
    p.add_argument("--depth-topic", default="/zed/zed_node/depth/depth_registered")
    p.add_argument("--depth-info-topic", default="/zed/zed_node/depth/depth_registered/camera_info")
    p.add_argument("--odom-topic", default="/zed/zed_node/odom")
    return p.parse_args()


def rotmat_to_quat_xyzw(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * S, (R[2, 1] - R[1, 2]) / S
        y, z = (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x = (R[2, 1] - R[1, 2]) / S, 0.25 * S
        y, z = (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S
        y, z = 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S
        y, z = (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def np_to_image_msg(arr, encoding, stamp, frame_id):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.strides[0]
    msg.data = arr.tobytes()
    return msg


class DatasetPlayer(Node):
    def __init__(self, args):
        super().__init__("splatam_dataset_player")
        self.args = args
        base = Path(args.dataset)
        with open(base / "transforms.json") as f:
            self.manifest = json.load(f)
        self.frames = self.manifest["frames"]
        self.base = base
        self.idx = 0

        self.rgb_pub = self.create_publisher(Image, args.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, args.depth_topic, 10)
        self.rgb_info_pub = self.create_publisher(CameraInfo, args.rgb_info_topic, 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, args.depth_info_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)

        self.cam_info = CameraInfo()
        self.cam_info.width = int(self.manifest["w"])
        self.cam_info.height = int(self.manifest["h"])
        self.cam_info.k = [
            float(self.manifest["fl_x"]), 0.0, float(self.manifest["cx"]),
            0.0, float(self.manifest["fl_y"]), float(self.manifest["cy"]),
            0.0, 0.0, 1.0,
        ]

        # Precompute CV-convention c2w for the whole trajectory
        self.c2ws = [P_GL @ np.array(fr["transform_matrix"]) @ P_GL
                     for fr in self.frames]

        self.t0 = self.get_clock().now().nanoseconds * 1e-9
        self.timer = self.create_timer(1.0 / args.rate, self.tick)
        if args.odom_rate > 0:
            self.odom_timer = self.create_timer(1.0 / args.odom_rate, self.odom_tick)
        self.get_logger().info(
            f"Replaying {len(self.frames)} frames from {base} at {args.rate} Hz"
            + (f", odom at {args.odom_rate} Hz" if args.odom_rate > 0 else "")
        )

    def publish_odom(self, c2w, stamp):
        q = rotmat_to_quat_xyzw(c2w[:3, :3])
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "camera_optical"
        odom.pose.pose.position.x = float(c2w[0, 3])
        odom.pose.pose.position.y = float(c2w[1, 3])
        odom.pose.pose.position.z = float(c2w[2, 3])
        odom.pose.pose.orientation.x = float(q[0])
        odom.pose.pose.orientation.y = float(q[1])
        odom.pose.pose.orientation.z = float(q[2])
        odom.pose.pose.orientation.w = float(q[3])
        self.odom_pub.publish(odom)

    def odom_tick(self):
        # Continuous pose along the replay timeline, independent of frame ticks
        now = self.get_clock().now()
        pos = (now.nanoseconds * 1e-9 - self.t0) * self.args.rate
        n = len(self.c2ws)
        if self.args.loop:
            pos = pos % (n - 1)
        pos = min(max(pos, 0.0), n - 1.001)
        i = int(pos)
        a = pos - i
        c0, c1 = self.c2ws[i], self.c2ws[i + 1]
        c2w = np.eye(4)
        c2w[:3, 3] = (1 - a) * c0[:3, 3] + a * c1[:3, 3]
        # Rotation: nearest-neighbor is fine at odom rates >> frame rate
        c2w[:3, :3] = (c0 if a < 0.5 else c1)[:3, :3]
        self.publish_odom(c2w, now.to_msg())

    def tick(self):
        if self.idx >= len(self.frames):
            if self.args.loop:
                self.idx = 0
            else:
                self.get_logger().info("Replay finished.")
                raise SystemExit
        frame = self.frames[self.idx]

        bgr = cv2.imread(str(self.base / frame["file_path"]), cv2.IMREAD_COLOR)
        depth_png = cv2.imread(
            str(self.base / frame["depth_path"]), cv2.IMREAD_UNCHANGED
        )
        depth_m = np.ascontiguousarray(depth_png.astype(np.float32) / DEPTH_PNG_SCALE)

        stamp = self.get_clock().now().to_msg()
        frame_id = "camera_optical"

        self.rgb_pub.publish(np_to_image_msg(bgr, "bgr8", stamp, frame_id))
        self.depth_pub.publish(np_to_image_msg(depth_m, "32FC1", stamp, frame_id))

        self.cam_info.header.stamp = stamp
        self.cam_info.header.frame_id = frame_id
        self.rgb_info_pub.publish(self.cam_info)
        self.depth_info_pub.publish(self.cam_info)

        if self.args.odom_rate <= 0:
            self.publish_odom(self.c2ws[self.idx], stamp)

        self.idx += 1


def main():
    args = parse_args()
    rclpy.init()
    node = DatasetPlayer(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
