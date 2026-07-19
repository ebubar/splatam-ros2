"""Live ROS2 capture of sparse keyframes from the ZED (via the Zenoh transfer).

Subscribes to the ZED RGB (+CameraInfo), depth and odometry topics, and saves a
*sparse* set of keyframes selected by camera motion. This replaces the live
SplaTAM node -- it only records frames + priors, it does not do SLAM.

Output (a normalized capture dir consumable by the SfM stage)::

    <out_dir>/images/frame_00000.png
    <out_dir>/depth/frame_00000.npy      # metres, float32 (if save_depth)
    <out_dir>/cameras.json               # intrinsics + optical c2w per frame

Run standalone::

    python -m fastsplat.ingest.ros2_capture --config configs/fastsplat/fastsplat.py
"""

import argparse
import os

import numpy as np

from ..utils.config import load_config
from ..utils.io import save_cameras_json
from ..utils.ros_image import (
    caminfo_to_K,
    depth_to_meters,
    odom_to_optical_c2w,
    ros_rgb_to_rgb,
)


def _rotation_angle_deg(Ra, Rb):
    R = Ra[:3, :3].T @ Rb[:3, :3]
    cos = (np.trace(R) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return np.degrees(np.arccos(cos))


def capture_from_ros2(cfg, out_dir):
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import CameraInfo, Image
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge
    from message_filters import ApproximateTimeSynchronizer, Subscriber

    ing = cfg["ingest"]
    ros = ing["ros"]
    images_dir = os.path.join(out_dir, "images")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(images_dir, exist_ok=True)
    if ing.get("save_depth"):
        os.makedirs(depth_dir, exist_ok=True)

    reliability = (
        ReliabilityPolicy.RELIABLE
        if ros.get("qos_reliability") == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    qos = QoSProfile(
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    class Capture(Node):
        def __init__(self):
            super().__init__("fastsplat_capture")
            self.bridge = CvBridge()
            self.frames = []
            self.received = 0
            self.last_c2w = None
            self.latest_info = None
            self.done = False

            self.create_subscription(
                CameraInfo, ros["rgb_info_topic"], self._info_cb, qos
            )
            subs = [
                Subscriber(self, Image, ros["rgb_topic"], qos_profile=qos),
                Subscriber(self, Image, ros["depth_topic"], qos_profile=qos),
            ]
            self.use_odom = bool(ing.get("save_poses", True)) and ros.get("odom_topic")
            if self.use_odom:
                subs.append(Subscriber(self, Odometry, ros["odom_topic"], qos_profile=qos))
            self.sync = ApproximateTimeSynchronizer(
                subs, queue_size=100, slop=0.15, allow_headerless=False
            )
            self.sync.registerCallback(self._synced_cb)
            self.get_logger().info(
                f"fastsplat capture: rgb={ros['rgb_topic']} odom={'on' if self.use_odom else 'off'}"
            )

        def _info_cb(self, msg):
            self.latest_info = msg

        def _accept(self, c2w):
            """Keyframe gate by motion (falls back to accept-all without odom)."""
            if c2w is None or self.last_c2w is None:
                return True
            dt = np.linalg.norm(c2w[:3, 3] - self.last_c2w[:3, 3])
            dr = _rotation_angle_deg(self.last_c2w, c2w)
            return dt >= ing["min_translation_m"] or dr >= ing["min_rotation_deg"]

        def _synced_cb(self, *msgs):
            if self.done:
                return
            self.received += 1
            if self.received % max(1, int(ing.get("process_every_n", 1))) != 0:
                return
            if self.latest_info is None:
                self.get_logger().warn("waiting for CameraInfo...")
                return

            rgb_msg, depth_msg = msgs[0], msgs[1]
            odom_msg = msgs[2] if self.use_odom and len(msgs) > 2 else None

            c2w = odom_to_optical_c2w(odom_msg) if odom_msg is not None else None
            if not self._accept(c2w):
                return

            idx = len(self.frames)
            rgb = ros_rgb_to_rgb(
                self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough"),
                rgb_msg.encoding,
            )
            if ing.get("resize_width") and ing.get("resize_height"):
                rgb = cv2.resize(rgb, (ing["resize_width"], ing["resize_height"]))
            name = f"frame_{idx:05d}.png"
            cv2.imwrite(os.path.join(images_dir, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            K = caminfo_to_K(self.latest_info)
            sx = rgb.shape[1] / float(self.latest_info.width)
            sy = rgb.shape[0] / float(self.latest_info.height)
            K = K.copy()
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy

            frame = {
                "file_path": os.path.join("images", name),
                "w": int(rgb.shape[1]),
                "h": int(rgb.shape[0]),
                "intrinsics": K.tolist(),
                "is_metric": True,
            }
            if odom_msg is not None:
                frame["c2w"] = c2w.tolist()
            if ing.get("save_depth"):
                depth = depth_to_meters(
                    self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"),
                    depth_msg.encoding,
                )
                if depth.ndim == 3:
                    depth = depth[..., 0]
                depth = np.nan_to_num(depth * ros.get("depth_unit_scale_m", 1.0))
                depth[(depth < ros["min_depth_m"]) | (depth > ros["max_depth_m"])] = 0.0
                if depth.shape[:2] != (rgb.shape[0], rgb.shape[1]):
                    depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                dpath = os.path.join("depth", f"frame_{idx:05d}.npy")
                np.save(os.path.join(out_dir, dpath), depth.astype(np.float32))
                frame["depth_path"] = dpath

            self.frames.append(frame)
            self.last_c2w = c2w if c2w is not None else self.last_c2w
            self.get_logger().info(f"captured keyframe {idx + 1}/{ing['max_frames']}")

            if len(self.frames) >= int(ing["max_frames"]):
                self.done = True

    rclpy.init()
    node = Capture()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        frames = node.frames
        node.destroy_node()
        rclpy.shutdown()

    save_cameras_json(os.path.join(out_dir, "cameras.json"), frames)
    print(f"[ingest] captured {len(frames)} sparse keyframes -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fastsplat/fastsplat.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = args.out or os.path.join(cfg["workdir"], cfg["run_name"], "capture")
    os.makedirs(out, exist_ok=True)
    capture_from_ros2(cfg, out)


if __name__ == "__main__":
    main()
