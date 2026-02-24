# datasets/gradslam_datasets/zed_ros2.py

import threading
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

try:
    from cv_bridge import CvBridge
except ImportError as e:
    raise ImportError(
        "cv_bridge not found. Make sure ROS2 Python environment is sourced and "
        "python3-cv-bridge is installed for your ROS distro."
    ) from e

try:
    import message_filters
except ImportError as e:
    raise ImportError(
        "message_filters not found. Install the ROS2 python message_filters package."
    ) from e


def _as_intrinsics_4x4(K3: np.ndarray) -> torch.Tensor:
    """Return 4x4 intrinsics with K in top-left."""
    intr = torch.eye(4, dtype=torch.float32)
    intr[:3, :3] = torch.from_numpy(K3.astype(np.float32))
    return intr


class _ZedROS2Node(Node):
    """
    Internal ROS2 node that synchronizes RGB + depth (registered) and stores
    frames into a ring buffer.
    """

    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        buffer_size: int = 300,
        sync_slop_s: float = 0.03,
        sync_queue: int = 10,
    ):
        super().__init__("splatam_zed_ros2_dataset")

        self.bridge = CvBridge()

        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic

        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()

        self._K: Optional[np.ndarray] = None  # 3x3 float64
        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None

        # CameraInfo subscriber (intrinsics)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._caminfo_cb, 10)

        # Synced RGB + Depth subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=sync_queue,
            slop=sync_slop_s,
        )
        self.ts.registerCallback(self._rgbd_cb)

        self.get_logger().info(
            f"ZED ROS2 Dataset subscribed to:\n"
            f"  RGB:   {self.rgb_topic}\n"
            f"  Depth: {self.depth_topic}\n"
            f"  K:     {self.camera_info_topic}\n"
            f"  Buffer size: {buffer_size}"
        )

    def _caminfo_cb(self, msg: CameraInfo):
        # K is row-major 3x3 in msg.k
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        with self.lock:
            self._K = K
            self._img_w = msg.width
            self._img_h = msg.height

    def _rgbd_cb(self, rgb_msg: Image, depth_msg: Image):
        # Convert RGB
        # Your ZED is bgra8. We convert to RGB (uint8 HxWx3).
        try:
            if rgb_msg.encoding.lower() == "bgra8":
                bgra = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgra8")
                bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif rgb_msg.encoding.lower() == "bgr8":
                bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif rgb_msg.encoding.lower() == "rgb8":
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            else:
                # Fallback: try to convert as is, then handle channels
                cv_img = self.bridge.imgmsg_to_cv2(rgb_msg)
                if cv_img.ndim == 3 and cv_img.shape[2] == 4:
                    # assume BGRA
                    bgr = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elif cv_img.ndim == 3 and cv_img.shape[2] == 3:
                    # assume BGR
                    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Unexpected RGB shape {cv_img.shape} encoding={rgb_msg.encoding}")
        except Exception as e:
            self.get_logger().warn(f"Failed converting RGB image: {e}")
            return

        # Convert Depth
        # Your depth is 32FC1. We decode float32 HxW.
        try:
            if depth_msg.encoding.upper() == "32FC1":
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
                depth = depth.astype(np.float32)
            elif depth_msg.encoding.upper() in ("16UC1", "MONO16"):
                # if someone switches ZED settings, handle mm depth
                depth_u16 = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
                depth = depth_u16.astype(np.float32)
            else:
                depth = self.bridge.imgmsg_to_cv2(depth_msg)
                depth = np.asarray(depth).astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f"Failed converting depth image: {e}")
            return

        # Replace invalid values (ZED often uses NaN for invalid)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # Store
        stamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.buffer.append((stamp, rgb, depth))


class ZEDROS2Dataset(torch.utils.data.Dataset):
    """
    Streaming dataset wrapper for ZED2i via ROS2 topics.
    Returns (color, depth, intrinsics_4x4, pose_4x4).

    color: torch.uint8-like float tensor later divided by 255 in splatam.py (we keep uint8 -> float)
           shape (H, W, 3)
    depth: float32 meters-ish, shape (H, W, 1)
    intrinsics: 4x4 with K scaled for desired resolution
    pose: 4x4 identity (no GT poses live)
    """

    def __init__(
        self,
        config_dict,
        basedir=None,
        sequence=None,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 360,
        desired_width: int = 640,
        device="cuda:0",
        dtype=torch.float,
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict["dataset_name"]
        self.device = device
        self.dtype = dtype

        cam = config_dict["camera_params"]
        self.orig_height = int(cam["image_height"])
        self.orig_width = int(cam["image_width"])

        # IMPORTANT: for 32FC1 depth in meters, use 1.0
        self.png_depth_scale = float(cam.get("png_depth_scale", 1.0))

        # Intrinsics from config (we can update from CameraInfo at runtime too)
        self.fx = float(cam["fx"])
        self.fy = float(cam["fy"])
        self.cx = float(cam["cx"])
        self.cy = float(cam["cy"])

        self.desired_height = int(desired_height)
        self.desired_width = int(desired_width)
        self.height_downsample_ratio = float(self.desired_height) / float(self.orig_height)
        self.width_downsample_ratio = float(self.desired_width) / float(self.orig_width)

        # ROS topics from config
        zed_cfg = config_dict.get("zed_ros2", {})
        self.rgb_topic = zed_cfg.get("rgb_topic", "/zed/zed_node/rgb/color/rect/image")
        self.depth_topic = zed_cfg.get("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.camera_info_topic = zed_cfg.get("camera_info_topic", "/zed/zed_node/rgb/color/rect/camera_info")
        self.buffer_size = int(zed_cfg.get("buffer_size", 300))
        self.sync_slop_s = float(zed_cfg.get("sync_slop_s", 0.03))
        self.sync_queue = int(zed_cfg.get("sync_queue", 10))

        # For splatam loop
        self.start = int(start or 0)
        self.end = int(end) if end is not None else -1
        self.stride = int(stride or 1)

        # Pose is identity for all frames (no GT)
        self._identity_pose = torch.eye(4, dtype=torch.float32)

        # ROS spin thread
        self._ros_thread = None
        self._ros_running = False
        self._node: Optional[_ZedROS2Node] = None

        self._start_ros()

        # Wait a moment to start receiving data
        t0 = time.time()
        while time.time() - t0 < 30.0:
            if self._node is not None and len(self._node.buffer) > 0 and self._node._K is not None:
                break
            time.sleep(0.05)

    def _start_ros(self):
        # Initialize ROS once per process
        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = _ZedROS2Node(
            rgb_topic=self.rgb_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            buffer_size=self.buffer_size,
            sync_slop_s=self.sync_slop_s,
            sync_queue=self.sync_queue,
        )
        self._ros_running = True

        def _spin():
            while self._ros_running and rclpy.ok():
                rclpy.spin_once(self._node, timeout_sec=0.05)

        self._ros_thread = threading.Thread(target=_spin, daemon=True)
        self._ros_thread.start()

    def shutdown(self):
        self._ros_running = False
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=1.0)
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __len__(self):
        """
        Finite length so splatam's range(num_frames) works.
        If end == -1, we report current buffered size (at least 1 once warmed up).
        Otherwise, we report (end-start)/stride.
        """
        if self.end != -1:
            n = max(0, (self.end - self.start + (self.stride - 1)) // self.stride)
            return n
        # live mode
        if self._node is None:
            return 0
        with self._node.lock:
            return max(1, len(self._node.buffer))

    def _get_latest_buffered(self, index: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int, int]:
        """
        Returns rgb(H,W,3), depth(H,W), K(3,3) if available, img_w, img_h
        """
        if self._node is None:
            raise RuntimeError("ROS2 node not initialized")

        with self._node.lock:
            if len(self._node.buffer) == 0:
                raise RuntimeError("No frames received yet from ROS2 topics")

            # Use an index into the ring buffer; if index exceeds current size, clamp to last
            idx = min(index, len(self._node.buffer) - 1)
            _, rgb, depth = self._node.buffer[idx]

            # Prefer runtime CameraInfo K if available
            K = self._node._K.copy() if self._node._K is not None else None
            iw = self._node._img_w if self._node._img_w is not None else self.orig_width
            ih = self._node._img_h if self._node._img_h is not None else self.orig_height

        return rgb, depth, K, int(iw), int(ih)

    def __getitem__(self, index: int):
        # Map requested dataset index to ring buffer position
        # If you want "always latest frame", set idx = -1.
        # Here we map increasing index -> increasing buffer entries.
        rgb_np, depth_np, K_runtime, iw, ih = self._get_latest_buffered(index)

        # Update orig sizes if CameraInfo differs
        self.orig_width = iw
        self.orig_height = ih
        self.height_downsample_ratio = float(self.desired_height) / float(self.orig_height)
        self.width_downsample_ratio = float(self.desired_width) / float(self.orig_width)

        # If runtime K exists, use it; else use config fx/fy/cx/cy
        if K_runtime is not None:
            K3 = K_runtime.astype(np.float32)
            fx = float(K3[0, 0]); fy = float(K3[1, 1]); cx = float(K3[0, 2]); cy = float(K3[1, 2])
        else:
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
            K3 = np.array([[fx, 0.0, cx],
                           [0.0, fy, cy],
                           [0.0, 0.0, 1.0]], dtype=np.float32)

        # Resize color
        rgb_resized = cv2.resize(rgb_np, (self.desired_width, self.desired_height), interpolation=cv2.INTER_LINEAR)

        # Resize depth (nearest), scale, add channel dim
        depth_resized = cv2.resize(depth_np.astype(np.float32), (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)
        depth_resized = np.nan_to_num(depth_resized, nan=0.0, posinf=0.0, neginf=0.0)
        depth_resized = (depth_resized / float(self.png_depth_scale)).astype(np.float32)
        depth_resized = np.expand_dims(depth_resized, axis=-1)  # (H,W,1)

        # Scale intrinsics for resized image
        K3_scaled = K3.copy()
        K3_scaled[0, 0] *= self.width_downsample_ratio
        K3_scaled[1, 1] *= self.height_downsample_ratio
        K3_scaled[0, 2] *= self.width_downsample_ratio
        K3_scaled[1, 2] *= self.height_downsample_ratio

        intrinsics_4x4 = _as_intrinsics_4x4(K3_scaled)

        # Convert to torch
        color_t = torch.from_numpy(rgb_resized.astype(np.float32)).to(self.device).type(self.dtype)
        depth_t = torch.from_numpy(depth_resized).to(self.device).type(self.dtype)
        intr_t = intrinsics_4x4.to(self.device).type(self.dtype)
        pose_t = self._identity_pose.to(self.device).type(self.dtype)

        return color_t, depth_t, intr_t, pose_t