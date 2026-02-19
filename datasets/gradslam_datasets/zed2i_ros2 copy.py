# datasets/gradslam_datasets/zed2i_ros2.py

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import cv2

# ROS2 imports (workstation must have ROS2 + these packages)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

def _decode_compressed_depth(self, msg: CompressedImage) -> np.ndarray:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"cv2.imdecode failed for compressedDepth, format={msg.format}")
    return depth


def _scale_intrinsics(K: np.ndarray, h_scale: float, w_scale: float) -> np.ndarray:
    """Scale intrinsics to match resized image."""
    K2 = K.copy()
    K2[0, 0] *= w_scale  # fx
    K2[1, 1] *= h_scale  # fy
    K2[0, 2] *= w_scale  # cx
    K2[1, 2] *= h_scale  # cy
    return K2


@dataclass
class RGBDPacket:
    stamp_sec: float
    color_bgr8: np.ndarray   # HxWx3 uint8 BGR
    depth_raw: np.ndarray    # HxW (uint16 mm OR float32 m)
    K: np.ndarray            # 3x3 float64
    frame_id: str


class _ZedRGBDSubscriber(Node):
    """
    Subscribes to rectified RGB + depth_registered + camera_info and keeps the latest synchronized packet.
    """
    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str,
        cam_info_topic: str,
        slop_sec: float,
        queue_size: int,
    ):
        super().__init__("zed_rgbd_subscriber")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest: Optional[RGBDPacket] = None
        self._latest_skew_sec: Optional[float] = None

        # Use message_filters to sync RGB+Depth, and we will use CameraInfo K as "latest known".
        # Many drivers publish CameraInfo at the same rate; safest is to subscribe to it normally.
        self._K: Optional[np.ndarray] = None
        self._frame_id: str = ""

        self.create_subscription(CameraInfo, cam_info_topic, self._on_cam_info, qos_profile_sensor_data)

        # Pick RGB message type based on topic name
        if rgb_topic.endswith("/compressed"):
            rgb_msg_type = CompressedImage
        else:
            rgb_msg_type = Image

        rgb_sub = Subscriber(self, rgb_msg_type, rgb_topic, qos_profile=qos_profile_sensor_data)

        # Pick message type based on topic name
        if depth_topic.endswith("/compressedDepth") or depth_topic.endswith("/compressed"):
            depth_msg_type = CompressedImage
        else:
            depth_msg_type = Image

        depth_sub = Subscriber(self, depth_msg_type, depth_topic, qos_profile=qos_profile_sensor_data)
        ats = ApproximateTimeSynchronizer(
            fs=[rgb_sub, depth_sub],
            queue_size=queue_size,
            slop=slop_sec,
            allow_headerless=False,
        )
        ats.registerCallback(self._on_rgb_depth)

        self.get_logger().info(f"Subscribing:\n  RGB: {rgb_topic}\n  Depth: {depth_topic}\n  Info: {cam_info_topic}")

    def _on_cam_info(self, msg: CameraInfo):
        # msg.k is row-major 3x3
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        with self._lock:
            self._K = K
            self._frame_id = msg.header.frame_id

    def _on_rgb_depth(
        self,
        rgb_msg: Union[Image, CompressedImage],
        depth_msg: Union[Image, CompressedImage],
    ):
        # timestamps (both types have header.stamp)
        t_rgb = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        t_d   = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
        skew = abs(t_rgb - t_d)

        # -----------------------
        # RGB conversion (robust)
        # -----------------------
        try:
            if isinstance(rgb_msg, CompressedImage):
                # Usually decodes to BGR8 already (OpenCV)
                rgb_cv = self._bridge.compressed_imgmsg_to_cv2(rgb_msg)
                # If it comes out as BGRA for some reason, drop alpha
                if rgb_cv is not None and rgb_cv.ndim == 3 and rgb_cv.shape[2] == 4:
                    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGRA2BGR)
            else:
                # Decode in native encoding first, then convert ourselves
                rgb_cv = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")

                enc = (rgb_msg.encoding or "").lower()

                # ZED often publishes bgra8
                if enc == "bgra8":
                    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGRA2BGR)
                elif enc == "rgba8":
                    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_RGBA2BGR)
                elif enc == "rgb8":
                    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR)
                elif enc == "bgr8":
                    pass  # already good
                else:
                    # Last resort: try to force bgr8 via cv_bridge conversion
                    rgb_cv = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")

            # Sanity check
            if rgb_cv is None or rgb_cv.ndim != 3 or rgb_cv.shape[2] != 3:
                self.get_logger().error(
                    f"RGB decode produced invalid shape: {None if rgb_cv is None else rgb_cv.shape}"
                )
                return

        except Exception as e:
            self.get_logger().error(f"RGB decode failed: {e}")
            return

        # -------------------------
        # Depth conversion (robust)
        # -------------------------
        try:
            if isinstance(depth_msg, CompressedImage):
                # cv_bridge can decode some compressed depth transports depending on setup
                depth_cv = self._bridge.compressed_imgmsg_to_cv2(depth_msg)

                if depth_cv is None:
                    self.get_logger().error("Depth decode returned None for CompressedImage")
                    return
            else:
                # Raw sensor_msgs/Image
                enc = (depth_msg.encoding or "").upper()

                depth_cv = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

                # Normalize dtype for common cases
                if enc == "16UC1":
                    depth_cv = depth_cv.astype(np.uint16, copy=False)
                elif enc == "32FC1":
                    depth_cv = depth_cv.astype(np.float32, copy=False)
                else:
                    # leave as-is but make sure it's numeric
                    if depth_cv.dtype not in (np.uint16, np.float32, np.float64):
                        depth_cv = depth_cv.astype(np.float32)

            # Sanity check depth shape
            if depth_cv is None or depth_cv.ndim != 2:
                self.get_logger().error(
                    f"Depth decode produced invalid shape: {None if depth_cv is None else depth_cv.shape}"
                )
                return

        except Exception as e:
            self.get_logger().error(f"Depth decode failed: {e}")
            return

        # -------------------------
        # Publish synchronized pkt
        # -------------------------
        with self._lock:
            if self._K is None:
                # Don’t publish packets without intrinsics
                return

            pkt = RGBDPacket(
                stamp_sec=t_rgb,  # choose RGB timestamp as packet stamp
                color_bgr8=rgb_cv,
                depth_raw=depth_cv,
                K=self._K.copy(),
                frame_id=self._frame_id or rgb_msg.header.frame_id,
            )
            self._latest = pkt
            self._latest_skew_sec = skew



    def pop_latest(self) -> Optional[Tuple[RGBDPacket, Optional[float]]]:
        with self._lock:
            pkt = self._latest
            skew = self._latest_skew_sec
        return (pkt, skew) if pkt is not None else None


class ZedRos2Dataset(torch.utils.data.Dataset):
    """
    Live ROS2 dataset adapter for SplaTAM.

    Returns tuples identical to GradSLAMDataset:
      (color(H,W,3), depth(H,W,1), intrinsics(4,4), pose(4,4))

    Notes:
    - pose is identity (no GT).
    - color is float with 0..255 range (SplaTAM divides by 255 later).
    - depth is float meters.
    """
    def __init__(
        self,
        config_dict,
        basedir: str,
        sequence: str,
        desired_height: int = 720,
        desired_width: int = 960,
        device: str = "cuda:0",
        dtype=torch.float,
        num_frames: int = 10_000_000,
        rgb_topic: str = "/zed/zed_node/rgb/color/rect/image",
        depth_topic: str = "/zed/zed_node/depth/depth_registered",
        cam_info_topic: str = "/zed/zed_node/rgb/color/rect/camera_info",
        slop_sec: float = 0.02,
        queue_size: int = 10,
        wait_timeout_sec: float = 2.0,
        debug_log_every_n: int = 50,
        **kwargs,
    ):
        
        super().__init__()
        self.name = config_dict.get("dataset_name", "zed_ros2")
        self.device = device
        self.dtype = dtype
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.num_frames = num_frames

        self._wait_timeout_sec = wait_timeout_sec
        self._debug_log_every_n = debug_log_every_n
        self._get_count = 0

        # ROS2 node runs in a background thread.
        self._ros_thread = None
        self._ros_node: Optional[_ZedRGBDSubscriber] = None
        self._executor = None

        self._rgb_topic = rgb_topic
        self._depth_topic = depth_topic
        self._cam_info_topic = cam_info_topic
        self._slop_sec = slop_sec
        self._queue_size = queue_size
        
        # Normalize accidental tuple/list configs -> string
        if isinstance(self._rgb_topic, (list, tuple)):
            self._rgb_topic = self._rgb_topic[0]
        if isinstance(self._depth_topic, (list, tuple)):
            self._depth_topic = self._depth_topic[0]
        if isinstance(self._cam_info_topic, (list, tuple)):
            self._cam_info_topic = self._cam_info_topic[0]

        self._start_ros()

    def _start_ros(self):
        # Initialize rclpy only once per process
        if not rclpy.ok():
            rclpy.init(args=None)

        self._ros_node = _ZedRGBDSubscriber(
            rgb_topic=self._rgb_topic,
            depth_topic=self._depth_topic,
            cam_info_topic=self._cam_info_topic,
            slop_sec=self._slop_sec,
            queue_size=self._queue_size,
        )

        # Use a SingleThreadedExecutor in a daemon thread
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._ros_node)

        def _spin():
            try:
                self._executor.spin()
            except Exception as e:
                # Don’t crash the whole training loop silently
                print(f"[ZedRos2Dataset] ROS2 spin thread error: {e}")

        self._ros_thread = threading.Thread(target=_spin, daemon=True)
        self._ros_thread.start()

    def __len__(self):
        return self.num_frames

    def _wait_for_packet(self) -> RGBDPacket:
        start = time.time()
        while True:
            out = self._ros_node.pop_latest() if self._ros_node is not None else None
            if out is not None:
                pkt, skew = out
                # Optional periodic debug log
                self._get_count += 1
                if self._debug_log_every_n and (self._get_count % self._debug_log_every_n == 0):
                    self._ros_node.get_logger().info(
                        f"Latest RGBD packet: skew={skew:.4f}s, rgb={pkt.color_bgr8.shape}, depth={pkt.depth_raw.shape}, depth_dtype={pkt.depth_raw.dtype}"
                    )
                return pkt
            if time.time() - start > self._wait_timeout_sec:
                raise TimeoutError(
                    f"Timed out waiting for synchronized RGB+Depth+CameraInfo on topics:\n"
                    f"  {self._rgb_topic}\n  {self._depth_topic}\n  {self._cam_info_topic}"
                )
            time.sleep(0.001)

    def __getitem__(self, index):
        pkt = self._wait_for_packet()

        rgb = pkt.color_bgr8  # HxWx3 uint8 BGR
        depth = pkt.depth_raw # HxW (uint16 mm OR float32 m)

        # Convert RGB to float 0..255 (SplaTAM will divide by 255 later)
        # Keep channel order as-is (BGR vs RGB) is not ideal, but consistent; we can swap to RGB if you want.
        rgb_f = rgb.astype(np.float32)

        # Convert depth to float meters, shape HxWx1
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            depth_m = depth.astype(np.float32)

        # Resize both consistently (if needed)
        h0, w0 = rgb_f.shape[0], rgb_f.shape[1]
        if (h0 != self.desired_height) or (w0 != self.desired_width):
            rgb_f = cv2.resize(rgb_f, (self.desired_width, self.desired_height), interpolation=cv2.INTER_LINEAR)
            depth_m = cv2.resize(depth_m, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)

            h_scale = float(self.desired_height) / float(h0)
            w_scale = float(self.desired_width) / float(w0)
            K_scaled = _scale_intrinsics(pkt.K, h_scale=h_scale, w_scale=w_scale)
        else:
            K_scaled = pkt.K

        depth_m = np.expand_dims(depth_m, axis=-1)  # HxWx1

        # Build intrinsics 4x4
        intrinsics = np.eye(4, dtype=np.float32)
        intrinsics[:3, :3] = K_scaled.astype(np.float32)

        # Live mode: identity pose (no GT)
        pose = np.eye(4, dtype=np.float32)

        # Torch tensors on GPU
        color_t = torch.from_numpy(rgb_f).to(self.device).type(self.dtype)
        depth_t = torch.from_numpy(depth_m).to(self.device).type(self.dtype)
        intr_t = torch.from_numpy(intrinsics).to(self.device).type(self.dtype)
        pose_t = torch.from_numpy(pose).to(self.device).type(self.dtype)

        return color_t, depth_t, intr_t, pose_t

