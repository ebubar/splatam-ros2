# datasets/gradslam_datasets/zed2i_ros2.py

import threading
import time
import struct
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _scale_intrinsics(K: np.ndarray, h_scale: float, w_scale: float) -> np.ndarray:
    """Scale intrinsics to match resized image."""
    K2 = K.copy()
    K2[0, 0] *= w_scale  # fx
    K2[1, 1] *= h_scale  # fy
    K2[0, 2] *= w_scale  # cx
    K2[1, 2] *= h_scale  # cy
    return K2


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _parse_compressed_depth_encoding(fmt: str) -> str:
    """
    CompressedImage.format often looks like:
      "32FC1; compressedDepth" or "16UC1; compressedDepth"
    We want the encoding part (32FC1 / 16UC1).
    """
    if not fmt:
        return ""
    if ";" in fmt:
        return fmt.split(";", 1)[0].strip()
    return fmt.strip()


def decode_compressed_rgb_to_bgr(msg: CompressedImage) -> np.ndarray:
    """
    Decode CompressedImage RGB topic to BGR uint8 (OpenCV order).
    ZED compressed RGB is typically JPEG.
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("RGB compressed decode failed (cv2.imdecode returned None)")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def decode_compressed_depth_to_meters(msg: CompressedImage) -> np.ndarray:
    """
    Robust decoder for ROS2 compressedDepth.

    Handles both:
      A) Raw PNG bytes start immediately (data begins with PNG magic)
      B) 12-byte header + PNG payload (common)

    Returns: depth float32 in METERS, shape HxW
    """
    enc = _parse_compressed_depth_encoding(msg.format).upper()

    data = msg.data
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)

    if len(data) < 8:
        raise RuntimeError("compressedDepth too small")

    # -------- Case A: raw PNG starts immediately --------
    if data[:8] == PNG_MAGIC:
        inv = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if inv is None:
            raise RuntimeError("Failed to decode depth PNG (raw)")
        if inv.ndim == 3:
            inv = inv[:, :, 0]

        if enc.startswith("16UC1"):
            # typically millimeters -> meters
            return inv.astype(np.float32) * 0.001

        if enc.startswith("32FC1"):
            # sometimes already float meters
            return inv.astype(np.float32)

        # Fallback: treat like mm if uint16, else float32
        if inv.dtype == np.uint16:
            return inv.astype(np.float32) * 0.001
        return inv.astype(np.float32)

    # -------- Case B: 12-byte header + PNG --------
    if len(data) < 12:
        raise RuntimeError("compressedDepth too small for header")

    # Common layout: <uint32, float32, float32> = 12 bytes
    # (matches your working script)
    _, depthQuantA, depthQuantB = struct.unpack("<iff", data[:12])
    payload = data[12:]

    # Sometimes payload may contain extra bytes before PNG. Find PNG magic.
    png_pos = payload.find(PNG_MAGIC)
    if png_pos < 0:
        raise RuntimeError(f"compressedDepth payload is not PNG (format='{msg.format}')")

    png_bytes = payload[png_pos:]
    inv = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if inv is None:
        raise RuntimeError("Failed to decode depth PNG (header+png)")
    if inv.ndim == 3:
        inv = inv[:, :, 0]

    # If 32FC1 header+png, inv is usually uint16 "inverse depth"
    # depth(m) = A / (inv - B)
    if enc.startswith("32FC1"):
        inv = inv.astype(np.float32)
        depth = np.zeros_like(inv, dtype=np.float32)
        mask = inv > 0
        depth[mask] = float(depthQuantA) / (inv[mask] - float(depthQuantB))
        return depth

    if enc.startswith("16UC1"):
        # usually mm in png
        return inv.astype(np.float32) * 0.001

    # fallback
    if inv.dtype == np.uint16:
        return inv.astype(np.float32) * 0.001
    return inv.astype(np.float32)


@dataclass
class RGBDPacket:
    stamp_sec: float
    color_bgr8: np.ndarray   # HxWx3 uint8 BGR
    depth_m: np.ndarray      # HxW float32 meters
    K: np.ndarray            # 3x3 float64
    frame_id: str
    skew_sec: float


class _ZedRGBDSubscriber(Node):
    """
    Subscribe to RGB + Depth + CameraInfo and synchronize them via message_filters.
    This avoids the “no K yet” / stale K issues and matches your working approach.
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

        # ZED topics show RELIABLE in your `ros2 topic info -v`
        self._qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(10, queue_size),
        )

        rgb_msg_type = CompressedImage if rgb_topic.endswith("/compressed") else Image
        depth_msg_type = CompressedImage if (depth_topic.endswith("/compressedDepth") or depth_topic.endswith("/compressed")) else Image

        self.rgb_sub = Subscriber(self, rgb_msg_type, rgb_topic, qos_profile=self._qos)
        self.depth_sub = Subscriber(self, depth_msg_type, depth_topic, qos_profile=self._qos)
        self.caminfo_sub = Subscriber(self, CameraInfo, cam_info_topic, qos_profile=self._qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.caminfo_sub],
            queue_size=queue_size,
            slop=slop_sec,
            allow_headerless=False,
        )
        self.sync.registerCallback(self._cb)

        self.get_logger().info(
            "Subscribing:\n"
            f"  RGB:  {rgb_topic}\n"
            f"  Depth:{depth_topic}\n"
            f"  Info: {cam_info_topic}\n"
            f"  ATS: slop={slop_sec}s, queue={queue_size}"
        )

    def _decode_rgb(self, msg: Union[Image, CompressedImage]) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            return decode_compressed_rgb_to_bgr(msg)

        rgb_cv = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        enc = (msg.encoding or "").lower()

        # ZED often publishes bgra8
        if enc == "bgra8":
            rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGRA2BGR)
        elif enc == "rgba8":
            rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_RGBA2BGR)
        elif enc == "rgb8":
            rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR)
        elif enc == "bgr8":
            pass
        else:
            rgb_cv = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if rgb_cv is None or rgb_cv.ndim != 3 or rgb_cv.shape[2] != 3:
            raise RuntimeError(f"RGB invalid after decode: {None if rgb_cv is None else rgb_cv.shape}")
        return rgb_cv

    def _decode_depth_m(self, msg: Union[Image, CompressedImage]) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            depth_m = decode_compressed_depth_to_meters(msg)
            if depth_m.ndim != 2:
                raise RuntimeError(f"Depth invalid after compressed decode: {depth_m.shape}")
            return depth_m.astype(np.float32, copy=False)

        depth_cv = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        enc = (msg.encoding or "").upper()

        if enc == "16UC1":
            depth_m = depth_cv.astype(np.float32, copy=False) * 0.001
        elif enc == "32FC1":
            depth_m = depth_cv.astype(np.float32, copy=False)
        else:
            # best effort
            depth_m = depth_cv.astype(np.float32, copy=False)

        if depth_m.ndim != 2:
            raise RuntimeError(f"Depth invalid after raw decode: {depth_m.shape}")
        return depth_m

    def _cb(self, rgb_msg, depth_msg, caminfo_msg: CameraInfo):
        try:
            t_rgb = _stamp_to_sec(rgb_msg.header.stamp)
            t_d = _stamp_to_sec(depth_msg.header.stamp)
            skew = abs(t_rgb - t_d)

            rgb_bgr = self._decode_rgb(rgb_msg)
            depth_m = self._decode_depth_m(depth_msg)

            K = np.array(caminfo_msg.k, dtype=np.float64).reshape(3, 3)
            frame_id = caminfo_msg.header.frame_id or rgb_msg.header.frame_id

        except Exception as e:
            self.get_logger().error(f"RGBD callback decode failed: {e}")
            return

        with self._lock:
            self._latest = RGBDPacket(
                stamp_sec=t_rgb,
                color_bgr8=rgb_bgr,
                depth_m=depth_m,
                K=K,
                frame_id=frame_id,
                skew_sec=skew,
            )

    def pop_latest(self) -> Optional[RGBDPacket]:
        with self._lock:
            return self._latest


class ZedRos2Dataset(torch.utils.data.Dataset):
    """
    Live ROS2 dataset adapter for SplaTAM.

    Returns tuples:
      (color(H,W,3), depth(H,W,1), intrinsics(4,4), pose(4,4))

    - color: float32 0..255 (SplaTAM divides by 255 later)
    - depth: float32 meters
    - pose: identity
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
        slop_sec: float = 0.1,
        queue_size: int = 30,
        wait_timeout_sec: float = 5.0,
        debug_log_every_n: int = 50,
        **kwargs,
    ):
        super().__init__()
        self.name = config_dict.get("dataset_name", "zed2i_ros2")
        self.device = device
        self.dtype = dtype
        self.desired_height = int(desired_height)
        self.desired_width = int(desired_width)
        self.num_frames = int(num_frames)

        self._wait_timeout_sec = float(wait_timeout_sec)
        self._debug_log_every_n = int(debug_log_every_n) if debug_log_every_n else 0
        self._get_count = 0

        self._rgb_topic = rgb_topic[0] if isinstance(rgb_topic, (list, tuple)) else rgb_topic
        self._depth_topic = depth_topic[0] if isinstance(depth_topic, (list, tuple)) else depth_topic
        self._cam_info_topic = cam_info_topic[0] if isinstance(cam_info_topic, (list, tuple)) else cam_info_topic

        self._slop_sec = float(slop_sec)
        self._queue_size = int(queue_size)

        self._ros_thread = None
        self._ros_node: Optional[_ZedRGBDSubscriber] = None
        self._executor = None

        self._start_ros()

    def _start_ros(self):
        if not rclpy.ok():
            rclpy.init(args=None)

        self._ros_node = _ZedRGBDSubscriber(
            rgb_topic=self._rgb_topic,
            depth_topic=self._depth_topic,
            cam_info_topic=self._cam_info_topic,
            slop_sec=self._slop_sec,
            queue_size=self._queue_size,
        )

        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._ros_node)

        def _spin():
            try:
                self._executor.spin()
            except Exception as e:
                print(f"[ZedRos2Dataset] ROS2 spin thread error: {e}")

        self._ros_thread = threading.Thread(target=_spin, daemon=True)
        self._ros_thread.start()

    def __len__(self):
        return self.num_frames

    def _wait_for_packet(self) -> RGBDPacket:
        start = time.time()
        while True:
            pkt = self._ros_node.pop_latest() if self._ros_node is not None else None
            if pkt is not None:
                self._get_count += 1
                if self._debug_log_every_n and (self._get_count % self._debug_log_every_n == 0):
                    # quick sanity depth stats
                    d = pkt.depth_m
                    valid = d > 0
                    if np.any(valid):
                        dv = d[valid]
                        self._ros_node.get_logger().info(
                            f"RGBD: skew={pkt.skew_sec:.4f}s rgb={pkt.color_bgr8.shape} "
                            f"depth={pkt.depth_m.shape} depth(m) min/med/max="
                            f"{float(dv.min()):.3f}/{float(np.median(dv)):.3f}/{float(dv.max()):.3f}"
                        )
                    else:
                        self._ros_node.get_logger().info(
                            f"RGBD: skew={pkt.skew_sec:.4f}s rgb={pkt.color_bgr8.shape} depth all zeros"
                        )
                return pkt

            if time.time() - start > self._wait_timeout_sec:
                raise TimeoutError(
                    "Timed out waiting for synchronized RGB+Depth+CameraInfo.\n"
                    f"  RGB:  {self._rgb_topic}\n"
                    f"  Depth:{self._depth_topic}\n"
                    f"  Info: {self._cam_info_topic}\n"
                    "If using compressed topics, ensure /compressed and /compressedDepth are actually publishing."
                )

            time.sleep(0.001)

    def __getitem__(self, index):
        pkt = self._wait_for_packet()

        rgb_f = pkt.color_bgr8.astype(np.float32)  # 0..255
        depth_m = pkt.depth_m.astype(np.float32)   # meters

        # Resize both consistently (if needed)
        h0, w0 = rgb_f.shape[:2]
        if (h0 != self.desired_height) or (w0 != self.desired_width):
            rgb_f = cv2.resize(rgb_f, (self.desired_width, self.desired_height), interpolation=cv2.INTER_LINEAR)
            depth_m = cv2.resize(depth_m, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)

            h_scale = float(self.desired_height) / float(h0)
            w_scale = float(self.desired_width) / float(w0)
            K_scaled = _scale_intrinsics(pkt.K, h_scale=h_scale, w_scale=w_scale)
        else:
            K_scaled = pkt.K

        depth_m = np.expand_dims(depth_m, axis=-1)  # HxWx1

        intrinsics = np.eye(4, dtype=np.float32)
        intrinsics[:3, :3] = K_scaled.astype(np.float32)

        pose = np.eye(4, dtype=np.float32)  # live mode: identity

        color_t = torch.from_numpy(rgb_f).to(self.device).type(self.dtype)
        depth_t = torch.from_numpy(depth_m).to(self.device).type(self.dtype)
        intr_t = torch.from_numpy(intrinsics).to(self.device).type(self.dtype)
        pose_t = torch.from_numpy(pose).to(self.device).type(self.dtype)

        return color_t, depth_t, intr_t, pose_t
