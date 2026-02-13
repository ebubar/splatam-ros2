# datasets/zed_ros_dataset.py
import time
import struct
import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def decode_compressed_rgb(msg: CompressedImage) -> np.ndarray:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode RGB compressed image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# REPLACE WITH (robust: supports raw-PNG and header+PNG, returns METERS)

def decode_compressed_depth(msg: CompressedImage) -> np.ndarray:
    """
    Robust decoder for ROS2 compressedDepth.

    Handles both:
      A) Raw PNG bytes start immediately (data begins with PNG magic)
      B) 12-byte header + PNG payload (common for 32FC1)

    Returns: depth float32 in METERS.
    """
    enc = msg.format.split(";")[0].strip() if ";" in msg.format else msg.format.strip()

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

        if enc.upper().startswith("16UC1"):
            # typically millimeters -> meters
            return inv.astype(np.float32) * 0.001

        if enc.upper().startswith("32FC1"):
            # sometimes already stored directly as float meters
            return inv.astype(np.float32)

        raise RuntimeError(f"Unsupported depth encoding (raw PNG): {enc}")

    # -------- Case B: 12-byte header + PNG --------
    if len(data) < 12:
        raise RuntimeError("compressedDepth too small for header")

    _, depthQuantA, depthQuantB = struct.unpack("<iff", data[:12])
    payload = data[12:]

    png_pos = payload.find(PNG_MAGIC)
    if png_pos < 0:
        raise RuntimeError(
            f"compressedDepth not PNG (format='{msg.format}'), RVL not supported"
        )

    png_bytes = payload[png_pos:]
    inv = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if inv is None:
        raise RuntimeError("Failed to decode depth PNG (header+png)")

    if enc.upper().startswith("32FC1"):
        inv = inv.astype(np.float32)
        depth = np.empty_like(inv, dtype=np.float32)
        mask = inv > 0
        depth[~mask] = 0.0
        depth[mask] = depthQuantA / (inv[mask] - depthQuantB)
        return depth  # meters

    if enc.upper().startswith("16UC1"):
        # commonly mm in PNG -> meters
        return inv.astype(np.float32) * 0.001

    raise RuntimeError(f"Unsupported depth encoding: {enc}")


class _ZedRosNode(Node):
    def __init__(self, rgb_topic: str, depth_topic: str, caminfo_topic: str):
        super().__init__("zed_ros_dataset")
        self.latest = None  # (rgb_np, depth_np, caminfo_msg)

        self.rgb_sub = Subscriber(self, CompressedImage, rgb_topic)
        self.depth_sub = Subscriber(self, CompressedImage, depth_topic)
        self.caminfo_sub = Subscriber(self, CameraInfo, caminfo_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.caminfo_sub],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self._rgbd_cb)

    def _rgbd_cb(self, rgb_msg: CompressedImage, depth_msg: CompressedImage, caminfo_msg: CameraInfo):
        rgb = decode_compressed_rgb(rgb_msg)
        depth = decode_compressed_depth(depth_msg)
        self.latest = (rgb, depth, caminfo_msg)


class ZedRosDataset:
    """
    Minimal dataset shim so SplaTAM's rgbd_slam() can stay unchanged.

    Returns:
        color: (H,W,3) uint8 torch tensor
        depth: (H,W,1) float32 torch tensor (meters)
        intrinsics: (4,4) float32 torch tensor
        pose: (4,4) float32 torch tensor (identity)
    """

    def __init__(
        self,
        desired_height: int,
        desired_width: int,
        device,
        num_frames: int,
        rgb_topic: str = "/zed/zed_node/rgb/color/rect/image/compressed",
        depth_topic: str = "/zed/zed_node/depth/depth_registered/compressedDepth",
        caminfo_topic: str = "/zed/zed_node/rgb/color/rect/camera_info",
        debug_print: bool = True,
    ):
        self.desired_height = int(desired_height)
        self.desired_width = int(desired_width)
        self.device = device
        self.num_frames = int(num_frames)
        self.debug_print = bool(debug_print)

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = _ZedRosNode(rgb_topic, depth_topic, caminfo_topic)

    def __len__(self):
        return self.num_frames

    def _wait_for_frame(self, timeout_s: float = 5.0):
        t0 = time.time()
        while (time.time() - t0) < timeout_s:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.latest is not None:
                frame = self.node.latest
                self.node.latest = None
                return frame
        raise RuntimeError("Timed out waiting for ZED frames")

    # REPLACE WITH (prints % valid + min/median/max in METERS)

    def __getitem__(self, idx):
        rgb, depth, caminfo = self._wait_for_frame()

        # ---- DEBUG: do we have real depth? ----
        valid = depth > 0
        pct = 100.0 * float(valid.mean())
        if valid.any():
            v = depth[valid]
            print(
                f"[DEPTH raw] valid={pct:.1f}%  "
                f"min/med/max={float(v.min()):.3f}/{float(np.median(v)):.3f}/{float(v.max()):.3f} (m)"
            )
        else:
            print("[DEPTH raw] valid=0% (all zeros)")


        # Save original decoded dimensions for K scaling fallback
        decoded_h, decoded_w = rgb.shape[0], rgb.shape[1]

        # Resize to desired
        rgb = cv2.resize(
            rgb, (self.desired_width, self.desired_height), interpolation=cv2.INTER_LINEAR
        )
        depth = cv2.resize(
            depth, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST
        )

        # ---- DEBUG: resized depth stats ----
        if self.debug_print:
            valid2 = depth[depth > 0]
            if valid2.size > 0:
                print(
                    "[ZED depth resized] min/median/max:",
                    float(valid2.min()),
                    float(np.median(valid2)),
                    float(valid2.max()),
                )

        # Depth shape (H,W,1)
        depth = np.expand_dims(depth.astype(np.float32), axis=-1)

        # Intrinsics from caminfo
        fx = float(caminfo.k[0])
        fy = float(caminfo.k[4])
        cx = float(caminfo.k[2])
        cy = float(caminfo.k[5])

        # Robust source size for scaling intrinsics:
        # prefer caminfo.width/height if valid, else use decoded RGB size
        src_w = int(caminfo.width) if int(caminfo.width) > 0 else int(decoded_w)
        src_h = int(caminfo.height) if int(caminfo.height) > 0 else int(decoded_h)

        sx = self.desired_width / float(src_w)
        sy = self.desired_height / float(src_h)

        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

        if self.debug_print:
            print(
                "[K scaled]",
                "src_w/h:", src_w, src_h,
                "dst_w/h:", self.desired_width, self.desired_height,
                "fx fy cx cy:", fx, fy, cx, cy,
            )

        intr = np.eye(4, dtype=np.float32)
        intr[0, 0] = fx
        intr[1, 1] = fy
        intr[0, 2] = cx
        intr[1, 2] = cy

        # Pose: identity (unless you choose to use /zed/zed_node/pose)
        pose = np.eye(4, dtype=np.float32)

        # Convert to torch on requested device
        color_t = torch.from_numpy(rgb).to(self.device)
        depth_t = torch.from_numpy(depth).to(self.device)
        intr_t = torch.from_numpy(intr).to(self.device)
        pose_t = torch.from_numpy(pose).to(self.device)

        return color_t, depth_t, intr_t, pose_t

    def close(self):
        """Clean shutdown of ROS node/context."""
        try:
            if hasattr(self, "node") and self.node is not None:
                self.node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def __del__(self):
        # best-effort cleanup
        self.close()
