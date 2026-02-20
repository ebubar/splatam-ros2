#!/usr/bin/env python3
"""ROS2 node: sanitize ZED compressed topics and forward only valid frames.

Subscribes to ZED pose + compressed RGB/depth, validates frames (decode + basic checks),
and republishes to `/splatam/zed/...` topics for downstream processing.

Basic validation:
- Minimum message size threshold
- Successful image decode via OpenCV
- For depth: minimum non-zero pixel ratio

This file is intended to be run inside the project's ROS2-capable container.
"""
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped

try:
    import cv2
except Exception:
    cv2 = None


class ZedSanitizerNode(Node):
    def __init__(self):
        super().__init__('zed_sanitizer')

        # Parameters
        self.declare_parameter('pose_in', '/zed/zed_node/pose')
        self.declare_parameter('rgb_in', '/zed/zed_node/rgb/color/rect/image/compressed')
        self.declare_parameter('depth_in', '/zed/zed_node/depth/depth_registered/compressedDepth')
        self.declare_parameter('pose_out', '/splatam/zed/pose')
        self.declare_parameter('rgb_out', '/splatam/zed/rgb/compressed')
        self.declare_parameter('depth_out', '/splatam/zed/depth/compressedDepth')
        self.declare_parameter('min_size_bytes', 512)  # small compressed frames are suspicious
        self.declare_parameter('depth_nonzero_ratio', 0.05)
        self.declare_parameter('log_interval_s', 5.0)

        pose_in = self.get_parameter('pose_in').get_parameter_value().string_value
        rgb_in = self.get_parameter('rgb_in').get_parameter_value().string_value
        depth_in = self.get_parameter('depth_in').get_parameter_value().string_value
        pose_out = self.get_parameter('pose_out').get_parameter_value().string_value
        rgb_out = self.get_parameter('rgb_out').get_parameter_value().string_value
        depth_out = self.get_parameter('depth_out').get_parameter_value().string_value

        self.min_size = self.get_parameter('min_size_bytes').get_parameter_value().integer_value
        self.depth_nonzero_ratio = self.get_parameter('depth_nonzero_ratio').get_parameter_value().double_value
        self.log_interval = self.get_parameter('log_interval_s').get_parameter_value().double_value

        self.pose_pub = self.create_publisher(PoseStamped, pose_out, 10)
        self.rgb_pub = self.create_publisher(CompressedImage, rgb_out, 2)
        self.depth_pub = self.create_publisher(CompressedImage, depth_out, 2)

        self.create_subscription(PoseStamped, pose_in, self.pose_cb, 50)
        self.create_subscription(CompressedImage, rgb_in, self.rgb_cb, 2)
        self.create_subscription(CompressedImage, depth_in, self.depth_cb, 2)

        self.stats = {
            'rgb_seen': 0,
            'rgb_forwarded': 0,
            'rgb_dropped': 0,
            'depth_seen': 0,
            'depth_forwarded': 0,
            'depth_dropped': 0,
            'pose_seen': 0,
            'pose_forwarded': 0,
        }

        self.lock = threading.Lock()
        self.last_log = time.time()

        self.get_logger().info('ZED sanitizer started')

    def pose_cb(self, msg: PoseStamped):
        with self.lock:
            self.stats['pose_seen'] += 1
            # Forward pose unchanged
            self.pose_pub.publish(msg)
            self.stats['pose_forwarded'] += 1

    def rgb_cb(self, msg: CompressedImage):
        with self.lock:
            self.stats['rgb_seen'] += 1

        if not self._valid_compressed(msg, is_depth=False):
            with self.lock:
                self.stats['rgb_dropped'] += 1
            return

        # forward original compressed message (keeps bandwidth low)
        self.rgb_pub.publish(msg)
        with self.lock:
            self.stats['rgb_forwarded'] += 1

        self._maybe_log()

    def depth_cb(self, msg: CompressedImage):
        with self.lock:
            self.stats['depth_seen'] += 1

        if not self._valid_compressed(msg, is_depth=True):
            with self.lock:
                self.stats['depth_dropped'] += 1
            return

        self.depth_pub.publish(msg)
        with self.lock:
            self.stats['depth_forwarded'] += 1

        self._maybe_log()

    def _valid_compressed(self, msg: CompressedImage, is_depth: bool) -> bool:
        # quick size check
        size = len(msg.data)
        fmt = getattr(msg, 'format', '')
        if size < self.min_size:
            self.get_logger().warn(f'dropping compressed msg: too small ({size} bytes) format="{fmt}"')
            return False

        # log brief info for diagnostics
        if size < self.min_size * 4:
            self.get_logger().debug(f'compressed msg size={size} format="{fmt}"')

        if cv2 is None:
            # If OpenCV not available, trust message size only
            return True

        try:
            # ZED compressed depth messages sometimes include a small header
            # before a PNG/JPEG payload (format contains 'compressedDepth').
            arr = None
            # normalize to bytes for consistent processing (handles array.array)
            try:
                raw = msg.data
                if not isinstance(raw, (bytes, bytearray)):
                    raw = bytes(raw)
            except Exception:
                raw = bytes(msg.data)

            if is_depth and ('compressedDepth' in fmt or '32FC1' in fmt):
                data = raw
                # look for PNG or JPEG signature inside payload
                png_sig = b'\x89PNG'
                jpe_sig = b'\xff\xd8\xff'
                idx = data.find(png_sig)
                if idx == -1:
                    idx = data.find(jpe_sig)
                if idx != -1:
                    arr = np.frombuffer(data[idx:], dtype=np.uint8)
                else:
                    # fallback: try whole buffer
                    arr = np.frombuffer(data, dtype=np.uint8)
            else:
                arr = np.frombuffer(raw, dtype=np.uint8)

            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            # log a short preview of payload for debugging corrupted frames
            try:
                preview = msg.data[:16]
                hexpreview = ' '.join(f"{b:02x}" for b in preview)
            except Exception:
                hexpreview = '<unavailable>'
            self.get_logger().warn(f'cv2 decode exception: {e}; size={size} format="{fmt}" preview={hexpreview}')
            return False

        if img is None:
            # For depth payloads try interpreting as raw float32 array
            if is_depth:
                try:
                    raw = msg.data if isinstance(msg.data, (bytes, bytearray)) else bytes(msg.data)
                    f32 = np.frombuffer(raw, dtype=np.float32)
                    if f32.size > 0:
                        nonzero = np.count_nonzero(f32)
                        ratio = float(nonzero) / float(f32.size)
                        if ratio < self.depth_nonzero_ratio:
                            self.get_logger().warn(f'depth raw-f32 low nonzero ratio {ratio:.3f}; size={size} format="{fmt}"')
                            return False
                        else:
                            return True
                except Exception:
                    pass

            try:
                preview = msg.data[:16] if isinstance(msg.data, (bytes, bytearray)) else bytes(msg.data)[:16]
                hexpreview = ' '.join(f"{b:02x}" for b in preview)
            except Exception:
                hexpreview = '<unavailable>'
            self.get_logger().warn(f'cv2 failed to decode compressed image; size={size} format="{fmt}" preview={hexpreview}')
            return False

        # basic sanity shape check
        if img.size == 0:
            self.get_logger().warn('decoded image has zero size')
            return False

        if is_depth:
            # depth images often encoded as single-channel 16-bit PNG; check non-zero ratio
            nonzero = np.count_nonzero(img)
            ratio = float(nonzero) / float(img.size)
            if ratio < self.depth_nonzero_ratio:
                self.get_logger().warn(f'dropping depth: low nonzero ratio {ratio:.3f}')
                return False

        return True

    def _maybe_log(self):
        now = time.time()
        if now - self.last_log < self.log_interval:
            return
        with self.lock:
            s = self.stats.copy()
        self.get_logger().info(
            f"stats rgb seen={s['rgb_seen']} fwd={s['rgb_forwarded']} dropped={s['rgb_dropped']} "
            f"depth seen={s['depth_seen']} fwd={s['depth_forwarded']} dropped={s['depth_dropped']} "
            f"pose seen={s['pose_seen']} fwd={s['pose_forwarded']}"
        )
        self.last_log = now


def main(args=None):
    rclpy.init(args=args)
    node = ZedSanitizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
