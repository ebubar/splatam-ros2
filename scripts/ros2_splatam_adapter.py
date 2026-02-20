#!/usr/bin/env python3
"""Adapter node: convert sanitized compressed topics into raw Image + Pose for SplaTAM.

Subscribes:
 - /splatam/zed/rgb/compressed       (sensor_msgs/CompressedImage)
 - /splatam/zed/depth/compressedDepth (sensor_msgs/CompressedImage)
 - /splatam/zed/pose                 (geometry_msgs/PoseStamped)

Publishes:
 - /splatam/input/image_rgb (sensor_msgs/Image, bgr8)
 - /splatam/input/image_depth (sensor_msgs/Image, 32FC1)
 - /splatam/input/pose (geometry_msgs/PoseStamped)

This keeps the downstream SplaTAM code unaware of compression and lets you toggle live vs bag playback.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped
import numpy as np
import threading

try:
    import cv2
except Exception:
    cv2 = None


class SplatamAdapter(Node):
    def __init__(self):
        super().__init__('splatam_adapter')

        self.declare_parameter('rgb_in', '/splatam/zed/rgb/compressed')
        self.declare_parameter('depth_in', '/splatam/zed/depth/compressedDepth')
        self.declare_parameter('pose_in', '/splatam/zed/pose')
        self.declare_parameter('rgb_out', '/splatam/input/image_rgb')
        self.declare_parameter('depth_out', '/splatam/input/image_depth')
        self.declare_parameter('pose_out', '/splatam/input/pose')

        rgb_in = self.get_parameter('rgb_in').get_parameter_value().string_value
        depth_in = self.get_parameter('depth_in').get_parameter_value().string_value
        pose_in = self.get_parameter('pose_in').get_parameter_value().string_value
        rgb_out = self.get_parameter('rgb_out').get_parameter_value().string_value
        depth_out = self.get_parameter('depth_out').get_parameter_value().string_value
        pose_out = self.get_parameter('pose_out').get_parameter_value().string_value

        self.rgb_pub = self.create_publisher(Image, rgb_out, 2)
        self.depth_pub = self.create_publisher(Image, depth_out, 2)
        self.pose_pub = self.create_publisher(PoseStamped, pose_out, 10)

        self.create_subscription(CompressedImage, rgb_in, self.rgb_cb, 2)
        self.create_subscription(CompressedImage, depth_in, self.depth_cb, 2)
        self.create_subscription(PoseStamped, pose_in, self.pose_cb, 50)

        self.lock = threading.Lock()
        self.get_logger().info('SplaTAM adapter started')

    def rgb_cb(self, msg: CompressedImage):
        # decode compressed to BGR8 Image
        if cv2 is None:
            self.get_logger().error('cv2 not available; cannot decode RGB')
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().warn('failed to decode RGB compressed frame')
                return
            img_msg = Image()
            img_msg.header = msg.header
            img_msg.height = bgr.shape[0]
            img_msg.width = bgr.shape[1]
            img_msg.encoding = 'bgr8'
            img_msg.step = bgr.shape[1] * 3
            img_msg.data = bgr.tobytes()
            self.rgb_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f'exception decoding rgb: {e}')

    def depth_cb(self, msg: CompressedImage):
        # try to extract PNG/JPEG embedded payload first
        if cv2 is None:
            self.get_logger().error('cv2 not available; cannot decode depth')
            return
        try:
            raw = msg.data if isinstance(msg.data, (bytes, bytearray)) else bytes(msg.data)
            # search for PNG/JPEG
            png_sig = b'\x89PNG'
            jpe_sig = b'\xff\xd8\xff'
            idx = raw.find(png_sig)
            if idx == -1:
                idx = raw.find(jpe_sig)
            if idx != -1:
                payload = raw[idx:]
                arr = np.frombuffer(payload, dtype=np.uint8)
                dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if dec is None:
                    # fallback to interpreting raw as float32
                    f32 = np.frombuffer(raw, dtype=np.float32)
                    self._publish_depth_array(f32, msg.header)
                    return
                # if decoded image is 16-bit or 8-bit depth, convert to 32FC1
                if dec.dtype == np.uint16:
                    depth32 = dec.astype(np.float32)
                elif dec.dtype == np.uint8:
                    depth32 = dec.astype(np.float32)
                else:
                    depth32 = dec.astype(np.float32)
                self._publish_depth_array(depth32, msg.header)
            else:
                # try raw float32 buffer
                f32 = np.frombuffer(raw, dtype=np.float32)
                self._publish_depth_array(f32, msg.header)
        except Exception as e:
            self.get_logger().warn(f'exception decoding depth: {e}')

    def _publish_depth_array(self, arr: np.ndarray, header):
        # arr may be 1D; try to infer shape from header or assume VGA/HD720 sizes
        if arr.size == 0:
            self.get_logger().warn('empty depth payload')
            return
        # Try common resolutions (VGA, HD720). Prefer preserving shape if possible.
        possible_shapes = [(480, 640), (720, 1280), (360, 640)]
        shape = None
        for h, w in possible_shapes:
            if h * w == arr.size:
                shape = (h, w)
                break
        if shape is None:
            # fallback to 1-row image
            shape = (1, arr.size)
        depth_img = arr.reshape(shape)
        img_msg = Image()
        img_msg.header = header
        img_msg.height = depth_img.shape[0]
        img_msg.width = depth_img.shape[1]
        img_msg.encoding = '32FC1'
        img_msg.step = depth_img.shape[1] * 4
        img_msg.data = depth_img.astype(np.float32).tobytes()
        self.depth_pub.publish(img_msg)

    def pose_cb(self, msg: PoseStamped):
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SplatamAdapter()
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
