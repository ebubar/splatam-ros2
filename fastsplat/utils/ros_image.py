"""Pure numpy helpers for converting ROS sensor messages to arrays.

These are lifted from ``scripts/zed2i_splat_live.py`` but kept dependency-free
(numpy + cv2 only, no torch, no SplaTAM imports) so the ingest node stays light.
"""

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 always present at runtime
    cv2 = None


def caminfo_to_K(cam_info):
    """CameraInfo -> 3x3 intrinsics matrix."""
    return np.array(cam_info.k, dtype=np.float32).reshape(3, 3)


def depth_to_meters(depth_cv, encoding):
    """Convert a raw depth image to float32 metres based on its encoding."""
    enc = (encoding or "").lower()
    if enc in ["32fc1", "32fc"]:
        return depth_cv.astype(np.float32)
    if enc in ["16uc1", "16uc", "mono16"]:
        return depth_cv.astype(np.float32) / 1000.0
    return depth_cv.astype(np.float32)


def ros_rgb_to_rgb(rgb_cv, encoding):
    """Convert a ROS image array (any common encoding) to HxWx3 uint8 RGB."""
    enc = (encoding or "").lower()
    if rgb_cv.ndim == 2:
        return cv2.cvtColor(rgb_cv, cv2.COLOR_GRAY2RGB)
    if rgb_cv.dtype != np.uint8:
        rgb_cv = np.clip(rgb_cv, 0, 255).astype(np.uint8)
    if rgb_cv.shape[2] == 4:
        if "bgra" in enc:
            return cv2.cvtColor(rgb_cv, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(rgb_cv, cv2.COLOR_RGBA2RGB)
    if rgb_cv.shape[2] == 3:
        if "bgr" in enc:
            return cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
        return rgb_cv
    raise ValueError(f"Unexpected RGB image shape: {rgb_cv.shape}")


def quat_xyzw_to_rotmat(x, y, z, w):
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix (numpy)."""
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float32,
    )


def odom_to_c2w(odom_msg):
    """nav_msgs/Odometry -> 4x4 camera-to-world matrix (numpy, ROS body frame)."""
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = quat_xyzw_to_rotmat(q.x, q.y, q.z, q.w)
    c2w[:3, 3] = [p.x, p.y, p.z]
    return c2w


# ZED body (ROS REP-103) -> left camera optical frame, from the ZED /tf_static.
# Same transform used by scripts/zed2i_splat_live.py::zed_link_to_left_optical.
def zed_link_to_left_optical():
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [-0.01, 0.06, 0.015]
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return T


def odom_to_optical_c2w(odom_msg):
    """ZED odom -> camera-to-world in the OpenCV optical convention.

    MapAnything expects OpenCV cam2world (x right, y down, z forward), which is
    exactly the ZED left optical frame.
    """
    return odom_to_c2w(odom_msg) @ zed_link_to_left_optical()
