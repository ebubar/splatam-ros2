"""Pure-NumPy pose / SE(3) math for the live ZED SplaTAM pipeline.

This is the accuracy-critical core of the "most accurate poses" path: turning
ZED VIO odometry into the camera-to-world transforms SplaTAM tracks. It lives
here, free of ROS2/torch imports, so it can be unit-tested on any machine (see
``tests/test_pose_utils.py``) instead of being buried inside the ROS node where
it could only be exercised with the camera plugged in.

Conventions:
* Quaternions are stored **wxyz** (SplaTAM / gaussian-splatting order).
* ROS geometry_msgs quaternions are **xyzw**; use :func:`pose_from_position_quat`.
* Transforms are 4x4 homogeneous matrices. ``c2w`` maps camera->world, ``w2c``
  maps world->camera. For a rigid transform the inverse is closed-form.
"""

from __future__ import annotations

import numpy as np


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        # Degenerate; fall back to identity rotation.
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Unit(-ised) quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = normalize_quat(q_wxyz)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (w, x, y, z), normalised.

    Branch-selecting form (Shepperd's method) for numerical stability -- matches
    the torch implementation the live node used previously.
    """
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return normalize_quat(np.array([w, x, y, z]))


def pose_from_position_quat(position_xyz, quat_xyzw) -> np.ndarray:
    """Build a 4x4 transform from a ROS position + quaternion (xyzw order)."""
    x, y, z, w = [float(v) for v in quat_xyzw]
    R = quat_wxyz_to_rotmat(np.array([w, x, y, z]))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid 4x4 transform: [R|t] -> [R^T | -R^T t]."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def relative_pose(reference_c2w: np.ndarray, current_c2w: np.ndarray) -> np.ndarray:
    """current expressed relative to reference: inv(reference) @ current."""
    return invert_transform(reference_c2w) @ np.asarray(current_c2w, dtype=np.float64)


# Static transform ZED body ``zed_camera_link`` -> left camera optical frame,
# taken from the ZED wrapper's /tf_static:
#   link -> center: (0, 0, 0.015);  center -> left_camera_frame: (-0.01, 0.06, 0)
#   left_camera_frame -> optical: quaternion (x=0.5, y=-0.5, z=0.5, w=-0.5)
def zed_link_to_left_optical() -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([-0.01, 0.06, 0.015])
    T[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    return T


def odom_to_optical_c2w(position_xyz, quat_xyzw) -> np.ndarray:
    """Full ZED VIO odom (body link) -> left-camera-optical camera-to-world."""
    link_c2w = pose_from_position_quat(position_xyz, quat_xyzw)
    return link_c2w @ zed_link_to_left_optical()


def camera_centers_from_w2c(w2c_stack: np.ndarray) -> np.ndarray:
    """Camera centres in world frame from a stack of 4x4 world-to-camera mats.

    For w2c = [R | t], the camera centre C = -R^T t.
    """
    w2c = np.asarray(w2c_stack, dtype=np.float64)
    if w2c.ndim == 2:
        w2c = w2c[None]
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)
