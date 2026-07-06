"""Unit tests for the accuracy-critical pose math (utils/pose_utils.py).

These run with numpy only -- no torch, ROS, or camera -- so the VIO->pose
pipeline can be regression-tested anywhere.
"""

import numpy as np
import pytest

from utils import pose_utils as pu


def _rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def _old_odom_formula(pos, quat_xyzw):
    """Replicates the original in-node torch formula (in numpy) so we can prove
    the refactor did not change behaviour."""
    x, y, z, w = quat_xyzw
    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def test_rotmat_is_orthonormal():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = pu.quat_wxyz_to_rotmat(_rand_quat(rng))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_quat_rotmat_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = _rand_quat(rng)
        R = pu.quat_wxyz_to_rotmat(q)
        q2 = pu.rotmat_to_quat_wxyz(R)
        # q and -q represent the same rotation; compare the rotation matrices.
        assert np.allclose(pu.quat_wxyz_to_rotmat(q2), R, atol=1e-8)


def test_rotmat_quat_roundtrip_identity_and_180():
    for R in [np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0])]:
        q = pu.rotmat_to_quat_wxyz(R)
        assert np.allclose(pu.quat_wxyz_to_rotmat(q), R, atol=1e-9)


def test_pose_from_position_quat_matches_old_formula():
    rng = np.random.default_rng(2)
    for _ in range(50):
        q = _rand_quat(rng)          # wxyz
        quat_xyzw = [q[1], q[2], q[3], q[0]]
        pos = rng.normal(size=3)
        new = pu.pose_from_position_quat(pos, quat_xyzw)
        old = _old_odom_formula(pos, quat_xyzw)
        assert np.allclose(new, old, atol=1e-9)


def test_invert_transform_is_true_inverse():
    rng = np.random.default_rng(3)
    for _ in range(50):
        T = pu.pose_from_position_quat(rng.normal(size=3), _rand_quat(rng)[[1, 2, 3, 0]])
        assert np.allclose(pu.invert_transform(T) @ T, np.eye(4), atol=1e-9)
        # closed-form inverse matches the general linear inverse
        assert np.allclose(pu.invert_transform(T), np.linalg.inv(T), atol=1e-9)


def test_relative_pose_self_is_identity():
    rng = np.random.default_rng(4)
    T = pu.pose_from_position_quat(rng.normal(size=3), _rand_quat(rng)[[1, 2, 3, 0]])
    assert np.allclose(pu.relative_pose(T, T), np.eye(4), atol=1e-9)


def test_zed_static_transform_is_rigid():
    T = pu.zed_link_to_left_optical()
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_camera_center_recovered_from_w2c():
    rng = np.random.default_rng(5)
    for _ in range(50):
        center = rng.normal(size=3)
        c2w = pu.pose_from_position_quat(center, _rand_quat(rng)[[1, 2, 3, 0]])
        w2c = pu.invert_transform(c2w)
        rec = pu.camera_centers_from_w2c(w2c)[0]
        assert np.allclose(rec, center, atol=1e-9)


def test_first_frame_relative_pose_gives_identity_w2c():
    """Frame 0 must map to identity w2c so the splat anchors to the first cam."""
    rng = np.random.default_rng(6)
    first = pu.odom_to_optical_c2w(rng.normal(size=3), _rand_quat(rng)[[1, 2, 3, 0]])
    rel = pu.relative_pose(first, first)
    w2c = pu.invert_transform(rel)
    assert np.allclose(w2c, np.eye(4), atol=1e-9)


def test_pure_translation_preserves_orientation():
    """If the ZED only translates, the relative rotation stays identity."""
    q = [0.0, 0.0, 0.0, 1.0]  # identity xyzw
    first = pu.odom_to_optical_c2w([0, 0, 0], q)
    later = pu.odom_to_optical_c2w([1.0, 2.0, 3.0], q)
    rel = pu.relative_pose(first, later)
    assert np.allclose(rel[:3, :3], np.eye(3), atol=1e-9)
    # translation of 1,2,3 in body maps through the fixed optical transform
    assert not np.allclose(rel[:3, 3], 0.0)
