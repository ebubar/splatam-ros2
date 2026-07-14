"""
Timestamp-based odometry buffer with SLERP interpolation.

Motivation (network robustness): the original live node fed odometry into a
3-way ``ApproximateTimeSynchronizer([rgb, depth, odom], slop=0.15)``. Over a
bandwidth-limited ROS2/Zenoh link the tiny odom message arrives fresh while the
large RGB/depth messages arrive late, so the synchronizer pairs a *fresh* pose
with a *stale* image (anything within 150 ms passes). The Gaussians are then
placed with a pose that does not match the pixels -> smearing that looks like
odometry drift but is really a timestamp-association bug.

Fix: sync only RGB+depth (2-way), buffer odometry here at full rate, and look up
the pose at the *image* header timestamp via SLERP (rotation) + lerp
(translation). The pose then always matches the image regardless of odom jitter.

This module is pure NumPy so it runs cheaply on the ROS callback thread with no
torch/CUDA involvement.
"""

import bisect
import threading

import numpy as np


def quat_xyzw_to_R(q):
    """Rotation matrix (3x3) from a quaternion in (x, y, z, w) order.

    Matches the formula used by the existing ``odom_to_c2w`` helper so the
    interpolated pose is bit-compatible with the legacy path at sample points.
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _slerp(q0, q1, t):
    """Spherical linear interpolation between two (x,y,z,w) quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)

    dot = float(np.dot(q0, q1))
    # Take the shorter arc.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        # Nearly parallel -> linear interpolate and renormalize.
        q = q0 + t * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


class PoseBuffer:
    """Thread-safe time-ordered buffer of odometry poses with interpolation."""

    def __init__(self, maxlen=400, max_extrapolation_s=0.05,
                 discontinuity_vel_mps=3.0, discontinuity_omega_rps=6.0):
        self._maxlen = int(maxlen)
        self._max_extrap = float(max_extrapolation_s)
        self._disc_vel = float(discontinuity_vel_mps)
        self._disc_omega = float(discontinuity_omega_rps)

        self._lock = threading.Lock()
        self._t = []       # sorted list of timestamps (sec)
        self._pos = []     # list of np.array([x, y, z])
        self._quat = []    # list of np.array([x, y, z, w])
        self.last_discontinuity = False  # set when the most recent add() jumped

    def add(self, stamp_sec, position, quat_xyzw):
        """Insert an odometry sample. Flags large jumps as discontinuities."""
        stamp_sec = float(stamp_sec)
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        quat = quat / (np.linalg.norm(quat) + 1e-12)

        with self._lock:
            disc = False
            if self._t:
                dt = stamp_sec - self._t[-1]
                if dt > 1e-6:
                    lin = np.linalg.norm(position - self._pos[-1]) / dt
                    # angular speed via quaternion dot -> angle
                    dot = abs(float(np.dot(quat, self._quat[-1])))
                    dot = min(1.0, max(-1.0, dot))
                    ang = 2.0 * np.arccos(dot) / dt
                    disc = (lin > self._disc_vel) or (ang > self._disc_omega)
            self.last_discontinuity = disc

            # Insert keeping timestamps sorted (they usually arrive monotonically).
            idx = bisect.bisect_left(self._t, stamp_sec)
            self._t.insert(idx, stamp_sec)
            self._pos.insert(idx, position)
            self._quat.insert(idx, quat)

            while len(self._t) > self._maxlen:
                self._t.pop(0)
                self._pos.pop(0)
                self._quat.pop(0)

    def is_stale(self, stamp_sec):
        """True if stamp is outside the buffered range beyond the extrap window."""
        with self._lock:
            if not self._t:
                return True
            return (stamp_sec < self._t[0] - self._max_extrap or
                    stamp_sec > self._t[-1] + self._max_extrap)

    def lookup(self, stamp_sec):
        """Interpolate the c2w pose at ``stamp_sec``.

        Returns (c2w[4x4] float64, quality) where quality is a dict with:
          'ok'          : bool (a usable pose was produced)
          'extrapolated': bool (stamp was just outside the buffered range)
          'stale'       : bool (stamp was beyond the extrapolation window)
          'gap_s'       : float (interpolation gap between bracketing samples)
        Returns (None, quality) when no usable pose exists.
        """
        stamp_sec = float(stamp_sec)
        with self._lock:
            n = len(self._t)
            quality = {'ok': False, 'extrapolated': False, 'stale': False, 'gap_s': 0.0}
            if n == 0:
                quality['stale'] = True
                return None, quality

            # Before the buffer.
            if stamp_sec <= self._t[0]:
                if self._t[0] - stamp_sec > self._max_extrap:
                    quality['stale'] = True
                    return None, quality
                quality.update(ok=True, extrapolated=(stamp_sec < self._t[0]))
                return self._pose_at_index(0), quality

            # After the buffer.
            if stamp_sec >= self._t[-1]:
                if stamp_sec - self._t[-1] > self._max_extrap:
                    quality['stale'] = True
                    return None, quality
                quality.update(ok=True, extrapolated=(stamp_sec > self._t[-1]))
                return self._pose_at_index(n - 1), quality

            # Interpolate between the two bracketing samples.
            hi = bisect.bisect_right(self._t, stamp_sec)
            lo = hi - 1
            t0, t1 = self._t[lo], self._t[hi]
            span = t1 - t0
            frac = 0.0 if span <= 1e-9 else (stamp_sec - t0) / span

            pos = (1.0 - frac) * self._pos[lo] + frac * self._pos[hi]
            quat = _slerp(self._quat[lo], self._quat[hi], frac)

            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = quat_xyzw_to_R(quat)
            c2w[:3, 3] = pos
            quality.update(ok=True, gap_s=float(span))
            return c2w, quality

    def _pose_at_index(self, i):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = quat_xyzw_to_R(self._quat[i])
        c2w[:3, 3] = self._pos[i]
        return c2w
