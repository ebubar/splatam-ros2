"""
Pluggable camera-pose source.

The map is built identically regardless of where per-frame poses come from, so
the pose provider sits behind one interface. Two implementations:

  * ``ZedOdomPoseSource`` (default) -- interpolates the ZED SDK odometry at the
    image timestamp (network-robust; see utils/pose_buffer.py) and expresses it
    in the left-optical frame relative to the first frame. No model-license
    exposure; real-time.

  * ``MapAnythingPoseSource`` -- runs Facebook's MapAnything **Apache-2.0**
    checkpoint (``facebook/map-anything-apache``) as a feed-forward SfM front-end
    over a window of frames to regress per-frame poses (+ optional intrinsics /
    point init). This is the defense-license-clean learned option; VGGT is
    intentionally NOT offered here because its license excludes military use.
    MapAnything is windowed / near-online, not per-frame real-time, so it is a
    comparison / bootstrap front-end rather than the live path.

All poses are returned as world->cam (w2c) in the first-frame world frame so
everything downstream (mapping, keyframes, save) is unchanged.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.pose_buffer import PoseBuffer


# Static transform zed_camera_link -> zed_left_camera_frame_optical, as a NumPy
# constant mirroring utils/zed link math in scripts/zed2i_splat_live.py.
_T_LINK_LEFT_OPTICAL = np.eye(4, dtype=np.float64)
_T_LINK_LEFT_OPTICAL[:3, :3] = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0],
], dtype=np.float64)
_T_LINK_LEFT_OPTICAL[:3, 3] = np.array([-0.01, 0.06, 0.015], dtype=np.float64)


@dataclass
class PoseResult:
    ok: bool
    w2c: Optional[np.ndarray] = None          # 4x4 float32, first-frame world frame
    K: Optional[np.ndarray] = None            # 3x3 intrinsics if the source provides them
    pts_init: Optional[np.ndarray] = None     # optional [M, 6] xyz+rgb init cloud
    quality: dict = field(default_factory=dict)


class PoseSource:
    """Interface: subclasses implement get_w2c()."""

    def get_w2c(self, stamp_sec, frame=None) -> PoseResult:
        raise NotImplementedError


class ZedOdomPoseSource(PoseSource):
    """ZED odometry, interpolated at the image timestamp and made first-frame relative."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.buf = PoseBuffer(
            maxlen=int(cfg.get('odom_buffer_len', 400)),
            max_extrapolation_s=float(cfg.get('max_extrapolation_s', 0.05)),
            discontinuity_vel_mps=float(cfg.get('discontinuity_vel_mps', 3.0)),
            discontinuity_omega_rps=float(cfg.get('discontinuity_omega_rps', 6.0)),
        )
        self._first_optical_c2w_inv = None

    def add_odom(self, stamp_sec, position, quat_xyzw):
        """Feed a raw Odometry sample (called from the ROS callback, cheap)."""
        self.buf.add(stamp_sec, position, quat_xyzw)

    def is_ready(self, stamp_sec):
        return not self.buf.is_stale(stamp_sec)

    def get_w2c(self, stamp_sec, frame=None) -> PoseResult:
        c2w_link, quality = self.buf.lookup(stamp_sec)
        quality = dict(quality)
        quality['discontinuity'] = bool(self.buf.last_discontinuity)
        if c2w_link is None:
            return PoseResult(ok=False, quality=quality)

        optical_c2w = c2w_link @ _T_LINK_LEFT_OPTICAL

        # Pin the first processed frame to identity; all later poses are relative.
        if self._first_optical_c2w_inv is None:
            self._first_optical_c2w_inv = np.linalg.inv(optical_c2w)

        rel_c2w = self._first_optical_c2w_inv @ optical_c2w
        w2c = np.linalg.inv(rel_c2w).astype(np.float32)
        return PoseResult(ok=True, w2c=w2c, quality=quality)


class MapAnythingPoseSource(PoseSource):
    """Feed-forward SfM poses from MapAnything (Apache-2.0 checkpoint).

    This is a windowed/near-online front-end used to A/B against ZED odometry.
    The heavy model + weights are optional and imported lazily; if unavailable,
    a clear error is raised so the realtime core stays lean.
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.model_name = cfg.get('mapanything_model', 'facebook/map-anything-apache')
        self.window = int(cfg.get('mapanything_window', 16))
        self.device = cfg.get('primary_device', 'cuda:0')
        self.use_depth_prior = bool(cfg.get('mapanything_use_depth_prior', True))
        self._model = None
        self._frames = []           # buffered (stamp, rgb, depth, K)
        self._poses = {}            # stamp -> w2c once computed
        self._first_c2w_inv = None

    def add_frame(self, stamp_sec, rgb, depth=None, K=None):
        self._frames.append((float(stamp_sec), rgb, depth, K))
        if len(self._frames) > self.window:
            self._frames.pop(0)

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            # Lazy import; only needed when this source is actually selected.
            from mapanything.models import MapAnything  # type: ignore
        except Exception as exc:  # pragma: no cover - optional heavy dependency
            raise RuntimeError(
                "MapAnythingPoseSource requires the 'mapanything' package and the "
                "'facebook/map-anything-apache' (Apache-2.0) weights. Install the "
                "optional extra and download the Apache checkpoint. VGGT is not "
                "supported here due to its military-use license exclusion."
            ) from exc
        self._model = MapAnything.from_pretrained(self.model_name).to(self.device).eval()

    def _run_window(self):
        """Run MapAnything over the buffered window, filling self._poses.

        Implementation note: MapAnything returns per-view extrinsics (up to a
        global metric frame). We express each as w2c relative to the first
        processed view so it matches the ZED path's convention. The exact tensor
        plumbing depends on the installed mapanything API version and is wired in
        when the optional dependency is present.
        """
        self._ensure_model()
        raise NotImplementedError(
            "MapAnything windowed inference plumbing is a stub pending the optional "
            "'mapanything' dependency; the ZED odom pose source is the live default."
        )

    def get_w2c(self, stamp_sec, frame=None) -> PoseResult:
        stamp_sec = float(stamp_sec)
        if stamp_sec in self._poses:
            return PoseResult(ok=True, w2c=self._poses[stamp_sec],
                              quality={'source': 'mapanything'})
        # Not yet computed: caller should have buffered enough frames and invoked
        # _run_window(). Report not-ready rather than blocking the ROS thread.
        return PoseResult(ok=False, quality={'source': 'mapanything', 'not_ready': True})


def make_pose_source(cfg) -> PoseSource:
    """Factory keyed on cfg['pose_source'] ('zed_odom' | 'mapanything')."""
    name = str(cfg.get('pose_source', 'zed_odom')).lower()
    if name in ('zed_odom', 'zed', 'odom'):
        return ZedOdomPoseSource(cfg)
    if name in ('mapanything', 'map_anything', 'ma'):
        return MapAnythingPoseSource(cfg)
    raise ValueError(f"Unknown pose_source '{name}' (expected 'zed_odom' or 'mapanything')")
