# utils/live_renderer.py

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Optional

import cv2
import numpy as np


@dataclass
class LiveRendererConfig:
    live_cam: bool = False
    live_depth: bool = False
    live_splat: bool = False
    max_fps: float = 30.0

    win_cam: str = "Live Camera"
    win_depth: str = "Live Depth"
    win_splat: str = "Live Splat"


class LiveRenderer:
    """
    Simple OpenCV live viewer for:
      - camera (BGR8)
      - depth (float meters)
      - splat preview (BGR8)

    Press 'q' or ESC in ANY window to quit.
    """

    def __init__(self, cfg: LiveRendererConfig):
        self.cfg = cfg
        self._last_show_t = 0.0
        self._min_dt = 1.0 / max(cfg.max_fps, 1e-6)

        self._cam_bgr: Optional[np.ndarray] = None
        self._depth_m: Optional[np.ndarray] = None
        self._splat_bgr: Optional[np.ndarray] = None

        self._windows_created = False

    def update_cam(self, cam_bgr: np.ndarray) -> None:
        """
        cam_bgr: uint8 BGR image (H,W,3)
        """
        if cam_bgr is None:
            return
        self._cam_bgr = cam_bgr

    def update_depth(self, depth_m: np.ndarray) -> None:
        """
        depth_m: float meters (H,W) or (H,W,1)
        """
        if depth_m is None:
            return
        if depth_m.ndim == 3 and depth_m.shape[2] == 1:
            depth_m = depth_m[..., 0]
        self._depth_m = depth_m

    def update_splat_preview(self, splat_bgr: np.ndarray) -> None:
        """
        splat_bgr: uint8 BGR image (H,W,3)
        """
        if splat_bgr is None:
            return
        self._splat_bgr = splat_bgr

    def _ensure_windows(self) -> None:
        if self._windows_created:
            return

        if self.cfg.live_cam:
            cv2.namedWindow(self.cfg.win_cam, cv2.WINDOW_NORMAL)

        if self.cfg.live_depth:
            cv2.namedWindow(self.cfg.win_depth, cv2.WINDOW_NORMAL)

        if self.cfg.live_splat:
            cv2.namedWindow(self.cfg.win_splat, cv2.WINDOW_NORMAL)

        self._windows_created = True

    def _render_depth_vis(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Convert depth (meters) to a visible 8-bit colormap image.
        """
        d = depth_m.copy().astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d[d < 0.0] = 0.0

        # Ignore zeros in normalization to avoid everything turning dark
        valid = d > 0.0
        if not np.any(valid):
            vis = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
            return vis

        d_valid = d[valid]
        lo = float(np.percentile(d_valid, 5))
        hi = float(np.percentile(d_valid, 95))
        hi = max(hi, lo + 1e-6)

        dn = (d - lo) / (hi - lo)
        dn = np.clip(dn, 0.0, 1.0)
        dn8 = (dn * 255.0).astype(np.uint8)

        vis = cv2.applyColorMap(dn8, cv2.COLORMAP_JET)
        return vis

    def tick(self) -> bool:
        """
        Show windows at <= max_fps.
        Returns False if user requested quit.
        """
        if not (self.cfg.live_cam or self.cfg.live_depth or self.cfg.live_splat):
            return True  # nothing to show

        self._ensure_windows()

        now = time.time()
        if (now - self._last_show_t) < self._min_dt:
            # Still need waitKey so OpenCV windows stay responsive
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return False
            return True

        self._last_show_t = now

        if self.cfg.live_cam and (self._cam_bgr is not None):
            cv2.imshow(self.cfg.win_cam, self._cam_bgr)

        if self.cfg.live_depth and (self._depth_m is not None):
            depth_vis = self._render_depth_vis(self._depth_m)
            cv2.imshow(self.cfg.win_depth, depth_vis)

        if self.cfg.live_splat and (self._splat_bgr is not None):
            cv2.imshow(self.cfg.win_splat, self._splat_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        return True

    def close(self) -> None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass