#!/usr/bin/env python3
"""
utils/live_renderer.py

Simple OpenCV-based live preview helper for:
- camera (BGR image)
- depth (meters -> colormap)
- splat preview (BGR image)

Usage:
    from utils.live_renderer import LiveRenderer, LiveRendererConfig
    live = LiveRenderer(LiveRendererConfig(live_cam=True, live_depth=True, live_splat=False, max_fps=30))

    live.update_camera(cam_bgr)
    live.update_depth(depth_m)
    live.update_splat_preview(splat_bgr)

    if not live.tick():
        # user requested quit (q or esc)
        ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2


@dataclass
class LiveRendererConfig:
    live_cam: bool = False
    live_depth: bool = False
    live_splat: bool = False
    max_fps: float = 30.0

    # Visualization tuning
    depth_max_m: float = 5.0          # clip depth for display
    depth_colormap: int = cv2.COLORMAP_JET

    # Window names
    win_cam: str = "Live Camera"
    win_depth: str = "Live Depth"
    win_splat: str = "Live Splat"


class LiveRenderer:
    """
    OpenCV imshow() based renderer with a FPS limiter and quit handling.

    - Call update_*() with latest frames whenever you have them.
    - Call tick() once per loop/frame to display (if enabled).
    - tick() returns False when user presses 'q' or ESC.
    """

    def __init__(self, cfg: LiveRendererConfig):
        self.cfg = cfg

        self._last_cam_bgr: Optional[np.ndarray] = None
        self._last_depth_m: Optional[np.ndarray] = None
        self._last_splat_bgr: Optional[np.ndarray] = None

        self._last_show_t: float = 0.0
        self._min_dt: float = 0.0 if (cfg.max_fps is None or cfg.max_fps <= 0) else (1.0 / float(cfg.max_fps))

        self._windows_created = False

    # ---------------------------
    # Update methods
    # ---------------------------
    def update_camera(self, cam_bgr: np.ndarray) -> None:
        """cam_bgr: HxWx3 uint8 BGR"""
        self._last_cam_bgr = cam_bgr

    def update_depth(self, depth_m: np.ndarray) -> None:
        """depth_m: HxW float32/float64 meters"""
        self._last_depth_m = depth_m

    def update_splat_preview(self, splat_bgr: np.ndarray) -> None:
        """splat_bgr: HxWx3 uint8 BGR"""
        self._last_splat_bgr = splat_bgr

    # ---------------------------
    # Main display tick
    # ---------------------------
    def tick(self) -> bool:
        """
        Shows any enabled windows, respects max_fps, handles quit.
        Returns:
            False if user requested quit (q or ESC), True otherwise.
        """
        if not (self.cfg.live_cam or self.cfg.live_depth or self.cfg.live_splat):
            return True  # nothing to show

        now = time.time()
        if self._min_dt > 0 and (now - self._last_show_t) < self._min_dt:
            # Still pump events so window doesn't freeze
            key = cv2.waitKey(1) & 0xFF
            return not self._is_quit_key(key)

        self._ensure_windows()

        if self.cfg.live_cam and self._last_cam_bgr is not None:
            cv2.imshow(self.cfg.win_cam, self._safe_bgr(self._last_cam_bgr))

        if self.cfg.live_depth and self._last_depth_m is not None:
            depth_vis = self._depth_to_vis(self._last_depth_m)
            cv2.imshow(self.cfg.win_depth, depth_vis)

        if self.cfg.live_splat and self._last_splat_bgr is not None:
            cv2.imshow(self.cfg.win_splat, self._safe_bgr(self._last_splat_bgr))

        self._last_show_t = now

        key = cv2.waitKey(1) & 0xFF
        if self._is_quit_key(key):
            self.close()
            return False

        return True

    # ---------------------------
    # Cleanup
    # ---------------------------
    def close(self) -> None:
        """Close any windows created by this renderer."""
        if not self._windows_created:
            return

        # Destroy only the windows we might have created
        try:
            if self.cfg.live_cam:
                cv2.destroyWindow(self.cfg.win_cam)
            if self.cfg.live_depth:
                cv2.destroyWindow(self.cfg.win_depth)
            if self.cfg.live_splat:
                cv2.destroyWindow(self.cfg.win_splat)
        except cv2.error:
            # If windows are already destroyed
            pass

        self._windows_created = False

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _is_quit_key(key: int) -> bool:
        return key in (ord("q"), 27)  # 'q' or ESC

    def _ensure_windows(self) -> None:
        if self._windows_created:
            return

        # Create only the windows we need.
        # WINDOW_NORMAL lets you resize; you can switch to WINDOW_AUTOSIZE if desired.
        if self.cfg.live_cam:
            cv2.namedWindow(self.cfg.win_cam, cv2.WINDOW_NORMAL)
        if self.cfg.live_depth:
            cv2.namedWindow(self.cfg.win_depth, cv2.WINDOW_NORMAL)
        if self.cfg.live_splat:
            cv2.namedWindow(self.cfg.win_splat, cv2.WINDOW_NORMAL)

        self._windows_created = True

    @staticmethod
    def _safe_bgr(img: np.ndarray) -> np.ndarray:
        """Ensure uint8 HxWx3 BGR for display."""
        if img is None:
            return img
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _depth_to_vis(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Convert depth meters -> 8-bit colormap for display.
        """
        d = np.array(depth_m, copy=True)

        # Accept HxWx1 too
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]

        # sanitize
        d[~np.isfinite(d)] = 0.0
        d[d < 0.0] = 0.0

        max_d = float(self.cfg.depth_max_m) if self.cfg.depth_max_m and self.cfg.depth_max_m > 0 else 5.0
        d = np.clip(d, 0.0, max_d)

        d8 = (d / max_d * 255.0).astype(np.uint8)
        vis = cv2.applyColorMap(d8, self.cfg.depth_colormap)
        return vis