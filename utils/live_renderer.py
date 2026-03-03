#!/usr/bin/env python3
# utils/live_renderer.py

from __future__ import annotations

import os
import time
from dataclasses import dataclass
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

    # ---- Recording ----
    # MP4s are written here (single output folder)
    record_dir: Optional[str] = None
    record_tag: str = "run"
    record_fps: float = 30.0
    record_cam: bool = True
    record_depth: bool = True
    record_splat: bool = True
    fourcc: str = "mp4v"  # try "avc1" if your OpenCV has H264, otherwise mp4v is safest


class LiveRenderer:
    """
    Simple OpenCV live viewer + optional mp4 recorder for:
      - camera (BGR8)
      - depth (float meters -> colormap BGR8)
      - splat preview (BGR8)

    Viewer quit: Press 'q' or ESC in ANY OpenCV window.
    """

    def __init__(self, cfg: LiveRendererConfig):
        self.cfg = cfg
        self._last_show_t = 0.0
        self._min_dt = 1.0 / max(float(cfg.max_fps), 1e-6)

        self._cam_bgr: Optional[np.ndarray] = None
        self._depth_m: Optional[np.ndarray] = None
        self._splat_bgr: Optional[np.ndarray] = None

        self._windows_created = False

        # video writers (lazy init once we know frame size)
        self._vw_cam: Optional[cv2.VideoWriter] = None
        self._vw_depth: Optional[cv2.VideoWriter] = None
        self._vw_splat: Optional[cv2.VideoWriter] = None
        self._fourcc = cv2.VideoWriter_fourcc(*self.cfg.fourcc)

    def update_cam(self, cam_bgr: np.ndarray) -> None:
        if cam_bgr is None:
            return
        self._cam_bgr = cam_bgr

    def update_depth(self, depth_m: np.ndarray) -> None:
        if depth_m is None:
            return
        if depth_m.ndim == 3 and depth_m.shape[2] == 1:
            depth_m = depth_m[..., 0]
        self._depth_m = depth_m

    def update_splat_preview(self, splat_bgr: np.ndarray) -> None:
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
        d = depth_m.copy().astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d[d < 0.0] = 0.0

        valid = d > 0.0
        if not np.any(valid):
            return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

        d_valid = d[valid]
        lo = float(np.percentile(d_valid, 5))
        hi = float(np.percentile(d_valid, 95))
        hi = max(hi, lo + 1e-6)

        dn = (d - lo) / (hi - lo)
        dn = np.clip(dn, 0.0, 1.0)
        dn8 = (dn * 255.0).astype(np.uint8)

        return cv2.applyColorMap(dn8, cv2.COLORMAP_JET)

    def _ensure_writer(self, kind: str, frame_bgr: np.ndarray) -> None:
        """
        kind: 'cam' | 'depth' | 'splat'
        """
        if not self.cfg.record_dir:
            return

        # IMPORTANT: make sure folder exists (otherwise OpenCV may silently fail)
        os.makedirs(self.cfg.record_dir, exist_ok=True)

        h, w = frame_bgr.shape[:2]
        fps = float(self.cfg.record_fps)

        def _open_writer(path: str) -> Optional[cv2.VideoWriter]:
            vw = cv2.VideoWriter(path, self._fourcc, fps, (w, h))
            if not vw.isOpened():
                print(f"[LiveRenderer] ERROR: could not open VideoWriter: {path}")
                return None
            return vw

        if kind == "cam" and self._vw_cam is None and self.cfg.record_cam:
            path = os.path.join(self.cfg.record_dir, f"cam_{self.cfg.record_tag}.mp4")
            self._vw_cam = _open_writer(path)

        elif kind == "depth" and self._vw_depth is None and self.cfg.record_depth:
            path = os.path.join(self.cfg.record_dir, f"depth_{self.cfg.record_tag}.mp4")
            self._vw_depth = _open_writer(path)

        elif kind == "splat" and self._vw_splat is None and self.cfg.record_splat:
            path = os.path.join(self.cfg.record_dir, f"splat_{self.cfg.record_tag}.mp4")
            self._vw_splat = _open_writer(path)

    def tick(self) -> bool:
        # nothing enabled
        if not (
            self.cfg.live_cam
            or self.cfg.live_depth
            or self.cfg.live_splat
            or self.cfg.record_cam
            or self.cfg.record_depth
            or self.cfg.record_splat
        ):
            return True

        # keep windows responsive if live display is used
        if self.cfg.live_cam or self.cfg.live_depth or self.cfg.live_splat:
            self._ensure_windows()

        now = time.time()
        if (now - self._last_show_t) < self._min_dt:
            if self.cfg.live_cam or self.cfg.live_depth or self.cfg.live_splat:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    return False
            return True

        self._last_show_t = now

        # CAMERA
        if self._cam_bgr is not None:
            if self.cfg.live_cam:
                cv2.imshow(self.cfg.win_cam, self._cam_bgr)
            self._ensure_writer("cam", self._cam_bgr)
            if self._vw_cam is not None:
                self._vw_cam.write(self._cam_bgr)

        # DEPTH
        if self._depth_m is not None:
            depth_vis = self._render_depth_vis(self._depth_m)
            if self.cfg.live_depth:
                cv2.imshow(self.cfg.win_depth, depth_vis)
            self._ensure_writer("depth", depth_vis)
            if self._vw_depth is not None:
                self._vw_depth.write(depth_vis)

        # SPLAT
        if self._splat_bgr is not None:
            if self.cfg.live_splat:
                cv2.imshow(self.cfg.win_splat, self._splat_bgr)
            self._ensure_writer("splat", self._splat_bgr)
            if self._vw_splat is not None:
                self._vw_splat.write(self._splat_bgr)

        # quit key only matters for windows
        if self.cfg.live_cam or self.cfg.live_depth or self.cfg.live_splat:
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return False

        return True

    def close(self) -> None:
        for vw in (self._vw_cam, self._vw_depth, self._vw_splat):
            try:
                if vw is not None:
                    vw.release()
            except Exception:
                pass

        self._vw_cam = None
        self._vw_depth = None
        self._vw_splat = None

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass