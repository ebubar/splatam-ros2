"""Thread-safe hand-off between the SLAM loop and the operator viewer.

The SLAM loop runs in the ROS2 callback thread; the viewer runs its own HTTP
server thread. They never touch each other's objects directly -- everything
goes through this small mailbox:

* SLAM  ->  viewer : ``publish_snapshot`` (latest downsampled point cloud) and
                     ``publish_status`` (frame counters, guidance hints).
* viewer ->  SLAM  : ``submit_refine_request`` / ``pop_refine_request``.

Only NumPy arrays and plain dicts cross the boundary, so there is no CUDA state
shared between threads.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from livesplat.roi import RefineRequest


class LiveSplatBridge:
    def __init__(self, max_preview_points: int = 150_000):
        self._lock = threading.Lock()
        self._snapshot: Optional[dict] = None
        self._status: dict = {}
        self._requests: "deque[RefineRequest]" = deque()
        self._guidance: dict = {}
        self._request_counter = 0
        self.max_preview_points = int(max_preview_points)
        self._snapshot_version = 0

    # ---- SLAM loop -> viewer -------------------------------------------------

    def publish_snapshot(
        self,
        means_xyz: np.ndarray,
        rgb: np.ndarray,
        opacities: Optional[np.ndarray] = None,
    ) -> None:
        """Store the latest point cloud for the viewer.

        Points are randomly subsampled to ``max_preview_points`` so streaming a
        multi-million-Gaussian map to a laptop stays cheap. Colours are expected
        in [0, 1]; they are quantised to uint8 for transport.
        """
        means = np.asarray(means_xyz, dtype=np.float32)
        cols = np.asarray(rgb, dtype=np.float32)
        n = means.shape[0]
        if n == 0:
            return

        if n > self.max_preview_points:
            # Deterministic stride keeps the preview stable frame-to-frame,
            # which looks far less noisy than re-drawing a fresh random subset.
            step = int(np.ceil(n / self.max_preview_points))
            sel = slice(0, n, step)
            means = means[sel]
            cols = cols[sel]
            if opacities is not None:
                opacities = np.asarray(opacities)[sel]

        cols_u8 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)

        with self._lock:
            self._snapshot = {
                "xyz": means,
                "rgb": cols_u8,
                "count_total": int(n),
                "count_preview": int(means.shape[0]),
            }
            self._snapshot_version += 1

    def publish_status(self, **status) -> None:
        with self._lock:
            self._status.update(status)

    def publish_guidance(self, guidance: dict) -> None:
        """Publish next-best-view hints from the capture advisor."""
        with self._lock:
            self._guidance = dict(guidance)

    # ---- viewer reads --------------------------------------------------------

    def snapshot_version(self) -> int:
        with self._lock:
            return self._snapshot_version

    def get_snapshot(self) -> Optional[dict]:
        with self._lock:
            return self._snapshot

    def get_status(self) -> dict:
        with self._lock:
            s = dict(self._status)
            s["guidance"] = dict(self._guidance)
            s["snapshot_version"] = self._snapshot_version
            return s

    # ---- viewer -> SLAM loop -------------------------------------------------

    def submit_refine_request(self, req: RefineRequest) -> int:
        with self._lock:
            self._request_counter += 1
            req.request_id = self._request_counter
            self._requests.append(req)
            return req.request_id

    def pop_refine_request(self) -> Optional[RefineRequest]:
        with self._lock:
            if self._requests:
                return self._requests.popleft()
            return None

    def pending_requests(self) -> int:
        with self._lock:
            return len(self._requests)
