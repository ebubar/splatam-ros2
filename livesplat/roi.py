"""Region-of-interest primitives shared by the viewer and the SLAM loop.

The operator draws an axis-aligned box in the *world* frame of the live splat
(same frame as ``params['means3D']`` -- i.e. the first camera's coordinate
frame, since SplaTAM anchors the map to the first frame). We keep the geometry
pure-Python/NumPy so it can be created, serialised to JSON for the browser, and
tested without torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class RegionOfInterest:
    """Axis-aligned bounding box in the splat world frame (metres)."""

    min_xyz: tuple  # (x, y, z)
    max_xyz: tuple  # (x, y, z)
    label: str = "roi"

    def as_numpy(self):
        lo = np.asarray(self.min_xyz, dtype=np.float32)
        hi = np.asarray(self.max_xyz, dtype=np.float32)
        # Be forgiving about which corner the operator dragged from.
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def center(self):
        lo, hi = self.as_numpy()
        return (lo + hi) * 0.5

    def extent(self):
        lo, hi = self.as_numpy()
        return hi - lo

    def radius(self) -> float:
        """Half the box diagonal -- a convenient scalar 'size' of the object."""
        return float(np.linalg.norm(self.extent()) * 0.5)

    def contains_points(self, points_xyz: np.ndarray) -> np.ndarray:
        """Boolean mask of which Nx3 points fall inside the box."""
        lo, hi = self.as_numpy()
        pts = np.asarray(points_xyz)
        return np.all((pts >= lo) & (pts <= hi), axis=-1)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RegionOfInterest":
        return RegionOfInterest(
            min_xyz=tuple(float(v) for v in d["min_xyz"]),
            max_xyz=tuple(float(v) for v in d["max_xyz"]),
            label=str(d.get("label", "roi")),
        )


@dataclass
class RefineRequest:
    """A request from the operator to produce a higher-quality splat of a ROI.

    ``quality`` scales how much extra compute the refinement spends (iteration
    counts, densification aggressiveness). ``guide_recapture`` asks the capture
    advisor to also emit next-best-view hints for the operator.
    """

    roi: RegionOfInterest
    quality: float = 2.0            # 1.0 == same as live mapping; >1 spends more
    guide_recapture: bool = True
    request_id: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["roi"] = self.roi.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(d: dict) -> "RefineRequest":
        return RefineRequest(
            roi=RegionOfInterest.from_dict(d["roi"]),
            quality=float(d.get("quality", 2.0)),
            guide_recapture=bool(d.get("guide_recapture", True)),
            request_id=int(d.get("request_id", 0)),
            extra=dict(d.get("extra", {})),
        )
