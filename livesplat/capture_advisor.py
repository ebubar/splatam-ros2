"""Viewpoint-coverage analysis and next-best-view guidance.

Purpose
-------
When the operator asks for an ultra-high-quality splat of a specific object,
raw "wave the camera around" capture is wasteful and leaves holes. Gaussian
splatting / NeRF benchmarks (Mip-NeRF 360, Tanks & Temples, DTU, ScanNet++)
all show the same thing: reconstruction quality of a bounded object is governed
by *angular coverage* of the object from a roughly constant radius, with
generous view overlap, rather than by sheer frame count. See
``docs/CAPTURE_PATTERNS.md`` for the full write-up and citations.

This module turns that into two concrete tools:

1. :func:`ideal_capture_plan` -- given an ROI, produce the target set of
   viewpoints (multi-elevation orbit) that a benchmark-quality capture needs.
2. :func:`analyse_coverage` -- given the poses actually captured so far,
   measure how much of that target sphere is covered and return the largest
   gaps as next-best-view hints for the operator.

Everything is NumPy-only and framed in the splat world frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from livesplat.roi import RegionOfInterest


# --- Benchmark-derived defaults ---------------------------------------------
# Rationale (see docs/CAPTURE_PATTERNS.md):
#   * 3 elevation rings (~15, ~45, ~70 deg) approximate the hemispherical
#     coverage used by DTU / object-centric captures without going fully
#     top-down, which stereo depth handles poorly.
#   * ~20-30 deg azimuth spacing gives the 60-80% pairwise view overlap that
#     photogrammetry and splatting both want for stable geometry.
#   * Capture radius ~ 2.5x object radius keeps the object filling the frame
#     while staying inside the ZED's reliable NEURAL-depth range.
DEFAULT_ELEVATIONS_DEG = (15.0, 45.0, 70.0)
DEFAULT_AZIMUTH_STEP_DEG = 24.0
DEFAULT_RADIUS_SCALE = 2.5
# Coverage binning resolution for gap analysis.
_AZ_BINS = 24
_EL_BINS = 6


@dataclass
class Viewpoint:
    """A recommended camera position + the point it should look at."""

    position: tuple      # world xyz of the camera
    look_at: tuple       # world xyz to aim at (ROI centre)
    elevation_deg: float
    azimuth_deg: float

    def to_dict(self) -> dict:
        return asdict(self)


def _camera_centers_from_w2c(w2c_stack: np.ndarray) -> np.ndarray:
    """Camera centres in world frame from a stack of 4x4 world-to-camera mats.

    For w2c = [R | t], the camera centre C = -R^T t.
    """
    w2c = np.asarray(w2c_stack, dtype=np.float64)
    if w2c.ndim == 2:
        w2c = w2c[None]
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    C = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)
    return C


def _dirs_to_azel(dirs: np.ndarray):
    """Unit direction vectors -> (azimuth_deg in [0,360), elevation_deg [-90,90]).

    Elevation is measured from the world XZ plane about +Y being "up". SplaTAM's
    first-frame optical frame has +Y down, but the advisor only needs a
    *consistent* spherical parameterisation to find gaps, not a gravity-aligned
    one, so any fixed convention works.
    """
    d = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-9)
    az = (np.degrees(np.arctan2(d[:, 2], d[:, 0])) + 360.0) % 360.0
    el = np.degrees(np.arcsin(np.clip(d[:, 1], -1.0, 1.0)))
    return az, el


def ideal_capture_plan(
    roi: RegionOfInterest,
    elevations_deg=DEFAULT_ELEVATIONS_DEG,
    azimuth_step_deg: float = DEFAULT_AZIMUTH_STEP_DEG,
    radius_scale: float = DEFAULT_RADIUS_SCALE,
) -> List[Viewpoint]:
    """Target viewpoints for a benchmark-quality object capture of ``roi``."""
    center = roi.center().astype(np.float64)
    radius = max(roi.radius(), 1e-3) * float(radius_scale)

    plan: List[Viewpoint] = []
    n_az = max(1, int(round(360.0 / float(azimuth_step_deg))))
    for el_deg in elevations_deg:
        el = math.radians(el_deg)
        for i in range(n_az):
            az = math.radians(i * 360.0 / n_az)
            offset = np.array(
                [
                    radius * math.cos(el) * math.cos(az),
                    radius * math.sin(el),
                    radius * math.cos(el) * math.sin(az),
                ],
                dtype=np.float64,
            )
            pos = center + offset
            plan.append(
                Viewpoint(
                    position=tuple(float(v) for v in pos),
                    look_at=tuple(float(v) for v in center),
                    elevation_deg=float(el_deg),
                    azimuth_deg=float(math.degrees(az)),
                )
            )
    return plan


def analyse_coverage(
    roi: RegionOfInterest,
    captured_w2c: np.ndarray,
    max_hints: int = 6,
):
    """Measure angular coverage of ``roi`` by the captured cameras.

    Returns a dict with a scalar ``coverage`` in [0, 1] (fraction of az/el bins
    that saw the object) and up to ``max_hints`` :class:`Viewpoint` suggestions
    aimed at the emptiest bins.
    """
    center = roi.center().astype(np.float64)
    radius = max(roi.radius(), 1e-3) * float(DEFAULT_RADIUS_SCALE)

    if captured_w2c is None or len(captured_w2c) == 0:
        return {"coverage": 0.0, "hints": [], "n_views": 0}

    centers = _camera_centers_from_w2c(captured_w2c)
    dirs = centers - center[None, :]
    az, el = _dirs_to_azel(dirs)

    # Occupancy grid over azimuth x elevation.
    grid = np.zeros((_AZ_BINS, _EL_BINS), dtype=np.int32)
    az_idx = np.clip((az / 360.0 * _AZ_BINS).astype(int), 0, _AZ_BINS - 1)
    # Elevation only spans the hemisphere we care about (0..90); fold negatives.
    el_clip = np.clip(np.abs(el), 0.0, 89.999)
    el_idx = np.clip((el_clip / 90.0 * _EL_BINS).astype(int), 0, _EL_BINS - 1)
    for a, e in zip(az_idx, el_idx):
        grid[a, e] += 1

    coverage = float(np.count_nonzero(grid)) / float(grid.size)

    # Rank empty bins by how isolated they are (largest coverage holes first).
    empty = np.argwhere(grid == 0)
    hints: List[Viewpoint] = []
    if empty.size > 0:
        # Distance (in bin space, wrapping azimuth) from each empty bin to the
        # nearest occupied bin -- bigger => more urgent to fill.
        occupied = np.argwhere(grid > 0)
        scores = []
        for (ai, ei) in empty:
            d_az = np.minimum(
                np.abs(occupied[:, 0] - ai),
                _AZ_BINS - np.abs(occupied[:, 0] - ai),
            )
            d_el = np.abs(occupied[:, 1] - ei)
            scores.append(float(np.min(d_az + d_el)) if len(occupied) else 1e9)
        order = np.argsort(scores)[::-1]
        for k in order[:max_hints]:
            ai, ei = empty[k]
            az_deg = (ai + 0.5) * 360.0 / _AZ_BINS
            el_deg = (ei + 0.5) * 90.0 / _EL_BINS
            el_r = math.radians(el_deg)
            az_r = math.radians(az_deg)
            offset = np.array(
                [
                    radius * math.cos(el_r) * math.cos(az_r),
                    radius * math.sin(el_r),
                    radius * math.cos(el_r) * math.sin(az_r),
                ],
                dtype=np.float64,
            )
            pos = center + offset
            hints.append(
                Viewpoint(
                    position=tuple(float(v) for v in pos),
                    look_at=tuple(float(v) for v in center),
                    elevation_deg=float(el_deg),
                    azimuth_deg=float(az_deg),
                )
            )

    return {
        "coverage": coverage,
        "hints": [h.to_dict() for h in hints],
        "n_views": int(len(captured_w2c)),
    }
