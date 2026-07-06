"""ZED depth utilities for the live SplaTAM pipeline.

The ZED SDK can produce depth from several modes. For SplaTAM we want the
NEURAL / NEURAL_PLUS modes, which use a learned stereo network and give dense,
metric depth good enough to replace a LiDAR as the geometry source. This module
centralises the (previously duplicated and slightly inconsistent) decoding,
unit-scaling, range-clipping and confidence-masking of that depth so the live
node and the offline dataset loader behave identically.

Nothing here imports torch at module load time so it can be unit-tested on a
machine without CUDA; the heavy tensors are only created by the live node.
"""

from __future__ import annotations

import numpy as np


# ZED publishes invalid depth as NaN/Inf (and occasionally 0). Everything below
# collapses those to a single sentinel of 0.0, which SplaTAM treats as "no
# measurement" via its depth > 0 masks.
_INVALID = 0.0


def decode_depth_to_meters(depth_raw: np.ndarray, encoding: str) -> np.ndarray:
    """Decode a ROS depth image into float32 metres.

    ZED's NEURAL depth is published as ``32FC1`` already in metres. Some
    configurations (or bag conversions) fall back to ``16UC1``/``mono16`` in
    millimetres. We normalise both to metres.
    """
    enc = (encoding or "").lower()

    depth = np.asarray(depth_raw)
    if depth.ndim == 3:
        depth = depth[..., 0]

    if enc in ("16uc1", "16uc", "mono16"):
        return depth.astype(np.float32) / 1000.0

    # 32FC1 / 32fc / unknown -> assume metres.
    return depth.astype(np.float32)


def sanitize_depth(
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    unit_scale_m: float = 1.0,
) -> np.ndarray:
    """Clean a metric depth map: apply unit scale, drop NaN/Inf, clip range.

    Depth outside ``[min_depth_m, max_depth_m]`` is set to 0.0 (invalid). This
    is important for the ZED NEURAL depth, whose far-field is noisy and, if
    left in, pulls Gaussians into a haze behind the real geometry.
    """
    depth = depth_m.astype(np.float32, copy=True)

    if unit_scale_m != 1.0:
        depth *= float(unit_scale_m)

    depth = np.nan_to_num(depth, nan=_INVALID, posinf=_INVALID, neginf=_INVALID)
    depth[(depth < float(min_depth_m)) | (depth > float(max_depth_m))] = _INVALID
    return depth


def apply_confidence_mask(
    depth_m: np.ndarray,
    confidence: np.ndarray | None,
    max_cost: float,
) -> np.ndarray:
    """Zero out depth pixels whose ZED confidence cost exceeds ``max_cost``.

    The ZED ``confidence/confidence_map`` topic is a per-pixel cost in [0, 100]
    where *lower is better*. Filtering on it removes the speckle around depth
    discontinuities that otherwise seeds floating Gaussians. If no confidence
    map is supplied (topic not enabled), the depth is returned unchanged.
    """
    if confidence is None:
        return depth_m

    conf = np.asarray(confidence)
    if conf.ndim == 3:
        conf = conf[..., 0]

    if conf.shape != depth_m.shape:
        # Confidence stream out of sync / different resolution; skip masking
        # rather than corrupt the depth.
        return depth_m

    out = depth_m.copy()
    out[conf > float(max_cost)] = _INVALID
    return out


def depth_fill_ratio(depth_m: np.ndarray) -> float:
    """Fraction of pixels with a valid depth measurement. Cheap health metric
    the live node logs so a user can tell at a glance whether the ZED depth mode
    is actually returning data (a black/empty depth image reads as ~0.0)."""
    total = depth_m.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(depth_m > _INVALID)) / float(total)
