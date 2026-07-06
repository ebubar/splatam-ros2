import struct

import numpy as np

from livesplat.bridge import LiveSplatBridge
from livesplat.roi import RegionOfInterest, RefineRequest
from livesplat.viewer_server import _pack_snapshot
from utils import zed_depth


def test_bridge_subsamples_preview():
    b = LiveSplatBridge(max_preview_points=1000)
    b.publish_snapshot(np.random.randn(50000, 3).astype(np.float32),
                       np.random.rand(50000, 3).astype(np.float32))
    snap = b.get_snapshot()
    assert snap["count_total"] == 50000
    assert snap["count_preview"] <= 1000


def test_bridge_snapshot_version_increments():
    b = LiveSplatBridge()
    v0 = b.snapshot_version()
    b.publish_snapshot(np.zeros((10, 3), np.float32), np.zeros((10, 3), np.float32))
    assert b.snapshot_version() == v0 + 1


def test_bridge_request_queue_fifo():
    b = LiveSplatBridge()
    roi = RegionOfInterest(min_xyz=(0, 0, 0), max_xyz=(1, 1, 1))
    r1 = b.submit_refine_request(RefineRequest(roi=roi, quality=1.0))
    r2 = b.submit_refine_request(RefineRequest(roi=roi, quality=2.0))
    assert (r1, r2) == (1, 2)
    assert b.pending_requests() == 2
    assert b.pop_refine_request().request_id == 1
    assert b.pop_refine_request().request_id == 2
    assert b.pop_refine_request() is None


def test_pack_snapshot_layout():
    b = LiveSplatBridge(max_preview_points=5000)
    b.publish_snapshot(np.random.randn(8000, 3).astype(np.float32),
                       np.random.rand(8000, 3).astype(np.float32))
    blob = _pack_snapshot(b.get_snapshot())
    n = struct.unpack("<I", blob[:4])[0]
    assert n <= 5000
    assert len(blob) == 4 + n * 12 + n * 3


def test_depth_decode_units():
    d16 = (np.ones((4, 4)) * 2000).astype(np.uint16)  # 2000 mm
    assert np.allclose(zed_depth.decode_depth_to_meters(d16, "16UC1"), 2.0)
    d32 = (np.ones((4, 4)) * 1.5).astype(np.float32)
    assert np.allclose(zed_depth.decode_depth_to_meters(d32, "32FC1"), 1.5)


def test_depth_sanitize_clips_and_nans():
    d = np.array([[0.1, 1.0], [np.nan, 10.0]], dtype=np.float32)
    s = zed_depth.sanitize_depth(d, min_depth_m=0.3, max_depth_m=6.0)
    assert s[0, 0] == 0 and s[0, 1] == 1.0 and s[1, 0] == 0 and s[1, 1] == 0
    assert np.isclose(zed_depth.depth_fill_ratio(s), 0.25)


def test_confidence_masking():
    depth = np.ones((2, 2), np.float32)
    conf = np.array([[10, 90], [10, 10]], dtype=np.float32)
    masked = zed_depth.apply_confidence_mask(depth, conf, max_cost=50.0)
    assert masked[0, 1] == 0 and masked[0, 0] == 1
    # mismatched shape -> unchanged
    assert np.array_equal(
        zed_depth.apply_confidence_mask(depth, np.ones((3, 3), np.float32), 50.0), depth
    )
    # no confidence -> unchanged
    assert np.array_equal(zed_depth.apply_confidence_mask(depth, None, 50.0), depth)
