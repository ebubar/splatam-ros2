import numpy as np

from livesplat.roi import RegionOfInterest, RefineRequest


def test_center_extent_radius():
    roi = RegionOfInterest(min_xyz=(0, 0, 0), max_xyz=(2, 4, 4))
    assert np.allclose(roi.center(), [1, 2, 2])
    assert np.allclose(roi.extent(), [2, 4, 4])
    assert np.isclose(roi.radius(), np.linalg.norm([2, 4, 4]) * 0.5)


def test_contains_points():
    roi = RegionOfInterest(min_xyz=(0, 0, 0), max_xyz=(1, 1, 1))
    pts = np.array([[0.5, 0.5, 0.5], [2, 2, 2], [0, 0, 0], [1, 1, 1]])
    assert roi.contains_points(pts).tolist() == [True, False, True, True]


def test_corner_order_is_forgiving():
    roi = RegionOfInterest(min_xyz=(1, 1, 1), max_xyz=(0, 0, 0))
    pts = np.array([[0.5, 0.5, 0.5], [2, 2, 2]])
    assert roi.contains_points(pts).tolist() == [True, False]


def test_roi_json_roundtrip():
    roi = RegionOfInterest(min_xyz=(0, 0, 0), max_xyz=(1, 1, 1), label="obj")
    assert RegionOfInterest.from_dict(roi.to_dict()) == roi


def test_refine_request_roundtrip():
    roi = RegionOfInterest(min_xyz=(0, 0, 0), max_xyz=(1, 1, 1))
    req = RefineRequest(roi=roi, quality=3.5, guide_recapture=False)
    back = RefineRequest.from_dict(req.to_dict())
    assert back.quality == 3.5 and back.guide_recapture is False
    assert back.roi == roi
