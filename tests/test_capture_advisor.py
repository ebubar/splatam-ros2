import numpy as np

from livesplat.roi import RegionOfInterest
from livesplat import capture_advisor as ca
from utils import pose_utils as pu


def _look_at_w2c(pos, target):
    pos = np.asarray(pos, float)
    target = np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(up, f)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(np.array([1.0, 0.0, 0.0]), f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([r, u, f], axis=1)
    c2w[:3, 3] = pos
    return pu.invert_transform(c2w)


def _roi():
    return RegionOfInterest(min_xyz=(0, 0, 0), max_xyz=(1, 1, 1))


def test_ideal_plan_covers_all_rings():
    roi = _roi()
    plan = ca.ideal_capture_plan(roi)
    els = {round(vp.elevation_deg) for vp in plan}
    assert els == {round(e) for e in ca.DEFAULT_ELEVATIONS_DEG}
    # every planned camera sits at roughly the target radius from centre
    R = roi.radius() * ca.DEFAULT_RADIUS_SCALE
    for vp in plan:
        d = np.linalg.norm(np.array(vp.position) - roi.center())
        assert np.isclose(d, R, rtol=1e-6)


def test_empty_capture_reports_zero_coverage():
    res = ca.analyse_coverage(_roi(), np.zeros((0, 4, 4)))
    assert res["coverage"] == 0.0 and res["hints"] == [] and res["n_views"] == 0


def test_partial_ring_leaves_gaps_and_hints_point_at_them():
    roi = _roi()
    center = roi.center()
    # cameras only around azimuth 0..150 deg
    w2c = [_look_at_w2c(center + 2.0 * np.array([np.cos(np.radians(a)), 0.3,
                                                 np.sin(np.radians(a))]), center)
           for a in range(0, 160, 30)]
    res = ca.analyse_coverage(roi, np.stack(w2c))
    assert 0.0 < res["coverage"] < 1.0
    assert len(res["hints"]) > 0
    # the biggest gap should be on the uncovered side (~180..330 deg)
    top = res["hints"][0]["azimuth_deg"]
    assert 170 < top < 350


def test_full_orbit_increases_coverage():
    roi = _roi()
    center = roi.center()
    sparse = [_look_at_w2c(center + 2.0 * np.array([np.cos(np.radians(a)), 0.3,
                                                    np.sin(np.radians(a))]), center)
              for a in range(0, 160, 30)]
    dense = [_look_at_w2c(center + 2.0 * np.array([np.cos(np.radians(a)), 0.3,
                                                   np.sin(np.radians(a))]), center)
             for a in range(0, 360, 20)]
    c_sparse = ca.analyse_coverage(roi, np.stack(sparse))["coverage"]
    c_dense = ca.analyse_coverage(roi, np.stack(dense))["coverage"]
    assert c_dense > c_sparse
