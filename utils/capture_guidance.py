"""
Online capture-guidance for high-quality Gaussian splatting.

Splat quality is dominated by *how* the scene is captured, not just how many
frames. The successful 3DGS benchmark captures share a geometry: an object /
region of interest densely sampled from many angles, 60-80% adjacent overlap,
translation (parallax) between frames rather than pure rotation, consistent
camera-to-subject distance, multi-elevation passes (~3 rings), and loop closure.

This module distills that into an online guide that, given the current Gaussian
map + trusted ZED poses + depth, tells the operator where to move next:

  1. Scene model:      running centroid c + radius R of observed geometry.
  2. Coverage shell:    3 elevation rings of target viewpoints around c.
  3. Keyframe gating:   accept a frame only if it adds parallax (baseline +
                        view-angle change) and keeps overlap in [min, max].
  4. Coverage scoring:  score unfilled slots by rendered alpha / depth-hole /
                        under-observed-Gaussian mass (render callback optional).
  5. Blur guard:        reject motion-blurred frames (variance of Laplacian).
  6. Operator prompt:   coarse instruction toward the best unfilled slot.

The heavy per-slot rendering is injected via a callback so this module stays
light and unit-testable; the geometry/gating/prompt logic is pure NumPy.
"""

import numpy as np


def _rot_of(w2c):
    return np.asarray(w2c)[:3, :3]


def _cam_center_world(w2c):
    """Camera center in world coords from a world->cam matrix."""
    w2c = np.asarray(w2c, dtype=np.float64)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return -R.T @ t


def _view_dir_world(w2c):
    """Camera +Z (optical viewing) axis expressed in world coords."""
    R = np.asarray(w2c, dtype=np.float64)[:3, :3]
    return R.T @ np.array([0.0, 0.0, 1.0])


def variance_of_laplacian(gray):
    """Blur metric: low variance -> blurry. Uses cv2 if available."""
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        g = np.asarray(gray, dtype=np.float64)
        lap = (-4 * g
               + np.roll(g, 1, 0) + np.roll(g, -1, 0)
               + np.roll(g, 1, 1) + np.roll(g, -1, 1))
        return float(lap.var())


class CaptureGuidance:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.rings_elev = [np.deg2rad(a) for a in cfg.get("rings_elevation_deg", [-20.0, 10.0, 40.0])]
        self.slots_per_ring = int(cfg.get("slots_per_ring", 15))
        self.radius_scale = float(cfg.get("radius_scale", 3.0))
        self.min_baseline_frac = float(cfg.get("min_baseline_frac", 0.1))
        self.min_view_angle = np.deg2rad(cfg.get("min_view_angle_deg", 12.0))
        self.min_overlap = float(cfg.get("min_overlap", 0.5))
        self.max_overlap = float(cfg.get("max_overlap", 0.9))
        self.blur_var_thresh = float(cfg.get("blur_var_thresh", 60.0))
        self.weights = cfg.get("weights", {"alpha": 1.0, "depth_hole": 0.5, "under_observed": 0.5})

        self.centroid = None       # np.array(3,)
        self.radius = None         # float
        self.slots = None          # [K,3] world positions of target viewpoints
        self.slot_filled = None    # [K] bool
        self.keyframe_centers = []  # accepted keyframe camera centers
        self.keyframe_dirs = []     # accepted keyframe view dirs

    # ------------------------------------------------------------- scene model
    def update_scene(self, gaussian_means):
        """Update centroid/radius and (re)build the coverage shell.

        gaussian_means: [N,3] array (numpy or torch tensor on CPU).
        """
        pts = np.asarray(gaussian_means, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return
        # Robust centroid/radius via median + percentile to resist outliers.
        c = np.median(pts, axis=0)
        d = np.linalg.norm(pts - c, axis=1)
        R = float(np.percentile(d, 90)) if d.size else 1.0
        R = max(R, 1e-3)
        self.centroid = c
        self.radius = R
        self._build_shell()

    def _build_shell(self):
        c, R = self.centroid, self.radius
        r = self.radius_scale * R
        slots = []
        for elev in self.rings_elev:
            for k in range(self.slots_per_ring):
                az = 2.0 * np.pi * k / self.slots_per_ring
                x = c[0] + r * np.cos(elev) * np.cos(az)
                y = c[1] + r * np.cos(elev) * np.sin(az)
                z = c[2] + r * np.sin(elev)
                slots.append([x, y, z])
        new_slots = np.array(slots, dtype=np.float64)
        if self.slots is None or self.slots.shape != new_slots.shape:
            self.slots = new_slots
            self.slot_filled = np.zeros(len(new_slots), dtype=bool)
        else:
            self.slots = new_slots  # geometry moved with the scene; keep fill state

    # --------------------------------------------------------- keyframe gating
    def should_keyframe(self, w2c):
        """Return (accept: bool, reason: str).

        Enforces parallax (translation baseline + view-angle change) and rejects
        near-duplicate views. Overlap is approximated from the view-angle change
        relative to the shell radius when no depth reprojection is supplied.
        """
        center = _cam_center_world(w2c)
        vdir = _view_dir_world(w2c)

        if not self.keyframe_centers:
            return True, "first-keyframe"

        # Nearest existing keyframe.
        centers = np.stack(self.keyframe_centers)
        dists = np.linalg.norm(centers - center, axis=1)
        j = int(np.argmin(dists))
        baseline = float(dists[j])

        # Angular change of viewing direction vs. the nearest keyframe.
        cos_ang = float(np.clip(np.dot(vdir, self.keyframe_dirs[j]), -1.0, 1.0))
        view_angle = float(np.arccos(cos_ang))

        r = self.radius_scale * (self.radius or 1.0)
        min_baseline = self.min_baseline_frac * r

        if baseline < min_baseline and view_angle < self.min_view_angle:
            return False, f"redundant (baseline={baseline:.3f}<{min_baseline:.3f}, ang={np.rad2deg(view_angle):.1f}deg)"
        return True, f"accept (baseline={baseline:.3f}, ang={np.rad2deg(view_angle):.1f}deg)"

    def register_keyframe(self, w2c):
        self.keyframe_centers.append(_cam_center_world(w2c))
        self.keyframe_dirs.append(_view_dir_world(w2c))
        self._mark_nearest_slot_filled(_cam_center_world(w2c))

    def _mark_nearest_slot_filled(self, center):
        if self.slots is None:
            return
        d = np.linalg.norm(self.slots - center, axis=1)
        j = int(np.argmin(d))
        # Only credit the slot if we're reasonably near it.
        if d[j] < 0.5 * self.radius_scale * (self.radius or 1.0):
            self.slot_filled[j] = True

    # ------------------------------------------------------------ blur guard
    def blur_ok(self, gray):
        return variance_of_laplacian(gray) >= self.blur_var_thresh

    # -------------------------------------------------------- coverage scoring
    def score_slots(self, render_alpha_fn=None):
        """Score unfilled slots. Higher = more worth capturing.

        render_alpha_fn(slot_w2c) -> dict(alpha_mean, depth_hole_ratio,
        under_observed) is optional; when absent, geometry-only scoring returns
        1.0 for every unfilled slot (fall back to nearest-unfilled guidance).
        """
        if self.slots is None:
            return None
        scores = np.zeros(len(self.slots), dtype=np.float64)
        for i in range(len(self.slots)):
            if self.slot_filled[i]:
                scores[i] = -1.0
                continue
            if render_alpha_fn is None:
                scores[i] = 1.0
                continue
            w2c = self._look_at_w2c(self.slots[i], self.centroid)
            m = render_alpha_fn(w2c) or {}
            scores[i] = (self.weights["alpha"] * (1.0 - m.get("alpha_mean", 0.0))
                         + self.weights["depth_hole"] * m.get("depth_hole_ratio", 0.0)
                         + self.weights["under_observed"] * m.get("under_observed", 0.0))
        return scores

    def _look_at_w2c(self, eye, target):
        """Build a world->cam matrix for a camera at `eye` looking at `target`."""
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        fwd = target - eye
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)          # +Z optical
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, fwd); right /= (np.linalg.norm(right) + 1e-9)
        true_up = np.cross(fwd, right)
        R_c2w = np.stack([right, true_up, fwd], axis=1)   # cols = cam axes in world
        w2c = np.eye(4)
        w2c[:3, :3] = R_c2w.T
        w2c[:3, 3] = -R_c2w.T @ eye
        return w2c

    # ------------------------------------------------------ operator prompting
    def next_instruction(self, w2c, render_alpha_fn=None):
        """Return a short human instruction toward the best unfilled slot."""
        if self.slots is None or self.centroid is None:
            return "Scanning... move slowly around the object."

        scores = self.score_slots(render_alpha_fn)
        unfilled = np.where(~self.slot_filled)[0]
        if unfilled.size == 0:
            return "Capture complete: coverage saturated."

        best = int(unfilled[np.argmax(scores[unfilled])])
        target = self.slots[best]
        center = _cam_center_world(w2c)

        # Direction to the target slot, expressed relative to current heading.
        to_slot = target - center
        dist = float(np.linalg.norm(to_slot))
        r = self.radius_scale * (self.radius or 1.0)

        # Radial (distance) vs. tangential (orbit) guidance.
        radial = float(np.linalg.norm(center - self.centroid))
        msg = []
        if radial > 1.2 * r:
            msg.append("move closer")
        elif radial < 0.8 * r:
            msg.append("step back to hold distance")

        # Azimuth of current vs. target around the scene centroid.
        cur_az = np.arctan2(center[1] - self.centroid[1], center[0] - self.centroid[0])
        tgt_az = np.arctan2(target[1] - self.centroid[1], target[0] - self.centroid[0])
        daz = np.rad2deg((tgt_az - cur_az + np.pi) % (2 * np.pi) - np.pi)
        if abs(daz) > 8.0:
            msg.append(f"orbit {'left' if daz > 0 else 'right'} ~{abs(daz):.0f} deg")

        # Elevation guidance.
        cur_el = np.arcsin(np.clip((center[2] - self.centroid[2]) / (radial + 1e-9), -1, 1))
        tgt_el = np.arcsin(np.clip((target[2] - self.centroid[2]) / (r + 1e-9), -1, 1))
        dele = np.rad2deg(tgt_el - cur_el)
        if abs(dele) > 8.0:
            msg.append(f"move {'higher' if dele > 0 else 'lower'}")

        filled = int(self.slot_filled.sum())
        total = len(self.slot_filled)
        prefix = f"[{filled}/{total} views] "
        return prefix + ("; ".join(msg) if msg else "good angle, keep panning slowly")
