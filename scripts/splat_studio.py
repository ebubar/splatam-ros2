#!/usr/bin/env python3
"""
Live tuning studio for the ZED SplaTAM node: a browser-based (viser) 3D view
of the growing Gaussian map that you can orbit while capturing, plus live
parameter controls — no restarts.

Enabled via the config: viz.studio=True (see configs/zed2i/zed2i_local_tune.py).
Open http://<machine>:8080 while the node runs.

Live controls (applied to the running node):
  - process every Nth frame, max depth
  - tracking iterations, mapping iterations, map every N
  - Pause/Resume   - Reset map (restart mapping, keep camera running)
  - Save snapshot  (params.npz written immediately, node keeps going)
"""

import threading
import time

import numpy as np
import torch


def _rotmat_to_wxyz(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        return np.array([0.25 * S, (R[2, 1] - R[1, 2]) / S,
                         (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    return q


class SplatStudio:
    def __init__(self, node, cfg):
        import viser  # deferred: only needed when studio is enabled

        self.node = node
        self.cfg = cfg
        self.paused = False
        self.reset_requested = False
        self.save_requested = False
        self._trail = []
        self._last_update = 0.0
        self._snapshot_count = 0

        port = int(cfg["viz"].get("studio_port", 8080))
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        # SplaTAM world is OpenCV convention: +y is down
        self.server.scene.set_up_direction("-y")

        gui = self.server.gui

        self._stats = gui.add_markdown("**Waiting for frames...**")

        with gui.add_folder("Capture"):
            self._sl_every = gui.add_slider(
                "process every Nth", min=1, max=10, step=1,
                initial_value=int(cfg["ros"].get("process_every_n", 1)))
            self._sl_maxd = gui.add_slider(
                "max depth (m)", min=1.0, max=10.0, step=0.5,
                initial_value=float(cfg["ros"].get("max_depth_m", 5.0)))

        with gui.add_folder("Optimization"):
            self._sl_trk = gui.add_slider(
                "tracking iters", min=0, max=100, step=5,
                initial_value=int(cfg["tracking"]["num_iters"]))
            self._sl_map = gui.add_slider(
                "mapping iters", min=10, max=300, step=10,
                initial_value=int(cfg["mapping"]["num_iters"]))
            self._sl_mapevery = gui.add_slider(
                "map every Nth", min=1, max=15, step=1,
                initial_value=int(cfg["map_every"]))

        with gui.add_folder("View"):
            self._sl_psize = gui.add_slider(
                "point size", min=0.002, max=0.05, step=0.002,
                initial_value=0.008)
            self._sl_maxpts = gui.add_slider(
                "max points (k)", min=50, max=500, step=50,
                initial_value=250)

        with gui.add_folder("Actions"):
            self._btn_pause = gui.add_button("Pause")
            self._btn_reset = gui.add_button("Reset map")
            self._btn_save = gui.add_button("Save snapshot")

        self._sl_every.on_update(self._apply_params)
        self._sl_maxd.on_update(self._apply_params)
        self._sl_trk.on_update(self._apply_params)
        self._sl_map.on_update(self._apply_params)
        self._sl_mapevery.on_update(self._apply_params)
        self._btn_pause.on_click(self._toggle_pause)
        self._btn_reset.on_click(lambda _: setattr(self, "reset_requested", True))
        self._btn_save.on_click(lambda _: setattr(self, "save_requested", True))

        print(f"[SplatStudio] Live view: http://localhost:{port}")

    # ---- GUI callbacks (viser thread) ----

    def _apply_params(self, _):
        self.node.process_every_n = int(self._sl_every.value)
        self.cfg["ros"]["max_depth_m"] = float(self._sl_maxd.value)
        self.cfg["tracking"]["num_iters"] = int(self._sl_trk.value)
        self.cfg["mapping"]["num_iters"] = int(self._sl_map.value)
        self.cfg["map_every"] = int(self._sl_mapevery.value)

    def _toggle_pause(self, _):
        self.paused = not self.paused
        self._btn_pause.name = "Resume" if self.paused else "Pause"

    # ---- called from the node (ROS thread) ----

    def clear_scene(self):
        self._trail = []
        try:
            self.server.scene.reset()
        except Exception:
            pass

    def update_stats(self, text):
        self._stats.content = text

    def update_scene(self, params, curr_c2w, min_interval_s=0.5):
        now = time.time()
        if now - self._last_update < min_interval_s:
            return
        self._last_update = now

        with torch.no_grad():
            means = params["means3D"]
            cols = params["rgb_colors"]
            opac = torch.sigmoid(params["logit_opacities"]).squeeze(-1)
            keep = opac > 0.1
            means = means[keep]
            cols = cols[keep]

            max_pts = int(self._sl_maxpts.value) * 1000
            n = means.shape[0]
            if n > max_pts:
                idx = torch.randperm(n, device=means.device)[:max_pts]
                means = means[idx]
                cols = cols[idx]

            pts = means.detach().cpu().numpy().astype(np.float32)
            rgb = (torch.clamp(cols, 0, 1) * 255).byte().detach().cpu().numpy()

        self.server.scene.add_point_cloud(
            "/splat",
            points=pts,
            colors=rgb,
            point_size=float(self._sl_psize.value),
            point_shape="circle",
        )

        if curr_c2w is not None:
            c2w = curr_c2w.detach().cpu().numpy() if torch.is_tensor(curr_c2w) else curr_c2w
            pos = c2w[:3, 3]
            self._trail.append(pos.copy())
            if len(self._trail) > 2000:
                self._trail = self._trail[-2000:]
            self.server.scene.add_camera_frustum(
                "/cam",
                fov=1.05,
                aspect=16 / 9,
                scale=0.12,
                color=(255, 120, 0),
                wxyz=_rotmat_to_wxyz(c2w[:3, :3]),
                position=pos,
            )
            self.server.scene.add_point_cloud(
                "/trail",
                points=np.asarray(self._trail, dtype=np.float32),
                colors=(255, 160, 0),
                point_size=0.01,
                point_shape="circle",
            )
