"""
Live browser viewer for zed2i_splat_live.py: watch the Gaussian map build in
realtime, in an orbitable 3D view, while walking the camera around an
environment. Built for room/environment-scale coverage — the question this
answers is "what have I actually captured, and what's still empty" — not
per-object photorealism preview.

Runs as a decoupled background thread (like the SLAM worker itself): it only
*reads* node.params / node.gt_w2c_all_frames on its own timer, so a slow or
stuck render can never block tracking/mapping. Uses viser (browser-based,
works over plain HTTP) rather than an OS window — this sidesteps native
GLFW/Wayland issues seen with viz_scripts/final_recon.py's Open3D viewer on
this desktop.

Enable via config: viz.live_viewer = True (see zed2i_local_direct.py).
Then open http://<host>:8080 (or the LAN IP, for the networked Orin setup)
while the capture runs.
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
    return np.array([1.0, 0.0, 0.0, 0.0])


def _wxyz_to_rotmat(wxyz):
    w, x, y, z = [float(v) for v in wxyz]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LiveViewer:
    def __init__(self, node, cfg):
        import viser

        self.node = node
        self.cfg = cfg
        viz_cfg = cfg.get("viz", {})

        port = int(viz_cfg.get("live_viewer_port", 8080))
        self.update_interval_s = float(viz_cfg.get("live_viewer_update_interval_s", 0.5))
        self.max_points = int(viz_cfg.get("live_viewer_max_points", 400_000))

        self._stop = False
        self._trail = []
        self._snapshot_count = 0
        self._save_requested = False
        self._recenter_requested = False

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        # SplaTAM/ZED world is OpenCV convention: +y is down.
        self.server.scene.set_up_direction("-y")

        gui = self.server.gui
        self._stats = gui.add_markdown("**Waiting for first frame...**")

        with gui.add_folder("View"):
            self._cb_true_render = gui.add_checkbox(
                "true splat render (heavier, occasional quality check)",
                initial_value=False,
            )
            self._sl_psize = gui.add_slider(
                "point size (m)", min=0.005, max=0.08, step=0.005,
                initial_value=0.02,
            )
            self._sl_maxpts = gui.add_slider(
                "max points shown (k)", min=50, max=1000, step=50,
                initial_value=min(400, self.max_points // 1000),
            )
            self._btn_recenter = gui.add_button("Recenter view on captured map")

        with gui.add_folder("Capture"):
            self._btn_save = gui.add_button(
                "Save snapshot (PLY) without stopping the walk"
            )

        # Live-tunable knobs — these are all re-read from node.cfg fresh on
        # every frame (confirmed in zed2i_splat_live.py), so writing to
        # node.cfg here takes effect on the next frame, no restart needed.
        # This is the "dials instead of command-line flags" panel: adjust
        # while walking if mapping feels too sparse, too slow, or depth
        # looks noisy at range.
        tcfg = cfg["tracking"]
        mcfg = cfg["mapping"]
        rcfg = cfg["ros"]
        with gui.add_folder("Live tuning (takes effect next frame)"):
            self._sl_map_every = gui.add_slider(
                "map every Nth frame", min=1, max=20, step=1,
                initial_value=int(cfg["map_every"]),
            )
            self._sl_track_iters = gui.add_slider(
                "tracking iterations", min=0, max=100, step=5,
                initial_value=int(tcfg["num_iters"]),
            )
            self._sl_map_iters = gui.add_slider(
                "mapping iterations", min=10, max=300, step=10,
                initial_value=int(mcfg["num_iters"]),
            )
            self._sl_min_depth = gui.add_slider(
                "min depth (m)", min=0.1, max=1.5, step=0.05,
                initial_value=float(rcfg.get("min_depth_m", 0.3)),
            )
            self._sl_max_depth = gui.add_slider(
                "max depth (m)", min=1.0, max=15.0, step=0.5,
                initial_value=float(rcfg.get("max_depth_m", 8.0)),
            )

        self._btn_recenter.on_click(lambda _: setattr(self, "_recenter_requested", True))
        self._btn_save.on_click(lambda _: setattr(self, "_save_requested", True))
        self._sl_map_every.on_update(self._apply_live_params)
        self._sl_track_iters.on_update(self._apply_live_params)
        self._sl_map_iters.on_update(self._apply_live_params)
        self._sl_min_depth.on_update(self._apply_live_params)
        self._sl_max_depth.on_update(self._apply_live_params)

    def _apply_live_params(self, _):
        cfg = self.node.cfg
        cfg["map_every"] = int(self._sl_map_every.value)
        cfg["tracking"]["num_iters"] = int(self._sl_track_iters.value)
        cfg["mapping"]["num_iters"] = int(self._sl_map_iters.value)
        cfg["ros"]["min_depth_m"] = float(self._sl_min_depth.value)
        cfg["ros"]["max_depth_m"] = float(self._sl_max_depth.value)

        self._render_thread = threading.Thread(target=self._loop, daemon=True)
        self._render_thread.start()

        print(f"[LiveViewer] Open http://localhost:{port} "
              f"(or this machine's LAN IP from another device) "
              f"and orbit while you walk.", flush=True)

    def stop(self):
        self._stop = True

    # ---- background loop (own thread; only reads node state) -------------- #

    def _loop(self):
        last_error_log = 0.0
        while not self._stop:
            time.sleep(self.update_interval_s)
            try:
                self._tick()
            except Exception as e:
                # A render hiccup (e.g. mid-mapping tensor resize) must never
                # take the viewer down or touch the SLAM thread.
                now = time.time()
                if now - last_error_log > 5.0:
                    last_error_log = now
                    print(f"[LiveViewer] tick error: {type(e).__name__}: {e}",
                          flush=True)

    def _tick(self):
        node = self.node
        params = node.params
        if params is None or params.get("means3D") is None:
            return

        n_frames = node.total_frames
        n_gaussians = int(params["means3D"].shape[0])
        dropped_total = max(0, node.received_frames - n_frames)
        self._stats.content = (
            f"**Frame {n_frames}/{node.num_frames}** &nbsp; | &nbsp; "
            f"Gaussians: {n_gaussians:,} &nbsp; | &nbsp; "
            f"Received: {node.received_frames} (dropped so far: {dropped_total})"
        )

        curr_c2w = None
        if node.gt_w2c_all_frames:
            w2c = node.gt_w2c_all_frames[-1]
            with torch.no_grad():
                curr_c2w = torch.linalg.inv(w2c).detach().cpu().numpy()

        if self._save_requested:
            self._save_requested = False
            self._save_snapshot()

        if self._cb_true_render.value:
            self._render_true_splat(params)
        else:
            self._update_point_cloud(params)

        self._update_trail(curr_c2w)

        if self._recenter_requested:
            self._recenter_requested = False
            self._recenter(params)

    # ---- visualization ------------------------------------------------------ #

    def _update_point_cloud(self, params):
        with torch.no_grad():
            means = params["means3D"]
            cols = params["rgb_colors"]
            opac = torch.sigmoid(params["logit_opacities"]).squeeze(-1)
            keep = opac > 0.08
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
            "/map",
            points=pts,
            colors=rgb,
            point_size=float(self._sl_psize.value),
            point_shape="circle",
        )

    def _render_true_splat(self, params):
        """Occasional true rasterized render, from the browser's own orbit
        camera, as a background image — for a real quality check without
        stopping the capture. Heavier than the point-cloud proxy."""
        import torch.nn.functional as F
        import sys
        sys.path.insert(0, ".")
        from utils.recon_helpers import setup_camera
        from diff_gaussian_rasterization import GaussianRasterizer as Renderer

        clients = list(self.server.get_clients().values())
        if not clients:
            return
        cam = clients[0].camera

        W, H = 640, 360
        GL2CV = np.diag([1.0, -1.0, -1.0])
        with torch.no_grad():
            R_gl = _wxyz_to_rotmat(cam.wxyz)
            c2w = np.eye(4)
            c2w[:3, :3] = R_gl @ GL2CV
            c2w[:3, 3] = np.array(cam.position)
            w2c = np.linalg.inv(c2w)

            fy = H / (2.0 * np.tan(float(cam.fov) / 2.0))
            k = np.array([[fy, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]])

            scales = torch.exp(params["log_scales"])
            if scales.shape[-1] == 1:
                scales = scales.tile((1, 3))
            rendervar = {
                "means3D": params["means3D"],
                "colors_precomp": params["rgb_colors"],
                "rotations": F.normalize(params["unnorm_rotations"]),
                "opacities": torch.sigmoid(params["logit_opacities"]),
                "scales": scales,
                "means2D": torch.zeros_like(params["means3D"]),
            }
            raster = setup_camera(W, H, k, w2c)
            im, _, _ = Renderer(raster_settings=raster)(**rendervar)
            rgb = (torch.clamp(im, 0, 1).permute(1, 2, 0)
                   .mul(255).byte().cpu().numpy())

        self.server.scene.set_background_image(rgb, format="jpeg", jpeg_quality=80)

    def _update_trail(self, curr_c2w):
        if curr_c2w is None:
            return
        self._trail.append(curr_c2w[:3, 3].copy())
        if len(self._trail) > 4000:
            self._trail = self._trail[-4000:]

        self.server.scene.add_point_cloud(
            "/trail",
            points=np.asarray(self._trail, dtype=np.float32),
            colors=(255, 140, 0),
            point_size=0.015,
            point_shape="circle",
        )
        self.server.scene.add_camera_frustum(
            "/cam",
            fov=1.05,
            aspect=16 / 9,
            scale=0.15,
            color=(255, 60, 60),
            wxyz=_rotmat_to_wxyz(curr_c2w[:3, :3]),
            position=curr_c2w[:3, 3],
        )

    def _recenter(self, params):
        with torch.no_grad():
            pts = params["means3D"].detach().cpu().numpy()
        if len(pts) == 0:
            return
        center = pts.mean(axis=0)
        extent = float(np.percentile(np.linalg.norm(pts - center, axis=1), 90))
        extent = max(extent, 1.0)
        for client in self.server.get_clients().values():
            client.camera.position = center + np.array([0.0, -extent * 0.4, -extent * 1.6])
            client.camera.look_at = center

    # ---- snapshot export without stopping the walk ------------------------- #

    def _save_snapshot(self):
        node = self.node
        if node.params is None:
            return

        import sys
        sys.path.insert(0, ".")
        from scripts.export_ply import save_ply

        snap_dir = node.output_dir / "live_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        ply_path = snap_dir / f"snapshot_{self._snapshot_count:03d}.ply"
        self._snapshot_count += 1

        with torch.no_grad():
            means = node.params["means3D"].detach().cpu().numpy()
            scales = node.params["log_scales"].detach().cpu().numpy()
            rotations = node.params["unnorm_rotations"].detach().cpu().numpy()
            rgbs = node.params["rgb_colors"].detach().cpu().numpy()
            opacities = node.params["logit_opacities"].detach().cpu().numpy()

        save_ply(str(ply_path), means, scales, rotations, rgbs, opacities)
        print(f"[LiveViewer] Snapshot saved (no stopping needed): {ply_path}\n"
              f"  Open it in https://playcanvas.com/supersplat/editor",
              flush=True)
