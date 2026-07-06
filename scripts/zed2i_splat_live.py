#!/usr/bin/env python3
"""Live SplaTAM from a ZED 2i over ROS2, with VIO-fused poses, ZED neural depth,
an operator web viewer, and operator-selectable high-quality ROI refinement.

Pipeline per frame:
  1. Sync RGB + ZED (neural) depth + VIO odometry over ROS2.
  2. Seed the camera pose from the ZED VIO odometry (accurate, drift-bounded).
  3. Refine that pose against the current Gaussian map (SplaTAM tracking) so the
     splat stays crisp -- the refined pose is what we keep (this was previously
     computed and then thrown away).
  4. Map: add Gaussians and optimise the map.
  5. Publish a downsampled snapshot to the operator's browser viewer.

When the operator draws a box in the viewer and hits "Refine", we run a focused,
higher-iteration optimisation of that region and emit next-best-view guidance.

See docs/CAPTURE_PATTERNS.md and docs/OPERATOR_VIEWER.md.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from importlib.machinery import SourceFileLoader

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from utils.common_utils import seed_everything, save_params
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation, prune_gaussians, densify
from utils.zed_depth import (
    decode_depth_to_meters,
    sanitize_depth,
    apply_confidence_mask,
    depth_fill_ratio,
)
from scripts.splatam import (
    get_loss,
    initialize_optimizer,
    initialize_params,
    initialize_camera_pose,
    get_pointcloud,
    add_new_gaussians,
)

from livesplat.bridge import LiveSplatBridge
from livesplat.viewer_server import OperatorViewerServer
from livesplat import capture_advisor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/zed2i/zed2i_splat_live.py", type=str)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Small geometry / message helpers                                            #
# --------------------------------------------------------------------------- #
def caminfo_to_K(cam_info):
    return np.array(cam_info.k, dtype=np.float32).reshape(3, 3)


def ros_rgb_to_rgb(rgb_cv, encoding):
    enc = (encoding or "").lower()
    if rgb_cv.ndim == 2:
        return cv2.cvtColor(rgb_cv, cv2.COLOR_GRAY2RGB)
    if rgb_cv.dtype != np.uint8:
        rgb_cv = np.clip(rgb_cv, 0, 255).astype(np.uint8)
    if rgb_cv.shape[2] == 4:
        if "bgra" in enc:
            return cv2.cvtColor(rgb_cv, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(rgb_cv, cv2.COLOR_RGBA2RGB)
    if rgb_cv.shape[2] == 3:
        if "bgr" in enc:
            return cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
        return rgb_cv
    raise ValueError(f"Unexpected RGB image shape: {rgb_cv.shape}")


def rotmat_to_quat_wxyz(R):
    """3x3 rotation matrix -> quaternion (w, x, y, z), SplaTAM order."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = torch.stack([w, x, y, z])
    return q / torch.linalg.norm(q)


def odom_to_c2w(odom_msg, device):
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = torch.tensor(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        device=device,
        dtype=torch.float32,
    )
    c2w = torch.eye(4, device=device, dtype=torch.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = torch.tensor([p.x, p.y, p.z], device=device, dtype=torch.float32)
    return c2w


def zed_link_to_left_optical(device):
    """Static transform ZED body link -> left camera optical frame (from /tf_static)."""
    T = torch.eye(4, device=device, dtype=torch.float32)
    T[:3, 3] = torch.tensor([-0.01, 0.06, 0.015], device=device, dtype=torch.float32)
    T[:3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    return T


def colorize_depth(depth_m, min_depth=0.4, max_depth=5.0):
    """Debug visualisation only (written to disk when explicitly enabled)."""
    valid = np.isfinite(depth_m) & (depth_m > min_depth) & (depth_m < max_depth)
    depth_color = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if np.count_nonzero(valid) == 0:
        return depth_color
    lo = np.percentile(depth_m[valid], 2)
    hi = np.percentile(depth_m[valid], 98)
    if hi <= lo:
        lo, hi = min_depth, max_depth
    depth_norm = np.zeros_like(depth_m, dtype=np.float32)
    depth_norm[valid] = 1.0 - np.clip((depth_m[valid] - lo) / (hi - lo), 0.0, 1.0)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_color[~valid] = (0, 0, 0)
    return depth_color


# --------------------------------------------------------------------------- #
# Live node                                                                   #
# --------------------------------------------------------------------------- #
class ZedSplatamOnline(Node):
    def __init__(self, config):
        super().__init__("ZedSplatamOnline")

        self.cfg = config
        self.bridge = CvBridge()
        self.device = torch.device(self.cfg.get("primary_device", "cuda:0"))
        self.ros_cfg = self.cfg["ros"]

        self.num_frames = int(self.cfg["num_frames"])
        self.total_frames = 0      # frames actually processed by SplaTAM
        self.received_frames = 0   # all synced ROS frames received
        self.process_every_n = int(self.ros_cfg.get("process_every_n", 1))

        # Pose fusion mode:
        #   "vio_seed_refine" -> VIO seeds pose, SplaTAM tracking refines it (default)
        #   "vio_only"        -> trust VIO directly, no tracking refinement
        #   "splatam_only"    -> ignore VIO, constant-velocity seed + tracking
        self.pose_mode = self.cfg.get("pose_mode", "vio_seed_refine")

        # Housekeeping / debug cadence (0 disables).
        hk = self.cfg.get("housekeeping", {})
        self.empty_cache_every = int(hk.get("empty_cache_every", 50))
        self.save_depth_every = int(self.cfg.get("debug", {}).get("save_depth_every", 0))

        self.workdir = Path(self.cfg["workdir"])
        self.run_name = self.cfg.get("run_name", "SplaTAM_ZED2i_ROS2")
        self.output_dir = self.workdir / self.run_name
        if self.cfg.get("overwrite", False) and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SLAM state
        self.params = None
        self.variables = None
        self.intrinsics = None
        self.first_frame_w2c = None
        self.cam = None
        self.densify_intrinsics = None
        self.densify_cam = None
        self.first_zed_optical_c2w = None

        self.keyframe_list = []
        self.keyframe_time_indices = []
        self.gt_w2c_all_frames = []

        self.tracking_iter_time_sum = 0.0
        self.tracking_iter_time_count = 0
        self.mapping_iter_time_sum = 0.0
        self.mapping_iter_time_count = 0

        self.latest_rgb_info = None
        self.latest_confidence = None  # most recent confidence map (np.float32)
        self.current_roi = None        # last RegionOfInterest requested by operator

        # ---- Operator viewer + refine bridge --------------------------------
        vcfg = self.cfg.get("viewer", {})
        self.viewer_enabled = bool(vcfg.get("enable", True))
        self.viewer_publish_every = int(vcfg.get("publish_every", 3))
        self.live_bridge = LiveSplatBridge(
            max_preview_points=int(vcfg.get("max_preview_points", 150_000))
        )
        self.viewer_server = None
        if self.viewer_enabled:
            self.viewer_server = OperatorViewerServer(
                self.live_bridge,
                host=vcfg.get("host", "0.0.0.0"),
                port=int(vcfg.get("port", 8080)),
            ).start()
            self.get_logger().info(
                f"Operator viewer at http://<this-machine>:{vcfg.get('port', 8080)}/"
            )

        # ---- ROS wiring ------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(CameraInfo, self.ros_cfg["rgb_info_topic"], self.rgb_info_cb, qos)

        conf_topic = self.ros_cfg.get("confidence_topic")
        if conf_topic:
            self.create_subscription(Image, conf_topic, self.confidence_cb, qos)

        self.rgb_sub = Subscriber(self, Image, self.ros_cfg["rgb_topic"], qos_profile=qos)
        self.depth_sub = Subscriber(self, Image, self.ros_cfg["depth_topic"], qos_profile=qos)
        self.odom_sub = Subscriber(self, Odometry, self.ros_cfg["odom_topic"], qos_profile=qos)

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=100,
            slop=float(self.ros_cfg.get("sync_slop_s", 0.15)),
            allow_headerless=False,
        )
        self.ts.registerCallback(self.synced_cb)

        # Service operator ROI requests on a timer, independent of frame arrival,
        # so refinement also works after capture ends or when a bag has stopped.
        # Single-threaded executor => this never races the frame callback.
        self.finished = False
        self.create_timer(0.2, self._on_timer)

        self.get_logger().info("ZedSplatamOnline ready.")
        self.get_logger().info(f"RGB:   {self.ros_cfg['rgb_topic']}")
        self.get_logger().info(f"Depth: {self.ros_cfg['depth_topic']}  (expecting ZED NEURAL depth)")
        self.get_logger().info(f"Odom:  {self.ros_cfg['odom_topic']}  pose_mode={self.pose_mode}")

    # ---- ROS callbacks ------------------------------------------------------
    def rgb_info_cb(self, msg):
        self.latest_rgb_info = msg

    def confidence_cb(self, msg):
        try:
            conf = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_confidence = np.asarray(conf, dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Failed to decode confidence map: {e}")

    # ---- Frame construction -------------------------------------------------
    def _decode_rgbd(self, rgb_msg, depth_msg):
        """Decode ROS RGB + ZED neural depth into cleaned numpy (rgb uint8, depth m)."""
        rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        rgb = ros_rgb_to_rgb(rgb_cv, rgb_msg.encoding)

        depth_m = decode_depth_to_meters(depth_cv, depth_msg.encoding)
        depth_m = sanitize_depth(
            depth_m,
            min_depth_m=float(self.ros_cfg.get("min_depth_m", 0.3)),
            max_depth_m=float(self.ros_cfg.get("max_depth_m", 8.0)),
            unit_scale_m=float(self.ros_cfg.get("depth_unit_scale_m", 1.0)),
        )
        # Confidence masking: only if the map matches the native depth resolution.
        conf = self.latest_confidence
        if conf is not None and conf.shape == depth_m.shape:
            depth_m = apply_confidence_mask(
                depth_m, conf, float(self.ros_cfg.get("confidence_max_cost", 50.0))
            )
        return rgb, depth_m

    def _to_tensor_frame(self, rgb, depth_m, rgb_info, width, height):
        rgb_rs = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        depth_rs = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
        depth_rs = np.expand_dims(depth_rs.astype(np.float32), axis=-1)

        color = torch.from_numpy(rgb_rs).to(self.device).float().permute(2, 0, 1) / 255.0
        depth = torch.from_numpy(depth_rs).to(self.device).float().permute(2, 0, 1)

        K = caminfo_to_K(rgb_info)
        sx = width / float(int(rgb_info.width))
        sy = height / float(int(rgb_info.height))
        K_scaled = K.copy()
        K_scaled[0, 0] *= sx
        K_scaled[1, 1] *= sy
        K_scaled[0, 2] *= sx
        K_scaled[1, 2] *= sy
        intrinsics = torch.tensor(K_scaled, device=self.device).float()
        return color, depth, intrinsics, K_scaled

    def make_frame(self, rgb, depth_m, rgb_info):
        W = int(self.cfg["data"]["desired_image_width"])
        H = int(self.cfg["data"]["desired_image_height"])
        return self._to_tensor_frame(rgb, depth_m, rgb_info, W, H)

    def make_densify_frame(self, rgb, depth_m, rgb_info):
        dW = int(self.cfg["data"]["densification_image_width"])
        dH = int(self.cfg["data"]["densification_image_height"])
        return self._to_tensor_frame(rgb, depth_m, rgb_info, dW, dH)

    # ---- Pose helpers -------------------------------------------------------
    def set_camera_pose_from_w2c(self, time_idx, w2c):
        q = rotmat_to_quat_wxyz(w2c[:3, :3])
        t = w2c[:3, 3]
        rot_slot = self.params["cam_unnorm_rots"][..., time_idx]
        trans_slot = self.params["cam_trans"][..., time_idx]
        self.params["cam_unnorm_rots"][..., time_idx] = q.reshape_as(rot_slot)
        self.params["cam_trans"][..., time_idx] = t.reshape_as(trans_slot)

    def curr_w2c_from_params(self, time_idx):
        with torch.no_grad():
            rot = F.normalize(self.params["cam_unnorm_rots"][..., time_idx].detach())
            tran = self.params["cam_trans"][..., time_idx].detach()
            w2c = torch.eye(4, device=self.device).float()
            w2c[:3, :3] = build_rotation(rot)
            w2c[:3, 3] = tran
        return w2c

    def zed_w2c_relative(self, odom_msg):
        """VIO odom -> world-to-cam in the map (first-frame) frame."""
        zed_link_c2w = odom_to_c2w(odom_msg, self.device)
        zed_optical_c2w = zed_link_c2w @ zed_link_to_left_optical(self.device)
        if self.first_zed_optical_c2w is None:
            self.first_zed_optical_c2w = zed_optical_c2w.detach().clone()
        rel_c2w = torch.linalg.inv(self.first_zed_optical_c2w) @ zed_optical_c2w
        return torch.linalg.inv(rel_c2w)

    # ---- First-frame map init ----------------------------------------------
    def initialize_first_frame(self, K_scaled, rgb, depth_m, rgb_info):
        self.first_frame_w2c = torch.eye(4, device=self.device).float()
        W = int(self.cfg["data"]["desired_image_width"])
        H = int(self.cfg["data"]["desired_image_height"])
        self.cam = setup_camera(W, H, K_scaled, self.first_frame_w2c.detach().cpu().numpy())

        densify_color, densify_depth, densify_intrinsics, Kd = self.make_densify_frame(
            rgb, depth_m, rgb_info
        )
        self.densify_intrinsics = densify_intrinsics
        dW = int(self.cfg["data"]["densification_image_width"])
        dH = int(self.cfg["data"]["densification_image_height"])
        self.densify_cam = setup_camera(dW, dH, Kd, self.first_frame_w2c.detach().cpu().numpy())

        mask = (densify_depth > 0).reshape(-1)
        init_pt_cld, mean3_sq_dist = get_pointcloud(
            densify_color,
            densify_depth,
            densify_intrinsics,
            self.first_frame_w2c,
            mask=mask,
            compute_mean_sq_dist=True,
            mean_sq_dist_method=self.cfg["mean_sq_dist_method"],
        )
        self.params, self.variables = initialize_params(
            init_pt_cld,
            self.num_frames,
            mean3_sq_dist,
            self.cfg.get("gaussian_distribution", "isotropic"),
        )
        self.variables["scene_radius"] = (
            torch.max(densify_depth) / self.cfg["scene_radius_depth_ratio"]
        )

    # ---- Tracking (VIO-seeded refinement) ----------------------------------
    def refine_pose_tracking(self, time_idx, tracking_curr_data):
        optimizer = initialize_optimizer(self.params, self.cfg["tracking"]["lrs"], tracking=True)
        candidate_rot = self.params["cam_unnorm_rots"][..., time_idx].detach().clone()
        candidate_trn = self.params["cam_trans"][..., time_idx].detach().clone()
        current_min_loss = float(1e20)

        for it in range(int(self.cfg["tracking"]["num_iters"])):
            iter_start = time.time()
            loss, self.variables, _ = get_loss(
                self.params,
                tracking_curr_data,
                self.variables,
                time_idx,
                self.cfg["tracking"]["loss_weights"],
                self.cfg["tracking"]["use_sil_for_loss"],
                self.cfg["tracking"]["sil_thres"],
                self.cfg["tracking"]["use_l1"],
                self.cfg["tracking"]["ignore_outlier_depth_loss"],
                tracking=True,
                visualize_tracking_loss=self.cfg["tracking"].get("visualize_tracking_loss", False),
                tracking_iteration=it,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_rot = self.params["cam_unnorm_rots"][..., time_idx].detach().clone()
                    candidate_trn = self.params["cam_trans"][..., time_idx].detach().clone()
            self.tracking_iter_time_sum += time.time() - iter_start
            self.tracking_iter_time_count += 1

        # Keep the best-seen pose (this is the fix: previously discarded).
        with torch.no_grad():
            self.params["cam_unnorm_rots"][..., time_idx] = candidate_rot
            self.params["cam_trans"][..., time_idx] = candidate_trn

    # ---- Mapping ------------------------------------------------------------
    def map_step(self, time_idx, color, depth, curr_w2c, curr_gt_w2c, rgb_np, depth_np, rgb_info):
        if self.cfg["mapping"]["add_new_gaussians"] and time_idx > 0:
            densify_color, densify_depth, densify_intrinsics, _ = self.make_densify_frame(
                rgb_np, depth_np, rgb_info
            )
            densify_curr_data = {
                "cam": self.densify_cam,
                "im": densify_color,
                "depth": densify_depth,
                "id": time_idx,
                "intrinsics": densify_intrinsics,
                "w2c": curr_w2c,
                "iter_gt_w2c_list": curr_gt_w2c,
            }
            self.params, self.variables = add_new_gaussians(
                self.params,
                self.variables,
                densify_curr_data,
                self.cfg["mapping"]["sil_thres"],
                time_idx,
                self.cfg["mean_sq_dist_method"],
                self.cfg.get("gaussian_distribution", "isotropic"),
            )

        with torch.no_grad():
            num_keyframes = int(self.cfg["mapping_window_size"]) - 2
            selected_keyframes = keyframe_selection_overlap(
                depth, curr_w2c, self.intrinsics, self.keyframe_list[:-1], num_keyframes
            )
            selected_time_idx = [self.keyframe_list[i]["id"] for i in selected_keyframes]
            if len(self.keyframe_list) > 0:
                selected_time_idx.append(self.keyframe_list[-1]["id"])
                selected_keyframes.append(len(self.keyframe_list) - 1)
            selected_time_idx.append(time_idx)
            selected_keyframes.append(-1)

        optimizer = initialize_optimizer(self.params, self.cfg["mapping"]["lrs"], tracking=False)
        for it in range(int(self.cfg["mapping"]["num_iters"])):
            iter_start = time.time()
            rand_idx = np.random.randint(0, len(selected_keyframes))
            selected_kf_idx = selected_keyframes[rand_idx]
            if selected_kf_idx == -1:
                iter_time_idx, iter_color, iter_depth = time_idx, color, depth
            else:
                iter_time_idx = self.keyframe_list[selected_kf_idx]["id"]
                iter_color = self.keyframe_list[selected_kf_idx]["color"]
                iter_depth = self.keyframe_list[selected_kf_idx]["depth"]

            iter_data = {
                "cam": self.cam,
                "im": iter_color,
                "depth": iter_depth,
                "id": iter_time_idx,
                "intrinsics": self.intrinsics,
                "w2c": self.first_frame_w2c,
                "iter_gt_w2c_list": self.gt_w2c_all_frames[: iter_time_idx + 1],
            }
            loss, self.variables, _ = get_loss(
                self.params,
                iter_data,
                self.variables,
                iter_time_idx,
                self.cfg["mapping"]["loss_weights"],
                self.cfg["mapping"]["use_sil_for_loss"],
                self.cfg["mapping"]["sil_thres"],
                self.cfg["mapping"]["use_l1"],
                self.cfg["mapping"]["ignore_outlier_depth_loss"],
                mapping=True,
            )
            loss.backward()
            with torch.no_grad():
                if self.cfg["mapping"]["prune_gaussians"]:
                    self.params, self.variables = prune_gaussians(
                        self.params, self.variables, optimizer, it, self.cfg["mapping"]["pruning_dict"]
                    )
                if self.cfg["mapping"]["use_gaussian_splatting_densification"]:
                    self.params, self.variables = densify(
                        self.params, self.variables, optimizer, it, self.cfg["mapping"]["densify_dict"]
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            self.mapping_iter_time_sum += time.time() - iter_start
            self.mapping_iter_time_count += 1

    # ---- Timer: operator ROI servicing (runs during and after capture) ------
    def _on_timer(self):
        if self.params is None:
            return
        if self.live_bridge.pending_requests() > 0:
            self.process_refine_requests()

    # ---- Operator ROI refinement -------------------------------------------
    def process_refine_requests(self):
        req = self.live_bridge.pop_refine_request()
        if req is None:
            return
        self.current_roi = req.roi
        self.get_logger().info(
            f"[ROI] Refine request #{req.request_id} label={req.roi.label} "
            f"quality={req.quality} -- pausing live capture."
        )
        self.run_roi_refinement(req)
        self.publish_guidance(force=True)

    def run_roi_refinement(self, req):
        """Focused, higher-iteration optimisation of the selected region.

        Strategy: pick the keyframes whose observed geometry falls inside the
        ROI, then run extra mapping iterations biased toward those views. More
        ``quality`` -> more iterations and (optionally) more Gaussians. This is
        the "spend more time for a better splat of this object" path.
        """
        if self.params is None or len(self.keyframe_list) == 0:
            return
        roi = req.roi
        means_np = self.params["means3D"].detach().cpu().numpy()
        inside = roi.contains_points(means_np)
        n_inside = int(inside.sum())
        self.get_logger().info(f"[ROI] {n_inside} Gaussians currently inside the box.")

        # Which keyframes actually see the ROI? Project the ROI centre into each
        # keyframe and keep those where it lands in front of the camera.
        roi_center = torch.tensor(roi.center(), device=self.device, dtype=torch.float32)
        roi_kfs = []
        for kf in self.keyframe_list:
            w2c = kf["est_w2c"]
            pc = (w2c @ torch.cat([roi_center, torch.ones(1, device=self.device)]))[:3]
            if float(pc[2]) > 0.05:  # in front of camera
                roi_kfs.append(kf)
        if not roi_kfs:
            roi_kfs = self.keyframe_list[-min(len(self.keyframe_list), 8):]

        base_iters = int(self.cfg.get("roi_refine", {}).get("base_iters", 300))
        n_iters = int(base_iters * max(1.0, float(req.quality)))
        self.get_logger().info(
            f"[ROI] Refining over {len(roi_kfs)} keyframes for {n_iters} iterations."
        )

        optimizer = initialize_optimizer(self.params, self.cfg["mapping"]["lrs"], tracking=False)
        for it in range(n_iters):
            kf = roi_kfs[np.random.randint(0, len(roi_kfs))]
            iter_time_idx = kf["id"]
            iter_data = {
                "cam": self.cam,
                "im": kf["color"],
                "depth": kf["depth"],
                "id": iter_time_idx,
                "intrinsics": self.intrinsics,
                "w2c": self.first_frame_w2c,
                "iter_gt_w2c_list": self.gt_w2c_all_frames[: iter_time_idx + 1],
            }
            loss, self.variables, _ = get_loss(
                self.params,
                iter_data,
                self.variables,
                iter_time_idx,
                self.cfg["mapping"]["loss_weights"],
                self.cfg["mapping"]["use_sil_for_loss"],
                self.cfg["mapping"]["sil_thres"],
                self.cfg["mapping"]["use_l1"],
                self.cfg["mapping"]["ignore_outlier_depth_loss"],
                mapping=True,
            )
            loss.backward()
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        self.publish_snapshot(force=True)
        # If capture already finished, persist the improved splat immediately.
        if self.finished:
            self._save_output()
        self.get_logger().info(f"[ROI] Refine request #{req.request_id} complete.")

    # ---- Viewer publishing --------------------------------------------------
    def publish_snapshot(self, force=False):
        if not self.viewer_enabled or self.params is None:
            return
        if not force and (self.total_frames % max(1, self.viewer_publish_every)) != 0:
            return
        with torch.no_grad():
            means = self.params["means3D"].detach().cpu().numpy()
            rgb = self.params["rgb_colors"].detach().cpu().numpy()
        self.live_bridge.publish_snapshot(means, rgb)

    def publish_guidance(self, force=False):
        if not self.viewer_enabled or self.current_roi is None:
            return
        if len(self.gt_w2c_all_frames) == 0:
            return
        w2c_stack = np.stack([m.detach().cpu().numpy() for m in self.gt_w2c_all_frames], axis=0)
        guidance = capture_advisor.analyse_coverage(self.current_roi, w2c_stack)
        self.live_bridge.publish_guidance(guidance)

    # ---- Main synced callback ----------------------------------------------
    def synced_cb(self, rgb_msg, depth_msg, odom_msg):
        self.received_frames += 1
        if self.latest_rgb_info is None:
            self.get_logger().warn("Waiting for RGB CameraInfo...")
            return
        if self.received_frames % self.process_every_n != 0:
            return
        if self.total_frames >= self.num_frames:
            return

        rgb_info = self.latest_rgb_info
        frame_start = time.time()
        time_idx = self.total_frames

        rgb_np, depth_np = self._decode_rgbd(rgb_msg, depth_msg)
        color, depth, intrinsics, K_scaled = self.make_frame(rgb_np, depth_np, rgb_info)

        if self.save_depth_every and (time_idx % self.save_depth_every == 0):
            dbg = self.output_dir / "depth_color_debug"
            dbg.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dbg / f"depth_{time_idx:05d}.png"),
                        colorize_depth(depth_np,
                                       float(self.ros_cfg.get("min_depth_m", 0.4)),
                                       float(self.ros_cfg.get("max_depth_m", 5.0))))

        if time_idx == 0:
            self.initialize_first_frame(K_scaled, rgb_np, depth_np, rgb_info)
            self.intrinsics = intrinsics
            self.get_logger().info(
                f"ODOM frame_id={odom_msg.header.frame_id} child={odom_msg.child_frame_id} "
                f"depth_fill={depth_fill_ratio(depth_np):.2f}"
            )

        # VIO-derived world-to-cam in the map frame.
        zed_w2c = self.zed_w2c_relative(odom_msg)

        # Seed + (optionally) refine the pose for this frame.
        if time_idx > 0:
            if self.pose_mode in ("vio_seed_refine", "vio_only"):
                self.params = initialize_camera_pose(self.params, time_idx, forward_prop=False)
                with torch.no_grad():
                    self.set_camera_pose_from_w2c(time_idx, zed_w2c)
            else:  # splatam_only
                self.params = initialize_camera_pose(
                    self.params, time_idx, forward_prop=self.cfg["tracking"].get("forward_prop", True)
                )

        # Store per-frame pose estimate (refined below for iter_gt_w2c_list slices).
        self.gt_w2c_all_frames.append(self.curr_w2c_from_params(time_idx).detach().clone())
        curr_gt_w2c = self.gt_w2c_all_frames

        if (
            time_idx > 0
            and self.pose_mode != "vio_only"
            and not self.cfg["tracking"]["use_gt_poses"]
        ):
            tracking_curr_data = {
                "cam": self.cam,
                "im": color,
                "depth": depth,
                "id": time_idx,
                "intrinsics": self.intrinsics,
                "w2c": zed_w2c,
                "iter_gt_w2c_list": curr_gt_w2c,
            }
            self.refine_pose_tracking(time_idx, tracking_curr_data)

        # Pose of record = refined params pose (fix: no longer overwritten by raw VIO).
        curr_w2c = self.curr_w2c_from_params(time_idx)
        self.gt_w2c_all_frames[time_idx] = curr_w2c.detach().clone()

        # Mapping.
        if time_idx == 0 or (time_idx + 1) % int(self.cfg["map_every"]) == 0:
            self.map_step(time_idx, color, depth, curr_w2c, curr_gt_w2c, rgb_np, depth_np, rgb_info)

        # Keyframing.
        if (
            time_idx == 0
            or (time_idx + 1) % int(self.cfg["keyframe_every"]) == 0
            or time_idx == self.num_frames - 2
        ):
            self.keyframe_list.append(
                {"id": time_idx, "est_w2c": curr_w2c.detach().clone(), "color": color, "depth": depth}
            )
            self.keyframe_time_indices.append(time_idx)

        self.total_frames += 1

        # Viewer + guidance + housekeeping.
        self.publish_snapshot()
        self.publish_guidance()
        self.live_bridge.publish_status(
            frame=self.total_frames,
            num_frames=self.num_frames,
            num_gaussians=int(self.params["means3D"].shape[0]),
            preview_points=int(min(self.params["means3D"].shape[0], self.live_bridge.max_preview_points)),
            depth_fill=round(depth_fill_ratio(depth_np), 3),
        )

        if self.empty_cache_every and (self.total_frames % self.empty_cache_every == 0):
            torch.cuda.empty_cache()

        frame_dt = time.time() - frame_start
        fps = 1.0 / frame_dt if frame_dt > 0 else 0.0
        self.get_logger().info(
            f"Frame {self.total_frames}/{self.num_frames} | FPS={fps:.2f} | "
            f"gaussians={int(self.params['means3D'].shape[0]):,} | "
            f"depth_fill={depth_fill_ratio(depth_np):.2f}"
        )

        if self.total_frames >= self.num_frames:
            self.finalize_and_exit()

    def _save_output(self):
        """Save params + SLAM metadata to params.npz.

        Builds a separate export dict rather than mutating ``self.params`` with
        non-tensor keys -- otherwise a later ROI refinement's optimizer would
        choke on the injected numpy/int entries.
        """
        export = dict(self.params)
        export["timestep"] = self.variables["timestep"]
        export["intrinsics"] = self.intrinsics.detach().cpu().numpy()
        export["w2c"] = self.first_frame_w2c.detach().cpu().numpy()
        export["org_width"] = self.cfg["data"]["desired_image_width"]
        export["org_height"] = self.cfg["data"]["desired_image_height"]
        export["gt_w2c_all_frames"] = np.stack(
            [m.detach().cpu().numpy() for m in self.gt_w2c_all_frames], axis=0
        )
        export["keyframe_time_indices"] = np.array(self.keyframe_time_indices)
        save_params(export, str(self.output_dir))
        self.get_logger().info(f"Saved SplaTAM output to: {self.output_dir}")

    def finalize_and_exit(self):
        if self.finished:
            return
        self.finished = True
        self.get_logger().info("Reached final frame. Saving output...")
        self._save_output()
        self.publish_snapshot(force=True)

        if self.viewer_server is not None:
            # Keep spinning so the operator can keep inspecting / refining the
            # final splat (the ROI timer stays live). Ctrl-C in the main thread
            # exits cleanly via main()'s finally block.
            self.get_logger().info(
                "Operator viewer still serving the final splat. "
                "Box a region and hit Refine, or Ctrl-C to exit."
            )
            return
        os._exit(0)


def main():
    args = parse_args()
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    cfg = experiment.config
    if "gaussian_distribution" not in cfg:
        cfg["gaussian_distribution"] = "isotropic"

    seed_everything(seed=cfg["seed"])
    rclpy.init()
    node = ZedSplatamOnline(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node.viewer_server is not None:
                node.viewer_server.stop()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
