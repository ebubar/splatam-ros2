#!/usr/bin/env python3
"""
Stream RGB + aligned depth from ZED2i ROS2 topics and run SplaTAM online (live).
This mirrors scripts/iphone_demo.py but uses ROS2 + message_filters.

Expected topics:
  RGB:   /zed/zed_node/rgb/color/rect/image
  RGB K: /zed/zed_node/rgb/color/rect/image/camera_info
  Depth: /zed/zed_node/depth/depth_registered
  D K:   /zed/zed_node/depth/depth_registered/camera_info
"""

import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from utils.common_utils import seed_everything, save_params
from utils.recon_helpers import setup_camera
from utils.keyframe_selection import keyframe_selection_overlap
from utils.slam_external import build_rotation, prune_gaussians, densify
from utils.live_renderer import LiveRenderer, LiveRendererConfig

# Reuse SplaTAM internals (same as iphone_demo.py)
from scripts.splatam import (
    get_loss,
    initialize_optimizer,
    initialize_params,
    initialize_camera_pose,
    get_pointcloud,
    add_new_gaussians,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./configs/zed/online_demo.py", type=str)

    # ---- Live Rendering Flags ----
    p.add_argument("--live_cam", action="store_true",
                   help="Enable live camera rendering")
    p.add_argument("--live_splat", action="store_true",
                   help="Enable live splat rendering")
    p.add_argument("--live_max_fps", type=float, default=None,
                   help="Max FPS for live renderer")
    p.add_argument("--live_depth", action="store_true",
                help="Enable live depth rendering (separate window)")
    return p.parse_args()


def caminfo_to_K(cam_info: CameraInfo) -> np.ndarray:
    K = np.array(cam_info.k, dtype=np.float32).reshape(3, 3)
    return K


def depth_to_meters(depth_cv: np.ndarray, encoding: str) -> np.ndarray:
    """
    Convert depth image to float32 meters (H, W).
    Handles common encodings:
      - 32FC1: already meters (float32)
      - 16UC1: often millimeters from depth sensors -> convert to meters
    """
    enc = (encoding or "").lower()
    if enc in ["32fc1", "32fc"]:
        if depth_cv.dtype != np.float32:
            depth_cv = depth_cv.astype(np.float32)
        return depth_cv

    if enc in ["16uc1", "16uc"]:
        # very common convention: mm
        if depth_cv.dtype != np.uint16:
            depth_cv = depth_cv.astype(np.uint16)
        return depth_cv.astype(np.float32) / 1000.0

    # Fallback: try float conversion (better than crashing)
    return depth_cv.astype(np.float32)

def pose_is_valid(params, time_idx, eps=1e-6) -> bool:
    """
    Returns False if the current estimated pose would likely produce a singular w2c.
    This prevents add_new_gaussians -> get_pointcloud -> inverse(w2c) crashes.
    """
    q = params["cam_unnorm_rots"][..., time_idx].detach()
    t = params["cam_trans"][..., time_idx].detach()

    # Finite check
    if (not torch.isfinite(q).all()) or (not torch.isfinite(t).all()):
        return False

    # Safe normalize quaternion
    qn = F.normalize(q, dim=-1, eps=eps)
    if not torch.isfinite(qn).all():
        return False

    # Rotation matrix should be finite and non-degenerate
    R = build_rotation(qn)
    if not torch.isfinite(R).all():
        return False

    detR = torch.det(R)
    if (not torch.isfinite(detR)) or (torch.abs(detR) < 1e-6):
        return False

    return True

def w2c_from_params(params, time_idx, device, eps=1e-6):
    q = params["cam_unnorm_rots"][..., time_idx].detach()
    t = params["cam_trans"][..., time_idx].detach()

    if (not torch.isfinite(q).all()) or (not torch.isfinite(t).all()):
        return None

    qn = F.normalize(q, dim=-1, eps=eps)
    if not torch.isfinite(qn).all():
        return None

    R = build_rotation(qn)
    if not torch.isfinite(R).all():
        return None

    detR = torch.det(R)
    if (not torch.isfinite(detR)) or (torch.abs(detR) < 1e-6):
        return None

    w2c = torch.eye(4, device=device).float()
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c

def torch_rgb_chw_to_bgr8(img_chw: torch.Tensor) -> np.ndarray:
    """
    img_chw: torch float tensor in [0,1], shape (3,H,W), RGB
    returns: uint8 BGR image (H,W,3)
    """
    img = img_chw.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # HWC RGB float
    img = (img * 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class ZedSplatamOnline(Node):
    def __init__(self, config: dict):
        super().__init__("zed_splatam_online")

        self.cfg = config
        self.bridge = CvBridge()
        self._done = False
        self._final_saved = False
        self._shutdown_timer = self.create_timer(0.1, self._shutdown_tick)
        # Where to save outputs
        self.workdir = Path(self.cfg["workdir"])
        self.run_name = self.cfg.get("run_name", "SplaTAM_ZED2i")
        self.output_dir = self.workdir / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.params = None
        self.variables = None
        self.intrinsics = None
        self.first_frame_w2c = None
        self.cam = None
        # Optional: overwrite behavior
        if self.cfg.get("overwrite", False) and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # QoS: ZED is RELIABLE (from your output). Match that.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        rgb_topic = self.cfg["ros"]["rgb_topic"]
        depth_topic = self.cfg["ros"]["depth_topic"]
        rgb_info_topic = self.cfg["ros"]["rgb_info_topic"]
        depth_info_topic = self.cfg["ros"]["depth_info_topic"]

        self.get_logger().info(f"Subscribing:\n  RGB: {rgb_topic}\n  DEPTH: {depth_topic}")

        self.rgb_sub = Subscriber(self, Image, rgb_topic, qos_profile=qos)
        self.depth_sub = Subscriber(self, Image, depth_topic, qos_profile=qos)
        self.rgb_info_sub = Subscriber(self, CameraInfo, rgb_info_topic, qos_profile=qos)
        self.depth_info_sub = Subscriber(self, CameraInfo, depth_info_topic, qos_profile=qos)

        # Approx sync (ZED stamps are usually close; we’ll allow a little slop)
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.rgb_info_sub, self.depth_info_sub],
            queue_size=20,
            slop=0.05,  # seconds
            allow_headerless=False,
        )
        self.ts.registerCallback(self.synced_cb)

        # SplaTAM state
        self.t0 = None
        self.total_frames = 0
        self.num_frames = int(self.cfg["num_frames"])
        self.device = torch.device(self.cfg.get("primary_device", "cuda:0"))
        self.live = LiveRenderer(
            LiveRendererConfig(
                live_cam=bool(self.cfg.get("live_cam", False)),
                live_depth=bool(self.cfg.get("live_depth", False)),   # ✅ ADD THIS
                live_splat=bool(self.cfg.get("live_splat", False)),
                max_fps=float(self.cfg.get("live_max_fps", 30.0)),
            )
        )
        self.get_logger().info(
        f"LiveRenderer: live_cam={self.cfg.get('live_cam')} "
        f"live_depth={self.cfg.get('live_depth')} "
        f"live_splat={self.cfg.get('live_splat')}"
         )   
        self.last_valid_w2c = torch.eye(4, device=self.device).float()
        self.last_valid_time_idx = 0
        self.params = None
        self.variables = None
        self.intrinsics = None
        self.first_frame_w2c = None
        self.cam = None

        self.densify_intrinsics = None
        self.densify_cam = None

        self.keyframe_list = []
        self.keyframe_time_indices = []

        # We’ll start with identity “gt pose list” like iphone demo can, but not used
        self.gt_w2c_all_frames = []

        self.get_logger().info("Ready. Waiting for synced frames...")

    def synced_cb(self, rgb_msg: Image, depth_msg: Image, rgb_info: CameraInfo, depth_info: CameraInfo):
        if self.total_frames >= self.num_frames:
            return

        if self.t0 is None:
            self.t0 = time.time()

        # --- Convert ROS images to numpy ---
        # RGB: cv_bridge gives BGR for "bgr8" and RGB for "rgb8"
        rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        rgb_encoding = (rgb_msg.encoding or "").lower()
        depth_encoding = (depth_msg.encoding or "").lower()

        # Ensure RGB is 3-channel uint8
        if rgb_cv.ndim == 2:
            rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_GRAY2BGR)
        if rgb_cv.dtype != np.uint8:
            rgb_cv = np.clip(rgb_cv, 0, 255).astype(np.uint8)

        # Depth -> float meters
        # Depth -> float meters
        depth_m = depth_to_meters(depth_cv, depth_encoding)
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]

        # Apply optional scale from config (ZED 32FC1 is already meters, so this is typically 1.0)
        depth_m = depth_m * float(self.cfg["ros"].get("depth_unit_scale_m", 1.0))

        # --- Resize to desired size in config (optional) ---
        W = int(self.cfg["data"]["desired_image_width"])
        H = int(self.cfg["data"]["desired_image_height"])
        rgb_rs = cv2.resize(rgb_cv, (W, H), interpolation=cv2.INTER_LINEAR)
        depth_rs = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

        # Sanity checks (helps a ton when encodings change)
        if rgb_rs.ndim != 3:
            self.get_logger().warn(f"Unexpected RGB shape: {rgb_rs.shape}")
            return
        if rgb_rs.shape[2] not in (3, 4):
            self.get_logger().warn(f"Unexpected RGB channels: {rgb_rs.shape}")
            return

        # ---- Live camera preview (OpenCV wants BGR 8-bit) ----
        if rgb_rs.shape[2] == 4:
            # Most likely ZED: bgra8
            if "bgra" in rgb_encoding:
                cam_bgr = cv2.cvtColor(rgb_rs, cv2.COLOR_BGRA2BGR)
            else:
                cam_bgr = cv2.cvtColor(rgb_rs, cv2.COLOR_RGBA2BGR)
        else:
            # 3 channel
            if "rgb" in rgb_encoding:
                cam_bgr = cv2.cvtColor(rgb_rs, cv2.COLOR_RGB2BGR)
            else:
                cam_bgr = rgb_rs  # assume already BGR

        # depth_rs is (H, W) float meters
        self.live.update_camera(cam_bgr)        
        self.live.update_depth(depth_rs)          
        # Call tick ONCE per frame
        if not self.live.tick():
            self.get_logger().info("LiveRenderer requested quit.")
            self._done = True
            return
        depth_rs = np.expand_dims(depth_rs, -1)

        # --- Torch tensors ---
        # --- Ensure RGB in numpy BEFORE torch ---
        # ZED gives bgra8 (you confirmed), but handle both cases.
        if rgb_rs.shape[2] == 4:
            # BGRA/RGBA -> RGB (we assume ZED's bgra8)
            if "bgra" in rgb_encoding:
                rgb_rs = cv2.cvtColor(rgb_rs, cv2.COLOR_BGRA2RGB)
            else:
                rgb_rs = cv2.cvtColor(rgb_rs, cv2.COLOR_RGBA2RGB)
        else:
            # BGR/RGB -> RGB
            if "bgr" in rgb_encoding:
                rgb_rs = cv2.cvtColor(rgb_rs, cv2.COLOR_BGR2RGB)
            # else assume already RGB

        # --- Torch tensors ---
        color = torch.from_numpy(rgb_rs).to(self.device).float().permute(2, 0, 1) / 255.0
        depth = torch.from_numpy(depth_rs).to(self.device).float().permute(2, 0, 1)

        if color.shape[0] != 3:
            self.get_logger().warn(f"Color tensor has {color.shape[0]} channels, expected 3")
            return
        # --- Intrinsics ---
        K = caminfo_to_K(rgb_info)
        # Scale intrinsics to match resized image if your CameraInfo was at the same res, this is identity.
        src_W = int(rgb_info.width)
        src_H = int(rgb_info.height)
        sx = W / float(src_W)
        sy = H / float(src_H)
        K_scaled = K.copy()
        K_scaled[0, 0] *= sx
        K_scaled[1, 1] *= sy
        K_scaled[0, 2] *= sx
        K_scaled[1, 2] *= sy

        intrinsics = torch.tensor(K_scaled, device=self.device).float()

        # --- Initialize on first frame ---
        time_idx = self.total_frames
        if time_idx == 0:
            self.intrinsics = intrinsics
            self.first_frame_w2c = torch.eye(4, device=self.device).float()
            self.cam = setup_camera(W, H, K_scaled, self.first_frame_w2c.detach().cpu().numpy())

            # Densification res (optional)
            dW = int(self.cfg["data"].get("densification_image_width", W))
            dH = int(self.cfg["data"].get("densification_image_height", H))
            densify_rgb = cv2.resize(rgb_rs, (dW, dH), interpolation=cv2.INTER_LINEAR)
            densify_depth = cv2.resize(depth_m, (dW, dH), interpolation=cv2.INTER_NEAREST)
            densify_depth = np.expand_dims(densify_depth, -1)

            if densify_rgb.shape[2] == 4:
                if "bgra" in rgb_encoding:
                    densify_rgb = cv2.cvtColor(densify_rgb, cv2.COLOR_BGRA2RGB)
                else:
                    densify_rgb = cv2.cvtColor(densify_rgb, cv2.COLOR_RGBA2RGB)
            else:
                if "bgr" in rgb_encoding:
                    densify_rgb = cv2.cvtColor(densify_rgb, cv2.COLOR_BGR2RGB)

            # Live preview (camera). rgb_rs is RGB uint8, depth_m is float meters (original res)
            densify_color = torch.from_numpy(densify_rgb).to(self.device).float().permute(2, 0, 1) / 255.0

            densify_depth = torch.from_numpy(densify_depth).to(self.device).float().permute(2, 0, 1)

            # Scale K to densify size
            sx_d = dW / float(src_W)
            sy_d = dH / float(src_H)
            Kd = K.copy()
            Kd[0, 0] *= sx_d
            Kd[1, 1] *= sy_d
            Kd[0, 2] *= sx_d
            Kd[1, 2] *= sy_d
            densify_intrinsics = torch.tensor(Kd, device=self.device).float()

            self.densify_intrinsics = densify_intrinsics
            self.densify_cam = setup_camera(dW, dH, Kd, self.first_frame_w2c.detach().cpu().numpy())

            # Init pointcloud
            mask = (densify_depth > 0).reshape(-1)
            init_pt_cld, mean3_sq_dist = get_pointcloud(
                densify_color, densify_depth, densify_intrinsics, self.first_frame_w2c,
                mask=mask, compute_mean_sq_dist=True, mean_sq_dist_method=self.cfg["mean_sq_dist_method"]
            )
            self.params, self.variables = initialize_params(
                init_pt_cld, self.num_frames, mean3_sq_dist, self.cfg.get("gaussian_distribution", "isotropic")
            )
            self.variables["scene_radius"] = torch.max(densify_depth) / self.cfg["scene_radius_depth_ratio"]

        # Fake GT list for compatibility (not used)
        self.gt_w2c_all_frames.append(torch.eye(4, device=self.device).float())
        curr_gt_w2c = self.gt_w2c_all_frames

        curr_data = {
            "cam": self.cam,
            "im": color,
            "depth": depth,
            "id": time_idx,
            "intrinsics": self.intrinsics,
            "w2c": self.first_frame_w2c,
            "iter_gt_w2c_list": curr_gt_w2c,
        }
        tracking_curr_data = curr_data

        # Initialize pose for this frame
        if time_idx > 0:
            self.params = initialize_camera_pose(self.params, time_idx, forward_prop=self.cfg["tracking"]["forward_prop"])

        # --- Tracking ---
        tracking_start = time.time()
        if time_idx > 0 and not self.cfg["tracking"]["use_gt_poses"]:
            optimizer = initialize_optimizer(self.params, self.cfg["tracking"]["lrs"], tracking=True)
            candidate_rot = self.params["cam_unnorm_rots"][..., time_idx].detach().clone()
            candidate_trn = self.params["cam_trans"][..., time_idx].detach().clone()
            current_min_loss = float(1e20)

            num_iters_tracking = int(self.cfg["tracking"]["num_iters"])
            for it in range(num_iters_tracking):
                loss, self.variables, losses = get_loss(
                    self.params, tracking_curr_data, self.variables, time_idx,
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

            with torch.no_grad():
                self.params["cam_unnorm_rots"][..., time_idx] = candidate_rot
                self.params["cam_trans"][..., time_idx] = candidate_trn

            # ---- Final quaternion safety normalization ----
            with torch.no_grad():
                self.params["cam_unnorm_rots"][..., time_idx] = F.normalize(
                    self.params["cam_unnorm_rots"][..., time_idx], dim=-1, eps=1e-6
                )
 
            # ---- Build a safe pose for this frame (used by mapping/keyframes/densify) ----
            curr_w2c = w2c_from_params(self.params, time_idx, self.device)

            if curr_w2c is None:
                self.get_logger().warn(
                    f"Frame {time_idx}: pose invalid/singular. Using last valid pose from frame {self.last_valid_time_idx}."
                )
                curr_w2c = self.last_valid_w2c
            else:
                self.last_valid_w2c = curr_w2c
                self.last_valid_time_idx = time_idx
        tracking_dt = time.time() - tracking_start

        # --- Mapping (keyframe window) ---
        if (time_idx == 0) or ((time_idx + 1) % int(self.cfg["map_every"]) == 0):
            # Ensure curr_w2c always exists (frame 0 won’t run tracking)
            if time_idx == 0:
                curr_w2c = self.first_frame_w2c.detach().clone()
                self.last_valid_w2c = curr_w2c
                self.last_valid_time_idx = 0           
        
            # Densify add-new-gaussians (optional)
            if self.cfg["mapping"]["add_new_gaussians"] and time_idx > 0:
                if pose_is_valid(self.params, time_idx):
                    densify_curr_data = curr_data
                    self.params, self.variables = add_new_gaussians(
                        self.params, self.variables, densify_curr_data,
                        self.cfg["mapping"]["sil_thres"], time_idx,
                        self.cfg["mean_sq_dist_method"], self.cfg.get("gaussian_distribution", "isotropic")
                    )
                else:
                    self.get_logger().warn(
                        f"Skipping add_new_gaussians at frame {time_idx}: invalid pose (would be singular)"
                    )
                
            #
            # Keyframe selection MUST happen regardless of densify
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

            # Full map optimization
            optimizer = initialize_optimizer(self.params, self.cfg["mapping"]["lrs"], tracking=False)
            num_iters_mapping = int(self.cfg["mapping"]["num_iters"])
            for it in range(num_iters_mapping):
                rand_idx = np.random.randint(0, len(selected_keyframes))
                kf_idx = selected_keyframes[rand_idx]

                if kf_idx == -1:
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    iter_time_idx = self.keyframe_list[kf_idx]["id"]
                    iter_color = self.keyframe_list[kf_idx]["color"]
                    iter_depth = self.keyframe_list[kf_idx]["depth"]

                iter_data = {
                    "cam": self.cam,
                    "im": iter_color,
                    "depth": iter_depth,
                    "id": iter_time_idx,
                    "intrinsics": self.intrinsics,
                    "w2c": self.first_frame_w2c,
                    "iter_gt_w2c_list": curr_gt_w2c,
                }

                loss, self.variables, losses = get_loss(
                    self.params, iter_data, self.variables, iter_time_idx,
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

        # Add keyframe
        if ((time_idx == 0) or ((time_idx + 1) % int(self.cfg["keyframe_every"]) == 0) or (time_idx == self.num_frames - 2)):
            with torch.no_grad():
                # curr_keyframe = {"id": time_idx, "est_w2c": torch.eye(4, device=self.device).float(),
                #                  "color": color, "depth": depth}
                curr_keyframe = {"id": time_idx, "est_w2c": curr_w2c.detach().clone(),
                                "color": color, "depth": depth}               
                self.keyframe_list.append(curr_keyframe)
                self.keyframe_time_indices.append(time_idx)

        self.total_frames += 1

        self.get_logger().info(
            f"Frame {self.total_frames}/{self.num_frames} | rgb_enc={rgb_encoding} depth_enc={depth_encoding} | track_dt={tracking_dt:.3f}s"
        )

        
        if self.total_frames == self.num_frames:
            # # Save like other pipelines
            # # Add camera + bookkeeping like splatam.py does
            # self.params["timestep"] = self.variables["timestep"]
            # self.params["intrinsics"] = self.intrinsics.detach().cpu().numpy()
            # self.params["w2c"] = self.first_frame_w2c.detach().cpu().numpy()
            # self.params["org_width"] = self.cfg["data"]["desired_image_width"]
            # self.params["org_height"] = self.cfg["data"]["desired_image_height"]
            # self.params["gt_w2c_all_frames"] = np.stack(
            #     [m.detach().cpu().numpy() for m in self.gt_w2c_all_frames], axis=0
            # )
            # self.params["keyframe_time_indices"] = np.array(self.keyframe_time_indices)

            # save_params(self.params, str(self.output_dir))
            # self.get_logger().info(f"Saved SplaTAM output to: {self.output_dir}")
            # self.get_logger().info("Done. You can now run viz_scripts/final_recon.py on the same config.")
            # rclpy.shutdown()
            self._done = True

            return


    def _shutdown_tick(self):
        """Runs in the executor but outside the subscription callback context."""
        if not self._done:
            return

        if not self._final_saved:
            self._final_saved = True

            # Save results (same logic you already have)
            self.params["timestep"] = self.variables["timestep"]
            self.params["intrinsics"] = self.intrinsics.detach().cpu().numpy()
            self.params["w2c"] = self.first_frame_w2c.detach().cpu().numpy()
            self.params["org_width"] = self.cfg["data"]["desired_image_width"]
            self.params["org_height"] = self.cfg["data"]["desired_image_height"]
            self.params["gt_w2c_all_frames"] = np.stack(
                [m.detach().cpu().numpy() for m in self.gt_w2c_all_frames], axis=0
            )
            self.params["keyframe_time_indices"] = np.array(self.keyframe_time_indices)

            save_params(self.params, str(self.output_dir))
            self.get_logger().info(f"Saved SplaTAM output to: {self.output_dir}")
            self.get_logger().info("Done. You can now run viz_scripts/final_recon.py on the same config.")

        # Now actually stop ROS
        self._shutdown_timer.cancel()
        self.destroy_node()
        rclpy.shutdown()

def main():
    args = parse_args()
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    cfg = experiment.config

    # ---- CLI overrides config ----
    cfg["live_cam"] = args.live_cam
    cfg["live_splat"] = args.live_splat
    cfg["live_depth"] = args.live_depth
    if args.live_max_fps is not None:
        cfg["live_max_fps"] = args.live_max_fps

    seed_everything(seed=cfg["seed"])

    rclpy.init()
    node = ZedSplatamOnline(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()