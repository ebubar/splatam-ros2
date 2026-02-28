#scripts/zed2i_ros2_demo.py:

#!/usr/bin/env python3
"""
ROS2 live ZED2i -> SplaTAM online demo.

This mirrors scripts/iphone_demo.py but uses ROS2 topics instead of NeRFCapture DDS.
It's designed so you can run in two modes:
  A) use_gt_poses=True  -> use ZED/ROS odom pose as "ground truth" camera pose
  B) use_gt_poses=False -> let SplaTAM estimate pose (tracking)

Topic names + encodings are intentionally configurable because you’re not in front of the camera yet.
"""

import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from importlib.machinery import SourceFileLoader
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

# Reuse SplaTAM building blocks (same as iphone_demo.py)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation, prune_gaussians, densify
from utils.keyframe_selection import keyframe_selection_overlap
from utils.slam_helpers import matrix_to_quaternion
from scripts.splatam import (
    get_loss,
    initialize_optimizer,
    initialize_params,
    initialize_camera_pose,
    get_pointcloud,
    add_new_gaussians,
)


def camera_info_to_k(info: CameraInfo) -> np.ndarray:
    # CameraInfo.K is 3x3 row-major
    k = np.array(info.k, dtype=np.float32).reshape(3, 3)
    return k

def odom_to_T(odom: Odometry) -> np.ndarray:
    """Return 4x4 transform from odom pose (frame is whatever your odom frame is)."""
    p = odom.pose.pose.position
    q = odom.pose.pose.orientation
    # quaternion -> rotation
    # q = (x,y,z,w)
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array([p.x, p.y, p.z], dtype=np.float32)
    return T

def relative_T(first_T: np.ndarray, curr_T: np.ndarray) -> np.ndarray:
    """Return T_rel = inv(first_T) @ curr_T"""
    return np.linalg.inv(first_T) @ curr_T

def ensure_empty_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"{path} exists. Set overwrite=True to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

# --------------------------
# ROS Node
# --------------------------

@dataclass
class LatestPacket:
    rgb: Optional[np.ndarray] = None          # HxWx3 uint8 BGR or RGB
    depth: Optional[np.ndarray] = None        # HxW float32 meters OR uint16 mm (we’ll normalize)
    cam_info: Optional[CameraInfo] = None
    odom_T: Optional[np.ndarray] = None       # 4x4 float32

class ZedRosBridge(Node):
    def __init__(self, cfg: dict):
        super().__init__("zed2i_splatam_bridge")

        self.cfg = cfg
        self.bridge = CvBridge()
        self.latest = LatestPacket()

        self.rgb_sub = self.create_subscription(Image, cfg["ros"]["rgb_topic"], self.on_rgb, 10)
        self.depth_sub = self.create_subscription(Image, cfg["ros"]["depth_topic"], self.on_depth, 10)
        self.info_sub = self.create_subscription(CameraInfo, cfg["ros"]["camera_info_topic"], self.on_info, 10)

        self.use_odom = cfg["ros"].get("use_odom", False)
        if self.use_odom:
            self.odom_sub = self.create_subscription(Odometry, cfg["ros"]["odom_topic"], self.on_odom, 50)

    def on_rgb(self, msg: Image):
        # We’ll decode to BGR8 by default; convert to RGB later if needed.
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.cfg["ros"].get("rgb_encoding", "bgr8"))
        self.latest.rgb = img

    def on_depth(self, msg: Image):
        # Depth could be 32FC1 or 16UC1. We decode "passthrough" then normalize later.
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.latest.depth = depth

    def on_info(self, msg: CameraInfo):
        self.latest.cam_info = msg

    def on_odom(self, msg: Odometry):
        self.latest.odom_T = odom_to_T(msg)

    def have_packet(self) -> bool:
        if self.latest.rgb is None or self.latest.depth is None or self.latest.cam_info is None:
            return False
        if self.use_odom and self.latest.odom_T is None:
            return False
        return True

# --------------------------
# Main SplaTAM loop (online)
# --------------------------

def run_online(cfg: dict):
    # Prepare output folder
    workdir = Path(cfg["workdir"])
    ensure_empty_dir(workdir, overwrite=cfg["overwrite"])
    (workdir / "rgb").mkdir(exist_ok=True)
    (workdir / "depth").mkdir(exist_ok=True)

    rclpy.init()
    node = ZedRosBridge(cfg)

    device = torch.device(cfg["primary_device"])
    seed_everything(seed=cfg["seed"])

    print("Waiting for first RGB+Depth+CameraInfo (and odom if enabled)...")
    while rclpy.ok() and not node.have_packet():
        rclpy.spin_once(node, timeout_sec=0.1)

    # --- First packet init ---
    pkt = node.latest
    first_rgb = pkt.rgb.copy()
    first_depth = pkt.depth.copy()
    K_full = camera_info_to_k(pkt.cam_info)

    # Depth normalization:
    # If uint16, we assume it’s in millimeters unless you set depth_unit_scale_m in config.
    depth_unit_scale_m = cfg["ros"].get("depth_unit_scale_m", None)
    if first_depth.dtype == np.uint16:
        scale = depth_unit_scale_m if depth_unit_scale_m is not None else 0.001
        first_depth_m = first_depth.astype(np.float32) * scale
    else:
        first_depth_m = first_depth.astype(np.float32)

    # Resize for tracking/mapping resolution
    W = cfg["data"]["desired_image_width"]
    H = cfg["data"]["desired_image_height"]
    rgb_small = cv2.resize(first_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    depth_small = cv2.resize(first_depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

    # Convert to torch
    # Convert BGR->RGB unless your topic is already rgb8.
    if cfg["ros"].get("rgb_is_bgr", True):
        rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
    color = torch.from_numpy(rgb_small).to(device).float().permute(2, 0, 1) / 255.0
    depth = torch.from_numpy(depth_small[..., None]).to(device).float().permute(2, 0, 1)

    # Intrinsics scaling
    # CameraInfo K is for full res; we scale by resize ratio.
    full_h, full_w = first_rgb.shape[0], first_rgb.shape[1]
    sx = W / float(full_w)
    sy = H / float(full_h)
    intrinsics = torch.tensor(K_full, device=device).float()
    intrinsics[0, :] *= sx
    intrinsics[1, :] *= sy
    intrinsics[2, 2] = 1.0

    first_frame_w2c = torch.eye(4, device=device).float()
    cam = setup_camera(W, H, intrinsics.detach().cpu().numpy(), first_frame_w2c.detach().cpu().numpy())

    # Densification resolution (optional)
    dW = cfg["data"].get("densification_image_width", W)
    dH = cfg["data"].get("densification_image_height", H)
    densify_rgb = cv2.resize(first_rgb, (dW, dH), interpolation=cv2.INTER_LINEAR)
    densify_depth = cv2.resize(first_depth_m, (dW, dH), interpolation=cv2.INTER_NEAREST)
    if cfg["ros"].get("rgb_is_bgr", True):
        densify_rgb = cv2.cvtColor(densify_rgb, cv2.COLOR_BGR2RGB)
    densify_color = torch.from_numpy(densify_rgb).to(device).float().permute(2, 0, 1) / 255.0
    densify_depth_t = torch.from_numpy(densify_depth[..., None]).to(device).float().permute(2, 0, 1)

    densify_intrinsics = torch.tensor(K_full, device=device).float()
    densify_intrinsics[0, :] *= (dW / float(full_w))
    densify_intrinsics[1, :] *= (dH / float(full_h))
    densify_intrinsics[2, 2] = 1.0
    densify_cam = setup_camera(dW, dH, densify_intrinsics.detach().cpu().numpy(), first_frame_w2c.detach().cpu().numpy())

    # Initialize Params
    mask = (densify_depth_t > 0).reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(
        densify_color, densify_depth_t, densify_intrinsics, first_frame_w2c,
        mask=mask,
        compute_mean_sq_dist=True,
        mean_sq_dist_method=cfg["mean_sq_dist_method"],
    )
    gaussian_distribution = cfg.get("gaussian_distribution", "isotropic")
    params, variables = initialize_params(init_pt_cld, cfg["num_frames"], mean3_sq_dist, gaussian_distribution)
    variables["scene_radius"] = torch.max(densify_depth_t) / cfg["scene_radius_depth_ratio"]

    # Pose handling
    use_gt_poses = cfg["tracking"]["use_gt_poses"]
    gt_w2c_all_frames = []
    first_odom_T = pkt.odom_T.copy() if (cfg["ros"].get("use_odom", False) and pkt.odom_T is not None) else None

    # Keyframes
    keyframe_list = []
    keyframe_time_indices = []

    num_frames = cfg["num_frames"]
    print(f"Starting online run for {num_frames} frames...")

    for time_idx in range(num_frames):
        # Wait for next packet
        while rclpy.ok() and not node.have_packet():
            rclpy.spin_once(node, timeout_sec=0.05)
        rclpy.spin_once(node, timeout_sec=0.01)
        pkt = node.latest

        rgb = pkt.rgb.copy()
        depth_raw = pkt.depth.copy()
        K_full = camera_info_to_k(pkt.cam_info)

        # depth normalize
        if depth_raw.dtype == np.uint16:
            scale = depth_unit_scale_m if depth_unit_scale_m is not None else 0.001
            depth_m = depth_raw.astype(np.float32) * scale
        else:
            depth_m = depth_raw.astype(np.float32)

        # Save raw frames (optional)
        if cfg.get("save_stream_frames", True):
            cv2.imwrite(str(workdir / "rgb" / f"{time_idx}.png"), rgb)
            # save depth as 16-bit millimeters for convenience
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(workdir / "depth" / f"{time_idx}.png"), depth_mm)

        # Resize + torch
        rgb_small = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        depth_small = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg["ros"].get("rgb_is_bgr", True):
            rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
        color = torch.from_numpy(rgb_small).to(device).float().permute(2, 0, 1) / 255.0
        depth = torch.from_numpy(depth_small[..., None]).to(device).float().permute(2, 0, 1)

        # intrinsics update (in case they change)
        full_h, full_w = rgb.shape[0], rgb.shape[1]
        sx = W / float(full_w)
        sy = H / float(full_h)
        intrinsics = torch.tensor(K_full, device=device).float()
        intrinsics[0, :] *= sx
        intrinsics[1, :] *= sy
        intrinsics[2, 2] = 1.0

        curr_data = {
            "cam": cam,
            "im": color,
            "depth": depth,
            "id": time_idx,
            "intrinsics": intrinsics,
            "w2c": first_frame_w2c,
            "iter_gt_w2c_list": gt_w2c_all_frames,  # filled below if using GT
        }

        # --- GT pose path (optional) ---
        if use_gt_poses:
            if pkt.odom_T is None:
                raise RuntimeError("use_gt_poses=True but no odom pose received.")
            if first_odom_T is None:
                first_odom_T = pkt.odom_T.copy()

            # Make relative pose to first frame
            rel = relative_T(first_odom_T, pkt.odom_T).astype(np.float32)

            # You may need a convention conversion here (ZED/ROS vs SplaTAM frame).
            # We leave it as identity for now and adjust once we see actual axes.
            gt_pose = torch.from_numpy(rel).to(device).float()
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            curr_data["iter_gt_w2c_list"] = gt_w2c_all_frames

        # Initialize camera pose estimate for current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=cfg["tracking"]["forward_prop"])

        # Tracking
        if time_idx > 0 and not use_gt_poses:
            optimizer = initialize_optimizer(params, cfg["tracking"]["lrs"], tracking=True)
            candidate_rot = params["cam_unnorm_rots"][..., time_idx].detach().clone()
            candidate_tr = params["cam_trans"][..., time_idx].detach().clone()
            current_min_loss = float(1e20)

            num_iters_tracking = cfg["tracking"]["num_iters"]
            pb = tqdm(range(num_iters_tracking), desc=f"Tracking t={time_idx}")
            for it in range(num_iters_tracking):
                loss, variables, losses = get_loss(
                    params, curr_data, variables, time_idx,
                    cfg["tracking"]["loss_weights"],
                    cfg["tracking"]["use_sil_for_loss"],
                    cfg["tracking"]["sil_thres"],
                    cfg["tracking"]["use_l1"],
                    cfg["tracking"]["ignore_outlier_depth_loss"],
                    tracking=True,
                    visualize_tracking_loss=cfg["tracking"].get("visualize_tracking_loss", False),
                    tracking_iteration=it,
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_rot = params["cam_unnorm_rots"][..., time_idx].detach().clone()
                        candidate_tr = params["cam_trans"][..., time_idx].detach().clone()
                pb.update(1)
            pb.close()

            with torch.no_grad():
                params["cam_unnorm_rots"][..., time_idx] = candidate_rot
                params["cam_trans"][..., time_idx] = candidate_tr

        elif time_idx > 0 and use_gt_poses:
            # Set pose from gt_w2c
            with torch.no_grad():
                rel_w2c = gt_w2c_all_frames[-1]
                rel_rot = rel_w2c[:3, :3].unsqueeze(0)
                rel_rot_q = matrix_to_quaternion(rel_rot)
                rel_tr = rel_w2c[:3, 3]
                params["cam_unnorm_rots"][..., time_idx] = rel_rot_q
                params["cam_trans"][..., time_idx] = rel_tr

        # Mapping every map_every frames
        if time_idx == 0 or (time_idx + 1) % cfg["map_every"] == 0:
            # Add new gaussians (densification) using densify resolution packet for this time
            if cfg["mapping"]["add_new_gaussians"] and time_idx > 0:
                densify_rgb = cv2.resize(rgb, (dW, dH), interpolation=cv2.INTER_LINEAR)
                densify_depth = cv2.resize(depth_m, (dW, dH), interpolation=cv2.INTER_NEAREST)
                if cfg["ros"].get("rgb_is_bgr", True):
                    densify_rgb = cv2.cvtColor(densify_rgb, cv2.COLOR_BGR2RGB)
                densify_color = torch.from_numpy(densify_rgb).to(device).float().permute(2, 0, 1) / 255.0
                densify_depth_t = torch.from_numpy(densify_depth[..., None]).to(device).float().permute(2, 0, 1)

                densify_curr_data = {
                    "cam": densify_cam,
                    "im": densify_color,
                    "depth": densify_depth_t,
                    "id": time_idx,
                    "intrinsics": densify_intrinsics,
                    "w2c": first_frame_w2c,
                    "iter_gt_w2c_list": gt_w2c_all_frames,
                }
                params, variables = add_new_gaussians(
                    params, variables, densify_curr_data,
                    cfg["mapping"]["sil_thres"],
                    time_idx,
                    cfg["mean_sq_dist_method"],
                    gaussian_distribution,
                )

            # Select keyframes (same idea as iphone_demo)
            with torch.no_grad():
                curr_cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
                curr_cam_tr = params["cam_trans"][..., time_idx].detach()
                curr_w2c = torch.eye(4, device=device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tr

                num_keyframes = cfg["mapping_window_size"] - 2
                selected = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_idx = [keyframe_list[i]["id"] for i in selected]
                if len(keyframe_list) > 0:
                    selected_idx.append(keyframe_list[-1]["id"])
                    selected.append(len(keyframe_list) - 1)
                selected_idx.append(time_idx)
                selected.append(-1)
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_idx}")

            optimizer = initialize_optimizer(params, cfg["mapping"]["lrs"], tracking=False)

            num_iters_mapping = cfg["mapping"]["num_iters"]
            pb = tqdm(range(num_iters_mapping), desc=f"Mapping t={time_idx}") if num_iters_mapping > 0 else None
            for it in range(num_iters_mapping):
                rand_i = np.random.randint(0, len(selected))
                kf = selected[rand_i]
                if kf == -1:
                    iter_time = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    iter_time = keyframe_list[kf]["id"]
                    iter_color = keyframe_list[kf]["color"]
                    iter_depth = keyframe_list[kf]["depth"]

                iter_data = {
                    "cam": cam,
                    "im": iter_color,
                    "depth": iter_depth,
                    "id": iter_time,
                    "intrinsics": intrinsics,
                    "w2c": first_frame_w2c,
                    "iter_gt_w2c_list": gt_w2c_all_frames,
                }

                loss, variables, losses = get_loss(
                    params, iter_data, variables, iter_time,
                    cfg["mapping"]["loss_weights"],
                    cfg["mapping"]["use_sil_for_loss"],
                    cfg["mapping"]["sil_thres"],
                    cfg["mapping"]["use_l1"],
                    cfg["mapping"]["ignore_outlier_depth_loss"],
                    mapping=True,
                )
                loss.backward()

                with torch.no_grad():
                    if cfg["mapping"]["prune_gaussians"]:
                        params, variables = prune_gaussians(params, variables, optimizer, it, cfg["mapping"]["pruning_dict"])
                    if cfg["mapping"]["use_gaussian_splatting_densification"]:
                        params, variables = densify(params, variables, optimizer, it, cfg["mapping"]["densify_dict"])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if pb is not None:
                    pb.update(1)
            if pb is not None:
                pb.close()

        # Add keyframe
        if ((time_idx == 0) or ((time_idx + 1) % cfg["keyframe_every"] == 0) or (time_idx == num_frames - 2)):
            with torch.no_grad():
                curr_keyframe = {"id": time_idx, "est_w2c": None, "color": color, "depth": depth}
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)

        torch.cuda.empty_cache()

    # Save final params
    params["timestep"] = variables["timestep"]
    params["intrinsics"] = intrinsics.detach().cpu().numpy()
    params["w2c"] = first_frame_w2c.detach().cpu().numpy()
    params["org_width"] = cfg["data"]["desired_image_width"]
    params["org_height"] = cfg["data"]["desired_image_height"]
    params["keyframe_time_indices"] = np.array(keyframe_time_indices)

    output_dir = os.path.join(cfg["workdir"], cfg["run_name"])
    os.makedirs(output_dir, exist_ok=True)
    save_params(params, output_dir)
    print("Saved SplaTAM splat to:", output_dir)

    node.destroy_node()
    rclpy.shutdown()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./configs/zed2i/online_ros2.py", type=str)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    run_online(experiment.config)