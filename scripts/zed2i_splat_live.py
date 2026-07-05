#!/usr/bin/env python3

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
from utils.slam_helpers import transform_to_frame, transformed_params2rendervar
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from scripts.splatam import (
    get_loss,
    initialize_optimizer,
    initialize_params,
    initialize_camera_pose,
    get_pointcloud,
    add_new_gaussians,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/zed2i/zed2i_splat_live.py",
        type=str,
    )
    return parser.parse_args()




def caminfo_to_K(cam_info):
    return np.array(cam_info.k, dtype=np.float32).reshape(3, 3)


def depth_to_meters(depth_cv, encoding):
    enc = (encoding or "").lower()

    if enc in ["32fc1", "32fc"]:
        return depth_cv.astype(np.float32)

    if enc in ["16uc1", "16uc", "mono16"]:
        return depth_cv.astype(np.float32) / 1000.0

    return depth_cv.astype(np.float32)

def colorize_depth(depth_m, min_depth=0.4, max_depth=5.0):
    valid = np.isfinite(depth_m) & (depth_m > min_depth) & (depth_m < max_depth)

    depth_color = np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    if np.count_nonzero(valid) == 0:
        return depth_color

    # Use percentiles instead of fixed min/max so the image has more contrast
    lo = np.percentile(depth_m[valid], 2)
    hi = np.percentile(depth_m[valid], 98)

    if hi <= lo:
        lo = min_depth
        hi = max_depth

    depth_norm = np.zeros_like(depth_m, dtype=np.float32)

    # Invert so closer objects are warmer/redder and far objects are cooler/bluer
    depth_norm[valid] = 1.0 - np.clip(
        (depth_m[valid] - lo) / (hi - lo),
        0.0,
        1.0,
    )

    depth_vis = (depth_norm * 255).astype(np.uint8)

    # Boost local contrast so walls/floor edges show more detail
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    depth_vis = clahe.apply(depth_vis)

    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Invalid / missing depth stays black
    depth_color[~valid] = (0, 0, 0)

    # Add white edges to make depth discontinuities easier to see
    edges = cv2.Canny(depth_vis, 40, 80)
    depth_color[edges > 0] = (255, 255, 255)

    return depth_color

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
    # Converts 3x3 rotation matrix to quaternion in SplaTAM/gaussian-splatting order: w, x, y, z
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
    q = q / torch.linalg.norm(q)
    return q

def odom_to_c2w(odom_msg, device):
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation

    x = q.x
    y = q.y
    z = q.z
    w = q.w

    R = torch.tensor(
        [
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
            [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y],
        ],
        device=device,
        dtype=torch.float32,
    )

    c2w = torch.eye(4, device=device, dtype=torch.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = torch.tensor([p.x, p.y, p.z], device=device, dtype=torch.float32)

    return c2w


def c2w_to_w2c(c2w):
    return torch.linalg.inv(c2w)

# def zed_body_to_optical(device):
#     T = torch.eye(4, device=device, dtype=torch.float32)

#     # ROS body: x forward, y left, z up
#     # Optical:  x right,   y down, z forward
#     T[:3, :3] = torch.tensor(
#         [
#             [0.0, -1.0,  0.0],
#             [0.0,  0.0, -1.0],
#             [1.0,  0.0,  0.0],
#         ],
#         device=device,
#         dtype=torch.float32,
#     )

#     return T

def zed_link_to_left_optical(device):
    T = torch.eye(4, device=device, dtype=torch.float32)

    # From /tf_static:
    # zed_camera_link -> zed_camera_center:      (0.0, 0.0, 0.015)
    # zed_camera_center -> zed_left_camera_frame: (-0.01, 0.06, 0.0)
    #
    # Combined translation link -> left_camera_frame:
    T[:3, 3] = torch.tensor(
        [-0.01, 0.06, 0.015],
        device=device,
        dtype=torch.float32,
    )

    # From /tf_static:
    # zed_left_camera_frame -> zed_left_camera_frame_optical
    # quaternion: x=0.5, y=-0.5, z=0.5, w=-0.5
    T[:3, :3] = torch.tensor(
        [
            [0.0,  0.0,  1.0],
            [-1.0, 0.0,  0.0],
            [0.0, -1.0,  0.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    return T

class ZedSplatamOnline(Node):
    def __init__(self, config):
        super().__init__("ZedSplatamOnline")

        self.cfg = config
        self.bridge = CvBridge()
        self.device = torch.device(self.cfg.get("primary_device", "cuda:0"))

        self.num_frames = int(self.cfg["num_frames"])

        # total_frames = frames actually processed by SplaTAM
        self.total_frames = 0

        # received_frames = all synced ROS RGB-D frames received
        self.received_frames = 0

        # Process every Nth ZED frame so video does not overwhelm SplaTAM
        self.process_every_n = int(self.cfg["ros"].get("process_every_n", 4))
        self.workdir = Path(self.cfg["workdir"])
        self.run_name = self.cfg.get("run_name", "SplaTAM_ZED2i_ROS2")
        self.output_dir = self.workdir / self.run_name

        if self.cfg.get("overwrite", False) and self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.params = None
        self.variables = None
        self.intrinsics = None
        self.first_frame_w2c = None
        self.cam = None
        self.densify_intrinsics = None
        self.densify_cam = None

        self.keyframe_list = []
        self.keyframe_time_indices = []
        self.gt_w2c_all_frames = []

        self.tracking_iter_time_sum = 0.0
        self.tracking_iter_time_count = 0
        self.mapping_iter_time_sum = 0.0
        self.mapping_iter_time_count = 0
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        ros_cfg = self.cfg["ros"]

        self.latest_rgb_info = None

        self.rgb_info_direct_sub = self.create_subscription(
            CameraInfo,
            ros_cfg["rgb_info_topic"],
            self.rgb_info_cb,
            qos,
        )
        self.rgb_sub = Subscriber(
            self,
            Image,
            ros_cfg["rgb_topic"],
            qos_profile=qos,
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            ros_cfg["depth_topic"],
            qos_profile=qos,
        )

        self.rgb_info_sub = Subscriber(
            self,
            CameraInfo,
            ros_cfg["rgb_info_topic"],
            qos_profile=qos,
        )

        self.depth_info_sub = Subscriber(
            self,
            CameraInfo,
            ros_cfg["depth_info_topic"],
            qos_profile=qos,
        )

        self.odom_sub = Subscriber(
            self,
            Odometry,
            ros_cfg["odom_topic"],
            qos_profile=qos,
        )

        self.ts = ApproximateTimeSynchronizer(
            [
                self.rgb_sub,
                self.depth_sub,
                self.odom_sub,
            ],
            queue_size=100,
            slop=0.15,
            allow_headerless=False,
        )

        self.ts.registerCallback(self.synced_cb)

        # Live splat rendering: publish the current reconstruction rendered
        # from the current camera pose so any machine on the ROS network can
        # watch it (e.g. rqt_image_view /splatam/live_render).
        viz_cfg = self.cfg.get("viz", {})
        self.render_every = int(viz_cfg.get("render_every", 1))
        self.render_save_every = int(viz_cfg.get("render_save_every", 0))
        self.render_pub = None

        if viz_cfg.get("publish_live_render", True):
            self.render_pub = self.create_publisher(
                Image,
                viz_cfg.get("render_topic", "/splatam/live_render"),
                qos,
            )

        self.get_logger().info("ZedSplatamOnline ready.")
        self.get_logger().info(f"RGB:   {ros_cfg['rgb_topic']}")
        self.get_logger().info(f"Depth: {ros_cfg['depth_topic']}")

    def rgb_info_cb(self, msg):
        self.latest_rgb_info = msg

    def make_frame(self, rgb_msg, depth_msg, rgb_info):
        rgb_cv = self.bridge.imgmsg_to_cv2(
            rgb_msg,
            desired_encoding="passthrough",
        )

        depth_cv = self.bridge.imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough",
        )

        rgb = ros_rgb_to_rgb(rgb_cv, rgb_msg.encoding)

        depth_m = depth_to_meters(depth_cv, depth_msg.encoding)

        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]

        depth_m = depth_m * float(self.cfg["ros"].get("depth_unit_scale_m", 1.0))
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

        min_depth = float(self.cfg["ros"].get("min_depth_m", 0.3))
        max_depth = float(self.cfg["ros"].get("max_depth_m", 8.0))

        depth_m[(depth_m < min_depth) | (depth_m > max_depth)] = 0.0

        W = int(self.cfg["data"]["desired_image_width"])
        H = int(self.cfg["data"]["desired_image_height"])

        rgb_rs = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        depth_rs = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_rs = np.expand_dims(depth_rs.astype(np.float32), axis=-1)

        color = torch.from_numpy(rgb_rs).to(self.device).float()
        color = color.permute(2, 0, 1) / 255.0

        depth = torch.from_numpy(depth_rs).to(self.device).float()
        depth = depth.permute(2, 0, 1)

        K = caminfo_to_K(rgb_info)

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

        return color, depth, intrinsics, K_scaled, rgb, depth_m

    def set_camera_pose_from_w2c(self, time_idx, w2c):
        q = rotmat_to_quat_wxyz(w2c[:3, :3])
        t = w2c[:3, 3]

        rot_slot = self.params["cam_unnorm_rots"][..., time_idx]
        trans_slot = self.params["cam_trans"][..., time_idx]

        self.params["cam_unnorm_rots"][..., time_idx] = q.reshape_as(rot_slot)
        self.params["cam_trans"][..., time_idx] = t.reshape_as(trans_slot)

    def make_densify_frame(self, rgb, depth_m, rgb_info):
        dW = int(self.cfg["data"].get("densification_image_width"))
        dH = int(self.cfg["data"].get("densification_image_height"))

        densify_rgb = cv2.resize(rgb, (dW, dH), interpolation=cv2.INTER_LINEAR)
        densify_depth = cv2.resize(depth_m, (dW, dH), interpolation=cv2.INTER_NEAREST)
        densify_depth = np.expand_dims(densify_depth.astype(np.float32), axis=-1)

        densify_color = torch.from_numpy(densify_rgb).to(self.device).float()
        densify_color = densify_color.permute(2, 0, 1) / 255.0

        densify_depth = torch.from_numpy(densify_depth).to(self.device).float()
        densify_depth = densify_depth.permute(2, 0, 1)

        K = caminfo_to_K(rgb_info)

        src_W = int(rgb_info.width)
        src_H = int(rgb_info.height)

        sx = dW / float(src_W)
        sy = dH / float(src_H)

        Kd = K.copy()
        Kd[0, 0] *= sx
        Kd[1, 1] *= sy
        Kd[0, 2] *= sx
        Kd[1, 2] *= sy

        densify_intrinsics = torch.tensor(Kd, device=self.device).float()

        return densify_color, densify_depth, densify_intrinsics, Kd

    def initialize_first_frame(self, intrinsics, K_scaled, rgb, depth_m, rgb_info):
        self.intrinsics = intrinsics
        self.first_frame_w2c = torch.eye(4, device=self.device).float()

        W = int(self.cfg["data"]["desired_image_width"])
        H = int(self.cfg["data"]["desired_image_height"])

        self.cam = setup_camera(
            W,
            H,
            K_scaled,
            self.first_frame_w2c.detach().cpu().numpy(),
        )

        densify_color, densify_depth, densify_intrinsics, Kd = self.make_densify_frame(
            rgb,
            depth_m,
            rgb_info,
        )

        self.densify_intrinsics = densify_intrinsics

        dW = int(self.cfg["data"]["densification_image_width"])
        dH = int(self.cfg["data"]["densification_image_height"])

        self.densify_cam = setup_camera(
            dW,
            dH,
            Kd,
            self.first_frame_w2c.detach().cpu().numpy(),
        )

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

    def publish_live_render(self, time_idx, stamp):
        if self.render_pub is None or self.params is None:
            return
        if self.render_every > 1 and time_idx % self.render_every != 0:
            return

        with torch.no_grad():
            transformed = transform_to_frame(
                self.params,
                time_idx,
                gaussians_grad=False,
                camera_grad=False,
            )
            rendervar = transformed_params2rendervar(self.params, transformed)
            im, _, _ = Renderer(raster_settings=self.cam)(**rendervar)

            rgb = (
                torch.clamp(im, 0.0, 1.0)
                .permute(1, 2, 0)
                .mul(255)
                .byte()
                .cpu()
                .numpy()
            )

        bgr = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Build the Image msg manually; cv_bridge's cv2_to_imgmsg is broken
        # under numpy 2.x (KeyError in its cvtype lookup table)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "splatam_render"
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = bgr.strides[0]
        msg.data = bgr.tobytes()
        self.render_pub.publish(msg)

        if self.render_save_every > 0 and time_idx % self.render_save_every == 0:
            render_dir = self.output_dir / "live_render_debug"
            render_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(render_dir / f"render_{time_idx:05d}.png"), bgr)

    def curr_w2c_from_params(self, time_idx):
        with torch.no_grad():
            curr_cam_rot = F.normalize(
                self.params["cam_unnorm_rots"][..., time_idx].detach()
            )

            curr_cam_tran = self.params["cam_trans"][..., time_idx].detach()

            curr_w2c = torch.eye(4, device=self.device).float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran

        return curr_w2c

    def synced_cb(self, rgb_msg, depth_msg, odom_msg):
        self.received_frames += 1

        if self.latest_rgb_info is None:
            self.get_logger().warn("Waiting for RGB CameraInfo...")
            return

        rgb_info = self.latest_rgb_info
        # Skip fast video frames. This makes ZED behave more like iPhone frame capture.
        if self.received_frames % self.process_every_n != 0:
            return

        if self.total_frames >= self.num_frames:
            return

        frame_start = time.time()
        time_idx = self.total_frames

        color, depth, intrinsics, K_scaled, rgb_np, depth_np = self.make_frame(
            rgb_msg,
            depth_msg,
            rgb_info,
        )

        depth_color = colorize_depth(
            depth_np,
            min_depth=float(self.cfg["ros"].get("min_depth_m", 0.4)),
            max_depth=float(self.cfg["ros"].get("max_depth_m", 5.0)),
        )

        depth_debug_dir = self.output_dir / "depth_color_debug"
        depth_debug_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(depth_debug_dir / f"depth_{time_idx:05d}.png"),
            depth_color,
        )

        if time_idx == 0:

            self.initialize_first_frame(
                intrinsics,
                K_scaled,
                rgb_np,
                depth_np,
                rgb_info,
            )
            self.get_logger().info(
                f"ODOM frame_id={odom_msg.header.frame_id}, child_frame_id={odom_msg.child_frame_id}"
            )

        # self.gt_w2c_all_frames.append(torch.eye(4, device=self.device).float())
        # curr_gt_w2c = self.gt_w2c_all_frames
        # 1. Compute ZED-derived initial pose
        zed_link_c2w = odom_to_c2w(odom_msg, self.device)
        if self.cfg["ros"].get("odom_frame", "zed_link") == "optical":
            # Odometry already poses the optical frame (e.g. dataset replay)
            zed_optical_c2w = zed_link_c2w
        else:
            T_link_left_optical = zed_link_to_left_optical(self.device)
            zed_optical_c2w = zed_link_c2w @ T_link_left_optical

        if time_idx == 0:
            self.first_zed_optical_c2w = zed_optical_c2w.detach().clone()

        rel_c2w = torch.linalg.inv(self.first_zed_optical_c2w) @ zed_optical_c2w
        zed_w2c = c2w_to_w2c(rel_c2w)

        # 2. Initialize SplaTAM camera pose from ZED
        if time_idx > 0:
            self.params = initialize_camera_pose(
                self.params,
                time_idx,
                forward_prop=False,
            )

            with torch.no_grad():
                self.set_camera_pose_from_w2c(time_idx, zed_w2c)

        # Store the ZED-initialized pose; tracking refinement below updates params in place
        curr_w2c = self.curr_w2c_from_params(time_idx)
        self.gt_w2c_all_frames.append(curr_w2c.detach().clone())
        curr_gt_w2c = self.gt_w2c_all_frames
        
        curr_data = {
            "cam": self.cam,
            "im": color,
            "depth": depth,
            "id": time_idx,
            "intrinsics": self.intrinsics,
            "w2c": zed_w2c,
            "iter_gt_w2c_list": curr_gt_w2c,
        }

        tracking_curr_data = curr_data

        if time_idx > 0 and not self.cfg["tracking"]["use_gt_poses"]:
            optimizer = initialize_optimizer(
                self.params,
                self.cfg["tracking"]["lrs"],
                tracking=True,
            )

            candidate_rot = self.params["cam_unnorm_rots"][..., time_idx].detach().clone()
            candidate_trn = self.params["cam_trans"][..., time_idx].detach().clone()
            current_min_loss = float(1e20)

            for it in range(int(self.cfg["tracking"]["num_iters"])):
                iter_start = time.time()

                loss, self.variables, losses = get_loss(
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
                    visualize_tracking_loss=self.cfg["tracking"].get(
                        "visualize_tracking_loss",
                        False,
                    ),
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

            with torch.no_grad():
                self.params["cam_unnorm_rots"][..., time_idx] = candidate_rot
                self.params["cam_trans"][..., time_idx] = candidate_trn

        curr_w2c = zed_w2c

        if time_idx == 0 or (time_idx + 1) % int(self.cfg["map_every"]) == 0:
            if self.cfg["mapping"]["add_new_gaussians"] and time_idx > 0:
                densify_color, densify_depth, densify_intrinsics, _ = self.make_densify_frame(
                    rgb_np,
                    depth_np,
                    rgb_info,
                )

                densify_curr_data = {
                    "cam": self.densify_cam,
                    "im": densify_color,
                    "depth": densify_depth,
                    "id": time_idx,
                    "intrinsics": densify_intrinsics,
                    "w2c": zed_w2c,
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
                    depth,
                    curr_w2c,
                    self.intrinsics,
                    self.keyframe_list[:-1],
                    num_keyframes,
                )

                selected_time_idx = [
                    self.keyframe_list[i]["id"] for i in selected_keyframes
                ]

                if len(self.keyframe_list) > 0:
                    selected_time_idx.append(self.keyframe_list[-1]["id"])
                    selected_keyframes.append(len(self.keyframe_list) - 1)

                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)

            optimizer = initialize_optimizer(
                self.params,
                self.cfg["mapping"]["lrs"],
                tracking=False,
            )

            for it in range(int(self.cfg["mapping"]["num_iters"])):
                iter_start = time.time()

                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_kf_idx = selected_keyframes[rand_idx]

                if selected_kf_idx == -1:
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
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

                loss, self.variables, losses = get_loss(
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
                            self.params,
                            self.variables,
                            optimizer,
                            it,
                            self.cfg["mapping"]["pruning_dict"],
                        )

                    if self.cfg["mapping"]["use_gaussian_splatting_densification"]:
                        self.params, self.variables = densify(
                            self.params,
                            self.variables,
                            optimizer,
                            it,
                            self.cfg["mapping"]["densify_dict"],
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                self.mapping_iter_time_sum += time.time() - iter_start
                self.mapping_iter_time_count += 1

        if (
            time_idx == 0
            or (time_idx + 1) % int(self.cfg["keyframe_every"]) == 0
            or time_idx == self.num_frames - 2
        ):
            self.keyframe_list.append(
                {
                    "id": time_idx,
                    "est_w2c": curr_w2c.detach().clone(),
                    "color": color,
                    "depth": depth,
                }
            )

            self.keyframe_time_indices.append(time_idx)

        self.publish_live_render(time_idx, rgb_msg.header.stamp)

        self.total_frames += 1

        num_gaussians = int(self.params["means3D"].shape[0])
        frame_dt = time.time() - frame_start
        fps = 1.0 / frame_dt if frame_dt > 0 else 0.0

        self.get_logger().info(
            f"Frame {self.total_frames}/{self.num_frames} | "
            f"FPS={fps:.2f} | "
            f"gaussians={num_gaussians:,}"
        )

        torch.cuda.empty_cache()

        if self.total_frames >= self.num_frames:
            self.finalize_and_exit()

    def finalize_and_exit(self):
        self.get_logger().info("Reached final frame. Saving output...")

        self.params["timestep"] = self.variables["timestep"]
        self.params["intrinsics"] = self.intrinsics.detach().cpu().numpy()
        self.params["w2c"] = self.first_frame_w2c.detach().cpu().numpy()
        self.params["org_width"] = self.cfg["data"]["desired_image_width"]
        self.params["org_height"] = self.cfg["data"]["desired_image_height"]

        self.params["gt_w2c_all_frames"] = np.stack(
            [m.detach().cpu().numpy() for m in self.gt_w2c_all_frames],
            axis=0,
        )

        self.params["keyframe_time_indices"] = np.array(
            self.keyframe_time_indices
        )

        save_params(self.params, str(self.output_dir))

        self.get_logger().info(f"Saved SplaTAM output to: {self.output_dir}")

        os._exit(0)


def main():
    args = parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.config),
        args.config,
    ).load_module()

    cfg = experiment.config

    # cfg["ros"]["use_odom"] = False
    # cfg["tracking"]["use_gt_poses"] = False

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
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()