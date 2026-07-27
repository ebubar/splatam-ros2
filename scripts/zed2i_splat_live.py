#!/usr/bin/env python3
"""
Live SplaTAM over ROS2 for the ZED2i camera.

Architecture (single-robot, real-time, network-tolerant):

  * ROS ingestion is DECOUPLED from SplaTAM processing.
    The synchronized subscriber callback is intentionally trivial: it drops the
    newest (rgb, depth, odom) triple into a single-slot buffer, overwriting any
    frame that has not yet been consumed. A dedicated worker loop always pulls
    the *freshest* frame and runs SplaTAM on it. Frame drops therefore become
    explicit and controlled ("always process newest") instead of being a
    non-deterministic DDS queue lottery under network jitter.

  * ZED VIO odometry is treated as a SEED ONLY, never as ground truth.
    Each frame's pose is initialized either from ZED odom or from SplaTAM's own
    constant-velocity motion model (config `ros.pose_init`), and is then refined
    by SplaTAM's dense tracking. The REFINED pose is authoritative and is what
    gets stored in the trajectory, keyframes, mapping, and the exported map. Raw
    VIO never contaminates the output.

  * The map output is structured for later multi-robot rtabmap melding.
    Per-frame and per-keyframe ROS timestamps are captured, and a TUM-format
    trajectory (camera-to-world, in the SplaTAM/VIO world frame) is exported
    alongside params.npz + splat.ply. That trajectory is the seam a ground
    station will use to align each robot's map.
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from importlib.machinery import SourceFileLoader

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
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
from utils.rtabmap_export import export_keyframe_dataset
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


# --------------------------------------------------------------------------- #
# Decoding / conversion helpers
# --------------------------------------------------------------------------- #

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


def mat_to_quat_xyzw(R):
    """3x3 numpy rotation matrix -> quaternion [x, y, z, w] (TUM/ROS order)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]

    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    return q


def write_tum_trajectory(path, stamps, w2c_list):
    """
    Write a TUM-format trajectory:  timestamp tx ty tz qx qy qz qw

    Poses are written as camera-to-world (the camera's position/orientation in
    the world frame), which is the TUM/evo/rtabmap convention. `w2c_list` holds
    world-to-camera 4x4 numpy matrices; we invert each to get c2w.
    """
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for stamp, w2c in zip(stamps, w2c_list):
            c2w = np.linalg.inv(w2c)
            t = c2w[:3, 3]
            q = mat_to_quat_xyzw(c2w[:3, :3])
            f.write(
                f"{stamp:.6f} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )


def stamp_to_sec(header_stamp):
    return float(header_stamp.sec) + float(header_stamp.nanosec) * 1e-9


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


# --------------------------------------------------------------------------- #
# Live SplaTAM node
# --------------------------------------------------------------------------- #

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

        ros_cfg = self.cfg["ros"]
        self.use_odom = bool(ros_cfg.get("use_odom", True))

        # Pose seeding strategy. VIO is a SEED ONLY; the refined SplaTAM pose is
        # always authoritative. If VIO is untrustworthy, use "constant_velocity"
        # (or set ros.use_odom=False) to keep VIO out of the pipeline entirely.
        #   "odom"              -> seed each frame from ZED odometry
        #   "constant_velocity" -> seed from SplaTAM's own motion model (no VIO)
        self.pose_init = str(ros_cfg.get("pose_init", "odom" if self.use_odom else "constant_velocity"))
        if not self.use_odom:
            self.pose_init = "constant_velocity"

        self.save_depth_debug = bool(ros_cfg.get("save_depth_debug", False))

        self.workdir = Path(self.cfg["workdir"])
        self.run_name = self.cfg.get("run_name", "SplaTAM_ZED2i_ROS2")
        self.output_dir = self.workdir / self.run_name

        if self.cfg.get("overwrite", False) and self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SplaTAM state
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
        self.gt_w2c_all_frames = []      # refined world-to-camera poses (authoritative)
        self.frame_stamps = []           # ROS stamp (sec) per processed frame
        self.keyframe_stamps = []        # ROS stamp (sec) per keyframe

        self.tracking_iter_time_sum = 0.0
        self.tracking_iter_time_count = 0
        self.mapping_iter_time_sum = 0.0
        self.mapping_iter_time_count = 0

        # --- Decoupling: single-slot latest-frame buffer -------------------- #
        self._buf_lock = threading.Lock()
        self._latest = None              # (rgb_msg, depth_msg, odom_msg_or_None, seq)
        self._recv_seq = 0               # incremented on every synced frame
        self._last_processed_seq = 0
        self._stop = False

        self.latest_rgb_info = None

        # --- ROS I/O -------------------------------------------------------- #
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.rgb_info_direct_sub = self.create_subscription(
            CameraInfo,
            ros_cfg["rgb_info_topic"],
            self.rgb_info_cb,
            qos,
        )

        self.rgb_sub = Subscriber(self, Image, ros_cfg["rgb_topic"], qos_profile=qos)
        self.depth_sub = Subscriber(self, Image, ros_cfg["depth_topic"], qos_profile=qos)

        sync_subs = [self.rgb_sub, self.depth_sub]

        if self.use_odom:
            self.odom_sub = Subscriber(self, Odometry, ros_cfg["odom_topic"], qos_profile=qos)
            sync_subs.append(self.odom_sub)

        self.ts = ApproximateTimeSynchronizer(
            sync_subs,
            queue_size=100,
            slop=float(ros_cfg.get("sync_slop_s", 0.15)),
            allow_headerless=False,
        )

        if self.use_odom:
            self.ts.registerCallback(self._cb_with_odom)
        else:
            self.ts.registerCallback(self._cb_no_odom)

        self.get_logger().info("ZedSplatamOnline ready.")
        self.get_logger().info(f"RGB:   {ros_cfg['rgb_topic']}")
        self.get_logger().info(f"Depth: {ros_cfg['depth_topic']}")
        self.get_logger().info(f"Pose seed: {self.pose_init} (use_odom={self.use_odom})")

    # ---- ROS callbacks (kept intentionally trivial) ----------------------- #

    def rgb_info_cb(self, msg):
        self.latest_rgb_info = msg

    def _cb_with_odom(self, rgb_msg, depth_msg, odom_msg):
        self._store_latest(rgb_msg, depth_msg, odom_msg)

    def _cb_no_odom(self, rgb_msg, depth_msg):
        self._store_latest(rgb_msg, depth_msg, None)

    def _store_latest(self, rgb_msg, depth_msg, odom_msg):
        with self._buf_lock:
            self._recv_seq += 1
            self.received_frames = self._recv_seq
            self._latest = (rgb_msg, depth_msg, odom_msg, self._recv_seq)

    def _take_latest(self):
        """Return the freshest unprocessed frame, or None if nothing new."""
        with self._buf_lock:
            if self._latest is None:
                return None
            if self._latest[3] == self._last_processed_seq:
                return None
            return self._latest

    def request_stop(self):
        self._stop = True

    # ---- Frame construction ------------------------------------------------ #

    def make_frame(self, rgb_msg, depth_msg, rgb_info):
        rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

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

    def set_camera_pose_from_w2c(self, time_idx, w2c):
        q = rotmat_to_quat_wxyz(w2c[:3, :3])
        t = w2c[:3, 3]

        rot_slot = self.params["cam_unnorm_rots"][..., time_idx]
        trans_slot = self.params["cam_trans"][..., time_idx]

        self.params["cam_unnorm_rots"][..., time_idx] = q.reshape_as(rot_slot)
        self.params["cam_trans"][..., time_idx] = t.reshape_as(trans_slot)

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

    def initialize_first_frame(self, intrinsics, K_scaled, rgb, depth_m, rgb_info):
        self.intrinsics = intrinsics
        self.first_frame_w2c = torch.eye(4, device=self.device).float()

        W = int(self.cfg["data"]["desired_image_width"])
        H = int(self.cfg["data"]["desired_image_height"])

        self.cam = setup_camera(
            W, H, K_scaled, self.first_frame_w2c.detach().cpu().numpy(),
        )

        densify_color, densify_depth, densify_intrinsics, Kd = self.make_densify_frame(
            rgb, depth_m, rgb_info,
        )

        self.densify_intrinsics = densify_intrinsics

        dW = int(self.cfg["data"]["densification_image_width"])
        dH = int(self.cfg["data"]["densification_image_height"])

        self.densify_cam = setup_camera(
            dW, dH, Kd, self.first_frame_w2c.detach().cpu().numpy(),
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

    # ---- Pose seeding (VIO is seed only) ---------------------------------- #

    def seed_pose(self, time_idx, odom_msg):
        """
        Seed the pose slot for `time_idx`. Returns the seed w2c (torch) if a VIO
        seed was applied, else None. The seed is only an initialization for the
        tracking refinement below; it is never used directly as the output pose.
        """
        if self.pose_init == "odom" and odom_msg is not None:
            zed_link_c2w = odom_to_c2w(odom_msg, self.device)
            T_link_left_optical = zed_link_to_left_optical(self.device)
            zed_optical_c2w = zed_link_c2w @ T_link_left_optical

            if self.first_zed_optical_c2w is None:
                self.first_zed_optical_c2w = zed_optical_c2w.detach().clone()

            rel_c2w = torch.linalg.inv(self.first_zed_optical_c2w) @ zed_optical_c2w
            zed_w2c = c2w_to_w2c(rel_c2w)

            # Copy previous refined pose into slot, then overwrite with VIO seed.
            self.params = initialize_camera_pose(self.params, time_idx, forward_prop=False)
            with torch.no_grad():
                self.set_camera_pose_from_w2c(time_idx, zed_w2c)
            return zed_w2c

        # VIO-free: seed from SplaTAM's own constant-velocity motion model.
        self.params = initialize_camera_pose(self.params, time_idx, forward_prop=True)
        return None

    def track_pose(self, time_idx, tracking_curr_data):
        """Refine the seeded pose with SplaTAM dense tracking (keep best loss)."""
        optimizer = initialize_optimizer(
            self.params, self.cfg["tracking"]["lrs"], tracking=True,
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

        with torch.no_grad():
            self.params["cam_unnorm_rots"][..., time_idx] = candidate_rot
            self.params["cam_trans"][..., time_idx] = candidate_trn

    # ---- Mapping ----------------------------------------------------------- #

    def run_mapping(self, time_idx, color, depth, curr_w2c, rgb_np, depth_np, rgb_info):
        if self.cfg["mapping"]["add_new_gaussians"] and time_idx > 0:
            densify_color, densify_depth, densify_intrinsics, _ = self.make_densify_frame(
                rgb_np, depth_np, rgb_info,
            )

            densify_curr_data = {
                "cam": self.densify_cam,
                "im": densify_color,
                "depth": densify_depth,
                "id": time_idx,
                "intrinsics": densify_intrinsics,
                "w2c": curr_w2c,                        # refined pose
                "iter_gt_w2c_list": self.gt_w2c_all_frames,
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
                depth, curr_w2c, self.intrinsics, self.keyframe_list[:-1], num_keyframes,
            )

            selected_time_idx = [self.keyframe_list[i]["id"] for i in selected_keyframes]

            if len(self.keyframe_list) > 0:
                selected_time_idx.append(self.keyframe_list[-1]["id"])
                selected_keyframes.append(len(self.keyframe_list) - 1)

            selected_time_idx.append(time_idx)
            selected_keyframes.append(-1)

        optimizer = initialize_optimizer(
            self.params, self.cfg["mapping"]["lrs"], tracking=False,
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
                        self.params, self.variables, optimizer, it,
                        self.cfg["mapping"]["pruning_dict"],
                    )

                if self.cfg["mapping"]["use_gaussian_splatting_densification"]:
                    self.params, self.variables = densify(
                        self.params, self.variables, optimizer, it,
                        self.cfg["mapping"]["densify_dict"],
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            self.mapping_iter_time_sum += time.time() - iter_start
            self.mapping_iter_time_count += 1

    # ---- Per-frame pipeline (single, consistent pose flow) ---------------- #

    def process_frame(self, rgb_msg, depth_msg, odom_msg, dropped):
        if self.latest_rgb_info is None:
            self.get_logger().warn("Waiting for RGB CameraInfo...")
            return

        rgb_info = self.latest_rgb_info

        if self.total_frames >= self.num_frames:
            return

        frame_start = time.time()
        time_idx = self.total_frames
        stamp = stamp_to_sec(rgb_msg.header.stamp)

        color, depth, intrinsics, K_scaled, rgb_np, depth_np = self.make_frame(
            rgb_msg, depth_msg, rgb_info,
        )

        if self.save_depth_debug:
            depth_debug_dir = self.output_dir / "depth_color_debug"
            depth_debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(depth_debug_dir / f"depth_{time_idx:05d}.png"),
                colorize_depth(
                    depth_np,
                    min_depth=float(self.cfg["ros"].get("min_depth_m", 0.4)),
                    max_depth=float(self.cfg["ros"].get("max_depth_m", 5.0)),
                ),
            )

        # ----- First frame: initialize map at identity pose ----------------- #
        if time_idx == 0:
            self.initialize_first_frame(intrinsics, K_scaled, rgb_np, depth_np, rgb_info)

            # Anchor the VIO world frame at the first frame (if using odom).
            if self.pose_init == "odom" and odom_msg is not None:
                zed_link_c2w = odom_to_c2w(odom_msg, self.device)
                zed_optical_c2w = zed_link_c2w @ zed_link_to_left_optical(self.device)
                self.first_zed_optical_c2w = zed_optical_c2w.detach().clone()

            curr_w2c = self.first_frame_w2c.clone()
        else:
            # 1. Seed the pose (VIO seed OR constant-velocity), never as output.
            seed_w2c = self.seed_pose(time_idx, odom_msg)

            # 2. Refine with dense tracking. The refined pose is authoritative.
            if not self.cfg["tracking"]["use_gt_poses"] and int(self.cfg["tracking"]["num_iters"]) > 0:
                tracking_curr_data = {
                    "cam": self.cam,
                    "im": color,
                    "depth": depth,
                    "id": time_idx,
                    "intrinsics": self.intrinsics,
                    "w2c": self.first_frame_w2c,
                    "iter_gt_w2c_list": self.gt_w2c_all_frames,
                }
                self.track_pose(time_idx, tracking_curr_data)

            # 3. Read back the refined pose (single source of truth).
            curr_w2c = self.curr_w2c_from_params(time_idx)

        # Store the authoritative pose + timestamp for this frame.
        self.gt_w2c_all_frames.append(curr_w2c.detach().clone())
        self.frame_stamps.append(stamp)

        # ----- Mapping ------------------------------------------------------ #
        if time_idx == 0 or (time_idx + 1) % int(self.cfg["map_every"]) == 0:
            self.run_mapping(time_idx, color, depth, curr_w2c, rgb_np, depth_np, rgb_info)

        # ----- Keyframe bookkeeping ---------------------------------------- #
        if (
            time_idx == 0
            or (time_idx + 1) % int(self.cfg["keyframe_every"]) == 0
            or time_idx == self.num_frames - 2
        ):
            self.keyframe_list.append(
                {
                    "id": time_idx,
                    "est_w2c": curr_w2c.detach().clone(),   # refined pose
                    "color": color,
                    "depth": depth,
                    "stamp": stamp,
                }
            )
            self.keyframe_time_indices.append(time_idx)
            self.keyframe_stamps.append(stamp)

        self.total_frames += 1

        num_gaussians = int(self.params["means3D"].shape[0])
        frame_dt = time.time() - frame_start
        fps = 1.0 / frame_dt if frame_dt > 0 else 0.0

        self.get_logger().info(
            f"Frame {self.total_frames}/{self.num_frames} | "
            f"FPS={fps:.2f} | dropped={dropped} | "
            f"gaussians={num_gaussians:,}"
        )

        if self.total_frames % 30 == 0:
            torch.cuda.empty_cache()

    # ---- Worker loop ------------------------------------------------------- #

    def run_worker(self):
        """Pull the freshest frame and process it, until the cap or a stop."""
        while rclpy.ok() and not self._stop:
            item = self._take_latest()
            if item is None:
                time.sleep(0.002)
                continue

            rgb_msg, depth_msg, odom_msg, seq = item
            dropped = max(0, seq - self._last_processed_seq - 1)
            self._last_processed_seq = seq

            try:
                self.process_frame(rgb_msg, depth_msg, odom_msg, dropped)
            except KeyboardInterrupt:
                self.get_logger().info("Interrupted — finalizing partial map...")
                break
            except Exception:
                # A per-frame failure (e.g. transient CUDA/decoder error) must
                # not throw away the map built so far. Log and stop cleanly so
                # finalize saves the partial result.
                import traceback
                self.get_logger().error(
                    f"process_frame failed at frame {self.total_frames}; "
                    f"finalizing partial map.\n{traceback.format_exc()}"
                )
                break

            if self.total_frames >= self.num_frames:
                break

        self.finalize_and_exit()

    # ---- Finalize / export ------------------------------------------------- #

    def finalize_and_exit(self):
        if self.params is None or self.total_frames == 0:
            self.get_logger().warn("No frames processed; nothing to save.")
            os._exit(0)

        self.get_logger().info("Saving output...")

        self.params["timestep"] = self.variables["timestep"]
        self.params["intrinsics"] = self.intrinsics.detach().cpu().numpy()
        self.params["w2c"] = self.first_frame_w2c.detach().cpu().numpy()
        self.params["org_width"] = self.cfg["data"]["desired_image_width"]
        self.params["org_height"] = self.cfg["data"]["desired_image_height"]

        w2c_np = [m.detach().cpu().numpy() for m in self.gt_w2c_all_frames]

        self.params["gt_w2c_all_frames"] = np.stack(w2c_np, axis=0)
        self.params["keyframe_time_indices"] = np.array(self.keyframe_time_indices)
        self.params["frame_stamps"] = np.array(self.frame_stamps, dtype=np.float64)

        save_params(self.params, str(self.output_dir))

        # --- rtabmap / evo trajectory export (VIO world frame) --------------- #
        try:
            write_tum_trajectory(
                str(self.output_dir / "traj_tum.txt"),
                self.frame_stamps,
                w2c_np,
            )
            kf_w2c_np = [self.gt_w2c_all_frames[i].detach().cpu().numpy() for i in self.keyframe_time_indices]
            write_tum_trajectory(
                str(self.output_dir / "traj_keyframes_tum.txt"),
                self.keyframe_stamps,
                kf_w2c_np,
            )
            meta = {
                "run_name": self.run_name,
                "num_frames": int(self.total_frames),
                "num_keyframes": int(len(self.keyframe_time_indices)),
                "pose_init": self.pose_init,
                "use_odom": self.use_odom,
                "world_frame": "first_frame_optical (SplaTAM/VIO origin at frame 0)",
                "trajectory_convention": "TUM: timestamp tx ty tz qx qy qz qw (camera-to-world)",
                "image_width": int(self.cfg["data"]["desired_image_width"]),
                "image_height": int(self.cfg["data"]["desired_image_height"]),
            }
            with open(self.output_dir / "map_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:  # export must never crash the run
            self.get_logger().warn(f"Trajectory export failed: {exc}")

        # --- Per-keyframe RGB-D dataset export for rtabmap ------------------ #
        # The currency a robot ships to the ground station: keyframe RGB-D +
        # refined poses + timestamps, as a TUM RGB-D dataset rtabmap ingests.
        if self.cfg.get("export_rtabmap", True) and len(self.keyframe_list) > 0:
            try:
                K = self.intrinsics.detach().cpu().numpy()
                fx, fy = float(K[0, 0]), float(K[1, 1])
                cx, cy = float(K[0, 2]), float(K[1, 2])
                W = int(self.cfg["data"]["desired_image_width"])
                H = int(self.cfg["data"]["desired_image_height"])

                kfs = []
                for kf in self.keyframe_list:
                    color = kf["color"].detach().clamp(0, 1)
                    rgb = (color * 255.0).byte().permute(1, 2, 0).cpu().numpy()
                    depth_m = kf["depth"].detach()[0].cpu().numpy()
                    c2w = np.linalg.inv(kf["est_w2c"].detach().cpu().numpy())
                    t = c2w[:3, 3]
                    q = mat_to_quat_xyzw(c2w[:3, :3])
                    kfs.append(
                        {
                            "stamp": kf["stamp"],
                            "rgb": rgb,
                            "depth": depth_m,
                            "tum": (
                                float(t[0]), float(t[1]), float(t[2]),
                                float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                            ),
                        }
                    )

                n = export_keyframe_dataset(
                    str(self.output_dir / "rtabmap_export"),
                    kfs, fx, fy, cx, cy, W, H,
                    depth_scale=float(self.cfg.get("rtabmap_depth_scale", 5000.0)),
                )
                self.get_logger().info(
                    f"Exported {n} keyframes for rtabmap -> "
                    f"{self.output_dir / 'rtabmap_export'}"
                )
            except Exception as exc:
                self.get_logger().warn(f"rtabmap keyframe export failed: {exc}")

        self.get_logger().info(f"Saved SplaTAM output to: {self.output_dir}")
        os._exit(0)


def main():
    args = parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.config),
        args.config,
    ).load_module()

    cfg = experiment.config

    if "gaussian_distribution" not in cfg:
        cfg["gaussian_distribution"] = "isotropic"

    seed_everything(seed=cfg["seed"])

    rclpy.init()
    node = ZedSplatamOnline(cfg)

    # ROS spins in a background thread; SplaTAM runs in the main-thread worker.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_worker()
    except KeyboardInterrupt:
        node.request_stop()
    finally:
        try:
            executor.shutdown()
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
