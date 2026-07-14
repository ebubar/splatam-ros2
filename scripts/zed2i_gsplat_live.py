#!/usr/bin/env python3
"""
Near-realtime ZED2i live splatting node (gsplat backend, trusted ZED poses,
network-robust ingestion, decoupled mapping thread).

Differences from the original scripts/zed2i_splat_live.py (which is kept as the
synchronous CUDA fallback):

  * Rendering runs through utils/render_backend.py -> selectable "gsplat"
    (Apache-2.0) or "cuda" engine via cfg['render_backend'].
  * Poses are trusted from the ZED SDK (cfg['pose_source']='zed_odom'), looked up
    by *image timestamp* with SLERP interpolation (utils/pose_source.py +
    pose_buffer.py) instead of a fragile 3-way message_filters sync. A short
    photometric pose-refinement fallback only fires when the pose looks
    unreliable (cfg['tracking']['mode']='auto').
  * RGB+depth are synced 2-way; odometry is buffered separately. Optional
    compressed transport (cfg['ros']['transport']='compressed') reuses the
    existing compressedDepth decoder.
  * Mapping runs on a dedicated thread fed by a bounded drop-oldest buffer, so
    the ROS callback never blocks (cfg['async_mapping']=True). Set False for a
    deterministic A/B against the CUDA backend.

Run:
  python3 scripts/zed2i_gsplat_live.py --config configs/zed2i/zed2i_gsplat_live.py
"""

import argparse
import os
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
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
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from utils.common_utils import seed_everything, save_params
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation, remove_points
from utils.pose_source import make_pose_source, ZedOdomPoseSource
from utils.capture_guidance import CaptureGuidance
from scripts.splatam import (
    get_loss,
    initialize_optimizer,
    initialize_params,
    initialize_camera_pose,
    get_pointcloud,
    add_new_gaussians,
)
# Reuse the tested ROS<->numpy helpers from the original live node.
from scripts.zed2i_splat_live import (
    caminfo_to_K,
    depth_to_meters,
    ros_rgb_to_rgb,
    rotmat_to_quat_wxyz,
)
# Reuse the existing compressed-transport decoders.
from datasets.zed_ros_dataset import decode_compressed_rgb, decode_compressed_depth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/zed2i/zed2i_gsplat_live.py", type=str)
    return parser.parse_args()


def stamp_to_sec(header):
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


@dataclass
class FrameMsg:
    seq: int
    stamp_sec: float
    rgb_np: np.ndarray      # HxWx3 RGB uint8 (working resolution)
    depth_np: np.ndarray    # HxW float32 meters (working resolution)
    K_scaled: np.ndarray    # 3x3 intrinsics at working resolution


class ZedGsplatOnline(Node):
    def __init__(self, config):
        super().__init__("ZedGsplatOnline")
        self.cfg = config
        self.bridge = CvBridge()
        self.device = torch.device(self.cfg.get("primary_device", "cuda:0"))

        self.num_frames = int(self.cfg["num_frames"])
        self.total_frames = 0          # frames processed by the mapper
        self.received_frames = 0       # synced RGB-D frames received
        self.process_every_n = int(self.cfg["ros"].get("process_every_n", 1))

        self.render_backend = self.cfg.get("render_backend", "gsplat")
        self.gsplat_cfg = self.cfg.get("gsplat", {})
        self.async_mapping = bool(self.cfg.get("async_mapping", True))
        self.tracking_mode = self.cfg.get("tracking", {}).get("mode", "auto")

        # System hardening (WS7): bounded memory, adaptive budget, checkpointing.
        self.hcfg = self.cfg.get("hardening", {})
        self._cur_map_iters = int(self.cfg["mapping"]["num_iters"])
        self._base_map_iters = self._cur_map_iters
        self._min_map_iters = int(self.hcfg.get("min_map_iters", 15))
        self._target_fps = float(self.hcfg.get("target_fps", 0.0))
        self._adaptive = bool(self.hcfg.get("adaptive_fps", True))
        self._checkpoint_every = int(self.hcfg.get("checkpoint_every", 0))

        self.workdir = Path(self.cfg["workdir"])
        self.run_name = self.cfg.get("run_name", "SplaTAM_ZED2i_gsplat")
        self.output_dir = self.workdir / self.run_name
        if self.cfg.get("overwrite", False) and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Map state (only ever touched by the mapper thread).
        self.params = None
        self.variables = None
        self.intrinsics = None
        self.first_frame_w2c = None
        self.cam = None
        self.keyframe_list = []
        self.keyframe_time_indices = []
        self.gt_w2c_all_frames = []

        # Pose source (default: ZED odom, interpolated by timestamp).
        self.pose_source = make_pose_source(self.cfg)

        # Optional online capture guidance (off by default).
        cg_cfg = self.cfg.get("capture_guidance", {})
        self.capture_guidance = CaptureGuidance(cg_cfg) if cg_cfg.get("enabled", False) else None

        # Bounded drop-oldest frame buffer + mapper thread.
        self.frame_buf = deque(maxlen=int(self.cfg.get("frame_buffer_size", 2)))
        self.buf_lock = threading.Lock()
        self.new_frame_evt = threading.Event()
        self.stop_evt = threading.Event()
        self.latest_rgb_info = None

        self._setup_subscriptions()

        self.mapper_thread = None
        if self.async_mapping:
            self.mapper_thread = threading.Thread(target=self._mapper_loop, daemon=True)
            self.mapper_thread.start()

        self.get_logger().info(
            f"ZedGsplatOnline ready | backend={self.render_backend} "
            f"pose_source={self.cfg.get('pose_source', 'zed_odom')} "
            f"async={self.async_mapping} tracking={self.tracking_mode}"
        )

    # ------------------------------------------------------------------ setup
    def _setup_subscriptions(self):
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        ros_cfg = self.cfg["ros"]
        self.transport = ros_cfg.get("transport", "raw")

        # Status topic for field monitoring (WS7).
        self.status_pub = None
        if self.hcfg.get("publish_status", True):
            self.status_pub = self.create_publisher(
                String, self.hcfg.get("status_topic", "/splatam/status"), 10)

        # Camera intrinsics: cache the latest RGB CameraInfo (small, reliable).
        self.create_subscription(CameraInfo, ros_cfg["rgb_info_topic"],
                                 self._rgb_info_cb, qos)

        # Odometry buffered separately (decoupled from the image sync).
        self.create_subscription(Odometry, ros_cfg["odom_topic"], self._odom_cb, qos)

        # RGB + depth synced 2-way only (no odom in the sync -> robust to jitter).
        if self.transport == "compressed":
            rgb_sub = Subscriber(self, CompressedImage, ros_cfg["rgb_topic_compressed"],
                                 qos_profile=qos)
            depth_sub = Subscriber(self, CompressedImage, ros_cfg["depth_topic_compressed"],
                                   qos_profile=qos)
        else:
            rgb_sub = Subscriber(self, Image, ros_cfg["rgb_topic"], qos_profile=qos)
            depth_sub = Subscriber(self, Image, ros_cfg["depth_topic"], qos_profile=qos)

        self.ts = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=int(ros_cfg.get("sync_queue", 30)),
            slop=float(ros_cfg.get("sync_slop", 0.05)),
            allow_headerless=False,
        )
        self.ts.registerCallback(self._synced_cb)

    def _rgb_info_cb(self, msg):
        self.latest_rgb_info = msg

    def _odom_cb(self, msg):
        # Cheap: push the sample into the pose buffer keyed by its own stamp.
        if isinstance(self.pose_source, ZedOdomPoseSource):
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self.pose_source.add_odom(stamp_to_sec(msg.header),
                                      (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

    # -------------------------------------------------------------- callback
    def _decode(self, rgb_msg, depth_msg):
        if self.transport == "compressed":
            rgb = decode_compressed_rgb(rgb_msg)              # BGR
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth_m = decode_compressed_depth(depth_msg)      # meters
        else:
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            rgb = ros_rgb_to_rgb(rgb_cv, rgb_msg.encoding)
            depth_m = depth_to_meters(depth_cv, depth_msg.encoding)
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        return rgb, depth_m.astype(np.float32)

    def _synced_cb(self, rgb_msg, depth_msg):
        self.received_frames += 1
        if self.latest_rgb_info is None:
            self.get_logger().warn("Waiting for RGB CameraInfo...", throttle_duration_sec=2.0)
            return
        if self.received_frames % self.process_every_n != 0:
            return
        if self.total_frames >= self.num_frames:
            return

        ros_cfg = self.cfg["ros"]
        rgb, depth_m = self._decode(rgb_msg, depth_msg)

        # Clamp + scrub depth (neural depth may have NaN/inf and out-of-range).
        depth_m = depth_m * float(ros_cfg.get("depth_unit_scale_m", 1.0))
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        min_d = float(ros_cfg.get("min_depth_m", 0.3))
        max_d = float(ros_cfg.get("max_depth_m", 8.0))
        depth_m[(depth_m < min_d) | (depth_m > max_d)] = 0.0

        # Resize to working resolution and scale intrinsics.
        W = int(self.cfg["data"]["desired_image_width"])
        H = int(self.cfg["data"]["desired_image_height"])
        rgb_rs = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        depth_rs = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

        info = self.latest_rgb_info
        K = caminfo_to_K(info)
        sx = W / float(info.width)
        sy = H / float(info.height)
        K_scaled = K.copy()
        K_scaled[0, 0] *= sx; K_scaled[1, 1] *= sy
        K_scaled[0, 2] *= sx; K_scaled[1, 2] *= sy

        # MapAnything source also needs the raw RGB window.
        if not isinstance(self.pose_source, ZedOdomPoseSource):
            self.pose_source.add_frame(stamp_to_sec(rgb_msg.header), rgb_rs, depth_rs, K_scaled)

        frame = FrameMsg(
            seq=self.received_frames,
            stamp_sec=stamp_to_sec(rgb_msg.header),
            rgb_np=rgb_rs,
            depth_np=depth_rs,
            K_scaled=K_scaled.astype(np.float32),
        )

        if self.async_mapping:
            with self.buf_lock:
                self.frame_buf.append(frame)   # drop-oldest via deque(maxlen)
            self.new_frame_evt.set()
        else:
            # Deterministic path: map inline (blocks ingestion; used for A/B).
            self._process_frame(frame)
            if self.total_frames >= self.num_frames:
                self._finalize()

    # ---------------------------------------------------------- mapper thread
    def _mapper_loop(self):
        while not self.stop_evt.is_set():
            self.new_frame_evt.wait(timeout=0.5)
            self.new_frame_evt.clear()
            while True:
                with self.buf_lock:
                    frame = self.frame_buf.pop() if self.frame_buf else None
                    self.frame_buf.clear()   # keep only the latest -> near-realtime
                if frame is None:
                    break
                if self.total_frames >= self.num_frames:
                    self._finalize()
                    return
                try:
                    self._process_frame(frame)
                except Exception as exc:  # keep the thread alive on a bad frame
                    self.get_logger().error(f"mapper frame {frame.seq} failed: {exc}")
                if self.total_frames >= self.num_frames:
                    self._finalize()
                    return

    # -------------------------------------------------------- frame processing
    def _to_torch(self, frame):
        color = torch.from_numpy(frame.rgb_np).to(self.device).float().permute(2, 0, 1) / 255.0
        depth = torch.from_numpy(frame.depth_np[None]).to(self.device).float()  # [1,H,W]
        intrinsics = torch.tensor(frame.K_scaled, device=self.device).float()
        return color, depth, intrinsics

    def _write_pose(self, time_idx, w2c):
        q = rotmat_to_quat_wxyz(w2c[:3, :3])
        t = w2c[:3, 3]
        rot_slot = self.params["cam_unnorm_rots"][..., time_idx]
        trn_slot = self.params["cam_trans"][..., time_idx]
        self.params["cam_unnorm_rots"][..., time_idx] = q.reshape_as(rot_slot)
        self.params["cam_trans"][..., time_idx] = t.reshape_as(trn_slot)

    def _init_first_frame(self, color, depth, intrinsics, K_scaled):
        self.intrinsics = intrinsics
        self.first_frame_w2c = torch.eye(4, device=self.device).float()
        W = int(self.cfg["data"]["desired_image_width"])
        H = int(self.cfg["data"]["desired_image_height"])
        self.cam = setup_camera(W, H, K_scaled, self.first_frame_w2c.detach().cpu().numpy())

        mask = (depth > 0).reshape(-1)
        init_pt_cld, mean3_sq_dist = get_pointcloud(
            color, depth, intrinsics, self.first_frame_w2c,
            mask=mask, compute_mean_sq_dist=True,
            mean_sq_dist_method=self.cfg["mean_sq_dist_method"],
        )
        self.params, self.variables = initialize_params(
            init_pt_cld, self.num_frames, mean3_sq_dist,
            self.cfg.get("gaussian_distribution", "isotropic"),
        )
        self.variables["scene_radius"] = torch.max(depth) / self.cfg["scene_radius_depth_ratio"]

    def _curr_data(self, color, depth, intrinsics, time_idx, w2c_for_add):
        return {
            "cam": self.cam,               # used by CUDA backend
            "im": color,
            "depth": depth,
            "id": time_idx,
            "intrinsics": intrinsics,
            "w2c": w2c_for_add,            # used by CUDA depth+sil render
            "iter_gt_w2c_list": self.gt_w2c_all_frames,
        }

    def _pose_needs_refine(self, quality):
        if self.tracking_mode == "off":
            return False
        if self.tracking_mode == "always":
            return True
        # auto: refine only when the pose looks unreliable.
        return bool(quality.get("discontinuity") or quality.get("extrapolated")
                    or quality.get("stale"))

    def _refine_pose(self, color, depth, intrinsics, time_idx):
        """Short photometric pose refinement fallback (gsplat/cuda backend)."""
        tcfg = self.cfg["tracking"]
        curr_data = self._curr_data(color, depth, intrinsics, time_idx,
                                    self.first_frame_w2c)
        optimizer = initialize_optimizer(self.params, tcfg["lrs"], tracking=True)
        best_rot = self.params["cam_unnorm_rots"][..., time_idx].detach().clone()
        best_trn = self.params["cam_trans"][..., time_idx].detach().clone()
        best_loss = float(1e20)
        for _ in range(int(tcfg.get("num_iters", 20))):
            loss, self.variables, _ = get_loss(
                self.params, curr_data, self.variables, time_idx,
                tcfg["loss_weights"], tcfg["use_sil_for_loss"], tcfg["sil_thres"],
                tcfg["use_l1"], tcfg["ignore_outlier_depth_loss"], tracking=True,
                render_backend=self.render_backend, gsplat_cfg=self.gsplat_cfg,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if loss < best_loss:
                    best_loss = float(loss)
                    best_rot = self.params["cam_unnorm_rots"][..., time_idx].detach().clone()
                    best_trn = self.params["cam_trans"][..., time_idx].detach().clone()
        with torch.no_grad():
            self.params["cam_unnorm_rots"][..., time_idx] = best_rot
            self.params["cam_trans"][..., time_idx] = best_trn

    def _process_frame(self, frame):
        frame_start = time.time()
        time_idx = self.total_frames

        # Resolve the pose at the *image* timestamp (network-robust association).
        res = self.pose_source.get_w2c(frame.stamp_sec, frame=frame)
        if not res.ok:
            # No usable pose yet (stale odom buffer / MapAnything window not ready).
            self.get_logger().warn(
                f"frame {frame.seq}: no pose ({res.quality}); skipping",
                throttle_duration_sec=1.0,
            )
            return
        w2c = torch.tensor(res.w2c, device=self.device).float()

        color, depth, intrinsics = self._to_torch(frame)

        if time_idx == 0:
            self._init_first_frame(color, depth, intrinsics, frame.K_scaled)

        # Seed / write the trusted pose into the map's per-frame camera params.
        if time_idx > 0:
            self.params = initialize_camera_pose(self.params, time_idx, forward_prop=False)
        with torch.no_grad():
            self._write_pose(time_idx, w2c)

        # Optional gated photometric refinement (fallback for unreliable poses).
        if time_idx > 0 and self._pose_needs_refine(res.quality):
            self._refine_pose(color, depth, intrinsics, time_idx)

        # Record the (possibly refined) pose actually used for this frame.
        curr_w2c = self._curr_w2c(time_idx)
        self.gt_w2c_all_frames.append(curr_w2c.detach().clone())

        # --- Mapping (add gaussians + reduced optimization on keyframes) -----
        do_map = (time_idx == 0) or ((time_idx + 1) % int(self.cfg["map_every"]) == 0)
        if do_map:
            if self.cfg["mapping"]["add_new_gaussians"] and time_idx > 0:
                add_data = self._curr_data(color, depth, intrinsics, time_idx, curr_w2c)
                self.params, self.variables = add_new_gaussians(
                    self.params, self.variables, add_data,
                    self.cfg["mapping"]["sil_thres"], time_idx,
                    self.cfg["mean_sq_dist_method"],
                    self.cfg.get("gaussian_distribution", "isotropic"),
                    render_backend=self.render_backend, gsplat_cfg=self.gsplat_cfg,
                )
            self._run_mapping(color, depth, time_idx, curr_w2c)

        # Keyframe bookkeeping.
        if (time_idx == 0
                or (time_idx + 1) % int(self.cfg["keyframe_every"]) == 0
                or time_idx == self.num_frames - 2):
            self.keyframe_list.append({
                "id": time_idx, "est_w2c": curr_w2c.detach().clone(),
                "color": color, "depth": depth,
            })
            self.keyframe_time_indices.append(time_idx)

        # Optional capture guidance: update scene model and emit an operator hint.
        if self.capture_guidance is not None:
            try:
                self.capture_guidance.update_scene(
                    self.params["means3D"].detach().cpu().numpy())
                self.capture_guidance.register_keyframe(curr_w2c.detach().cpu().numpy())
                hint = self.capture_guidance.next_instruction(curr_w2c.detach().cpu().numpy())
                self.get_logger().info(f"[capture] {hint}")
            except Exception as exc:
                self.get_logger().warn(f"capture guidance skipped: {exc}")

        self.total_frames += 1
        dt = time.time() - frame_start
        fps = 1.0 / dt if dt > 0 else 0.0
        drop = self.received_frames - self.total_frames
        self.get_logger().info(
            f"Frame {self.total_frames}/{self.num_frames} | FPS={fps:.2f} | "
            f"gaussians={int(self.params['means3D'].shape[0]):,} | recv={self.received_frames} "
            f"drop={drop} | map_iters={self._cur_map_iters}"
        )

        # WS7 hardening: adapt budget, checkpoint, publish status.
        self._adjust_budget(fps)
        self._maybe_checkpoint(time_idx)
        self._publish_status(fps, drop, res.quality)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _curr_w2c(self, time_idx):
        with torch.no_grad():
            rot = F.normalize(self.params["cam_unnorm_rots"][..., time_idx].detach())
            trn = self.params["cam_trans"][..., time_idx].detach()
            w2c = torch.eye(4, device=self.device).float()
            w2c[:3, :3] = build_rotation(rot)
            w2c[:3, 3] = trn
        return w2c

    def _run_mapping(self, color, depth, time_idx, curr_w2c):
        mcfg = self.cfg["mapping"]
        with torch.no_grad():
            num_kf = int(self.cfg["mapping_window_size"]) - 2
            selected = keyframe_selection_overlap(
                depth, curr_w2c, self.intrinsics, self.keyframe_list[:-1], num_kf)
            if len(self.keyframe_list) > 0:
                selected.append(len(self.keyframe_list) - 1)
            selected.append(-1)  # current frame

        optimizer = initialize_optimizer(self.params, mcfg["lrs"], tracking=False)
        for _ in range(int(self._cur_map_iters)):
            kf = selected[np.random.randint(0, len(selected))]
            if kf == -1:
                it_idx, it_color, it_depth = time_idx, color, depth
            else:
                it_idx = self.keyframe_list[kf]["id"]
                it_color = self.keyframe_list[kf]["color"]
                it_depth = self.keyframe_list[kf]["depth"]
            iter_data = {
                "cam": self.cam, "im": it_color, "depth": it_depth, "id": it_idx,
                "intrinsics": self.intrinsics, "w2c": self.first_frame_w2c,
                "iter_gt_w2c_list": self.gt_w2c_all_frames[: it_idx + 1],
            }
            loss, self.variables, _ = get_loss(
                self.params, iter_data, self.variables, it_idx,
                mcfg["loss_weights"], mcfg["use_sil_for_loss"], mcfg["sil_thres"],
                mcfg["use_l1"], mcfg["ignore_outlier_depth_loss"], mapping=True,
                render_backend=self.render_backend, gsplat_cfg=self.gsplat_cfg,
            )
            loss.backward()
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Bounded memory: enforce the Gaussian budget with the mapping optimizer
        # in scope (keeps params/variables/optimizer-state consistent).
        self._enforce_budget(optimizer)

    # ------------------------------------------------------------ WS7 hardening
    def _enforce_budget(self, optimizer):
        """Opacity prune + hard Gaussian cap so memory stays bounded on long runs."""
        max_g = int(self.hcfg.get("max_gaussians", 0))
        opac_thresh = float(self.hcfg.get("prune_opacity_below", 0.0))
        n = int(self.params["means3D"].shape[0])
        if n == 0 or (max_g <= 0 and opac_thresh <= 0.0):
            return
        with torch.no_grad():
            opac = torch.sigmoid(self.params["logit_opacities"]).squeeze(-1)  # [N]
            to_remove = torch.zeros(n, dtype=torch.bool, device=opac.device)
            if opac_thresh > 0.0:
                to_remove |= (opac < opac_thresh)
            keep = int((~to_remove).sum())
            if max_g > 0 and keep > max_g:
                # Drop the lowest-opacity surplus among the currently-kept splats.
                surplus = keep - max_g
                kept_idx = (~to_remove).nonzero(as_tuple=True)[0]
                drop_local = torch.topk(opac[kept_idx], surplus, largest=False).indices
                to_remove[kept_idx[drop_local]] = True
            if bool(to_remove.any()):
                removed = int(to_remove.sum())
                self.params, self.variables = remove_points(
                    to_remove, self.params, self.variables, optimizer)
                self.get_logger().info(
                    f"[budget] pruned {removed} gaussians -> "
                    f"{int(self.params['means3D'].shape[0]):,}")

    def _adjust_budget(self, fps):
        """Gently adapt mapping iterations to hold a target processing rate."""
        if not self._adaptive or self._target_fps <= 0.0 or fps <= 0.0:
            return
        if fps < 0.9 * self._target_fps:
            self._cur_map_iters = max(self._min_map_iters,
                                      int(self._cur_map_iters * 0.9))
        elif fps > 1.2 * self._target_fps:
            self._cur_map_iters = min(self._base_map_iters,
                                      int(self._cur_map_iters * 1.1) + 1)

    def _maybe_checkpoint(self, time_idx):
        if self._checkpoint_every <= 0 or time_idx <= 0:
            return
        if (time_idx + 1) % self._checkpoint_every != 0:
            return
        try:
            ckpt = dict(self.params)
            ckpt["timestep"] = self.variables["timestep"]
            ckpt["intrinsics"] = self.intrinsics.detach().cpu().numpy()
            ckpt["w2c"] = self.first_frame_w2c.detach().cpu().numpy()
            ckpt["org_width"] = self.cfg["data"]["desired_image_width"]
            ckpt["org_height"] = self.cfg["data"]["desired_image_height"]
            ckpt["gt_w2c_all_frames"] = np.stack(
                [m.detach().cpu().numpy() for m in self.gt_w2c_all_frames], axis=0)
            ckpt["keyframe_time_indices"] = np.array(self.keyframe_time_indices)
            save_params(ckpt, str(self.output_dir))
            self.get_logger().info(f"[checkpoint] saved at frame {time_idx + 1}")
        except Exception as exc:
            self.get_logger().warn(f"checkpoint failed: {exc}")

    def _publish_status(self, fps, drop, quality):
        if self.status_pub is None:
            return
        import json
        msg = String()
        msg.data = json.dumps({
            "frame": self.total_frames,
            "num_frames": self.num_frames,
            "fps": round(float(fps), 2),
            "recv": self.received_frames,
            "drop": int(drop),
            "gaussians": int(self.params["means3D"].shape[0]) if self.params else 0,
            "map_iters": int(self._cur_map_iters),
            "pose_ok": bool(quality.get("ok", False)),
            "pose_discontinuity": bool(quality.get("discontinuity", False)),
            "pose_stale": bool(quality.get("stale", False)),
            "backend": self.render_backend,
        })
        self.status_pub.publish(msg)

    # ---------------------------------------------------------------- finalize
    def _finalize(self):
        if self.stop_evt.is_set():
            return
        self.stop_evt.set()
        self.get_logger().info("Reached final frame. Saving output...")
        self.params["timestep"] = self.variables["timestep"]
        self.params["intrinsics"] = self.intrinsics.detach().cpu().numpy()
        self.params["w2c"] = self.first_frame_w2c.detach().cpu().numpy()
        self.params["org_width"] = self.cfg["data"]["desired_image_width"]
        self.params["org_height"] = self.cfg["data"]["desired_image_height"]
        self.params["gt_w2c_all_frames"] = np.stack(
            [m.detach().cpu().numpy() for m in self.gt_w2c_all_frames], axis=0)
        self.params["keyframe_time_indices"] = np.array(self.keyframe_time_indices)
        save_params(self.params, str(self.output_dir))
        self.get_logger().info(f"Saved SplaTAM output to: {self.output_dir}")
        # Ask the executor to stop; main() handles rclpy shutdown/join.
        try:
            rclpy.shutdown()
        except Exception:
            pass


def main():
    args = parse_args()
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    cfg = experiment.config
    cfg.setdefault("gaussian_distribution", "isotropic")
    seed_everything(seed=cfg["seed"])

    rclpy.init()
    node = ZedGsplatOnline(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_evt.set()
        if node.mapper_thread is not None:
            node.mapper_thread.join(timeout=5.0)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
