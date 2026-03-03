# datasets/gradslam_datasets/zed2i_dataset.py

import os
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch


@dataclass
class Zed2iDatasetConfig:
    # Folder layout (inside basedir/sequence):
    #   rgb/000000.png (or .jpg)
    #   depth/000000.png (16UC1 mm) OR depth/000000.npy (float meters)
    #   intrinsics.txt  (optional) OR intrinsics.npy (optional)
    #   poses.txt       (optional) OR poses.npy (optional)
    #
    # If intrinsics/poses are missing, defaults are used.
    rgb_dir: str = "rgb"
    depth_dir: str = "depth"

    intrinsics_txt: str = "intrinsics.txt"   # 3x3 or 4x4; we take top-left 3x3
    intrinsics_npy: str = "intrinsics.npy"   # 3x3 or 4x4
    poses_txt: str = "poses.txt"             # N x 4x4 in one of the formats below
    poses_npy: str = "poses.npy"             # (N,4,4)

    # Depth encoding assumptions
    depth_png_is_mm: bool = True
    depth_scale_m: float = 1.0  # extra multiplier after conversion

    # Resize
    desired_height: Optional[int] = None
    desired_width: Optional[int] = None


def _load_intrinsics(path_txt: str, path_npy: str) -> np.ndarray:
    if os.path.isfile(path_npy):
        K = np.load(path_npy)
        K = np.asarray(K, dtype=np.float32)
    elif os.path.isfile(path_txt):
        vals = []
        with open(path_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals.extend([float(x) for x in line.replace(",", " ").split()])
        K = np.array(vals, dtype=np.float32)
        if K.size == 9:
            K = K.reshape(3, 3)
        elif K.size == 16:
            K = K.reshape(4, 4)[:3, :3]
        else:
            raise ValueError(f"intrinsics.txt must have 9 or 16 floats, got {K.size}")
    else:
        # Fallback: safe-ish defaults (you should provide real intrinsics)
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = 500.0
        K[1, 1] = 500.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
    if K.shape == (4, 4):
        K = K[:3, :3]
    if K.shape != (3, 3):
        raise ValueError(f"Intrinsics must be 3x3; got {K.shape}")
    return K


def _load_poses(path_txt: str, path_npy: str, n_frames: int) -> np.ndarray:
    if os.path.isfile(path_npy):
        P = np.load(path_npy)
        P = np.asarray(P, dtype=np.float32)
        if P.ndim != 3 or P.shape[1:] != (4, 4):
            raise ValueError(f"poses.npy must be (N,4,4); got {P.shape}")
        return P

    if os.path.isfile(path_txt):
        mats: List[np.ndarray] = []
        vals: List[float] = []
        with open(path_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(",", " ").split() if p]
                # Support:
                #  - 16 floats per line (row-major 4x4)
                #  - 12 floats per line (row-major 3x4), we append [0 0 0 1]
                nums = [float(x) for x in parts]
                if len(nums) in (12, 16):
                    if len(nums) == 12:
                        M = np.array(nums, dtype=np.float32).reshape(3, 4)
                        M = np.vstack([M, np.array([0, 0, 0, 1], dtype=np.float32)])
                    else:
                        M = np.array(nums, dtype=np.float32).reshape(4, 4)
                    mats.append(M)
                else:
                    # Allow "free-form": accumulate floats until we can form a 4x4
                    vals.extend(nums)
                    while len(vals) >= 16:
                        M = np.array(vals[:16], dtype=np.float32).reshape(4, 4)
                        mats.append(M)
                        vals = vals[16:]

        if len(mats) == 0:
            raise ValueError("poses.txt found but no poses parsed")
        P = np.stack(mats, axis=0).astype(np.float32)
        return P

    # Fallback: identity poses
    P = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], n_frames, axis=0)
    return P


def _list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files


def _match_depth(rgb_paths: List[str], depth_folder: str) -> List[str]:
    # Matches by filename stem: rgb/000123.png -> depth/000123.(png|npy)
    out: List[str] = []
    for rp in rgb_paths:
        stem = os.path.splitext(os.path.basename(rp))[0]
        cand_png = os.path.join(depth_folder, stem + ".png")
        cand_npy = os.path.join(depth_folder, stem + ".npy")
        if os.path.isfile(cand_npy):
            out.append(cand_npy)
        elif os.path.isfile(cand_png):
            out.append(cand_png)
        else:
            raise FileNotFoundError(f"Missing depth for {rp}: expected {cand_png} or {cand_npy}")
    return out


class Zed2iDataset(torch.utils.data.Dataset):
    """
    Returns: (color, depth, intrinsics4x4, pose4x4)

    color: (H,W,3) uint8 on CPU unless device specified
    depth: (H,W,1) float32 meters
    intrinsics: 4x4 float32 (top-left 3x3 is K)
    pose: 4x4 float32 (camera-to-world or world-to-camera depends on your pipeline;
          here we output pose as 4x4 "pose" like other loaders often do (c2w).
          If your downstream expects a different convention, swap/invert there.)
    """
    def __init__(
        self,
        config_dict: dict,
        basedir: str,
        sequence: str,
        start: int = 0,
        end: int = -1,
        stride: int = 1,
        desired_height: Optional[int] = None,
        desired_width: Optional[int] = None,
        device: Optional[torch.device] = None,
        relative_pose: bool = True,
        ignore_bad: bool = False,
        use_train_split: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.relative_pose = bool(relative_pose)
        self.ignore_bad = bool(ignore_bad)

        # Optional config overrides
        self.cfg = Zed2iDatasetConfig(
            desired_height=desired_height,
            desired_width=desired_width,
        )

        root = os.path.join(basedir, sequence)
        self.root = root

        rgb_folder = os.path.join(root, self.cfg.rgb_dir)
        depth_folder = os.path.join(root, self.cfg.depth_dir)

        rgb_all = _list_images(rgb_folder)
        depth_all = _match_depth(rgb_all, depth_folder)

        if end is None or end < 0:
            end = len(rgb_all)
        rgb_all = rgb_all[start:end:stride]
        depth_all = depth_all[start:end:stride]

        self.rgb_paths = rgb_all
        self.depth_paths = depth_all

        K = _load_intrinsics(
            os.path.join(root, self.cfg.intrinsics_txt),
            os.path.join(root, self.cfg.intrinsics_npy),
        )
        self.K = K

        poses = _load_poses(
            os.path.join(root, self.cfg.poses_txt),
            os.path.join(root, self.cfg.poses_npy),
            n_frames=len(self.rgb_paths),
        )
        if poses.shape[0] != len(self.rgb_paths):
            # If poses include full sequence but we sliced start/end/stride:
            # we slice poses the same way as images.
            full_rgb_count = _list_images(rgb_folder)
            if poses.shape[0] == len(full_rgb_count):
                poses = poses[start:end:stride]
            else:
                raise ValueError(f"Pose count {poses.shape[0]} != frame count {len(self.rgb_paths)}")

        if self.relative_pose and poses.shape[0] > 0:
            # Make poses relative to first pose: pose_rel = inv(pose0) @ pose_i
            pose0 = poses[0]
            inv0 = np.linalg.inv(pose0)
            poses = (inv0[None, :, :] @ poses).astype(np.float32)

        self.poses = poses.astype(np.float32)

        # Intrinsics as 4x4
        intr4 = np.eye(4, dtype=np.float32)
        intr4[:3, :3] = self.K.astype(np.float32)
        self.intr4 = intr4

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def _load_rgb(self, path: str) -> np.ndarray:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(path)
        # Convert to RGB uint8
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)

        if self.cfg.desired_width and self.cfg.desired_height:
            im = cv2.resize(im, (self.cfg.desired_width, self.cfg.desired_height), interpolation=cv2.INTER_LINEAR)

        return im

    def _load_depth(self, path: str) -> np.ndarray:
        if path.endswith(".npy"):
            d = np.load(path).astype(np.float32)
        else:
            d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if d is None:
                raise FileNotFoundError(path)
            if d.dtype == np.uint16 and self.cfg.depth_png_is_mm:
                d = d.astype(np.float32) / 1000.0
            else:
                d = d.astype(np.float32)

        if d.ndim == 3:
            d = d[..., 0]

        if self.cfg.desired_width and self.cfg.desired_height:
            d = cv2.resize(d, (self.cfg.desired_width, self.cfg.desired_height), interpolation=cv2.INTER_NEAREST)

        d = d * float(self.cfg.depth_scale_m)
        d = np.expand_dims(d, axis=-1).astype(np.float32)
        return d

    def _scaled_intrinsics4(self, orig_w: int, orig_h: int) -> np.ndarray:
        if not (self.cfg.desired_width and self.cfg.desired_height):
            return self.intr4.copy()

        sx = float(self.cfg.desired_width) / float(orig_w)
        sy = float(self.cfg.desired_height) / float(orig_h)

        intr4 = self.intr4.copy()
        intr4[0, 0] *= sx
        intr4[1, 1] *= sy
        intr4[0, 2] *= sx
        intr4[1, 2] *= sy
        return intr4

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]

        # Read original for intrinsics scaling
        orig = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        if orig is None:
            raise FileNotFoundError(rgb_path)
        orig_h, orig_w = orig.shape[:2]

        color = self._load_rgb(rgb_path)          # (H,W,3) uint8 RGB
        depth = self._load_depth(depth_path)      # (H,W,1) float meters
        intr4 = self._scaled_intrinsics4(orig_w, orig_h)  # (4,4)
        pose = self.poses[idx]                    # (4,4)

        # To torch
        color_t = torch.from_numpy(color)         # uint8
        depth_t = torch.from_numpy(depth)         # float32
        intr_t = torch.from_numpy(intr4)          # float32
        pose_t = torch.from_numpy(pose)           # float32

        # Move to device if requested
        if self.device is not None:
            color_t = color_t.to(self.device)
            depth_t = depth_t.to(self.device)
            intr_t = intr_t.to(self.device)
            pose_t = pose_t.to(self.device)

        return color_t, depth_t, intr_t, pose_t