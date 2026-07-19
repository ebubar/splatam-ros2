"""Config loading with deep-merged defaults.

Configs are plain Python modules that expose a ``config = dict(...)`` at module
level (same convention as the SplaTAM configs in ``configs/``). The loader fills
in any missing keys from :data:`DEFAULTS` so config files can stay small.
"""

import copy
import os
from importlib.machinery import SourceFileLoader


# ---------------------------------------------------------------------------
# Default configuration. Every stage reads from a single nested dict.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    seed=0,
    device="cuda:0",
    workdir="experiments/FastSplat",
    run_name="object_splat",
    overwrite=True,

    # --- Stage 1: ingest sparse keyframes ---------------------------------
    ingest=dict(
        source="folder",              # "folder" | "ros2"
        image_dir="data/object/images",
        image_glob="*.png",
        # Live ROS2 capture (ZED via Zenoh transfer).
        ros=dict(
            rgb_topic="/zed/zed_node/rgb/color/rect/image",
            rgb_info_topic="/zed/zed_node/rgb/color/rect/camera_info",
            depth_topic="/zed/zed_node/depth/depth_registered",
            depth_info_topic="/zed/zed_node/depth/depth_registered/camera_info",
            odom_topic="/zed/zed_node/odom",
            rgb_encoding="bgr8",
            depth_encoding="32FC1",
            depth_unit_scale_m=1.0,
            min_depth_m=0.3,
            max_depth_m=6.0,
            qos_reliability="best_effort",  # "best_effort" | "reliable"
        ),
        max_frames=40,                # cap on sparse keyframes
        # Keyframe selection by camera motion (uses odom when available).
        min_translation_m=0.03,
        min_rotation_deg=3.0,
        process_every_n=1,            # decimate incoming stream first
        save_depth=True,
        save_intrinsics=True,
        save_poses=True,
        resize_width=None,            # optional downscale of saved images
        resize_height=None,
    ),

    # --- Stage 2: MapAnything feed-forward SfM ----------------------------
    sfm=dict(
        model="facebook/map-anything-apache",   # Apache 2.0 checkpoint
        # How much of the ZED signal to trust as a prior.
        #   "rgb"                  : images only (MapAnything estimates K + poses)
        #   "rgb_intrinsics"       : + trusted ZED intrinsics  (DEFAULT)
        #   "rgb_intrinsics_depth" : + ZED depth for metric scale
        #   "full_priors"          : + ZED odom poses (re-introduces drift risk)
        # NOTE: raw ZED odometry is the usual cause of "ghost splats at different
        # heights". The default deliberately lets MapAnything re-estimate poses
        # jointly across views, which removes the multi-view inconsistency.
        input_mode="rgb_intrinsics",
        memory_efficient_inference=True,
        minibatch_size=None,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=True,
        confidence_percentile=10,     # drop lowest-confidence points
        use_multiview_confidence=True,
        is_metric_scale=True,
        max_points=500000,            # subsample fused cloud for speed
    ),

    # --- Stage 3: object isolation ----------------------------------------
    object=dict(
        method="auto_crop",           # "auto_crop" | "sam2" | "none"
        auto_crop=dict(
            depth_percentile=(1.0, 60.0),   # keep near-field band of the cloud
            conf_percentile=30.0,           # extra confidence gate
            outlier_nb=20,                  # statistical outlier removal
            outlier_std=2.0,
            dbscan_eps=0.05,                # metres (metric scale)
            dbscan_min_points=30,
            keep_largest_cluster=True,
            box_padding=0.10,               # fraction of box extent
        ),
        sam2=dict(
            checkpoint="checkpoints/sam2.1_hiera_small.pt",
            model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
            prompt="center",                # "center" click point
            dilate_px=5,
            min_mask_area_frac=0.01,
        ),
        # When True, background pixels are masked out of the gsplat photometric
        # loss so Gaussians concentrate on the object (used by sam2 mode; also
        # available for auto_crop via reprojected silhouette).
        mask_supervision=True,
    ),

    # --- Stage 4: gsplat time-boxed optimization --------------------------
    splat=dict(
        max_steps=3000,
        time_budget_s=240,            # hard wall-clock cap (keeps total < 5 min)
        sh_degree=0,                  # 0 = direct RGB (fastest); >0 enables SH
        ssim_lambda=0.2,
        init_opacity=0.5,
        init_scale=1.0,               # multiplier on kNN-derived scale
        densify=True,
        strategy=dict(
            refine_start_iter=200,
            refine_stop_iter=2000,
            refine_every=100,
            reset_every=1500,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            prune_opa=0.005,
        ),
        lr=dict(
            means=1.6e-4,
            scales=5e-3,
            quats=1e-3,
            opacities=5e-2,
            sh0=2.5e-3,
            shN=2.5e-3 / 20.0,
        ),
        packed=True,
        absgrad=False,
        background="white",           # "white" | "black" | "random"
        near_plane=0.01,
        far_plane=100.0,
        batch_size=1,
        log_every=100,
        eval_holdout=0,               # hold out N views for a quick PSNR check
    ),

    # --- Stage 5: export ---------------------------------------------------
    export=dict(
        ply=True,
        ply_name="object_splat.ply",
    ),
)


def _deep_merge(base, override):
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path):
    """Load a config module and deep-merge it over :data:`DEFAULTS`."""
    module = SourceFileLoader(os.path.basename(path), path).load_module()
    user_cfg = getattr(module, "config", None)
    if user_cfg is None:
        raise ValueError(f"Config file {path!r} must define a top-level `config` dict")
    return _deep_merge(DEFAULTS, user_cfg)


def default_config():
    """Return a fresh copy of the default config (no file needed)."""
    return copy.deepcopy(DEFAULTS)
