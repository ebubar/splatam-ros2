import os
from os.path import join as p_join

primary_device = "cuda:0"
seed = 0

base_dir = "./experiments/ZED2i_Captures"
scene_name = "zed2i_gsplat_demo"
run_name = "SplaTAM_ZED2i_gsplat"

num_frames = 45
overwrite = True

# ZED working resolution
full_res_width = 640
full_res_height = 360
downscale_factor = 1.0
densify_downscale_factor = 1.0

# ---- Fast-path engine + robustness switches -------------------------------
render_backend = "gsplat"      # "gsplat" (Apache-2.0, fast) | "cuda" (fallback)
pose_source = "zed_odom"       # "zed_odom" (trusted ZED SDK) | "mapanything"
async_mapping = True           # decouple mapping from ingestion (set False for A/B)
frame_buffer_size = 2          # bounded drop-oldest buffer

# ROS topics (raw)
zed_rgb_topic = "/zed/zed_node/rgb/color/rect/image"
zed_rgb_info_topic = "/zed/zed_node/rgb/color/rect/image/camera_info"
zed_depth_topic = "/zed/zed_node/depth/depth_registered"
zed_depth_info_topic = "/zed/zed_node/depth/depth_registered/camera_info"
zed_odom_topic = "/zed/zed_node/odom"

# ROS topics (compressed transport — big bandwidth reduction over a limited link)
zed_rgb_topic_compressed = "/zed/zed_node/rgb/color/rect/image/compressed"
zed_depth_topic_compressed = "/zed/zed_node/depth/depth_registered/compressedDepth"

map_every = 5
keyframe_every = 1 if num_frames >= 25 else max(1, int(num_frames // 5))
mapping_window_size = 24

# gsplat per-iter is cheaper than the CUDA rasterizer, so the mapping budget is
# reduced to fit the realtime map interval; tune up if quality is short.
mapping_iters = 50
tracking_iters = 20   # only used by the gated pose-refinement fallback

config = dict(
    workdir=f"{base_dir}/{scene_name}",
    run_name=run_name,
    overwrite=overwrite,

    num_frames=num_frames,
    seed=seed,
    primary_device=primary_device,

    render_backend=render_backend,
    pose_source=pose_source,
    async_mapping=async_mapping,
    frame_buffer_size=frame_buffer_size,

    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=100,
    eval_every=1,

    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",

    report_iter_progress=False,
    load_checkpoint=False,
    save_checkpoints=False,
    use_wandb=False,

    data=dict(
        desired_image_height=int(full_res_height // downscale_factor),
        desired_image_width=int(full_res_width // downscale_factor),
        densification_image_height=int(full_res_height // densify_downscale_factor),
        densification_image_width=int(full_res_width // densify_downscale_factor),
        downscale_factor=downscale_factor,
        densify_downscale_factor=densify_downscale_factor,
    ),

    # gsplat rasterization options.
    gsplat=dict(
        render_mode="RGB+ED",   # expected/metric depth for the depth-L1 loss; "RGB+D" for CUDA-parity
        packed=False,           # unpacked keeps per-Gaussian radii shape [1,N]
        near_plane=0.01,
        far_plane=100.0,
        bg=None,   # None -> gsplat default (black). Set [r,g,b] only if verified for render_mode.
    ),

    # Trust ZED poses by default; refine only when the pose looks unreliable.
    tracking=dict(
        mode="auto",            # "off" | "auto" (gated fallback) | "always"
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.02, depth=5.0),
        lrs=dict(
            means3D=0.0, rgb_colors=0.0, unnorm_rotations=0.0,
            logit_opacities=0.0, log_scales=0.0,
            cam_unnorm_rots=0.0004, cam_trans=0.002,
        ),
    ),

    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_sil_for_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0001, rgb_colors=0.0025, unnorm_rotations=0.001,
            logit_opacities=0.05, log_scales=0.001,
            cam_unnorm_rots=0.0, cam_trans=0.0,
        ),
        # gsplat densify/prune disabled in the live path (silhouette-based
        # add_new_gaussians handles growth); the CUDA fallback can re-enable.
        prune_gaussians=False,
        pruning_dict=dict(
            start_after=20, remove_big_after=40, stop_after=300, prune_every=15,
            removal_opacity_threshold=0.002, final_removal_opacity_threshold=0.002,
            reset_opacities=False, reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500, remove_big_after=3000, stop_after=5000, densify_every=100,
            grad_thresh=0.0002, num_to_split_into=2,
            removal_opacity_threshold=0.005, final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),

    ros=dict(
        transport="raw",        # "raw" | "compressed"
        rgb_topic=zed_rgb_topic,
        rgb_info_topic=zed_rgb_info_topic,
        depth_topic=zed_depth_topic,
        depth_info_topic=zed_depth_info_topic,
        rgb_topic_compressed=zed_rgb_topic_compressed,
        depth_topic_compressed=zed_depth_topic_compressed,
        odom_topic=zed_odom_topic,

        depth_unit_scale_m=1.0,
        min_depth_m=0.3,
        max_depth_m=6.0,
        process_every_n=1,

        # 2-way RGB+depth sync (odom is buffered separately).
        sync_queue=30,
        sync_slop=0.05,

        # Odometry interpolation buffer (utils/pose_buffer.py).
        odom_buffer_len=400,
        max_extrapolation_s=0.05,
        discontinuity_vel_mps=3.0,
        discontinuity_omega_rps=6.0,
    ),

    # System hardening for long / unattended edge runs (WS7).
    hardening=dict(
        # Bounded memory: cap the Gaussian count and drop very transparent splats.
        max_gaussians=1_200_000,        # 0 = unbounded; hard ceiling to avoid OOM
        prune_opacity_below=0.005,      # 0 = disabled; remove near-invisible gaussians
        # Adaptive mapping budget: back off iterations to hold a target rate.
        adaptive_fps=True,
        target_fps=0.0,                 # 0 = disabled (no throttling); e.g. 5.0 on Thor
        min_map_iters=15,               # floor when throttling
        # Periodic checkpoint so a long run is recoverable.
        checkpoint_every=0,             # 0 = disabled; e.g. 100 frames
        # Status topic for field monitoring.
        status_topic="/splatam/status",
        publish_status=True,
    ),

    # Capture-guidance module (utils/capture_guidance.py). Off by default.
    capture_guidance=dict(
        enabled=False,
        rings_elevation_deg=[-20.0, 10.0, 40.0],
        slots_per_ring=15,
        radius_scale=3.0,
        min_baseline_frac=0.1,
        min_view_angle_deg=12.0,
        min_overlap=0.5,
        max_overlap=0.9,
        blur_var_thresh=60.0,
        weights=dict(alpha=1.0, depth_hole=0.5, under_observed=0.5),
    ),

    viz=dict(
        render_mode="color",
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=600, viz_h=340, viz_near=0.01, viz_far=100.0,
        view_scale=2, viz_fps=5,
        enter_interactive_post_online=False,
    ),
)
