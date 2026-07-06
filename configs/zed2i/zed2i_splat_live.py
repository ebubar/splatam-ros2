import os
from os.path import join as p_join

primary_device = "cuda:0"
seed = 0

base_dir = "./experiments/ZED2i_Captures"
scene_name = "zed2i_ros2_demo"
run_name = "SplaTAM_ZED2i_ROS2"

num_frames = 45
overwrite = True

# ZED working resolution
full_res_width = 640
full_res_height = 360
downscale_factor = 1.0
densify_downscale_factor = 1.0

# Recording
live_stream_dir = "experiments/ZED2i_Live/zeros2 bag info zed2i_splatam_tf_bag/dTest/LiveStream"
mp4_dir = "experiments/mp4"
record_cam = True
record_depth = True
record_splat = True
record_fps = 30.0

# ROS Topics
zed_rgb_topic = "/zed/zed_node/rgb/color/rect/image"
zed_rgb_info_topic = "/zed/zed_node/rgb/color/rect/image/camera_info"

zed_depth_topic = "/zed/zed_node/depth/depth_registered"
zed_depth_info_topic = "/zed/zed_node/depth/depth_registered/camera_info"

use_odom = True
zed_odom_topic = "/zed/zed_node/odom"

# ZED NEURAL depth confidence map (optional). Enable
# `depth.depth_confidence` publishing in the ZED wrapper to use this; it masks
# out low-confidence stereo pixels before they seed Gaussians. Set to None to
# disable confidence masking.
zed_confidence_topic = "/zed/zed_node/confidence/confidence_map"

# Encodings
zed_rgb_encoding = "bgr8"
zed_depth_encoding = "32FC1"

# Pose fusion:
#   "vio_seed_refine" -> ZED VIO odom seeds each pose, SplaTAM tracking refines
#                        it against the map (most accurate; recommended).
#   "vio_only"        -> trust ZED VIO directly (fastest, no refinement).
#   "splatam_only"    -> ignore VIO, constant-velocity + tracking (baseline).
pose_mode = "vio_seed_refine"

map_every = 10

if num_frames < 25:
    keyframe_every = max(1, int(num_frames // 5))
else:
    keyframe_every = 1

mapping_window_size = 32
tracking_iters = 30
mapping_iters = 180

config = dict(
    workdir=f"{base_dir}/{scene_name}",
    run_name=run_name,
    overwrite=overwrite,

    num_frames=num_frames,
    seed=seed,
    primary_device=primary_device,
    pose_mode=pose_mode,

    # Periodic housekeeping (0 disables). Per-frame torch.cuda.empty_cache()
    # tanks throughput, so we only do it occasionally.
    housekeeping=dict(empty_cache_every=50),

    # Debug depth PNGs are expensive disk I/O; disabled by default (0). Set to
    # e.g. 20 to dump one colourised depth image every 20 processed frames.
    debug=dict(save_depth_every=0),

    # Operator browser viewer + region-of-interest refinement.
    viewer=dict(
        enable=True,
        host="0.0.0.0",
        port=8080,
        publish_every=3,            # push a splat snapshot every N frames
        max_preview_points=150000,  # cap points streamed to the laptop
    ),

    # Focused high-quality refinement when the operator selects a region.
    roi_refine=dict(
        base_iters=300,             # * quality slider = total refine iterations
    ),

    save_stream_frames=True,

    live_stream_dir=live_stream_dir,
    mp4_dir=mp4_dir,
    record_cam=record_cam,
    record_depth=record_depth,
    record_splat=record_splat,
    record_fps=record_fps,

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
    checkpoint_time_idx=130,
    save_checkpoints=False,
    checkpoint_interval=5,
    use_wandb=False,

    data=dict(
        desired_image_height=int(full_res_height // downscale_factor),
        desired_image_width=int(full_res_width // downscale_factor),
        densification_image_height=int(full_res_height // densify_downscale_factor),
        densification_image_width=int(full_res_width // densify_downscale_factor),
        downscale_factor=downscale_factor,
        densify_downscale_factor=densify_downscale_factor,
    ),

    # Hybrid mode:
    # ZED odom initializes each pose in the script.
    # SplaTAM then refines that pose locally.
    tracking=dict(
        use_gt_poses=False,
        forward_prop=False,
        visualize_tracking_loss=False,
        num_iters=tracking_iters,

        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        use_depth_loss_thres=True,
        depth_loss_thres=20000,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,

        loss_weights=dict(
            im=0.02,
            depth=5.0,
        ),

        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.00005,
            cam_trans=0.0001,
        ),
    ),

    mapping=dict(
        num_iters=180,
        add_new_gaussians=True,
        sil_thres=0.45,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_sil_for_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,

        loss_weights=dict(
            im=0.6,
            depth=1.0,
        ),

        lrs=dict(
            means3D=0.00007,
            rgb_colors=0.003,
            unnorm_rotations=0.0005,
            logit_opacities=0.015,
            log_scales=0.00005,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),

        prune_gaussians=False,

        pruning_dict=dict(
            start_after=20,
            remove_big_after=40,
            stop_after=300,
            prune_every=15,
            removal_opacity_threshold=0.002,
            final_removal_opacity_threshold=0.002,
            reset_opacities=False,
            reset_opacities_every=500,
        ),

        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),


    ros=dict(
        rgb_topic=zed_rgb_topic,
        rgb_info_topic=zed_rgb_info_topic,
        depth_topic=zed_depth_topic,
        depth_info_topic=zed_depth_info_topic,

        use_odom=use_odom,
        odom_topic=zed_odom_topic,

        confidence_topic=zed_confidence_topic,
        confidence_max_cost=50.0,   # ZED confidence cost in [0,100]; lower=better

        rgb_encoding=zed_rgb_encoding,
        depth_encoding=zed_depth_encoding,

        rgb_is_bgr=True,
        depth_unit_scale_m=1.0,

        min_depth_m=0.3,
        max_depth_m=6.0,
        process_every_n=1,
        sync_slop_s=0.15,
    ),

    viz=dict(
        render_mode="color",
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=600,
        viz_h=340,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=False,
    ),
)