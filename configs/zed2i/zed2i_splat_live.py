import os
from os.path import join as p_join

primary_device = "cuda:0"
seed = 0

base_dir = "./experiments/ZED2i_Captures"
scene_name = "zed2i_ros2_demo"
run_name = "SplaTAM_ZED2i_ROS2"

num_frames = 50
overwrite = True

# ZED working resolution
full_res_width = 640
full_res_height = 360
downscale_factor = 1.0
densify_downscale_factor = 1.0

# Recording
live_stream_dir = "experiments/ZED2i_Live/zedTest/LiveStream"
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

# Encodings
zed_rgb_encoding = "bgr8"
zed_depth_encoding = "32FC1"

map_every = 1

if num_frames < 25:
    keyframe_every = max(1, int(num_frames // 5))
else:
    keyframe_every = 2

mapping_window_size = 32
tracking_iters = 30
mapping_iters = 60

config = dict(
    workdir=f"{base_dir}/{scene_name}",
    run_name=run_name,
    overwrite=overwrite,

    num_frames=num_frames,
    seed=seed,
    primary_device=primary_device,

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
            im=0.1,
            depth=3.0,
        ),

        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.00015,
            cam_trans=0.0005,
        ),
    ),

    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_sil_for_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,

        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),

        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.02,
            log_scales=0.0003,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),

        prune_gaussians=True,
        pruning_dict=dict(
            start_after=30,
            remove_big_after=60,
            stop_after=200,
            prune_every=30,
            removal_opacity_threshold=0.001,
            final_removal_opacity_threshold=0.001,
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

        rgb_encoding=zed_rgb_encoding,
        depth_encoding=zed_depth_encoding,

        rgb_is_bgr=True,
        depth_unit_scale_m=1.0,

        min_depth_m=0.5,
        max_depth_m=5.0,
        process_every_n=1,
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