primary_device = "cuda:0"
seed = 0

base_dir = "./experiments/ZED2i_Captures"
scene_name = "zed2i_ros2_demo"
run_name = "SplaTAM_ZED2i_ROS2"

num_frames = 300


# ALWAYS-ON LiveStream Recording 
live_stream_dir = "experiments/ZED2i_Live/zedTest/LiveStream"
mp4_dir = "experiments/mp4"
record_cam = True
record_depth = True
record_splat = True
record_fps = 30.0


# ROS Topics

zed_rgb_topic = "/zed/zed_node/rgb/color/rect/image"
#zed_rgb_topic = "/zed/zed_node/rgb/color/rect/image/compressed"

zed_rgb_info_topic = "/zed/zed_node/rgb/color/rect/image/camera_info"
zed_rgb_info_topic = "/zed/zed_node/rgb/color/rect/camera_info"

zed_depth_topic = "/zed/zed_node/depth/depth_registered"
#zed_depth_topic = "/zed/zed_node/depth/depth_registered/compressed"

zed_depth_info_topic = "/zed/zed_node/depth/depth_registered/camera_info"

use_odom = True
zed_odom_topic = "/zed/zed_node/odom"

# Encodings (MATCH YOUR ros2 echo)
zed_rgb_encoding = "bgra8"
zed_depth_encoding = "32FC1"


config = dict(
    
    # Run / Output
    workdir=f"{base_dir}/{scene_name}",
    run_name=run_name,
    overwrite=True,
    num_frames=num_frames,
    save_stream_frames=True,

    
    # Always-On LiveStream Recording
    live_stream_dir=live_stream_dir,  
    mp4_dir=mp4_dir,      
    record_cam=record_cam,
    record_depth=record_depth,
    record_splat=record_splat,
    record_fps=record_fps,

    
    # Core SLAM Settings
    map_every=1,
    keyframe_every=10,
    mapping_window_size=32,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    primary_device=primary_device,
    seed=seed,

    
    # Image Data Settings
    data=dict(
        desired_image_width=640,
        desired_image_height=360,
        densification_image_width=320,
        densification_image_height=180,
        downscale_factor=1.0,
        densify_downscale_factor=2.0,
    ),

    
    # Tracking Settings
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        visualize_tracking_loss=False,
        num_iters=5,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        use_depth_loss_thres=False,
        depth_loss_thres=20000,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.001,
            cam_trans=0.004,
        ),
    ),

    
    # Mapping Settings
    mapping=dict(
        num_iters=5,
        add_new_gaussians=False,
        sil_thres=0.5,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_sil_for_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
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

    
    # ROS Interface
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
    ),

    
    # Visualization
    viz=dict(
        viz_w=640,
        viz_h=360,
        viz_near=0.01,
        viz_far=50.0,
        viz_fps=5, 
        view_scale=1.0,
        render_mode="rgb",  # rgb | depth | centers
        show_sil=False,
        visualize_cams=False,
        offset_first_viz_cam=False,

    ),
)