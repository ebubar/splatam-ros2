# configs/splatam_live.py

primary_device = "cuda:0"
seed = 0

num_frames=100


zed_rgb_topic = "/zed/zed_node/rgb/color/rect/image"          
zed_depth_topic = "/zed/zed_node/depth/depth_registered" 
zed_cam_info_topic = "/zed/zed_node/rgb/color/rect/camera_info"
# zed_rgb_topic      = "/zed/zed_node/rgb/color/rect/image/compressed"
# zed_depth_topic    = "/zed/zed_node/depth/depth_registered/compressedDepth"
# zed_cam_info_topic = "/zed/zed_node/rgb/color/rect/camera_info"

slop_sec = 0.05
queue_size = 30
wait_timeout_sec = 20

config = dict(
    workdir="./experiments/ZED2i_Live",
    run_name="SplaTAM_ZED2i_Live",
    overwrite=True,
    seed=seed,
    primary_device=primary_device,
    report_iter_progress = False,

    num_frames=num_frames,

    map_every=1,
    keyframe_every=3,
    mapping_window_size=32,

    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    scene_radius_depth_ratio=3,
    report_global_progress_every=50,
    eval_every=999999,   
    use_wandb=False,
    load_checkpoint = False,
    checkpoint_time_idx = 0,
    save_checkpoints = True,
    checkpoint_interval = 5,

    data=dict(
        dataset_name="zed2i_ros2",
        basedir=".",
        sequence="zed2i",

        desired_image_height=360,
        desired_image_width=640,

        start=0,
        end=-1,
        stride=1,
        num_frames=num_frames,

        rgb_topic=zed_rgb_topic,
        depth_topic=zed_depth_topic,
        cam_info_topic=zed_cam_info_topic,

        # Sync tuning (we’ll tighten after you paste stamps)
        slop_sec=slop_sec,
        queue_size=queue_size,
        wait_timeout_sec=wait_timeout_sec,
    ),

    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        visualize_tracking_loss=False,
        num_iters=20,
        use_sil_for_loss=False,
        sil_thres=0.9,
        use_l1=True,
        use_depth_loss_thres=False,
        depth_loss_thres=20000,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0, rgb_colors=0.0, unnorm_rotations=0.0, logit_opacities=0.0, log_scales=0.0,
            cam_unnorm_rots=0.001, cam_trans=0.002,
        ),
    ),

    mapping=dict(
        num_iters=30,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        ignore_outlier_depth_loss=True,
        use_sil_for_loss=True,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0001, rgb_colors=0.0025, unnorm_rotations=0.001, logit_opacities=0.05, log_scales=0.001,
            cam_unnorm_rots=0.0, cam_trans=0.0,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0, remove_big_after=0, stop_after=20, prune_every=20,
            removal_opacity_threshold=0.005, final_removal_opacity_threshold=0.005,
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
)
