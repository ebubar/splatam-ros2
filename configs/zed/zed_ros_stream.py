# configs/zed/zed_ros_stream.py

primary_device = "cuda:0"
seed = 0

base_dir = "./zedTest/zed_Captures"
scene_name = "zed_stream"
num_frames = 300  # total frames to process

overwrite = True

config = dict(
    workdir=f"{base_dir}/{scene_name}",
    run_name="SplaTAM_zed_ros",
    overwrite=overwrite,
    seed=seed,
    primary_device=primary_device,

    # How often mapping runs
    map_every=10,
    keyframe_every=max(1, num_frames // 5),
    mapping_window_size=32,
    report_global_progress_every=100,
    eval_every=1,

    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",

    report_iter_progress=False,
    load_checkpoint=False,
    save_checkpoints=False,
    checkpoint_interval=5,
    use_wandb=False,

    data=dict(
        dataset_name="zed_ros",
        basedir=".",              # required by framework
        sequence="zed_stream",    # required by framework
        start=0,
        end=-1,
        stride=1,
        num_frames=num_frames,
        desired_image_width=640,
        desired_image_height=360,
        densification_image_width=320,
        densification_image_height=180,
    ),

    tracking=dict(
        use_gt_poses=False,
        forward_prop=False,
        visualize_tracking_loss=False,
        num_iters=5,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        use_depth_loss_thres=False,
        depth_loss_thres=20000,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
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

    mapping=dict(
        num_iters=60,
        add_new_gaussians=True,
        sil_thres=0.7,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_sil_for_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
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

    live_view=dict(
        enabled=True,
        during_tracking=False,
        tracking_render_every=10,
        show_depth=False,
    ),
)
