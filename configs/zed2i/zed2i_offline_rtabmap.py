"""
Offline high-quality SplaTAM run on a dataset produced by
scripts/rtabmap2dataset.py (ZED2i bag + RTAB-Map loop-closed poses).

Poses are trusted (use_gt_poses=True), so the per-frame tracking optimization
is skipped and the budget goes to mapping: every frame is mapped, with
pruning enabled. To let SplaTAM photometrically refine the RTAB-Map poses
instead, set tracking use_gt_poses=False.

Usage:
    python3 scripts/splatam.py configs/zed2i/zed2i_offline_rtabmap.py

Scene selection: dataset is read from <base_dir>/<scene_name>; override the
scene without editing this file via the SPLATAM_SCENE environment variable.
"""

import os

primary_device = "cuda:0"
seed = 0

base_dir = "./experiments/ZED2i_Captures"
scene_name = os.environ.get("SPLATAM_SCENE", "offline_rtabmap")

# Dataset is stored at the camera's native resolution; SplaTAM works at these
# dims (overridable via env for other cameras/aspect ratios).
desired_width = int(os.environ.get("SPLATAM_IMAGE_WIDTH", 640))
desired_height = int(os.environ.get("SPLATAM_IMAGE_HEIGHT", 360))

map_every = 1
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 40   # only used if use_gt_poses=False
mapping_iters = int(os.environ.get("SPLATAM_MAPPING_ITERS", 60))

config = dict(
    workdir=f"{base_dir}/{scene_name}",
    # Override per-experiment so A/B runs don't overwrite each other
    run_name=os.environ.get("SPLATAM_RUN_NAME", "SplaTAM_ZED2i_Offline"),
    overwrite=True,
    depth_scale=10.0,  # matches integer_depth_scale written by rtabmap2dataset.py
    num_frames=-1,     # use all frames in the dataset
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=100,
    eval_every=1,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution=os.environ.get("SPLATAM_GAUSSIAN_DIST", "isotropic"),  # "anisotropic" for sharper edges
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,
    use_wandb=False,
    data=dict(
        dataset_name="nerfcapture",
        basedir=base_dir,
        sequence=scene_name,
        desired_image_height=desired_height,
        desired_image_width=desired_width,
        densification_image_height=desired_height,
        densification_image_width=desired_width,
        start=int(os.environ.get("SPLATAM_FRAME_START", 0)),
        end=int(os.environ.get("SPLATAM_FRAME_END", -1)),
        stride=1,
        num_frames=-1,
    ),
    tracking=dict(
        use_gt_poses=True,  # trust RTAB-Map poses; False -> photometric refinement
        forward_prop=True,
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
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0005,
            cam_trans=0.001,
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
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),
        # Prune low-opacity floaters; schedule is in mapping iterations
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=max(20, mapping_iters - 20),
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
