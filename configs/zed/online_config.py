import os
from os.path import join as p_join

# --- SETTINGS ---
config = dict(
    seed=0,
    scene_name="ZED_Live_3D",
    workdir="./experiments/ZED_Captures",
    run_name="Live_Pro_Test",
    primary_device="cuda:0",
    
    # --- SPEED & KEYFRAME TUNING ---
    use_wandb=False,
    map_every=5,            
    keyframe_every=10,      
    mapping_window_size=8,
    report_global_progress_every=10,
    checkpoint_interval=100,
    save_checkpoints=False,
    load_checkpoint=False,
    
    # --- DATA CONFIG ---
    overwrite=True,
    num_frames=1000,
    depth_scale=1.0, 
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    scene_radius_depth_ratio=3,
    
    data=dict(
        desired_image_height=360, 
        desired_image_width=640,
        downscale_factor=2.0,
        densification_image_height=180, 
        densification_image_width=320,
        densify_downscale_factor=4.0, 
    ),
    
    # --- TRACKING ---
    tracking=dict(
        use_gt_poses=True,
        forward_prop=True,
        num_iters=0, 
        use_l1=True,
        use_sil_for_loss=True,
        sil_thres=0.99,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            log_scales=0.0,
            logit_opacities=0.0,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),
    ),
    
    # --- MAPPING ---
    mapping=dict(
        num_iters=15, 
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        use_gaussian_splatting_densification=True,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.001,
            rgb_colors=0.002,
            unnorm_rotations=0.001,
            log_scales=0.001,
            logit_opacities=0.05,
        ),
    ),
    # Visualizer settings (optional but good for debugging)
    viz=dict(
        render_true_depth=True,
        unnorm_rotations=True,
        upscale_prediction=True,
    ),
)

