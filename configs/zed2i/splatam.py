"""SplaTAM configuration for ZED2i camera over ROS2.

Conservative settings optimized for:
- Real-time processing on limited hardware (Jetson Orin NX for subscriber, GPU container for SplaTAM)
- Handling frame rate drops and network latency
- Dual-rate architecture (fast pose, slower video)

Assumptions:
- Input from ROS2 topics: /splatam/input/image_rgb, /splatam/input/image_depth, /splatam/input/pose
- VGA (640x480) resolution
- 10 FPS effective frame rate (limited by ZED USB + network)
"""

import os
from os.path import join as p_join

primary_device = "cuda:0"
seed = 0

# ============================================================================
# Frame Input & Preprocessing
# ============================================================================
image_width = 640
image_height = 480
downsample_factor = 2  # For tracking: 320x240 (faster, lower memory)

# Depth processing
depth_scale = 1000.0  # millimeters to meters
depth_min = 0.1       # minimum depth threshold (m)
depth_max = 10.0      # maximum depth threshold (m)

# ============================================================================
# Tracking & Mapping Iteration Counts
# ============================================================================
tracking_iters = 40  # Reduced from 200 (TUM); aggressive for real-time
mapping_iters = 15  # Reduced from 30-60; only on keyframes
map_every = 10  # Process keyframe every 10 input frames (reduces overhead)
keyframe_every = 10  # Keep keyframe every 10 frames
mapping_window_size = 12  # Conservative window for limited hardware

# ============================================================================
# SplaTAM Config Dict
# ============================================================================
group_name = "ZED2i"
run_name = f"zed2i_test_{seed}"

config = dict(
    workdir=f"./outputs/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=500,
    eval_every=5,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=True,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,
    use_wandb=False,  # Disable wandb for simplicity
    
    # ========================================================================
    # Data source: ROS2 subscriber instead of dataset
    # ========================================================================
    data=dict(
        basedir="./data",
        gradslam_data_cfg="./configs/data/zed2i.yaml",
        sequence="zed2i_ros2",
        input_folder="",  # Not used; data comes from ROS2 topics
        output_type="ros2",  # Signal to use ROS2 input
        desired_image_height=image_height,
        desired_image_width=image_width,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),
    
    # ========================================================================
    # Tracking: Per-Frame Fast Loop
    # ========================================================================
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
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
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    
    # ========================================================================
    # Mapping: Keyframe Batch Processing
    # ========================================================================
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
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
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=mapping_iters,
            prune_every=mapping_iters,
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
    
    # ========================================================================
    # Visualization: Conservative settings for real-time
    # ========================================================================
    viz=dict(
        render_mode='color',
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=640,
        viz_h=480,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=2,
        viz_fps=1,  # Slow visualization (1 FPS) to avoid blocking
        enter_interactive_post_online=False,  # Skip interactive mode
    ),
)
lambda_l1 = 0.1

# ============================================================================
# Regularization
# ============================================================================
use_l2_regularization = True
l2_weight = 0.0001

# ============================================================================
# Pruning & Compaction (cleanup during runtime)
# ============================================================================
prune_opacity_threshold = 0.005
reset_opacity_interval = 500

# ============================================================================
# Output & Logging
# ============================================================================
output_dir = "./outputs/zed2i"
save_ply_every = 100
log_every_n_iters = 10
verbose = True

# ============================================================================
# Debugging & Profiling
# ============================================================================
profile_timing = False
save_intermediate_frames = False  # Disable to save disk I/O on limited systems
