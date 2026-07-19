"""Default fastsplat config.

Only override what differs from ``fastsplat.utils.config.DEFAULTS`` -- missing
keys are filled in automatically. Run with::

    python -m fastsplat.run_pipeline --config configs/fastsplat/fastsplat.py
"""

config = dict(
    device="cuda:0",
    workdir="experiments/FastSplat",
    run_name="object_splat",
    overwrite=True,

    ingest=dict(
        # "folder": use an existing directory of images (fastest to iterate).
        # "ros2":   capture sparse keyframes live from the ZED via Zenoh.
        source="folder",
        image_dir="data/object/images",
        image_glob="*.png",
        max_frames=40,
        save_depth=True,
        save_poses=True,
    ),

    sfm=dict(
        model="facebook/map-anything-apache",
        # Default trusts ZED intrinsics but re-estimates poses (fixes the
        # ghost-splat-at-different-heights artifact caused by odom drift).
        input_mode="rgb_intrinsics",
        apply_confidence_mask=True,
        confidence_percentile=10,
    ),

    object=dict(
        method="auto_crop",        # switch to "sam2" for a clean cutout
    ),

    splat=dict(
        max_steps=3000,
        time_budget_s=240,
        sh_degree=0,
        background="white",
        densify=True,
    ),

    export=dict(
        ply=True,
        ply_name="object_splat.ply",
    ),
)
