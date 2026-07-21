"""Speed-tuned fastsplat config for benchmark datasets (no ZED priors).

Used by ``python -m fastsplat.benchmarks``. Trimmed frame count, point budget
and step count so a splat comes out in ~2-3 min on a modern dGPU. Only overrides
what differs from ``fastsplat.utils.config.DEFAULTS``.
"""

config = dict(
    device="cuda:0",
    workdir="experiments/FastSplat",
    run_name="benchmark",
    overwrite=True,

    ingest=dict(
        source="folder",
        image_dir="data/benchmarks/fox/images",
        image_glob="*.jpg",
        max_frames=30,          # subsample for sparse, fast SfM
    ),

    sfm=dict(
        model="facebook/map-anything-apache",
        # Public dataset has no ZED intrinsics/odometry -> pure RGB SfM.
        input_mode="rgb",
        apply_confidence_mask=True,
        confidence_percentile=10,
        max_points=250000,
    ),

    object=dict(
        method="auto_crop",
        auto_crop=dict(
            depth_percentile=(1.0, 85.0),   # keep more of the central object
            box_padding=0.15,
            keep_largest_cluster=True,
        ),
    ),

    splat=dict(
        max_steps=1500,
        time_budget_s=150,
        sh_degree=0,
        background="white",
        densify=True,
    ),

    export=dict(ply=True, ply_name="object_splat.ply"),
)
