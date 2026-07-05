# Splat Quality Tuning Guide

All knobs live in `configs/zed2i/zed2i_splat_live.py`. Changes here trade
quality against per-frame latency and GPU memory — test on a recorded bag
(`docs/README.md`, Option B) so runs are repeatable before going live.

> **Biggest lever of all:** if you can spare a few minutes of post-processing,
> use the offline quality path (RTAB-Map loop-closed poses + offline re-splat)
> described in `docs/README.md` — it removes drift ghosting that no live
> tuning can fix.

## Highest-impact levers, in rough order

### 1. Prune floaters (enabled on this branch)

`mapping.prune_gaussians=True` removes low-opacity Gaussians during mapping.
Floaters — semi-transparent blobs hanging in free space — are the most visible
quality defect in ZED runs because stereo depth is noisy at range. The
`pruning_dict` schedule is expressed in *mapping iterations* (currently 180
per mapped frame), so `start_after`/`prune_every` must stay below
`mapping.num_iters` to ever fire.

### 2. Tighten the depth range

`ros.max_depth_m=6.0` is optimistic for a ZED2i indoors; stereo depth error
grows quadratically with distance. Dropping to `4.0–5.0` discards the noisiest
points instead of baking them into the map. Keep `min_depth_m` ≥ 0.3.

### 3. More frames, slower motion

`num_frames=45` is a very short capture. More frames (150–300) with slow,
deliberate camera motion and view overlap gives mapping many more constraints.
If the node can't keep up live, raise `ros.process_every_n` (e.g. 2–4) rather
than moving the camera faster — SplaTAM tolerates lower frame rate much better
than motion blur and large inter-frame baselines.

### 4. Resolution

`full_res_width/height = 640×360` bounds the detail the splat can represent.
If the GPU has headroom, try 1280×720 for the working resolution, or keep
640×360 for tracking and set `densify_downscale_factor < 1` equivalent by
raising only `densification_image_*` — new Gaussians are seeded from the
densification frames, so densify resolution matters most for detail.

### 5. Mapping effort

* `map_every=10`: mapping runs every 10th processed frame. Lowering it (e.g. 5)
  maps more often at the cost of throughput.
* `mapping.num_iters=180`: more iterations sharpen the map; diminishing returns
  past ~200 for short captures. If you raise it, revisit the pruning schedule.
* `mapping_window_size=32`: number of keyframes optimized together. Larger is
  better for consistency, more VRAM.

### 6. Tracking quality (pose accuracy → sharpness)

Blurry or ghosted splats often mean pose error, not mapping error. This
pipeline seeds each pose from ZED odometry, then refines with
`tracking.num_iters=30`. If ghosting appears:

* raise `tracking.num_iters` to 40–60;
* verify the ZED wrapper runs with `pos_tracking_mode:=GEN_3`;
* keep `use_sil_for_loss=True` / `sil_thres=0.99` (ignores not-yet-mapped
  regions when tracking).

### 7. Anisotropic Gaussians

`gaussian_distribution="anisotropic"` lets Gaussians stretch along surfaces —
better thin structures and edges — at higher memory and some robustness cost.
Worth an A/B test on a bag once the rest is dialed in.

## Suggested experiment protocol

1. Record one good bag (slow indoor loop, 60–120 s).
2. Change **one** knob at a time; keep `seed` fixed.
3. Compare exported PLYs in SuperSplat and the per-frame
   `depth_color_debug/` output (black regions = discarded depth).
4. Watch the per-frame log line: FPS and Gaussian count show the cost side.
