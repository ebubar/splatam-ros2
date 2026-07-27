# Real-time SplaTAM architecture (single robot → multi-robot rtabmap)

This document describes how the live ZED2i → ROS2 → SplaTAM pipeline is
structured for solid real-time operation over a network, and how that design
sets up the eventual multi-robot map-melding goal with rtabmap.

Entry point: `scripts/zed2i_splat_live.py`, config: `configs/zed2i/zed2i_splat_live.py`.

## The core problem

SplaTAM produces excellent splats when the camera and the GPU are on the same
machine. The quality collapses when frames arrive over a network. The culprit is
**not** steady-state bandwidth — it is **latency variance (jitter) and frame
drops**, which starve the tracking front-end. Two things had to change to make
the networked path as solid as the local one.

## 1. Ingestion is decoupled from processing

Previously, all GPU work (tracking + mapping, ~hundreds of optimizer iterations)
ran *inside* the ROS synchronized subscriber callback. While SplaTAM ground on
frame *N*, incoming frames piled into the DDS queue and were dropped
**non-deterministically** (BEST_EFFORT, `KEEP_LAST`, depth 10). The transport,
not the SLAM, decided which frames were processed.

Now:

- The synchronized callback is trivial. It drops the newest
  `(rgb, depth, odom, stamp)` into a **single-slot buffer**, overwriting any
  frame not yet consumed.
- A dedicated **worker loop** (`run_worker`) always pulls the *freshest*
  unprocessed frame and runs SplaTAM on it. ROS spins in a background thread.
- Frame dropping is therefore **explicit and controlled** ("always process
  newest"), and each processed frame logs how many frames were dropped since the
  last one — a direct read on network health.

This is safe **because the pose is anchored to an absolute source** (see below),
so skipping intermediate frames does not break tracking the way pure
frame-to-frame SplaTAM would.

## 2. ZED VIO is a seed only; the refined pose is authoritative

ZED VIO poses are **not** trusted as ground truth (they have proven too
inaccurate for pose-dependent splatting elsewhere). So:

- Each frame's pose is **seeded** either from ZED odometry (`pose_init="odom"`)
  or from SplaTAM's own constant-velocity motion model
  (`pose_init="constant_velocity"`, or `use_odom=False` — VIO never enters the
  pipeline at all).
- SplaTAM dense tracking then **refines** that seed. The **refined pose is the
  single source of truth** — it is what gets stored in the trajectory, the
  keyframes, the mapping step, and the exported map. Raw VIO is never written to
  the output.

The tracking learning rates (`tracking.lrs.cam_unnorm_rots`,
`tracking.lrs.cam_trans`) govern how strongly refinement can pull off the seed.
They must be large enough to correct VIO error but small enough to avoid
jitter — **tune these on-robot.**

## 3. Outputs are structured for multi-robot melding

On finalize (either reaching `num_frames` or Ctrl-C — a partial map is still
saved), the run writes, into `<workdir>/<run_name>/`:

| File | Contents |
|------|----------|
| `params.npz` | Gaussian map + refined poses + `frame_stamps` + keyframe indices |
| `splat.ply` | Exported Gaussian splat (via `scripts/export_ply.py`) |
| `traj_tum.txt` | Per-frame trajectory, TUM format, camera-to-world |
| `traj_keyframes_tum.txt` | Per-keyframe trajectory, TUM format |
| `map_meta.json` | Frame counts, pose mode, world-frame + trajectory conventions |

The trajectory is TUM format (`timestamp tx ty tz qx qy qz qw`, camera-to-world)
so it is directly consumable by `evo` and by rtabmap tooling. Every pose and
keyframe carries its **ROS header timestamp**, which multi-robot temporal
alignment needs.

**World frame:** each robot's map is expressed in its own frame, anchored at the
camera pose of its first frame. Metric scale is consistent across robots because
the ZED provides metric depth — which makes rigid cross-robot alignment
well-posed (no scale ambiguity).

## Roadmap to multi-robot (rtabmap)

The division of labor keeps the latency-sensitive tracking loop on the edge and
sends only latency-tolerant, map-scale data to a ground station:

1. **Per robot (edge):** run this pipeline. Sensor + GPU co-located. Produces a
   dense Gaussian map + a timestamped, refined keyframe trajectory in its own
   metric frame.
2. **Ship maps, not pixels:** each robot sends compact keyframes + poses (and its
   `.ply`) to the ground station — not raw RGB-D streams. This routes around the
   networking bottleneck entirely.
3. **Ground station (rtabmap):** multi-session place recognition + loop closure
   across robots produces globally consistent rigid transforms between each
   robot's world frame (and can also correct intra-robot drift — SplaTAM has no
   loop closure of its own).
4. **Merge:** apply those transforms to place every robot's Gaussian cloud into
   one common frame.

**Next step** toward this: a per-keyframe RGB-D export (images + poses +
timestamps in an rtabmap-ingestable layout), so a robot is immediately ready to
ship maps to the ground station. The timestamped keyframe trajectory this pass
already emits is the seam that step plugs into.

## Config knobs that matter for real-time

- `ros.pose_init` — `"odom"` vs `"constant_velocity"`.
- `ros.use_odom` — `False` drops VIO from the sync entirely.
- `tracking.num_iters`, `mapping.num_iters` — the dominant per-frame cost.
- `tracking.lrs.cam_*` — how hard tracking corrects the VIO seed.
- `map_every`, `keyframe_every`, `mapping_window_size` — mapping cadence/cost.
- `ros.sync_slop_s` — rgb/depth/odom time-sync tolerance.
- `ros.save_depth_debug` — leave `False` for real-time (per-frame PNG writes).
- `num_frames` — currently a hard cap (pose arrays are preallocated to this
  size); set generously for long runs.
