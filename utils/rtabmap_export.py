#!/usr/bin/env python3
"""
Export a SplaTAM run's keyframes as a TUM RGB-D dataset for rtabmap.

The output folder follows the TUM RGB-D layout, which rtabmap ingests directly
via `rtabmap-rgbd_dataset`. This is the currency a robot ships to the ground
station: compact per-keyframe RGB-D + poses + timestamps, in the robot's own
metric (SplaTAM/VIO-anchored) world frame. The ground station runs rtabmap
multi-session place recognition / loop closure across robots to recover the
rigid transforms that align each robot's map, then those transforms are applied
to merge the Gaussian clouds.

Layout produced::

    <out_dir>/
      rgb/<stamp>.png            8-bit RGB keyframe
      depth/<stamp>.png          16-bit depth PNG (millimetres * DEPTH_SCALE/1000)
      rgb.txt                    "stamp rgb/<stamp>.png"
      depth.txt                  "stamp depth/<stamp>.png"
      associations.txt           "dstamp depth/... rgbstamp rgb/..." (1:1, synced)
      groundtruth.txt            TUM trajectory, camera-to-world
      calibration.yaml           OpenCV-format rectified intrinsics
      README.md                  how to feed this to rtabmap

The pure-text writers below (index files, TUM trajectory, calibration, README)
have no numpy/cv2 dependency so they can be unit-tested without the robot
runtime; only `export_keyframe_dataset` touches cv2/numpy, and it imports them
lazily.
"""

import os


# TUM's canonical depth scale: depth_metres = pixel_value / 5000.
DEFAULT_DEPTH_SCALE = 5000.0


def tum_stamp(stamp):
    """Canonical fixed-point timestamp string used for filenames and indices."""
    return f"{float(stamp):.6f}"


def write_index_file(path, entries):
    """entries: iterable of (stamp, rel_path). Writes 'stamp rel_path' lines."""
    with open(path, "w") as f:
        for stamp, rel_path in entries:
            f.write(f"{tum_stamp(stamp)} {rel_path}\n")


def write_associations(path, rows):
    """rows: iterable of (depth_stamp, depth_path, rgb_stamp, rgb_path)."""
    with open(path, "w") as f:
        for dstamp, dpath, rgbstamp, rgbpath in rows:
            f.write(
                f"{tum_stamp(dstamp)} {dpath} "
                f"{tum_stamp(rgbstamp)} {rgbpath}\n"
            )


def write_tum_lines(path, poses):
    """
    poses: iterable of (stamp, tx, ty, tz, qx, qy, qz, qw).

    Written verbatim as TUM: 'timestamp tx ty tz qx qy qz qw'. Poses are
    expected to be camera-to-world (the TUM/evo/rtabmap convention).
    """
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for p in poses:
            stamp, tx, ty, tz, qx, qy, qz, qw = p
            f.write(
                f"{tum_stamp(stamp)} "
                f"{tx:.6f} {ty:.6f} {tz:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )


def write_opencv_calibration(path, fx, fy, cx, cy, width, height,
                             camera_name="zed2i_left"):
    """
    Write intrinsics in OpenCV FileStorage YAML, which rtabmap reads.

    Input frames are already rectified (ZED `.../rect/...` topics), so distortion
    is zero and the rectification matrix is identity.
    """
    def _mat(rows, cols, data):
        data_str = ", ".join(f"{v:.8g}" for v in data)
        return (
            f"   rows: {rows}\n"
            f"   cols: {cols}\n"
            f"   dt: d\n"
            f"   data: [ {data_str} ]\n"
        )

    camera_matrix = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    distortion = [0.0, 0.0, 0.0, 0.0, 0.0]
    rectification = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    projection = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write(f'camera_name: "{camera_name}"\n')
        f.write(f"image_width: {int(width)}\n")
        f.write(f"image_height: {int(height)}\n")
        f.write("camera_matrix: !!opencv-matrix\n")
        f.write(_mat(3, 3, camera_matrix))
        f.write("distortion_coefficients: !!opencv-matrix\n")
        f.write(_mat(1, 5, distortion))
        f.write('distortion_model: "plumb_bob"\n')
        f.write("rectification_matrix: !!opencv-matrix\n")
        f.write(_mat(3, 3, rectification))
        f.write("projection_matrix: !!opencv-matrix\n")
        f.write(_mat(3, 4, projection))


def write_readme(path, num_keyframes, depth_scale, camera_name):
    text = f"""# rtabmap keyframe export

TUM RGB-D dataset exported from a SplaTAM run: {num_keyframes} keyframes.

- `rgb/`, `depth/` — 8-bit RGB and 16-bit depth PNGs, named by ROS timestamp.
- Depth scale: pixel_value / {depth_scale:g} = metres (TUM convention).
- `rgb.txt`, `depth.txt`, `associations.txt` — timestamp -> file indices.
  RGB and depth share each keyframe's timestamp, so association is 1:1.
- `groundtruth.txt` — TUM trajectory (camera-to-world) in this robot's own
  metric world frame (anchored at frame 0). Scale is metric (ZED depth).
- `calibration.yaml` — OpenCV-format rectified intrinsics ({camera_name}).

## Build an rtabmap database from this folder

    rtabmap-rgbd_dataset \\
        --Rtabmap/PublishRAMUsage true \\
        --Mem/IncrementalMemory true \\
        --calib calibration.yaml \\
        .

This produces `rtabmap.db`. For multi-robot melding, build one `.db` per robot
and merge on the ground station (e.g. `rtabmap-reprocess "robotA.db;robotB.db" merged.db`),
which runs cross-session loop closure to recover the inter-robot transforms.
Apply those transforms to place each robot's `splat.ply` into a common frame.
"""
    with open(path, "w") as f:
        f.write(text)


def export_keyframe_dataset(out_dir, keyframes, fx, fy, cx, cy, width, height,
                            depth_scale=DEFAULT_DEPTH_SCALE,
                            camera_name="zed2i_left"):
    """
    Write the full TUM RGB-D dataset.

    keyframes: list of dicts, each with
        'stamp' : float ROS timestamp (seconds)
        'rgb'   : HxWx3 uint8 numpy array, RGB order
        'depth' : HxW float32 numpy array, metres
        'tum'   : (tx, ty, tz, qx, qy, qz, qw) camera-to-world

    Returns the number of keyframes written.
    """
    import cv2
    import numpy as np

    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    rgb_index = []
    depth_index = []
    assoc_rows = []
    poses = []

    for kf in keyframes:
        stamp = float(kf["stamp"])
        name = f"{tum_stamp(stamp)}.png"
        rgb_rel = f"rgb/{name}"
        depth_rel = f"depth/{name}"

        rgb = kf["rgb"]
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, rgb_rel), bgr)

        depth_m = np.asarray(kf["depth"], dtype=np.float32)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_u16 = np.clip(depth_m * depth_scale, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(out_dir, depth_rel), depth_u16)

        rgb_index.append((stamp, rgb_rel))
        depth_index.append((stamp, depth_rel))
        assoc_rows.append((stamp, depth_rel, stamp, rgb_rel))
        poses.append((stamp,) + tuple(kf["tum"]))

    write_index_file(os.path.join(out_dir, "rgb.txt"), rgb_index)
    write_index_file(os.path.join(out_dir, "depth.txt"), depth_index)
    write_associations(os.path.join(out_dir, "associations.txt"), assoc_rows)
    write_tum_lines(os.path.join(out_dir, "groundtruth.txt"), poses)
    write_opencv_calibration(
        os.path.join(out_dir, "calibration.yaml"),
        fx, fy, cx, cy, width, height, camera_name,
    )
    write_readme(
        os.path.join(out_dir, "README.md"),
        len(keyframes), depth_scale, camera_name,
    )

    return len(keyframes)
