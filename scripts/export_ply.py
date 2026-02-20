import os
import argparse
from importlib.machinery import SourceFileLoader

import numpy as np
from plyfile import PlyData, PlyElement

# Spherical harmonic constant
C0 = 0.28209479177387814


def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh*C0 + 0.5


def save_ply(path, means, scales, rotations, rgbs, opacities, normals=None):
    if normals is None:
        normals = np.zeros_like(means)

    colors = rgb_to_spherical_harmonic(rgbs)

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3',]

    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"Saved PLY format Splat to {path}")


def save_ply_rgb(path, means, rgbs):
    """Save a simple RGB PLY with position + uint8 RGB for WebGL viewers."""
    # Ensure rgb in 0..1, convert to 0..255 uint8
    rgb_vals = (rgbs * 255.0).clip(0, 255).astype(np.uint8)

    dtype_simple = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(means.shape[0], dtype=dtype_simple)

    elements['x'] = means[:, 0].astype(np.float32)
    elements['y'] = means[:, 1].astype(np.float32)
    elements['z'] = means[:, 2].astype(np.float32)
    elements['red'] = rgb_vals[:, 0]
    elements['green'] = rgb_vals[:, 1]
    elements['blue'] = rgb_vals[:, 2]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"Saved simple RGB PLY to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load SplaTAM config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    work_path = config['workdir']
    run_name = config['run_name']
    params_path = os.path.join(work_path, run_name, "params.npz")

    params = dict(np.load(params_path, allow_pickle=True))
    means = params['means3D']
    scales = params['log_scales']
    rotations = params['unnorm_rotations']
    rgbs = params['rgb_colors']
    opacities = params['logit_opacities']

    ply_path = os.path.join(work_path, run_name, "splat.ply")

    save_ply(ply_path, means, scales, rotations, rgbs, opacities)
    # Also write a simple RGB PLY (uint8 colors) for WebGL viewers
    rgb_ply_path = os.path.join(work_path, run_name, "splat_rgb.ply")
    save_ply_rgb(rgb_ply_path, means, rgbs)