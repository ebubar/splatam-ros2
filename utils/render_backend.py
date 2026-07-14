"""
Rendering-backend seam for SplaTAM.

The only place SplaTAM rasterizes in the live path is the RGB + depth/silhouette
render inside ``get_loss`` and ``add_new_gaussians`` (scripts/splatam.py). This
module isolates that render behind a single ``render_frame`` call so the engine
can be switched at runtime between:

  * ``"cuda"``   -- the original INRIA diff-gaussian-rasterization (research /
                    non-commercial license). Kept byte-for-byte as the fallback.
  * ``"gsplat"`` -- nerfstudio's gsplat (Apache-2.0). Faster, and license-clean
                    for defense use. gsplat is imported lazily so the module
                    still imports on machines that only have the CUDA rasterizer.

Both backends return the same normalized outputs so the loss/masking code above
``get_loss``'s ``nan_mask`` is identical regardless of engine.

Returned dict:
    im         : [3, H, W]   rendered RGB
    depth      : [1, H, W]   rendered depth (metric)
    silhouette : [H, W]      per-pixel coverage in [0, 1] (alpha)
    uncertainty: [1, H, W]   depth variance proxy (0 for gsplat; used only by
                             the CUDA tracking-uncertainty mask)
    radius     : [N] or None screen-space radii (CUDA densification bookkeeping)
    means2D    : tensor|None  screen-space means w/ retained grad (CUDA densify)
"""

import torch

from utils.slam_helpers import (
    transform_to_frame,
    transformed_params2rendervar,
    transformed_params2depthplussilhouette,
    params2gsplat_inputs,
    build_w2c_from_cam_params,
)


def _render_cuda(params, time_idx, curr_data, variables, gaussians_grad, camera_grad):
    """Original two-pass CUDA rasterizer path (unchanged behavior)."""
    # Imported lazily so environments without the CUDA rasterizer can still use gsplat.
    from diff_gaussian_rasterization import GaussianRasterizer as Renderer

    transformed_gaussians = transform_to_frame(
        params, time_idx, gaussians_grad=gaussians_grad, camera_grad=camera_grad
    )
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(
        params, curr_data['w2c'], transformed_gaussians
    )

    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    means2D = rendervar['means2D']  # gradient accum for densification

    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = (depth_sq - depth ** 2).detach()

    return {
        'im': im,
        'depth': depth,
        'silhouette': silhouette,
        'uncertainty': uncertainty,
        'radius': radius,
        'means2D': means2D,
    }


def _render_gsplat(params, time_idx, curr_data, gaussians_grad, camera_grad, gsplat_cfg):
    """gsplat (Apache-2.0) single-call path.

    gsplat applies the camera pose via ``viewmats`` (world->cam) and returns
    depth + alpha natively, so SplaTAM's depth+silhouette packing hack is not
    needed here.
    """
    import gsplat  # lazy: only required when this backend is selected

    gsplat_cfg = gsplat_cfg or {}
    means, quats, scales, opacities, colors = params2gsplat_inputs(params, gaussians_grad)

    # world->cam viewmat from the per-frame pose params (differentiable when
    # camera_grad=True so the optional pose-refinement fallback can optimize it).
    w2c = build_w2c_from_cam_params(params, time_idx, camera_grad)
    viewmats = w2c.unsqueeze(0)                      # [1, 4, 4]
    Ks = curr_data['intrinsics'].unsqueeze(0)        # [1, 3, 3]

    # Derive image size from the GT frame so the new node need not thread extra keys.
    H = int(curr_data['im'].shape[1])
    W = int(curr_data['im'].shape[2])

    render_mode = gsplat_cfg.get('render_mode', 'RGB+ED')
    bg = gsplat_cfg.get('bg', None)
    backgrounds = None
    if bg is not None:
        backgrounds = torch.tensor([bg], dtype=colors.dtype, device=colors.device)

    render_colors, render_alphas, info = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=None,               # colors are precomputed RGB, not SH
        render_mode=render_mode,
        packed=bool(gsplat_cfg.get('packed', False)),
        near_plane=float(gsplat_cfg.get('near_plane', 0.01)),
        far_plane=float(gsplat_cfg.get('far_plane', 1e10)),
        backgrounds=backgrounds,
    )

    im = render_colors[0, ..., :3].permute(2, 0, 1)  # [3, H, W]
    if render_mode.endswith('D'):                     # "RGB+D" or "RGB+ED"
        depth = render_colors[0, ..., 3:4].permute(2, 0, 1)  # [1, H, W]
    else:
        depth = torch.zeros((1, H, W), device=im.device, dtype=im.dtype)
    silhouette = render_alphas[0, ..., 0]             # [H, W] coverage / alpha

    # Screen-space radii for CUDA-style densification bookkeeping. In unpacked
    # mode (packed=False) this is [1, N]; collapse to [N] so the caller can index
    # per-Gaussian. gsplat densify/prune is disabled in the live config, so this
    # is best-effort and may be None.
    radius = None
    radii = info.get('radii') if isinstance(info, dict) else None
    if radii is not None and radii.dim() == 2 and radii.shape[0] == 1:
        radius = radii[0]

    return {
        'im': im,
        'depth': depth,
        'silhouette': silhouette,
        'uncertainty': torch.zeros_like(depth),  # gsplat has no depth^2 channel
        'radius': radius,
        'means2D': None,
    }


def render_frame(params, time_idx, curr_data, variables,
                 gaussians_grad, camera_grad,
                 backend='cuda', gsplat_cfg=None):
    """Dispatch to the selected rendering backend.

    Args mirror ``transform_to_frame``'s grad gating: ``gaussians_grad`` enables
    gradients to the Gaussian params (mapping), ``camera_grad`` enables gradients
    to the camera pose (tracking / pose-refinement fallback).
    """
    if backend == 'gsplat':
        return _render_gsplat(params, time_idx, curr_data,
                              gaussians_grad, camera_grad, gsplat_cfg)
    return _render_cuda(params, time_idx, curr_data, variables,
                        gaussians_grad, camera_grad)
