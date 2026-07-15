import torch

# The CUDA rasterizer's camera settings type is only needed by the "cuda" render
# backend. Import it lazily/optionally so the default gsplat path (and gsplat-only
# installs) don't require diff-gaussian-rasterization to be built.
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
except Exception:  # pragma: no cover - CUDA rasterizer not installed
    Camera = None


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    if Camera is None:
        raise RuntimeError(
            "setup_camera() needs the CUDA rasterizer (diff-gaussian-rasterization), "
            "which is not installed. Use render_backend='gsplat' (default), or install "
            "the optional fallback: bash bash_scripts/install.bash --with-cuda-fallback"
        )
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam
