"""Desktop x86 CUDA profile for the realtime gsplat node.

Thin override of configs/zed2i/zed2i_gsplat_live.py tuned for a discrete desktop
GPU: full working resolution, a larger mapping budget, a higher Gaussian ceiling,
and no FPS throttling. Build gsplat with TORCH_CUDA_ARCH_LIST matching your GPU
(e.g. 8.6 Ampere, 8.9 Ada, 9.0 Hopper).

  python3 scripts/zed2i_gsplat_live.py --config configs/zed2i/zed2i_gsplat_desktop.py
"""
import os
from importlib.machinery import SourceFileLoader

_base = SourceFileLoader(
    "zed2i_gsplat_live",
    os.path.join(os.path.dirname(__file__), "zed2i_gsplat_live.py"),
).load_module()

config = _base.config
config["run_name"] = "SplaTAM_ZED2i_gsplat_desktop"

# Desktop can afford a larger mapping budget and no throttle.
config["mapping"]["num_iters"] = 60
config["map_every"] = 5
config["hardening"]["max_gaussians"] = 2_500_000
config["hardening"]["target_fps"] = 0.0        # unthrottled
config["hardening"]["checkpoint_every"] = 0
