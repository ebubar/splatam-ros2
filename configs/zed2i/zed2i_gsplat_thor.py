"""Jetson Thor profile for the realtime gsplat node.

Thin override of configs/zed2i/zed2i_gsplat_live.py tuned for the Jetson Thor
(Blackwell, sm_110): lower working resolution, a smaller mapping budget, a tighter
Gaussian ceiling, FPS throttling on, and periodic checkpointing for long
unattended runs. Build gsplat with TORCH_CUDA_ARCH_LIST="11.0".

  python3 scripts/zed2i_gsplat_live.py --config configs/zed2i/zed2i_gsplat_thor.py
"""
import os
from importlib.machinery import SourceFileLoader

_base = SourceFileLoader(
    "zed2i_gsplat_live",
    os.path.join(os.path.dirname(__file__), "zed2i_gsplat_live.py"),
).load_module()

config = _base.config
config["run_name"] = "SplaTAM_ZED2i_gsplat_thor"

# Lower resolution to fit the edge compute/memory budget.
config["data"]["desired_image_width"] = 512
config["data"]["desired_image_height"] = 288
config["data"]["densification_image_width"] = 512
config["data"]["densification_image_height"] = 288

# Smaller mapping budget; map slightly less often.
config["mapping"]["num_iters"] = 40
config["map_every"] = 6

# Bounded memory + adaptive throttle + checkpoints for long runs.
config["hardening"]["max_gaussians"] = 800_000
config["hardening"]["adaptive_fps"] = True
config["hardening"]["target_fps"] = 5.0
config["hardening"]["min_map_iters"] = 12
config["hardening"]["checkpoint_every"] = 100

# Compressed transport helps over the Orin/Thor network link.
config["ros"]["transport"] = "raw"   # set "compressed" if bandwidth-limited
