"""
Local single-machine test: ZED2i plugged directly into the SplaTAM laptop
(docs/RUN_SINGLE_ROBOT_TEST.md). Same pipeline and topics as
zed2i_splat_live.py, overriding only what's specific to a direct/wired
single-machine test rather than the networked Orin/WiFi deployment:

  - use_compressed=False: compressed transport exists to survive the WiFi
    hop (docs/FULL_STACK_SETUP.md); on a direct USB/wired connection it buys
    nothing and requires the compressed-image-transport plugins to be
    installed. Verify with `ros2 topic list | grep compress` — if empty,
    leave this False (or install the plugins per FULL_STACK_SETUP.md A2).
  - num_frames / map_every: the base config's 45 frames / map_every=10 is a
    fast confirmation run (~5 mapping events) by design, not a quality run.
    Raise both for an honest look at reconstruction quality.

    python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_local_direct.py
"""

import copy
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from configs.zed2i.zed2i_splat_live import config as _base

config = copy.deepcopy(_base)

config["ros"]["use_compressed"] = False

config["num_frames"] = int(os.environ.get("SPLATAM_NUM_FRAMES", 150))
config["map_every"] = int(os.environ.get("SPLATAM_MAP_EVERY", 8))
