"""
Environment-scale live capture with the browser viewer
(scripts/live_viewer.py), for the networked deployment: ZED2i + wrapper
running on an Orin NX, publishing compressed RGB/depth/odom over WiFi; this
script runs on a separate GPU "splatting machine" ground station and
subscribes to those topics (docs/FULL_STACK_SETUP.md).

Same as zed2i_local_live_view.py, except it keeps the base config's
use_compressed=True (required to survive the WiFi hop -- do not disable it
here) and doesn't touch ros.pose_source/topics, since those come from
whatever the Orin's wrapper actually publishes (verify with
`ros2 topic list` on the Orin per FULL_STACK_SETUP.md A6 -- topic names can
differ across zed-ros2-wrapper versions, see the note in zed2i_splat_live.py).

Runs until you Ctrl-C (graceful -- saves whatever's been captured so far).

    python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_networked_live_view.py

Then open http://localhost:8080 on the SPLATTING machine (not the Orin --
the viewer runs wherever this script runs), or that machine's LAN IP from
another device, and walk the ZED/Orin around.
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

# Run until Ctrl-C, guided by the live viewer, rather than a preset cap.
config["num_frames"] = int(os.environ.get("SPLATAM_NUM_FRAMES", 20000))
config["map_every"] = int(os.environ.get("SPLATAM_MAP_EVERY", 8))

config["viz"]["live_viewer"] = True
config["viz"]["live_viewer_port"] = int(os.environ.get("SPLATAM_VIEWER_PORT", 8080))
