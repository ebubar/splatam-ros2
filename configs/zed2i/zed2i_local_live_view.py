"""
Environment-scale live capture with the browser viewer
(scripts/live_viewer.py): walk the ZED around a room and watch the map
build in an orbitable 3D view, so you can see what's covered and what's
still empty — the actual failure mode this config is built for ("got the
curtains, missed the rest of the room").

Runs until you Ctrl-C (graceful — saves whatever's been captured so far),
not a fixed frame count: this is a scene, not a 45-frame confirmation test.
See docs/RUN_SINGLE_ROBOT_TEST.md for the fixed-length version
(zed2i_local_direct.py) and docs/FULL_STACK_SETUP.md for the networked
Orin/WiFi deployment (same live_viewer flag works there too — set
live_viewer=True on whichever machine runs this script, and open its IP).

    python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_local_live_view.py

Then open http://localhost:8080 (or this machine's LAN IP from another
device) and start walking. Use the "Save snapshot" button anytime to export
a PLY without interrupting the walk.
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

# Run until Ctrl-C, guided by the live viewer, rather than a preset cap.
config["num_frames"] = int(os.environ.get("SPLATAM_NUM_FRAMES", 20000))
config["map_every"] = int(os.environ.get("SPLATAM_MAP_EVERY", 8))

config["viz"]["live_viewer"] = True
config["viz"]["live_viewer_port"] = int(os.environ.get("SPLATAM_VIEWER_PORT", 8080))
