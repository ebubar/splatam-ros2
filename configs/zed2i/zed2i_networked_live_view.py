"""
Environment-scale live capture with the browser viewer
(scripts/live_viewer.py), for the networked deployment: ZED2i + wrapper
running on an Orin NX, publishing compressed RGB/depth/odom over WiFi; this
script runs on a separate GPU "splatting machine" ground station and
subscribes to those topics (docs/FULL_STACK_SETUP.md).

Same as zed2i_local_live_view.py, except it keeps the base config's
use_compressed=True (required to survive the WiFi hop -- do not disable it
here). The RGB topic paths ARE overridden below -- unlike the base config's
pinned wrapper tag (humble-v4.2.5), Orins on this branch have been running
newer wrapper builds (e.g. humble-v5.1.0-14-g72ee77c) that publish RGB under
".../rgb/color/rect/image[/camera_info]" instead of
".../rgb/image_rect_color". If you swap in yet another Orin/wrapper build,
re-verify with `ros2 topic list | grep rgb` on the Orin and update this file
(docs/QUICKSTART.md has the general troubleshooting notes).

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

# This Orin's wrapper build (humble-v5.1.0-14-g72ee77c) publishes RGB under
# a different path than the base config's pinned humble-v4.2.5 -- confirmed
# live via `ros2 topic hz` (the base config's paths are advertised but never
# actually publish on this wrapper version).
config["ros"]["rgb_topic"] = "/zed/zed_node/rgb/color/rect/image"
config["ros"]["rgb_info_topic"] = "/zed/zed_node/rgb/color/rect/camera_info"
config["ros"]["rgb_compressed_topic"] = config["ros"]["rgb_topic"] + "/compressed"

# Run until Ctrl-C, guided by the live viewer, rather than a preset cap.
config["num_frames"] = int(os.environ.get("SPLATAM_NUM_FRAMES", 20000))
# Lower than the base config's map_every=10: more frequent, individually
# CHEAPER mapping passes (same mapping_iters/densify_downscale_factor per
# pass) spread density growth out more evenly instead of occasional bigger
# stalls. Went from 8 -> 4 after densify_downscale_factor=1.5 (a bigger-
# stall approach) caused a hard tracking break -- one 84-dropped-frame gap
# split the map into two disconnected pieces. densify_downscale_factor is
# back at 2.0 (zed2i_splat_live.py); this is the "add density a different
# way" follow-up, not stacked on top of the failed 1.5 attempt.
config["map_every"] = int(os.environ.get("SPLATAM_MAP_EVERY", 4))

config["viz"]["live_viewer"] = True
config["viz"]["live_viewer_port"] = int(os.environ.get("SPLATAM_VIEWER_PORT", 8080))
