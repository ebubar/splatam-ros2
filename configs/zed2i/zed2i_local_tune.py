"""
Local ZED2i tuning config: same pipeline as zed2i_splat_live.py but with the
topic names of the locally-built zed-ros2-wrapper humble-v4.2.5 (which uses
rgb/image_rect_color + rgb/camera_info instead of the rgb/color/rect/... names
of the wrapper on the Orin). Used for camera-in-hand parameter iteration.

    python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_local_tune.py
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

config["workdir"] = "./experiments/ZED2i_Captures/local_tune"
config["run_name"] = os.environ.get("SPLATAM_RUN_NAME", "SplaTAM_local")
# With the studio you normally stop via Save snapshot + Ctrl-C; the frame
# cap is just a safety net.
config["num_frames"] = int(os.environ.get("SPLATAM_NUM_FRAMES", 5000))
config["viz"]["studio"] = True
config["ros"]["odom_frame"] = os.environ.get("SPLATAM_ODOM_FRAME", "zed_link")

config["ros"].update(
    rgb_topic="/zed/zed_node/rgb/image_rect_color",
    rgb_info_topic="/zed/zed_node/rgb/camera_info",
    depth_topic="/zed/zed_node/depth/depth_registered",
    depth_info_topic="/zed/zed_node/depth/camera_info",
    odom_topic="/zed/zed_node/odom",
    pose_source=os.environ.get("SPLATAM_POSE_SOURCE", "pose"),
    pose_topic="/zed/zed_node/pose",
    min_depth_m=0.3,
    max_depth_m=float(os.environ.get("SPLATAM_MAX_DEPTH", 5.0)),
    process_every_n=int(os.environ.get("SPLATAM_PROCESS_EVERY_N", 1)),
)

config["viz"]["render_save_every"] = 5
