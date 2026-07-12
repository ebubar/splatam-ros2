"""
Config for testing the live SplaTAM node without a camera, by replaying a
converted dataset with scripts/dataset_player.py (which publishes
optical-frame odometry). Inherits the live ZED2i config.

    Terminal A: python3 scripts/zed2i_splat_live.py --config configs/zed2i/zed2i_replay_test.py
    Terminal B: python3 scripts/dataset_player.py --dataset <dataset_dir> --rate 5
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

config["workdir"] = "./experiments/ZED2i_Captures/replay_test"
config["num_frames"] = int(os.environ.get("SPLATAM_NUM_FRAMES", 40))
config["ros"]["odom_frame"] = "optical"
config["ros"]["pose_source"] = "odom"  # dataset_player publishes Odometry
config["viz"]["studio"] = os.environ.get("SPLATAM_STUDIO", "0") == "1"
config["viz"]["render_save_every"] = 5
