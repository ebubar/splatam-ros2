# Intentionally does NOT eagerly `from .gradslam_datasets import *`.
#
# The offline benchmark loaders (Replica/TUM/ScanNet/...) live in
# datasets.gradslam_datasets and pull heavy deps. Importing them eagerly here
# would force the realtime ROS node (which only needs datasets.zed_ros_dataset)
# to load the entire offline dataset stack. Offline code imports the loaders
# explicitly via `from datasets.gradslam_datasets import ...`, so nothing here
# is needed.
