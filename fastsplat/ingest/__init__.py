"""Stage 1: produce a normalized capture directory of sparse keyframes."""

from .folder import prepare_from_folder


def run_ingest(cfg, out_dir):
    """Dispatch to the configured ingest source. Returns the capture dir."""
    source = cfg["ingest"]["source"]
    if source == "folder":
        return prepare_from_folder(cfg, out_dir)
    if source == "ros2":
        # Imported lazily so the folder path never requires rclpy.
        from .ros2_capture import capture_from_ros2
        return capture_from_ros2(cfg, out_dir)
    raise ValueError(f"Unknown ingest source: {source!r}")
