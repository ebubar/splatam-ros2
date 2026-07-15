#!/usr/bin/env python3
"""
Startup preflight: check that everything the realtime gsplat node needs is in
place, and print an actionable fix for anything missing. Run this before the node
so failures surface as a clear checklist instead of a stack trace mid-run.

  python3 scripts/tools/preflight.py
  python3 scripts/tools/preflight.py --config configs/zed2i/zed2i_gsplat_thor.py

Exit code 0 if all REQUIRED checks pass, 1 otherwise.
"""

import argparse
import importlib
import importlib.util
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASE)

GREEN, RED, YELLOW, END = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


class Check:
    def __init__(self):
        self.failed_required = False

    def report(self, name, ok, required, detail="", fix=""):
        tag = f"{GREEN}PASS{END}" if ok else (f"{RED}FAIL{END}" if required else f"{YELLOW}WARN{END}")
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            if fix:
                print(f"         fix: {fix}")
            if required:
                self.failed_required = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zed2i/zed2i_gsplat_live.py")
    args = ap.parse_args()
    c = Check()

    print("== SplaTAM gsplat preflight ==")

    # --- torch + CUDA (required) ---
    torch = None
    try:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        c.report("PyTorch + CUDA", cuda, required=True,
                 detail=f"torch {torch.__version__}, cuda {torch.version.cuda}, available={cuda}",
                 fix="install a CUDA build of torch 2.x, e.g. "
                     "pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121")
    except Exception as exc:
        c.report("PyTorch + CUDA", False, required=True, detail=str(exc),
                 fix="pip install torch (CUDA build); see docs/running_locally.md")

    # --- at least one render backend (required) ---
    has_gsplat = importlib.util.find_spec("gsplat") is not None
    has_cuda_rast = importlib.util.find_spec("diff_gaussian_rasterization") is not None
    c.report("gsplat engine (default)", has_gsplat, required=False,
             fix="bash bash_scripts/install.bash")
    c.report("cuda fallback engine (optional)", has_cuda_rast, required=False,
             fix="bash bash_scripts/install.bash --with-cuda-fallback")
    c.report("at least one render backend", has_gsplat or has_cuda_rast, required=True,
             fix="bash bash_scripts/install.bash")

    # --- config loads (required) ---
    try:
        from importlib.machinery import SourceFileLoader
        cfg = SourceFileLoader("cfg", args.config).load_module().config
        c.report("config loads", True, required=True,
                 detail=f"{args.config} (backend={cfg.get('render_backend')}, "
                        f"pose_source={cfg.get('pose_source')})")
    except Exception as exc:
        c.report("config loads", False, required=True, detail=str(exc),
                 fix=f"check {args.config}")

    # --- ROS2 python (required to run the live node) ---
    ros_distro = os.environ.get("ROS_DISTRO", "")
    for mod, required in [("rclpy", True), ("cv_bridge", True), ("message_filters", True),
                          ("sensor_msgs.msg", True), ("nav_msgs.msg", True), ("std_msgs.msg", True)]:
        ok = importlib.util.find_spec(mod.split(".")[0]) is not None
        c.report(f"ROS: {mod}", ok, required=required,
                 fix="source /opt/ros/<distro>/setup.bash  (and apt install the ros-<distro>-* packages)")
    c.report("ROS environment sourced", bool(ros_distro), required=False,
             detail=f"ROS_DISTRO={ros_distro or '<unset>'}",
             fix="source /opt/ros/humble/setup.bash  (or jazzy on Thor)")

    print()
    if c.failed_required:
        print(f"{RED}Preflight FAILED — resolve the required items above before running the node.{END}")
        sys.exit(1)
    print(f"{GREEN}Preflight OK — you can launch the node.{END}")
    sys.exit(0)


if __name__ == "__main__":
    main()
