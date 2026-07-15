#!/bin/bash
# Create a Python virtual environment that can see the system ROS2 packages.
#
# ROS2 ships rclpy / cv_bridge / message_filters as apt packages built against the
# SYSTEM Python for your ROS distro (Humble -> 3.10, Jazzy -> 3.12). A plain conda
# env with its own Python cannot import them (cv_bridge is a compiled extension
# linked to the system Python ABI). The reliable fix is a venv created FROM the
# matching system Python WITH --system-site-packages, so it inherits the system
# packages while still letting you pip-install torch/gsplat on top.
#
# This script only CREATES the venv and prints the activation + install steps —
# a child process cannot activate an env in your parent shell.
#
# Usage:
#   bash bash_scripts/make_venv.bash                       # autodetect distro from $ROS_DISTRO
#   bash bash_scripts/make_venv.bash --distro humble       # or jazzy
#   bash bash_scripts/make_venv.bash --path ~/venvs/splatam
set -uo pipefail

info='\033[96m'; ok='\033[92m'; warn='\033[93m'; err='\033[91m'; END='\033[0m'
say() { printf "${1}%b${END}\n" "$2"; }
die() { say "$err" "[ERROR] $1"; exit 1; }

DISTRO="${ROS_DISTRO:-}"
VENV_PATH="$HOME/venvs/splatam"
while [ $# -gt 0 ]; do
    case "$1" in
        --distro) DISTRO="$2"; shift 2 ;;
        --path)   VENV_PATH="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

[ -n "$DISTRO" ] || die "could not determine ROS distro. Pass --distro humble|jazzy, or 'source /opt/ros/<distro>/setup.bash' first."

# Map ROS distro -> the system Python it was built against.
case "$DISTRO" in
    humble) PYVER="3.10" ;;
    jazzy)  PYVER="3.12" ;;
    *) say "$warn" "[WARN] unrecognized distro '$DISTRO'; defaulting to python3. Verify the version matches your ROS build."; PYVER="" ;;
esac

SYS_PY="python${PYVER}"
command -v "$SYS_PY" >/dev/null 2>&1 || die "$SYS_PY not found. Install it (e.g. 'sudo apt install python${PYVER}-venv python${PYVER}-dev') — it must match your ROS $DISTRO Python."

say "$info" "[INFO] ROS distro : $DISTRO"
say "$info" "[INFO] system py  : $($SYS_PY --version 2>&1) ($(command -v "$SYS_PY"))"
say "$info" "[INFO] venv path  : $VENV_PATH"

if [ -d "$VENV_PATH" ]; then
    say "$warn" "[WARN] $VENV_PATH already exists; leaving it as-is."
else
    "$SYS_PY" -m venv --system-site-packages "$VENV_PATH" \
        || die "venv creation failed. Ensure the '${SYS_PY}-venv' apt package is installed."
    say "$ok" "[OK] created venv with --system-site-packages"
fi

# Verify ROS is importable through the venv (needs ROS sourced in THIS shell).
say "$info" "[INFO] checking ROS import via the new venv..."
if [ -f "/opt/ros/$DISTRO/setup.bash" ]; then
    # shellcheck disable=SC1090
    ( set +u; source "/opt/ros/$DISTRO/setup.bash"; source "$VENV_PATH/bin/activate";
      python -c "import rclpy, cv_bridge; print('    rclpy + cv_bridge import OK')" ) \
      || say "$warn" "    could not import rclpy/cv_bridge yet — check that ros-$DISTRO-cv-bridge is apt-installed and the Python versions match."
else
    say "$warn" "    /opt/ros/$DISTRO/setup.bash not found; install ROS $DISTRO first (see docs/install.md)."
fi

cat <<EOF

$(say "$ok" "Next steps — run these in every new shell (order matters):")
  source /opt/ros/$DISTRO/setup.bash        # put ROS python packages on PYTHONPATH
  source $VENV_PATH/bin/activate            # activate the venv (same Python as ROS)

$(say "$ok" "Then install torch + the pipeline:")
  # x86 desktop:
  pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
  # (Jetson Thor: install the JetPack/NGC torch wheel instead — see docs/install.md)
  bash bash_scripts/install.bash            # core deps + gsplat
  python scripts/tools/preflight.py         # verify everything

Tip: if 'conda' auto-activates a base env and shadows this venv's python, run 'conda deactivate' first.
EOF
