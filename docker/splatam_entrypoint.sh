#!/bin/bash
set -e

# Install ROS only if missing
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    echo "[INFO] Installing ROS2 Jazzy..."

    apt-get update
    apt-get install -y curl gnupg2 lsb-release software-properties-common

    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')

    curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"

    dpkg -i /tmp/ros2-apt-source.deb

    apt-get update
    apt-get install -y \
        ros-jazzy-ros-base \
        ros-jazzy-cv-bridge \
        ros-jazzy-message-filters

    echo "[INFO] ROS install complete"
fi

# Wait until ROS is actually available
while [ ! -f /opt/ros/jazzy/setup.bash ]; do
    echo "[INFO] Waiting for ROS installation..."
    sleep 2
done
# Fix torch / ROS conflict by prioritizing HPC-X
export LD_LIBRARY_PATH="/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH}"

echo "[INFO] Fixed LD_LIBRARY_PATH for Torch"
# Source ROS
source /opt/ros/jazzy/setup.bash

echo "[INFO] ROS ready"

# Run original command
exec "$@"