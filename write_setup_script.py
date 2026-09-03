#!/usr/bin/env python3
import os

content = """#!/usr/bin/env bash
set -e

echo "=== [1/5] Configuring APT and ROS Repositories ==="
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --fix-missing curl gnupg2 lsb-release ca-certificates

# Add ROS Noetic key and repo
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - || \
  apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list

echo "=== [2/5] Installing ROS Noetic and Dependencies ==="
apt-get update
apt-get install -y --fix-missing \
  ros-noetic-desktop-full \
  ros-noetic-cv-bridge \
  ros-noetic-image-transport \
  ros-noetic-image-transport-plugins \
  ros-noetic-pcl-ros \
  ros-noetic-pcl-conversions \
  ros-noetic-eigen-conversions \
  ros-noetic-tf \
  ros-noetic-cmake-modules \
  build-essential \
  cmake \
  git \
  libgoogle-glog-dev \
  libgflags-dev \
  libatlas-base-dev \
  libsuitesparse-dev \
  python3-catkin-tools \
  python3-rosdep \
  python3-rosinstall \
  python3-rosinstall-generator \
  python3-wstool \
  python3-pip \
  lz4

if ! grep -q "source /opt/ros/noetic/setup.bash" /root/.bashrc; then
  echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
fi

echo "=== [3/5] Building Sophus ==="
cd /tmp
rm -rf Sophus
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
sed -i 's/unit_complex_\\.real() = 1\\.;/unit_complex_.real(1.);/g' sophus/so2.cpp
sed -i 's/unit_complex_\\.imag() = 0\\.;/unit_complex_.imag(0.);/g' sophus/so2.cpp

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install
cd /tmp && rm -rf Sophus

cat << 'SOF' > /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake
find_path(Sophus_INCLUDE_DIRS sophus/se3.h PATHS /usr/local/include /usr/include)
find_library(Sophus_LIBRARIES Sophus PATHS /usr/local/lib /usr/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sophus DEFAULTMSG Sophus_LIBRARIES Sophus_INCLUDE_DIRS)
SOF

cp /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake /usr/share/cmake-3.16/Modules/FindSophus.cmake || true

echo "=== [4/5] Setting up Catkin Workspace ==="
mkdir -p /root/catkin_ws/src
cd /root/catkin_ws/src

if [ ! -d "rpg_vikit" ]; then
  git clone https://github.com/xuankuzcr/rpg_vikit.git
fi

mkdir -p /root/catkin_ws/src/rpg_vikit/vikit_common/CMakeModules
mkdir -p /root/catkin_ws/src/rpg_vikit/vikit_ros/CMakeModules
cp /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake /root/catkin_ws/src/rpg_vikit/vikit_common/CMakeModules/FindSophus.cmake
cp /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake /root/catkin_ws/src/rpg_vikit/vikit_ros/CMakeModules/FindSophus.cmake

rm -rf /root/catkin_ws/src/fast_livo
ln -s /mnt/c/Users/User/Documents/APLICATIVOS/RAVEN-SCAN-INSTA360-COLORIZATION /root/catkin_ws/src/fast_livo

echo "=== [5/5] Building Catkin Workspace ==="
source /opt/ros/noetic/setup.bash
cd /root/catkin_ws
rm -rf build devel
catkin_make -DCMAKE_BUILD_TYPE=Release

if ! grep -q "source /root/catkin_ws/devel/setup.bash" /root/.bashrc; then
  echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc
fi

echo "=== WSL2 Environment and FAST-LIVO2 Build Complete! ==="
"""

target_path = r"C:\Users\User\Documents\APLICATIVOS\RAVEN-SCAN-INSTA360-COLORIZATION\scripts\setup_wsl_env_c.sh"
with open(target_path, "w", newline="\n", encoding="utf-8") as f:
    f.write(content)
print(f"Created {target_path} successfully")
