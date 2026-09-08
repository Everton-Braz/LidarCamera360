#!/usr/bin/env bash
set -e

echo "=== [1/6] Configuring APT & ROS Repositories ==="
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --fix-missing curl gnupg2 lsb-release ca-certificates

# Add ROS Noetic key and repo
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - || \
  apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

echo "=== [2/6] Updating APT and installing ROS Noetic & Dependencies ==="
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
  python3-wstool

# Source ROS in bashrc if not present
if ! grep -q "source /opt/ros/noetic/setup.bash" /root/.bashrc; then
  echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
fi

echo "=== [3/6] Building and Installing Sophus ==="
cd /tmp
rm -rf Sophus sophus_test /usr/local/lib/cmake/Sophus
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
# Fix std::complex assignment for GCC 9+
sed -i 's/unit_complex_\.real() = 1\.;/unit_complex_.real(1.);/g' sophus/so2.cpp
sed -i 's/unit_complex_\.imag() = 0\.;/unit_complex_.imag(0.);/g' sophus/so2.cpp

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install
cd /tmp && rm -rf Sophus

# Install FindSophus.cmake globally to ROS and CMake modules
cat << 'EOF' > /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake
find_path(Sophus_INCLUDE_DIRS sophus/se3.h PATHS /usr/local/include /usr/include)
find_library(Sophus_LIBRARIES Sophus PATHS /usr/local/lib /usr/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sophus DEFAULT_MSG Sophus_LIBRARIES Sophus_INCLUDE_DIRS)
EOF

cp /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake /usr/share/cmake-3.16/Modules/FindSophus.cmake || true

echo "=== [4/6] Setting up Catkin Workspace ==="
mkdir -p /root/catkin_ws/src
cd /root/catkin_ws/src

if [ ! -d "rpg_vikit" ]; then
  git clone https://github.com/xuankuzcr/rpg_vikit.git
fi

mkdir -p /root/catkin_ws/src/rpg_vikit/vikit_common/CMakeModules
mkdir -p /root/catkin_ws/src/rpg_vikit/vikit_ros/CMakeModules
cp /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake /root/catkin_ws/src/rpg_vikit/vikit_common/CMakeModules/FindSophus.cmake
cp /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake /root/catkin_ws/src/rpg_vikit/vikit_ros/CMakeModules/FindSophus.cmake

# Link or copy FAST-LIVO2
if [ -d "/mnt/d/APLICATIVOS/FAST-LIVO2" ]; then
  rm -rf /root/catkin_ws/src/fast_livo
  ln -s /mnt/d/APLICATIVOS/FAST-LIVO2 /root/catkin_ws/src/fast_livo
fi

echo "=== [5/6] Building Catkin Workspace ==="
source /opt/ros/noetic/setup.bash
cd /root/catkin_ws
rm -rf build devel
catkin_make -DCMAKE_BUILD_TYPE=Release

if ! grep -q "source /root/catkin_ws/devel/setup.bash" /root/.bashrc; then
  echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc
fi

echo "=== [6/6] Setup Completed Successfully! ==="
