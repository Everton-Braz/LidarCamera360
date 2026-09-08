"""Build an auditable native adaptation without modifying the supplied ROS tree.

Only middleware, filesystem and compiler portability code is adapted. Estimator
and camera mathematics are compiled from the supplied FAST-LIVO2/Vikit sources.
"""
import argparse
from pathlib import Path
import re
import shutil


def replace_body(text, signature, body):
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[:opening] + '{\n' + body + '\n}' + text[end:]


def main():
    p = argparse.ArgumentParser()
    for key in ('upstream', 'output', 'sophus', 'vikit'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ('include', 'src'):
        shutil.copytree(a.upstream / name, a.output / name, dirs_exist_ok=True)
    shutil.copytree(a.sophus / 'sophus', a.output / 'sophus/sophus', dirs_exist_ok=True)
    shutil.copytree(a.vikit / 'vikit_common', a.output / 'vikit', dirs_exist_ok=True)
    for f in (a.output / 'sophus').rglob('*.cpp'):
        s = f.read_text()
        s = s.replace('unit_complex_.real() = 1.;', 'unit_complex_.real(1.);')
        s = s.replace('unit_complex_.imag() = 0.;', 'unit_complex_.imag(0.);')
        f.write_text(s)
    for f in [*(a.output / 'include').rglob('*.h'), *(a.output / 'src').glob('*.cpp')]:
        s = f.read_text(encoding='utf-8')
        s = re.sub(r'#include <(?:ros/|sensor_msgs/|nav_msgs/|tf/|visualization_msgs/|cv_bridge/|image_transport/|vikit/camera_loader)[^>]*>', '#include "native_runtime.h"', s)
        s = s.replace('#include <unistd.h>', '').replace('#include <vikit/performance_monitor.h>', '')
        s = s.replace('const bool time_list(', 'inline bool time_list(')
        s = s.replace('omp_set_num_threads(MP_PROC_NUM)', 'omp_set_num_threads(native::threads)')
        s = s.replace('int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;', 'int grid_size=0, patch_size=0, grid_n_width=0, grid_n_height=0, patch_pyrimid_level=0;')
        if f.name == 'LIVMapper.cpp':
            for signature in ('void LIVMapper::initializeSubscribersAndPublishers', 'void LIVMapper::run',
                              'void LIVMapper::standard_pcl_cbk', 'void LIVMapper::livox_pcl_cbk',
                              'void LIVMapper::publish_odometry', 'void LIVMapper::publish_path'):
                s = replace_body(s, signature, '// Offline runner owns ingestion and output; no ROS middleware.')
            s = replace_body(s, 'void LIVMapper::initializeFiles', '''
  for (const auto& dir : {"Log/pcd", "Log/result", "Log/image", "Log/Colmap/sparse/0", "Log/Colmap/images"})
    std::filesystem::create_directories(native::output_root + dir);
  if(colmap_output_en) fout_points.open(native::output_root + "Log/Colmap/sparse/0/points3D.txt");
  if(pcd_save_en) fout_lidar_pos.open(native::output_root + "Log/pcd/lidar_poses.txt");
  if(img_save_en) fout_visual_pos.open(native::output_root + "Log/image/image_poses.txt");
''')
            s = s.replace('if (!vk::camera_loader::loadFromRosNs("laserMapping", vio_manager->cam))', 'if (!native::load_camera(vio_manager->cam))')
        if f.name == 'common_lib.h':
            s = '#include "native_runtime.h"\n' + s
        s = s.replace('"Log/', '"')
        f.write_text(s, encoding='utf-8')
    (a.output / 'include/preprocess.h').write_text('''#pragma once
#include "common_lib.h"
// Offline input already contains XYZ, intensity and per-point milliseconds.
// Filtering is performed by the native reader before the estimator sees a scan.
class Preprocess { public:
 int lidar_type=5, point_filter_num=1, N_SCANS=16;
 double blind=0.4, blind_sqr=0.16;
 bool feature_enabled=false;
};
using PreprocessPtr = std::shared_ptr<Preprocess>;
''')


if __name__ == '__main__':
    main()
