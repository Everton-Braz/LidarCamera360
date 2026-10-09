#pragma once

#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace mola_tools {

struct Pose3D {
    double timestamp = 0.0;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();

    Eigen::Matrix4d toMatrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = rotation.toRotationMatrix();
        T.block<3, 1>(0, 3) = translation;
        return T;
    }

    static Pose3D fromMatrix(double ts, const Eigen::Matrix4d& T) {
        Pose3D p;
        p.timestamp = ts;
        p.translation = T.block<3, 1>(0, 3);
        p.rotation = Eigen::Quaterniond(T.block<3, 3>(0, 0));
        p.rotation.normalize();
        return p;
    }
};

struct SubmapNode {
    size_t id = 0;
    double timestamp_start = 0.0;
    double timestamp_end = 0.0;
    Pose3D pose;
    std::string pcd_path;
    size_t point_count = 0;
    Eigen::Vector3d min_bound = Eigen::Vector3d::Zero();
    Eigen::Vector3d max_bound = Eigen::Vector3d::Zero();
};

struct LoopConstraint {
    size_t from_id = 0;
    size_t to_id = 0;
    Eigen::Vector3d relative_translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond relative_rotation = Eigen::Quaterniond::Identity();
    Eigen::Matrix<double, 6, 6> information_matrix = Eigen::Matrix<double, 6, 6>::Identity();
    double confidence = 1.0;
    double inlier_ratio = 1.0;
};

struct OptimizationStats {
    size_t node_count = 0;
    size_t odometry_edges = 0;
    size_t loop_edges = 0;
    double initial_chi2 = 0.0;
    double final_chi2 = 0.0;
    size_t iterations = 0;
    bool converged = false;
    double elapsed_ms = 0.0;
};

} // namespace mola_tools
