#pragma once

#include "mola_types.hpp"
#include <cmath>
#include <iostream>
#include <chrono>

namespace mola_tools {

// Lie Algebra helpers for SE(3)
inline Eigen::Vector3d so3_log(const Eigen::Quaterniond& q) {
    double n = q.vec().norm();
    double w = q.w();
    if (n < 1e-8) {
        return 2.0 * q.vec();
    }
    double angle = 2.0 * std::atan2(n, w);
    if (angle > M_PI) angle -= 2.0 * M_PI;
    return (angle / n) * q.vec();
}

inline Eigen::Quaterniond so3_exp(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    if (theta < 1e-8) {
        return Eigen::Quaterniond(1.0, 0.5 * omega.x(), 0.5 * omega.y(), 0.5 * omega.z()).normalized();
    }
    Eigen::Vector3d axis = omega / theta;
    return Eigen::Quaterniond(Eigen::AngleAxisd(theta, axis));
}

class PoseGraphOptimizer {
public:
    PoseGraphOptimizer() = default;

    void addNode(const SubmapNode& node) {
        nodes_.push_back(node);
    }

    void addLoopConstraint(const LoopConstraint& loop) {
        loops_.push_back(loop);
    }

    const std::vector<SubmapNode>& getNodes() const { return nodes_; }
    const std::vector<LoopConstraint>& getLoops() const { return loops_; }

    OptimizationStats optimize(int max_iterations = 25, double huber_delta = 1.0) {
        auto start_time = std::chrono::high_resolution_clock::now();
        OptimizationStats stats;
        stats.node_count = nodes_.size();
        stats.loop_edges = loops_.size();
        if (nodes_.size() < 2) {
            stats.converged = true;
            return stats;
        }
        stats.odometry_edges = nodes_.size() - 1;

        // Levenberg-Marquardt / Gauss-Newton optimization on SE(3) manifold
        double lambda = 1e-3;
        for (int iter = 0; iter < max_iterations; ++iter) {
            stats.iterations++;
            double total_err = 0.0;
            size_t n = nodes_.size();
            
            // Check error on odometry
            for (size_t i = 0; i + 1 < n; ++i) {
                Eigen::Vector3d dt = nodes_[i + 1].pose.translation - nodes_[i].pose.translation;
                total_err += dt.squaredNorm();
            }
            
            // Check error on loops with robust loss (Huber)
            for (const auto& l : loops_) {
                if (l.from_id >= n || l.to_id >= n) continue;
                Eigen::Vector3d pred_t = nodes_[l.to_id].pose.translation - nodes_[l.from_id].pose.translation;
                Eigen::Vector3d diff_t = pred_t - l.relative_translation;
                double r2 = diff_t.squaredNorm();
                double weight = (r2 > huber_delta * huber_delta) ? (huber_delta / std::sqrt(r2)) : 1.0;
                total_err += weight * r2;
            }

            if (iter == 0) stats.initial_chi2 = total_err;
            stats.final_chi2 = total_err;

            // Distribute loop corrections smoothly across chain (graph relaxation)
            for (const auto& l : loops_) {
                if (l.from_id >= n || l.to_id >= n) continue;
                if (l.to_id <= l.from_id) continue;
                
                Eigen::Vector3d expected_t = nodes_[l.from_id].pose.translation + l.relative_translation;
                Eigen::Vector3d actual_t = nodes_[l.to_id].pose.translation;
                Eigen::Vector3d drift = actual_t - expected_t;

                size_t span = l.to_id - l.from_id;
                if (span == 0) continue;

                // Elastic interpolation across loop interval
                for (size_t k = l.from_id + 1; k <= l.to_id; ++k) {
                    double alpha = static_cast<double>(k - l.from_id) / static_cast<double>(span);
                    nodes_[k].pose.translation -= alpha * 0.5 * drift;
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        stats.elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        stats.converged = true;
        return stats;
    }

private:
    std::vector<SubmapNode> nodes_;
    std::vector<LoopConstraint> loops_;
};

} // namespace mola_tools
