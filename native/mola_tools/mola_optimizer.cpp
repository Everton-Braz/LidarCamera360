#include "mola_types.hpp"
#include "mola_pose_graph.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>

using namespace mola_tools;

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --trajectory <file>         Input trajectory TXT/CSV (timestamp x y z qx qy qz qw)\n"
              << "  --scan-ranges <file>        Input scan ranges CSV\n"
              << "  --output-trajectory <file>  Output optimized trajectory\n"
              << "  --output-stats <file>       Output optimization stats JSON\n"
              << "  --submap-seconds <sec>      Submap time duration (default: 10.0)\n"
              << "  --max-loop-distance <dist>  Max distance for loop candidate search (default: 8.0m)\n";
}

std::vector<Pose3D> loadTrajectory(const std::string& path) {
    std::vector<Pose3D> poses;
    std::ifstream in(path);
    if (!in.is_open()) return poses;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 't') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream ss(line);
        double ts, x, y, z, qx, qy, qz, qw;
        if (ss >> ts >> x >> y >> z >> qx >> qy >> qz >> qw) {
            Pose3D p;
            p.timestamp = ts;
            p.translation = Eigen::Vector3d(x, y, z);
            p.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
            poses.push_back(p);
        }
    }
    return poses;
}

bool saveTrajectory(const std::string& path, const std::vector<Pose3D>& poses) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "# timestamp x y z qx qy qz qw\n";
    out << std::fixed << std::setprecision(7);
    for (const auto& p : poses) {
        out << p.timestamp << " "
            << p.translation.x() << " "
            << p.translation.y() << " "
            << p.translation.z() << " "
            << p.rotation.x() << " "
            << p.rotation.y() << " "
            << p.rotation.z() << " "
            << p.rotation.w() << "\n";
    }
    return true;
}

int main(int argc, char** argv) {
    std::string traj_in, scan_ranges_in, traj_out, stats_out;
    double submap_seconds = 10.0;
    double max_loop_dist = 8.0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trajectory" && i + 1 < argc) traj_in = argv[++i];
        else if (arg == "--scan-ranges" && i + 1 < argc) scan_ranges_in = argv[++i];
        else if (arg == "--output-trajectory" && i + 1 < argc) traj_out = argv[++i];
        else if (arg == "--output-stats" && i + 1 < argc) stats_out = argv[++i];
        else if (arg == "--submap-seconds" && i + 1 < argc) submap_seconds = std::stod(argv[++i]);
        else if (arg == "--max-loop-distance" && i + 1 < argc) max_loop_dist = std::stod(argv[++i]);
        else if (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
    }

    if (traj_in.empty() || traj_out.empty()) {
        std::cerr << "[-] Error: missing --trajectory or --output-trajectory\n";
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "[*] MOLA Standalone Pose Graph Optimizer v0.1\n";
    std::cout << "[*] Loading trajectory: " << traj_in << "\n";
    auto poses = loadTrajectory(traj_in);
    if (poses.empty()) {
        std::cerr << "[-] Error: failed to load poses or trajectory empty\n";
        return 1;
    }
    std::cout << "[+] Loaded " << poses.size() << " poses\n";

    // Build submap keyframe nodes
    PoseGraphOptimizer pgo;
    std::vector<size_t> pose_to_node(poses.size(), 0);
    size_t current_node_id = 0;
    double submap_start_t = poses[0].timestamp;

    SubmapNode current_node;
    current_node.id = 0;
    current_node.timestamp_start = submap_start_t;
    current_node.pose = poses[0];
    pgo.addNode(current_node);

    for (size_t i = 0; i < poses.size(); ++i) {
        if (poses[i].timestamp - submap_start_t >= submap_seconds) {
            current_node_id++;
            submap_start_t = poses[i].timestamp;
            SubmapNode next_node;
            next_node.id = current_node_id;
            next_node.timestamp_start = submap_start_t;
            next_node.pose = poses[i];
            pgo.addNode(next_node);
        }
        pose_to_node[i] = current_node_id;
    }

    auto nodes = pgo.getNodes();
    std::cout << "[+] Partitioned into " << nodes.size() << " submap keyframe nodes\n";

    // Find loop candidate pairs
    size_t loops_found = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = i + 3; j < nodes.size(); ++j) {
            double time_gap = nodes[j].pose.timestamp - nodes[i].pose.timestamp;
            if (time_gap < 20.0) continue; // Minimum revisit temporal distance

            double dist = (nodes[j].pose.translation - nodes[i].pose.translation).norm();
            if (dist < max_loop_dist) {
                LoopConstraint loop;
                loop.from_id = i;
                loop.to_id = j;
                loop.relative_translation = nodes[j].pose.translation - nodes[i].pose.translation;
                loop.relative_rotation = nodes[i].pose.rotation.inverse() * nodes[j].pose.rotation;
                loop.confidence = 1.0 / (1.0 + dist);
                pgo.addLoopConstraint(loop);
                loops_found++;
            }
        }
    }
    std::cout << "[+] Detected " << loops_found << " spatial revisit / loop candidates\n";

    auto stats = pgo.optimize(20, 1.5);
    std::cout << "[+] Pose graph optimization finished in " << stats.elapsed_ms << " ms\n"
              << "    Initial Chi2: " << stats.initial_chi2 << ", Final Chi2: " << stats.final_chi2 << "\n";

    // Interpolate corrections back onto continuous high-rate trajectory
    const auto& opt_nodes = pgo.getNodes();
    for (size_t i = 0; i < poses.size(); ++i) {
        size_t nid = pose_to_node[i];
        if (nid < opt_nodes.size() && nid < nodes.size()) {
            Eigen::Vector3d delta_t = opt_nodes[nid].pose.translation - nodes[nid].pose.translation;
            poses[i].translation += delta_t;
        }
    }

    if (!saveTrajectory(traj_out, poses)) {
        std::cerr << "[-] Error saving optimized trajectory to " << traj_out << "\n";
        return 1;
    }
    std::cout << "[+] Saved optimized trajectory to: " << traj_out << "\n";

    if (!stats_out.empty()) {
        std::ofstream so(stats_out);
        if (so.is_open()) {
            so << "{\n"
               << "  \"status\": \"accepted\",\n"
               << "  \"optimizer\": \"mola_standalone_pgo\",\n"
               << "  \"node_count\": " << stats.node_count << ",\n"
               << "  \"odometry_edges\": " << stats.odometry_edges << ",\n"
               << "  \"loop_edges\": " << stats.loop_edges << ",\n"
               << "  \"initial_chi2\": " << stats.initial_chi2 << ",\n"
               << "  \"final_chi2\": " << stats.final_chi2 << ",\n"
               << "  \"elapsed_ms\": " << stats.elapsed_ms << ",\n"
               << "  \"iterations\": " << stats.iterations << "\n"
               << "}\n";
        }
    }

    return 0;
}
