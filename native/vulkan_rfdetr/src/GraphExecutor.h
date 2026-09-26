#pragma once
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
namespace rvk {
struct Output {
  std::string name;
  std::vector<int> shape;
  std::vector<float> data;
};
// Fixed-shape float32 graph. Preparation is offline; all runtime tensor math is
// Vulkan.
class GraphExecutor {
public:
  GraphExecutor(const std::filesystem::path &model,
                const std::filesystem::path &shaders, int device = -1);
  ~GraphExecutor();
  GraphExecutor(const GraphExecutor &) = delete;
  GraphExecutor &operator=(const GraphExecutor &) = delete;
  std::vector<Output> run(const std::vector<float> &input);
  const std::vector<int> &input_shape() const;
  const std::string &device_name() const;
  size_t arena_bytes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace rvk
