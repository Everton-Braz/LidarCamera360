#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rvk {

struct Dispatch {
  uint32_t pipeline{};
  uint32_t record{};
  uint32_t x{};
  uint32_t y{};
  uint32_t z{};
};

class VulkanCompute {
public:
  explicit VulkanCompute(int device = -1);
  ~VulkanCompute();

  VulkanCompute(const VulkanCompute &) = delete;
  VulkanCompute &operator=(const VulkanCompute &) = delete;
  VulkanCompute(VulkanCompute &&) = delete;
  VulkanCompute &operator=(VulkanCompute &&) = delete;

  const std::string &device_name() const;

  void prepare(size_t arena_elements, const std::vector<float> &initial_arena,
               const std::vector<int32_t> &metadata,
               const std::vector<Dispatch> &dispatches,
               const std::filesystem::path &shader_dir);
  void write(size_t float_offset, const float *data, size_t count);
  std::vector<float> read(size_t offset, size_t count) const;
  void run();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rvk
