#include "VulkanCompute.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rvk {
namespace {

void check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                             std::to_string(result));
}

struct Buffer {
  VkBuffer handle{};
  VkDeviceMemory memory{};
  VkDeviceSize size{};
  void *mapped{};
};

constexpr std::array<const char *, 3> kShaderNames{
    "ops.comp.spv", "matmul.comp.spv", "reduce.comp.spv"};
constexpr uint64_t kFenceTimeoutNanoseconds = 120000000000ull;

} // namespace

struct VulkanCompute::Impl {
  VkInstance instance{};
  VkPhysicalDevice physical{};
  VkDevice device{};
  VkQueue queue{};
  uint32_t queue_family{};
  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memory_properties{};

  VkCommandPool command_pool{};
  VkCommandBuffer command{};
  VkCommandBuffer transfer_command{};
  VkFence fence{};
  VkDescriptorSetLayout descriptor_layout{};
  VkPipelineLayout pipeline_layout{};

  std::array<VkPipeline, 3> pipelines{};
  VkDescriptorPool descriptor_pool{};
  VkDescriptorSet descriptor_set{};
  Buffer arena{};
  Buffer metadata{};
  Buffer readback{};
  size_t arena_elements{};
  bool prepared{};
  std::string name;

  explicit Impl(int requested_device) {
    if (requested_device < -1)
      throw std::invalid_argument(
          "Vulkan device index must be -1 or non-negative");
    try {
      initialize_device(requested_device);
      create_shared_layouts();
      create_command_resources();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Impl() { cleanup(); }

  void initialize_device(int requested_device) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Raven RF-DETR Vulkan executor";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    check(vkCreateInstance(&instance_info, nullptr, &instance),
          "create Vulkan instance");

    uint32_t device_count = 0;
    check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
          "enumerate Vulkan devices");
    if (!device_count)
      throw std::runtime_error("No Vulkan physical device found");
    std::vector<VkPhysicalDevice> devices(device_count);
    check(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()),
          "read Vulkan devices");

    int best_score = -1;
    for (uint32_t index = 0; index < device_count; ++index) {
      if (requested_device >= 0 &&
          index != static_cast<uint32_t>(requested_device))
        continue;
      VkPhysicalDeviceProperties candidate_properties{};
      vkGetPhysicalDeviceProperties(devices[index], &candidate_properties);
      uint32_t family_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &family_count,
                                               nullptr);
      std::vector<VkQueueFamilyProperties> families(family_count);
      vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &family_count,
                                               families.data());

      for (uint32_t family = 0; family < family_count; ++family) {
        if (!(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) ||
            !families[family].queueCount)
          continue;
        int score = 1;
        switch (candidate_properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
          score = 100;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
          score = 50;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
          score = 25;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
          score = 10;
          break;
        default:
          break;
        }
        if (score > best_score) {
          best_score = score;
          physical = devices[index];
          queue_family = family;
          properties = candidate_properties;
        }
        break;
      }
    }
    if (!physical)
      throw std::runtime_error(
          "No Vulkan compute device matches the requested device index");

    name = properties.deviceName;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    check(vkCreateDevice(physical, &device_info, nullptr, &device),
          "create Vulkan device");
    vkGetDeviceQueue(device, queue_family, 0, &queue);
  }

  void create_shared_layouts() {
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t index = 0; index < 2; ++index) {
      bindings[index].binding = index;
      bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[index].descriptorCount = 1;
      bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptor_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_info.bindingCount = 2;
    descriptor_info.pBindings = bindings;
    check(vkCreateDescriptorSetLayout(device, &descriptor_info, nullptr,
                                      &descriptor_layout),
          "create descriptor-set layout");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(uint32_t);
    VkPipelineLayoutCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_info.setLayoutCount = 1;
    pipeline_info.pSetLayouts = &descriptor_layout;
    pipeline_info.pushConstantRangeCount = 1;
    pipeline_info.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(device, &pipeline_info, nullptr,
                                 &pipeline_layout),
          "create pipeline layout");
  }

  void create_command_resources() {
    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(device, &pool_info, nullptr, &command_pool),
          "create command pool");

    VkCommandBufferAllocateInfo command_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 2;
    VkCommandBuffer buffers[2]{};
    check(vkAllocateCommandBuffers(device, &command_info, buffers),
          "allocate command buffers");
    command = buffers[0];
    transfer_command = buffers[1];

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device, &fence_info, nullptr, &fence),
          "create execution fence");
  }

  uint32_t host_memory_type(uint32_t allowed_bits) const {
    const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags preferred =
        required | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount;
         ++index) {
      const auto flags = memory_properties.memoryTypes[index].propertyFlags;
      if ((allowed_bits & (1u << index)) && (flags & preferred) == preferred)
        return index;
    }
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount;
         ++index) {
      const auto flags = memory_properties.memoryTypes[index].propertyFlags;
      if ((allowed_bits & (1u << index)) && (flags & required) == required)
        return index;
    }
    throw std::runtime_error(
        "No host-visible coherent Vulkan memory type is available");
  }

  uint32_t readback_memory_type(uint32_t allowed_bits) const {
    const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags cached =
        required | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount;
         ++index) {
      const auto flags = memory_properties.memoryTypes[index].propertyFlags;
      if ((allowed_bits & (1u << index)) && (flags & cached) == cached)
        return index;
    }
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount;
         ++index) {
      const auto flags = memory_properties.memoryTypes[index].propertyFlags;
      if ((allowed_bits & (1u << index)) && (flags & required) == required &&
          !(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        return index;
    }
    return host_memory_type(allowed_bits);
  }

  Buffer make_buffer(VkDeviceSize size, bool staging = false) {
    if (!size)
      throw std::invalid_argument(
          "Vulkan storage buffer size must be non-zero");
    Buffer buffer{};
    buffer.size = size;
    try {
      VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      info.size = size;
      info.usage = staging ? VK_BUFFER_USAGE_TRANSFER_DST_BIT
                           : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      check(vkCreateBuffer(device, &info, nullptr, &buffer.handle),
            "create storage buffer");

      VkMemoryRequirements requirements{};
      vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
      VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex =
          staging ? readback_memory_type(requirements.memoryTypeBits)
                  : host_memory_type(requirements.memoryTypeBits);
      check(vkAllocateMemory(device, &allocation, nullptr, &buffer.memory),
            "allocate storage-buffer memory");
      check(vkBindBufferMemory(device, buffer.handle, buffer.memory, 0),
            "bind storage-buffer memory");
      check(vkMapMemory(device, buffer.memory, 0, size, 0, &buffer.mapped),
            "map storage buffer");
    } catch (...) {
      destroy_buffer(buffer);
      throw;
    }
    return buffer;
  }

  void destroy_buffer(Buffer &buffer) noexcept {
    if (device && buffer.mapped && buffer.memory)
      vkUnmapMemory(device, buffer.memory);
    if (device && buffer.handle)
      vkDestroyBuffer(device, buffer.handle, nullptr);
    if (device && buffer.memory)
      vkFreeMemory(device, buffer.memory, nullptr);
    buffer = {};
  }

  static std::vector<uint32_t> load_spirv(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
      throw std::runtime_error("Cannot open Vulkan shader: " + path.u8string());
    const auto end = file.tellg();
    if (end <= 0 || static_cast<uint64_t>(end) % sizeof(uint32_t) != 0)
      throw std::runtime_error("Invalid SPIR-V file size: " + path.u8string());
    const size_t word_count = static_cast<size_t>(end) / sizeof(uint32_t);
    std::vector<uint32_t> code(word_count);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(code.data()),
                   static_cast<std::streamsize>(word_count * sizeof(uint32_t))))
      throw std::runtime_error("Cannot read Vulkan shader: " + path.u8string());
    return code;
  }

  void create_pipelines(const std::filesystem::path &shader_dir) {
    for (size_t index = 0; index < kShaderNames.size(); ++index) {
      const auto code = load_spirv(shader_dir / kShaderNames[index]);
      VkShaderModuleCreateInfo module_info{
          VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module_info.codeSize = code.size() * sizeof(uint32_t);
      module_info.pCode = code.data();
      VkShaderModule module{};
      check(vkCreateShaderModule(device, &module_info, nullptr, &module),
            "create compute shader module");
      try {
        VkComputePipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.layout = pipeline_layout;
        pipeline_info.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                       &pipeline_info, nullptr,
                                       &pipelines[index]),
              "create RF-DETR compute pipeline");
      } catch (...) {
        vkDestroyShaderModule(device, module, nullptr);
        throw;
      }
      vkDestroyShaderModule(device, module, nullptr);
    }
  }

  void create_descriptors() {
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool),
          "create descriptor pool");

    VkDescriptorSetAllocateInfo allocate{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = descriptor_pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &descriptor_layout;
    check(vkAllocateDescriptorSets(device, &allocate, &descriptor_set),
          "allocate descriptor set");

    VkDescriptorBufferInfo infos[2]{{arena.handle, 0, arena.size},
                                    {metadata.handle, 0, metadata.size}};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t index = 0; index < 2; ++index) {
      writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[index].dstSet = descriptor_set;
      writes[index].dstBinding = index;
      writes[index].descriptorCount = 1;
      writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[index].pBufferInfo = &infos[index];
    }
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
  }

  void record_graph(const std::vector<Dispatch> &dispatches) {
    check(vkResetCommandBuffer(command, 0), "reset command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(command, &begin), "begin graph command buffer");

    VkMemoryBarrier host_to_compute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host_to_compute.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host_to_compute.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &host_to_compute, 0, nullptr, 0, nullptr);

    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    for (const Dispatch &dispatch : dispatches) {
      if (dispatch.pipeline >= pipelines.size() ||
          !pipelines[dispatch.pipeline])
        throw std::invalid_argument(
            "Dispatch references an unavailable Vulkan pipeline");
      const uint32_t groups[3]{dispatch.x, dispatch.y, dispatch.z};
      for (uint32_t axis = 0; axis < 3; ++axis) {
        if (!groups[axis] ||
            groups[axis] > properties.limits.maxComputeWorkGroupCount[axis])
          throw std::invalid_argument("Vulkan dispatch group count is zero or "
                                      "exceeds the device limit");
      }
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelines[dispatch.pipeline]);
      vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(dispatch.record), &dispatch.record);
      vkCmdDispatch(command, dispatch.x, dispatch.y, dispatch.z);

      VkMemoryBarrier compute_to_compute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      compute_to_compute.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      compute_to_compute.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &compute_to_compute, 0, nullptr, 0, nullptr);
    }

    VkMemoryBarrier compute_to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    compute_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &compute_to_host, 0,
                         nullptr, 0, nullptr);
    check(vkEndCommandBuffer(command), "end graph command buffer");
  }

  void prepare(size_t element_count, const std::vector<float> &initial_arena,
               const std::vector<int32_t> &initial_metadata,
               const std::vector<Dispatch> &dispatches,
               const std::filesystem::path &shader_dir) {
    if (element_count == 0)
      throw std::invalid_argument(
          "Vulkan arena must contain at least one float");
    if (!initial_arena.empty() && initial_arena.size() != element_count)
      throw std::invalid_argument(
          "Initial arena length does not match arena_elements");
    if (element_count >
        std::numeric_limits<VkDeviceSize>::max() / sizeof(float))
      throw std::overflow_error(
          "Vulkan arena byte size overflows VkDeviceSize");

    const VkDeviceSize arena_bytes =
        static_cast<VkDeviceSize>(element_count) * sizeof(float);
    const VkDeviceSize metadata_bytes = std::max<VkDeviceSize>(
        sizeof(int32_t),
        static_cast<VkDeviceSize>(initial_metadata.size()) * sizeof(int32_t));
    if (arena_bytes > properties.limits.maxStorageBufferRange ||
        metadata_bytes > properties.limits.maxStorageBufferRange)
      throw std::length_error(
          "Vulkan arena or metadata exceeds maxStorageBufferRange");
    for (const Dispatch &dispatch : dispatches) {
      if (dispatch.pipeline >= pipelines.size())
        throw std::invalid_argument("Dispatch pipeline must be 0, 1, or 2");
      const uint32_t groups[3]{dispatch.x, dispatch.y, dispatch.z};
      for (uint32_t axis = 0; axis < 3; ++axis) {
        if (!groups[axis] ||
            groups[axis] > properties.limits.maxComputeWorkGroupCount[axis])
          throw std::invalid_argument("Vulkan dispatch group count is zero or "
                                      "exceeds the device limit");
      }
    }

    check(vkDeviceWaitIdle(device), "wait before re-preparing Vulkan executor");
    clear_prepared();
    try {
      arena = make_buffer(arena_bytes);
      metadata = make_buffer(metadata_bytes);
      std::memset(arena.mapped, 0, static_cast<size_t>(arena_bytes));
      if (!initial_arena.empty())
        std::memcpy(arena.mapped, initial_arena.data(),
                    static_cast<size_t>(arena_bytes));
      std::memset(metadata.mapped, 0, static_cast<size_t>(metadata_bytes));
      if (!initial_metadata.empty())
        std::memcpy(metadata.mapped, initial_metadata.data(),
                    initial_metadata.size() * sizeof(int32_t));

      create_pipelines(shader_dir);
      create_descriptors();
      record_graph(dispatches);
      arena_elements = element_count;
      prepared = true;
    } catch (...) {
      clear_prepared();
      throw;
    }
  }

  void require_prepared() const {
    if (!prepared)
      throw std::runtime_error(
          "Call VulkanCompute::prepare before using the executor");
  }

  void write(size_t offset, const float *data, size_t count) {
    require_prepared();
    if (offset > arena_elements || count > arena_elements - offset)
      throw std::out_of_range("Vulkan arena write is out of bounds");
    if (count && !data)
      throw std::invalid_argument("Vulkan arena write data is null");
    if (count)
      std::memcpy(static_cast<float *>(arena.mapped) + offset, data,
                  count * sizeof(float));
  }

  std::vector<float> read(size_t offset, size_t count) {
    require_prepared();
    if (offset > arena_elements || count > arena_elements - offset)
      throw std::out_of_range("Vulkan arena read is out of bounds");
    std::vector<float> output(count);
    if (!count)
      return output;

    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(float);
    if (readback.size < bytes) {
      destroy_buffer(readback);
      readback = make_buffer(bytes, true);
    }
    check(vkResetCommandBuffer(transfer_command, 0), "reset output transfer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(transfer_command, &begin),
          "begin output transfer");
    VkMemoryBarrier compute_to_transfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    compute_to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(transfer_command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &compute_to_transfer, 0, nullptr, 0, nullptr);
    VkBufferCopy region{};
    region.srcOffset = static_cast<VkDeviceSize>(offset) * sizeof(float);
    region.size = bytes;
    vkCmdCopyBuffer(transfer_command, arena.handle, readback.handle, 1,
                    &region);
    VkMemoryBarrier transfer_to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    transfer_to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    transfer_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(transfer_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &transfer_to_host, 0,
                         nullptr, 0, nullptr);
    check(vkEndCommandBuffer(transfer_command), "end output transfer");
    check(vkResetFences(device, 1, &fence), "reset output transfer fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &transfer_command;
    check(vkQueueSubmit(queue, 1, &submit, fence), "submit output transfer");
    check(vkWaitForFences(device, 1, &fence, VK_TRUE, kFenceTimeoutNanoseconds),
          "wait for output transfer");
    std::memcpy(output.data(), readback.mapped, static_cast<size_t>(bytes));
    return output;
  }

  void run() {
    require_prepared();
    check(vkResetFences(device, 1, &fence), "reset execution fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(queue, 1, &submit, fence), "submit Vulkan graph");
    check(vkWaitForFences(device, 1, &fence, VK_TRUE, kFenceTimeoutNanoseconds),
          "wait for Vulkan graph");
  }

  void clear_prepared() noexcept {
    prepared = false;
    arena_elements = 0;
    if (device) {
      for (VkPipeline &pipeline : pipelines) {
        if (pipeline)
          vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
      }
      if (descriptor_pool)
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
      descriptor_pool = VK_NULL_HANDLE;
      descriptor_set = VK_NULL_HANDLE;
      destroy_buffer(arena);
      destroy_buffer(metadata);
      destroy_buffer(readback);
    }
  }

  void cleanup() noexcept {
    if (device) {
      vkDeviceWaitIdle(device);
      clear_prepared();
      if (fence)
        vkDestroyFence(device, fence, nullptr);
      if (command_pool)
        vkDestroyCommandPool(device, command_pool, nullptr);
      if (pipeline_layout)
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
      if (descriptor_layout)
        vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (instance)
      vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
    physical = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
  }
};

VulkanCompute::VulkanCompute(int device)
    : impl_(std::make_unique<Impl>(device)) {}
VulkanCompute::~VulkanCompute() = default;

const std::string &VulkanCompute::device_name() const { return impl_->name; }

void VulkanCompute::prepare(size_t arena_elements,
                            const std::vector<float> &initial_arena,
                            const std::vector<int32_t> &metadata,
                            const std::vector<Dispatch> &dispatches,
                            const std::filesystem::path &shader_dir) {
  impl_->prepare(arena_elements, initial_arena, metadata, dispatches,
                 shader_dir);
}

void VulkanCompute::write(size_t float_offset, const float *data,
                          size_t count) {
  impl_->write(float_offset, data, count);
}

std::vector<float> VulkanCompute::read(size_t offset, size_t count) const {
  return impl_->read(offset, count);
}

void VulkanCompute::run() { impl_->run(); }

} // namespace rvk
