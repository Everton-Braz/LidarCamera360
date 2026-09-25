#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

static void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with VkResult " + std::to_string(result));
}

struct Buffer {
    VkBuffer handle{};
    VkDeviceMemory memory{};
    VkDeviceSize size{};
    void* mapped{};
};

class VulkanSoftmax {
public:
    explicit VulkanSoftmax(int requested_device = -1) { init_device(requested_device); }
    VulkanSoftmax(const VulkanSoftmax&) = delete;
    VulkanSoftmax& operator=(const VulkanSoftmax&) = delete;

    ~VulkanSoftmax() {
        if (device_) {
            vkDeviceWaitIdle(device_);
            if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
            if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
            if (descriptor_layout_) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
            if (descriptor_pool_) vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
            if (fence_) vkDestroyFence(device_, fence_, nullptr);
            if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
            for (auto& buffer : buffers_) destroy_buffer(buffer);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) vkDestroyInstance(instance_, nullptr);
    }

    const std::string& device_name() const { return device_name_; }

    void load_shader(const fs::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open Vulkan shader: " + path.u8string());
        const auto bytes = file.tellg();
        if (bytes <= 0 || bytes % 4 != 0) throw std::runtime_error("Invalid SPIR-V shader size");
        std::vector<uint32_t> code(static_cast<size_t>(bytes) / 4);
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(code.data()), bytes))
            throw std::runtime_error("Cannot read Vulkan shader");

        VkDescriptorSetLayoutBinding bindings[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = 2;
        layout_info.pBindings = bindings;
        check(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_layout_), "create descriptor layout");

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = sizeof(uint32_t) * 2;
        VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_layout_;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push;
        check(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_), "create pipeline layout");

        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = static_cast<size_t>(bytes);
        module_info.pCode = code.data();
        VkShaderModule module{};
        check(vkCreateShaderModule(device_, &module_info, nullptr, &module), "create shader module");
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.layout = pipeline_layout_;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        const VkResult pipeline_result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                                                   &pipeline_info, nullptr, &pipeline_);
        vkDestroyShaderModule(device_, module, nullptr);
        check(pipeline_result, "create softmax pipeline");

        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 32;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_), "create descriptor pool");
    }

    std::vector<float> run(const std::vector<float>& input, uint32_t rows, uint32_t columns) {
        if (!pipeline_) throw std::runtime_error("Load the softmax shader before dispatch");
        if (!rows || !columns || input.size() != static_cast<size_t>(rows) * columns)
            throw std::runtime_error("Softmax input dimensions do not match the buffer");
        const VkDeviceSize bytes = VkDeviceSize(input.size()) * sizeof(float);
        if (bytes > properties_.limits.maxStorageBufferRange)
            throw std::runtime_error("Softmax tensor exceeds this device's storage-buffer limit");

        const size_t input_id = make_buffer(bytes);
        const size_t output_id = make_buffer(bytes);
        auto& input_buffer = buffers_[input_id];
        auto& output_buffer = buffers_[output_id];
        std::memcpy(input_buffer.mapped, input.data(), static_cast<size_t>(bytes));

        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = descriptor_pool_;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &descriptor_layout_;
        VkDescriptorSet set{};
        check(vkAllocateDescriptorSets(device_, &allocate, &set), "allocate descriptor set");
        VkDescriptorBufferInfo infos[2]{{input_buffer.handle, 0, bytes}, {output_buffer.handle, 0, bytes}};
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

        check(vkResetCommandBuffer(command_, 0), "reset command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command_, &begin), "begin command buffer");
        VkMemoryBarrier host_to_compute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_to_compute.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        host_to_compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &host_to_compute, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &set, 0, nullptr);
        const uint32_t shape[2]{rows, columns};
        vkCmdPushConstants(command_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shape), shape);
        vkCmdDispatch(command_, rows, 1, 1);
        VkMemoryBarrier compute_to_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        compute_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        compute_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &compute_to_host, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(command_), "end command buffer");
        check(vkResetFences(device_, 1, &fence_), "reset fence");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_;
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "submit softmax dispatch");
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, 120000000000ull), "wait for softmax dispatch");

        std::vector<float> output(input.size());
        std::memcpy(output.data(), output_buffer.mapped, static_cast<size_t>(bytes));
        return output;
    }

private:
    VkInstance instance_{};
    VkPhysicalDevice physical_{};
    VkDevice device_{};
    VkQueue queue_{};
    uint32_t queue_family_{};
    VkPhysicalDeviceProperties properties_{};
    VkCommandPool command_pool_{};
    VkCommandBuffer command_{};
    VkFence fence_{};
    VkDescriptorSetLayout descriptor_layout_{};
    VkPipelineLayout pipeline_layout_{};
    VkPipeline pipeline_{};
    VkDescriptorPool descriptor_pool_{};
    std::vector<Buffer> buffers_;
    std::string device_name_;

    void init_device(int requested_device) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Raven RF-DETR Vulkan ops prototype";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &app;
        check(vkCreateInstance(&instance_info, nullptr, &instance_), "create Vulkan instance");

        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "enumerate Vulkan devices");
        if (!count) throw std::runtime_error("No Vulkan physical device found");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "read Vulkan devices");
        int best = -1;
        for (uint32_t i = 0; i < count; ++i) {
            if (requested_device >= 0 && i != static_cast<uint32_t>(requested_device)) continue;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(devices[i], &properties);
            uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, families.data());
            for (uint32_t q = 0; q < family_count; ++q) {
                if (!(families[q].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
                const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 100 : 1;
                if (score > best) {
                    best = score;
                    physical_ = devices[i];
                    queue_family_ = q;
                    properties_ = properties;
                }
                break;
            }
        }
        if (!physical_) throw std::runtime_error("No Vulkan compute device matches the requested index");
        device_name_ = properties_.deviceName;
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_family_;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        check(vkCreateDevice(physical_, &device_info, nullptr, &device_), "create Vulkan device");
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.queueFamilyIndex = queue_family_;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_), "create command pool");
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = command_pool_;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device_, &command_info, &command_), "allocate command buffer");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device_, &fence_info, nullptr, &fence_), "create fence");
    }

    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) const {
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &memory);
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
        throw std::runtime_error("No Vulkan memory type supports host-visible coherent buffers");
    }

    size_t make_buffer(VkDeviceSize size) {
        Buffer buffer{};
        buffer.size = size;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &info, nullptr, &buffer.handle), "create storage buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type(requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device_, &allocation, nullptr, &buffer.memory), "allocate storage buffer memory");
        check(vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0), "bind storage buffer memory");
        check(vkMapMemory(device_, buffer.memory, 0, size, 0, &buffer.mapped), "map storage buffer");
        buffers_.push_back(buffer);
        return buffers_.size() - 1;
    }

    void destroy_buffer(Buffer& buffer) {
        if (buffer.mapped) vkUnmapMemory(device_, buffer.memory);
        if (buffer.handle) vkDestroyBuffer(device_, buffer.handle, nullptr);
        if (buffer.memory) vkFreeMemory(device_, buffer.memory, nullptr);
        buffer = {};
    }
};

static void run_self_test(VulkanSoftmax& gpu) {
    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> values(-12.0f, 12.0f);
    for (const auto [rows, columns] : {std::pair<uint32_t, uint32_t>{1, 1}, {2, 7}, {3, 513}}) {
        std::vector<float> input(static_cast<size_t>(rows) * columns);
        for (float& value : input) value = values(generator);
        for (uint32_t row = 0; row < rows; ++row) input[static_cast<size_t>(row) * columns] += 40.0f;

        const auto actual = gpu.run(input, rows, columns);
        for (uint32_t row = 0; row < rows; ++row) {
            const size_t base = static_cast<size_t>(row) * columns;
            const float maximum = *std::max_element(input.begin() + base, input.begin() + base + columns);
            double sum = 0.0;
            for (uint32_t column = 0; column < columns; ++column)
                sum += std::exp(static_cast<double>(input[base + column] - maximum));
            double actual_sum = 0.0;
            for (uint32_t column = 0; column < columns; ++column) {
                const double expected = std::exp(static_cast<double>(input[base + column] - maximum)) / sum;
                const double observed = actual[base + column];
                if (!std::isfinite(observed) || std::abs(observed - expected) > 2.0e-6)
                    throw std::runtime_error("Vulkan softmax parity failed at row " + std::to_string(row) +
                                            ", column " + std::to_string(column));
                actual_sum += observed;
            }
            if (std::abs(actual_sum - 1.0) > 2.0e-6)
                throw std::runtime_error("Vulkan softmax row does not sum to one");
        }
        std::cout << "softmax parity passed for " << rows << " x " << columns << std::endl;
    }
}

static int run(int argc, char** argv) {
    bool probe = false;
    int device = -1;
    fs::path shader = fs::absolute(fs::u8path(argv[0])).parent_path() / "shaders/softmax.comp.spv";
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            std::cout << "vulkan_rfdetr_ops_test [--probe] [--self-test] [--device N] [--shader PATH]\n";
            return 0;
        }
        if (option == "--probe") { probe = true; continue; }
        if ((option == "--device" || option == "--shader") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (option == "--device") device = std::stoi(value);
            else shader = fs::u8path(value);
            continue;
        }
        if (option == "--self-test") continue;
        throw std::runtime_error("Unknown or incomplete option: " + option);
    }
    VulkanSoftmax gpu(device);
    std::cout << "Vulkan device: " << gpu.device_name() << std::endl;
    if (!probe) {
        gpu.load_shader(shader);
        run_self_test(gpu);
    }
    return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** wide_argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        const int size = WideCharToMultiByte(CP_UTF8, 0, wide_argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string value(static_cast<size_t>(size), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide_argv[i], -1, value.data(), size, nullptr, nullptr);
        value.pop_back();
        args.push_back(std::move(value));
    }
    std::vector<char*> argv;
    for (auto& value : args) argv.push_back(value.data());
    try { return run(argc, argv.data()); }
    catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
}
#else
int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
}
#endif
