// Derived from Spirula Studio's TensorRT inference wrapper.
// Copyright (C) Spirula Studio contributors. Licensed under GPL-3.0;
// see the adjacent LICENSE file. Adapted for a headless standalone batch tool.
#include "TensorRt.h"

#include <stdexcept>

#ifdef SS_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace nn {
#ifdef SS_TENSORRT
namespace {
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Library {
    void* handle = nullptr;
    explicit Library(const char* name) {
#ifdef _WIN32
        handle = LoadLibraryA(name);
#else
        handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle) throw std::runtime_error(std::string("Cannot load TensorRT library: ") + name);
    }
    ~Library() {
#ifdef _WIN32
        if (handle) FreeLibrary(static_cast<HMODULE>(handle));
#else
        if (handle) dlclose(handle);
#endif
    }
    template<class T> T symbol(const char* name) {
#ifdef _WIN32
        auto value = GetProcAddress(static_cast<HMODULE>(handle), name);
#else
        auto value = dlsym(handle, name);
#endif
        if (!value) throw std::runtime_error(std::string("Missing TensorRT symbol: ") + name);
        return reinterpret_cast<T>(value);
    }
};

struct Logger : nvinfer1::ILogger {
    std::string error;
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) {
            try { error = message; } catch (...) {}
        }
    }
};

std::vector<char> read_file(const std::string& path) {
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open model: " + path);
    const auto size = file.tellg();
    if (size <= 0 || size > (int64_t(4) << 30)) throw std::runtime_error("Invalid model size");
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(bytes.data(), size)) throw std::runtime_error("Cannot read model: " + path);
    return bytes;
}

std::string fingerprint(const std::vector<char>& bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char b : bytes) { hash ^= b; hash *= 1099511628211ull; }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

std::string uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "uuid:" << std::hex << std::setfill('0');
    for (unsigned char b : uuid.bytes) out << std::setw(2) << unsigned(b);
    return out.str();
}
}

struct TensorRt::Impl {
#ifdef _WIN32
    Library library{"nvinfer_10.dll"};
#else
    Library library{"libnvinfer.so.10"};
#endif
    Logger logger;
    int device = -1;
    cudaStream_t stream = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<void*> buffers;
    std::vector<TrtTensor> outputs;
    std::string input_name;
    int w = 0, h = 0;
    size_t input_count = 0;

    ~Impl() {
        if (device >= 0) cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        context.reset();
        for (void* p : buffers) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }

    void build(const std::string& path, const std::vector<char>& bytes,
               const std::filesystem::path& cache, int dynamic_size) {
#ifdef _WIN32
        Library parser_library("nvonnxparser_10.dll");
#else
        Library parser_library("libnvonnxparser.so.10");
#endif
        // The parser module must outlive its parser, network and builder.
        using MakeBuilder = void* (*)(void*, int);
        std::unique_ptr<nvinfer1::IBuilder> builder(static_cast<nvinfer1::IBuilder*>(
            library.symbol<MakeBuilder>("createInferBuilder_INTERNAL")(&logger, NV_TENSORRT_VERSION)));
        if (!builder) throw std::runtime_error("Cannot create TensorRT builder: " + logger.error);
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
        using MakeParser = void* (*)(void*, void*, int);
        std::unique_ptr<nvonnxparser::IParser> parser(static_cast<nvonnxparser::IParser*>(
            parser_library.symbol<MakeParser>("createNvOnnxParser_INTERNAL")(
                network.get(), &logger, NV_ONNX_PARSER_VERSION)));
        if (!parser || !parser->parse(bytes.data(), bytes.size(), path.c_str()))
            throw std::runtime_error("ONNX import failed: " + logger.error);
        if (network->getNbInputs() != 1) throw std::runtime_error("Expected one RGB model input");
        auto* input = network->getInput(0);
        auto dims = input->getDimensions();
        if (dims.nbDims != 4) throw std::runtime_error("Expected NCHW model input");
        const int expected[] = {1, 3, dynamic_size, dynamic_size};
        bool dynamic = false;
        for (int i = 0; i < 4; ++i) {
            if (dims.d[i] == -1) dynamic = true;
            else if (dims.d[i] != expected[i] && i < 2)
                throw std::runtime_error("Expected batch-one RGB model");
        }
        if ((dims.d[0] != 1 && dims.d[0] != -1) || (dims.d[1] != 3 && dims.d[1] != -1))
            throw std::runtime_error("Expected batch-one RGB model");
        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (dynamic) {
            auto* profile = builder->createOptimizationProfile();
            const nvinfer1::Dims4 selected(1, 3, dynamic_size, dynamic_size);
            if (!profile || !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, selected) ||
                !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, selected) ||
                !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, selected) ||
                config->addOptimizationProfile(profile) < 0)
                throw std::runtime_error("Cannot configure dynamic TensorRT input");
        }
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t(2) << 30);
        if (!dynamic) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            // FP32 normalization avoids transformer overflow in otherwise FP16 engines.
            config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
            for (int i = 0; i < network->getNbLayers(); ++i) {
                auto* layer = network->getLayer(i);
                if (layer->getType() == nvinfer1::LayerType::kNORMALIZATION)
                    layer->setPrecision(nvinfer1::DataType::kFLOAT);
            }
        }
        std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        if (!plan) throw std::runtime_error("TensorRT engine build failed: " + logger.error);
        engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine) throw std::runtime_error("Cannot load built TensorRT engine: " + logger.error);
        const auto tmp = cache.u8string() + "." + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".tmp";
        std::ofstream file(std::filesystem::u8path(tmp), std::ios::binary);
        if (file && file.write(static_cast<const char*>(plan->data()), plan->size())) {
            file.close();
            std::error_code ec;
            std::filesystem::rename(std::filesystem::u8path(tmp), cache, ec);
            if (ec) std::filesystem::remove(std::filesystem::u8path(tmp), ec);
        }
    }
};

TensorRt::TensorRt() = default;
TensorRt::~TensorRt() = default;

void TensorRt::load(const std::string& path, const std::string& device_uuid,
                    const std::string& cache_dir, int dynamic_size) {
    if (dynamic_size < 128 || dynamic_size > 8192)
        throw std::runtime_error("Invalid dynamic TensorRT input size");
    auto next = std::make_unique<Impl>();
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp props{};
        cuda_check(cudaGetDeviceProperties(&props, i));
        if (uuid_string(props.uuid) == device_uuid) { next->device = i; break; }
    }
    if (next->device < 0) throw std::runtime_error("Selected GPU is not a CUDA device");
    cuda_check(cudaSetDevice(next->device));
    using MakeRuntime = void* (*)(void*, int);
    next->runtime.reset(static_cast<nvinfer1::IRuntime*>(
        next->library.symbol<MakeRuntime>("createInferRuntime_INTERNAL")(&next->logger, NV_TENSORRT_VERSION)));
    if (!next->runtime) throw std::runtime_error("Cannot create TensorRT runtime");
    const auto bytes = read_file(path);
    if (std::filesystem::u8path(path).extension() == ".onnx") {
        int driver = 0;
        cuda_check(cudaDriverGetVersion(&driver));
        auto cache_root = cache_dir.empty() ? std::filesystem::u8path(path).parent_path()
                                            : std::filesystem::u8path(cache_dir);
        if (cache_root.empty()) cache_root = std::filesystem::current_path();
        std::filesystem::create_directories(cache_root);
        const auto cache = cache_root / (std::filesystem::u8path(path).filename().u8string() + "." + fingerprint(bytes) + "." +
            device_uuid.substr(5) + "." + std::to_string(NV_TENSORRT_VERSION) + "." +
            std::to_string(driver) + (dynamic_size == 512 ? "" : ".s" + std::to_string(dynamic_size)) +
            (dynamic_size == 512 ? ".fp16-v1.engine" : ".fp32-v3.engine"));
        if (std::filesystem::exists(cache)) {
            try {
                const auto plan = read_file(cache.u8string());
                next->engine.reset(next->runtime->deserializeCudaEngine(plan.data(), plan.size()));
            } catch (const std::exception&) {}
        }
        if (!next->engine) next->build(path, bytes, cache, dynamic_size);
    } else {
        next->engine.reset(next->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    }
    if (!next->engine) throw std::runtime_error("Incompatible TensorRT engine: " + next->logger.error);
    next->context.reset(next->engine->createExecutionContext());
    if (!next->context) throw std::runtime_error("Cannot allocate TensorRT execution context");
    for (int i = 0; i < next->engine->getNbIOTensors(); ++i) {
        const char* name = next->engine->getIOTensorName(i);
        if (next->engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
        const auto dims = next->context->getTensorShape(name);
        if (dims.nbDims == 4 && (dims.d[0] < 0 || dims.d[2] < 0 || dims.d[3] < 0) &&
            !next->context->setInputShape(name, nvinfer1::Dims4(1, 3, dynamic_size, dynamic_size)))
            throw std::runtime_error("Cannot select dynamic TensorRT input shape");
    }
    cuda_check(cudaStreamCreate(&next->stream));
    int inputs = 0;
    std::vector<void*> output_buffers;
    void* input_buffer = nullptr;
    for (int i = 0; i < next->engine->getNbIOTensors(); ++i) {
        const char* name = next->engine->getIOTensorName(i);
        const auto dims = next->context->getTensorShape(name);
        if (next->engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
            next->engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
            next->engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR)
            throw std::runtime_error("Model I/O must be linear device float32 tensors");
        TrtTensor tensor;
        tensor.name = name;
        size_t elements = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] <= 0 || dims.d[j] > 65536 || elements > (size_t(1) << 28) / dims.d[j])
                throw std::runtime_error("Model has unresolved or excessive tensor dimensions");
            tensor.shape.push_back(static_cast<int>(dims.d[j]));
            elements *= dims.d[j];
        }
        void* buffer = nullptr;
        cuda_check(cudaMalloc(&buffer, elements * sizeof(float)));
        next->buffers.push_back(buffer);
        if (!next->context->setTensorAddress(name, buffer)) throw std::runtime_error("Cannot bind model tensor");
        if (next->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            ++inputs;
            if (tensor.shape.size() != 4 || tensor.shape[0] != 1 || tensor.shape[1] != 3)
                throw std::runtime_error("Expected static [1,3,H,W] model input");
            next->h = tensor.shape[2]; next->w = tensor.shape[3];
            next->input_count = elements;
            next->input_name = name;
            input_buffer = buffer;
        } else {
            tensor.data.resize(elements);
            next->outputs.push_back(std::move(tensor));
            output_buffers.push_back(buffer);
        }
    }
    if (inputs != 1) throw std::runtime_error("Expected one model input");
    next->buffers.clear();
    next->buffers.push_back(input_buffer);
    next->buffers.insert(next->buffers.end(), output_buffers.begin(), output_buffers.end());
    impl_ = std::move(next);
}

int TensorRt::width() const { return impl_ ? impl_->w : 0; }
int TensorRt::height() const { return impl_ ? impl_->h : 0; }
const std::vector<TrtTensor>& TensorRt::outputs() const {
    if (!impl_) throw std::runtime_error("TensorRT model is not loaded");
    return impl_->outputs;
}

const std::vector<TrtTensor>& TensorRt::run(const std::vector<float>& input) {
    if (!impl_ || input.size() != impl_->input_count) throw std::runtime_error("Invalid TensorRT input");
    auto& s = *impl_;
    cuda_check(cudaSetDevice(s.device));
    cuda_check(cudaMemcpyAsync(s.buffers[0], input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    if (!s.context->enqueueV3(s.stream)) throw std::runtime_error("TensorRT inference failed: " + s.logger.error);
    for (size_t i = 0; i < s.outputs.size(); ++i)
        cuda_check(cudaMemcpyAsync(s.outputs[i].data.data(), s.buffers[i + 1],
            s.outputs[i].data.size() * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    cuda_check(cudaStreamSynchronize(s.stream));
    return s.outputs;
}
#else
struct TensorRt::Impl {};
TensorRt::TensorRt() = default;
TensorRt::~TensorRt() = default;
void TensorRt::load(const std::string&, const std::string&, const std::string&, int) {
    throw std::runtime_error("This build has no TensorRT support; rebuild with SS_ENABLE_TENSORRT=ON");
}
int TensorRt::width() const { return 0; }
int TensorRt::height() const { return 0; }
const std::vector<TrtTensor>& TensorRt::outputs() const {
    throw std::runtime_error("TensorRT is unavailable");
}
const std::vector<TrtTensor>& TensorRt::run(const std::vector<float>&) {
    throw std::runtime_error("TensorRT is unavailable");
}
#endif
}
