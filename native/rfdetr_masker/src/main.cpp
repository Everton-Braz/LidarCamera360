// LidarCamera360 contributors. See ../LICENSE for the native worker license.
#include "ImageIo.h"
#include "RfDetrDecode.h"
#include "TensorRt.h"
#include <cuda_runtime_api.h>
#include <Windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

int run(int argc, char** argv) {
    fs::path input, output, model, cache;
    float threshold = .5f, margin = .03f;
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "rfdetr-masker --input DIR --output DIR --model MODEL.onnx|MODEL.engine\n"
                         "  [--cache-dir DIR] [--threshold 0.5] [--margin 0.03] [--device 0]\n"
                         "Recursive batch, original resolution PNGs, white=keep / black=person.\n";
            return 0;
        }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--input") input = fs::u8path(value);
        else if (arg == "--output") output = fs::u8path(value);
        else if (arg == "--model") model = fs::u8path(value);
        else if (arg == "--cache-dir") cache = fs::u8path(value);
        else if (arg == "--threshold" || arg == "--margin") {
            size_t consumed = 0;
            float parsed = std::stof(value, &consumed);
            if (consumed != value.size()) throw std::runtime_error("Invalid number: " + value);
            (arg == "--threshold" ? threshold : margin) = parsed;
        } else if (arg == "--device") {
            size_t consumed = 0;
            device = std::stoi(value, &consumed);
            if (consumed != value.size()) throw std::runtime_error("Invalid CUDA ordinal");
        } else throw std::runtime_error("Unknown argument: " + arg);
    }
    if (!fs::is_directory(input) || output.empty() || !fs::is_regular_file(model))
        throw std::runtime_error("Existing --input directory, --model and --output are required");
    if (!std::isfinite(threshold) || threshold <= 0 || threshold >= 1 ||
        !std::isfinite(margin) || std::abs(margin) > 1)
        throw std::runtime_error("Threshold must be in (0,1), margin in [-1,1]");
    input = fs::weakly_canonical(input);
    output = fs::weakly_canonical(output);
    auto relative_output = output.lexically_relative(input);
    if (relative_output.empty() || *relative_output.begin() != "..")
        throw std::runtime_error("Output must be outside the input image directory");
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(input)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("No supported images in input");
    cudaDeviceProp props{};
    auto status = cudaGetDeviceProperties(&props, device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    std::ostringstream uuid;
    uuid << "uuid:" << std::hex << std::setfill('0');
    for (unsigned char b : props.uuid.bytes) uuid << std::setw(2) << unsigned(b);
    std::cout << "GPU: " << props.name << " (" << uuid.str() << ")\n" << std::flush;
    const auto load_start = Clock::now();
    nn::TensorRt engine;
    engine.load(model.u8string(), uuid.str(), cache.u8string());
    std::cout << "MODEL_LOAD_SECONDS=" << elapsed(load_start) << " INPUT="
              << engine.width() << "x" << engine.height() << '\n' << std::flush;
    const auto batch_start = Clock::now();
    auto pending = std::async(std::launch::async, rfdetr::read_rgb_image, files.front());
    std::deque<std::future<void>> writers;
    std::vector<float> nchw;
    for (size_t i = 0; i < files.size(); ++i) {
        auto image = pending.get();
        if (i + 1 < files.size()) pending = std::async(std::launch::async, rfdetr::read_rgb_image, files[i + 1]);
        rfdetr::preprocess_rgb(image.pixels.data(), image.width, image.height,
                               engine.width(), engine.height(), nchw);
        const auto& tensors = engine.run(nchw);
        std::vector<uint8_t> mask(size_t(image.width) * image.height, 0);
        rfdetr::decode_people(tensors, image.width, image.height, threshold, margin, mask);
        for (auto& pixel : mask) pixel = pixel ? 0 : 255;
        fs::path target = output / files[i].lexically_relative(input);
        target += L".png";
        fs::create_directories(target.parent_path());
        if (writers.size() >= 2) { writers.front().get(); writers.pop_front(); }
        writers.push_back(std::async(std::launch::async,
            [target, width=image.width, height=image.height, mask=std::move(mask)] {
                auto temporary = target; temporary += L".tmp";
                rfdetr::write_gray_png(temporary, width, height, mask);
                if (!MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                    throw std::runtime_error("Cannot publish mask: " + target.u8string());
            }));
        if ((i + 1) % 10 == 0 || i + 1 == files.size())
            std::cout << "MASKS=" << i + 1 << '/' << files.size() << '\n' << std::flush;
    }
    for (auto& writer : writers) writer.get();
    const double seconds = elapsed(batch_start);
    std::cout << "BATCH_SECONDS=" << seconds << " IMAGES=" << files.size()
              << " IMAGES_PER_SECOND=" << files.size() / seconds << '\n';
    return 0;
}

int wmain(int argc, wchar_t** wide) {
    try {
        std::vector<std::string> args;
        for (int i = 0; i < argc; ++i) {
            int n = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s(n, '\0');
            WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, s.data(), n, nullptr, nullptr);
            s.pop_back(); args.push_back(std::move(s));
        }
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        return run(argc, argv.data());
    } catch (const std::exception& e) {
        std::cerr << "RF-DETR: " << e.what() << '\n';
        return 1;
    }
}
