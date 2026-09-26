#include "GraphExecutor.h"
#include "ImageIo.h"
#include "RfDetrDecode.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <windows.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void check(bool b, const std::string &s) {
  if (!b)
    throw std::runtime_error(s);
}
fs::path shader_directory() {
  std::vector<wchar_t> path(32768);
  auto n = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
  check(n > 0 && n < path.size(), "Cannot locate executable");
  return fs::path(std::wstring(path.data(), n)).parent_path() / "shaders";
}
size_t elements(const std::vector<int> &shape) {
  size_t n = 1;
  for (int d : shape)
    n *= size_t(d);
  return n;
}
std::vector<float> read_tensor(const fs::path &path, size_t count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  check(bool(f) && f.tellg() == std::streamoff(count * 4),
        "Tensor input file size mismatch");
  std::vector<float> x(count);
  f.seekg(0);
  check(bool(f.read(reinterpret_cast<char *>(x.data()), count * 4)),
        "Cannot read input tensor");
  return x;
}
void dump(const std::vector<rvk::Output> &outputs, const fs::path &dir) {
  fs::create_directories(dir);
  for (const auto &o : outputs) {
    check(!o.name.empty() && o.name.find_first_not_of(
                                 "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrs"
                                 "tuvwxyz0123456789_-") == std::string::npos,
          "Unsafe output tensor name");
    std::ofstream f(dir / (o.name + ".f32"), std::ios::binary);
    check(bool(f.write(reinterpret_cast<const char *>(o.data.data()),
                       o.data.size() * 4)),
          "Cannot write tensor output");
  }
}
void dump_input(const std::vector<float> &input, const fs::path &path) {
  fs::create_directories(path.parent_path());
  std::ofstream f(path, std::ios::binary);
  check(bool(f.write(reinterpret_cast<const char *>(input.data()),
                     input.size() * sizeof(float))),
        "Cannot write normalized input tensor");
}
struct InferenceResult {
  std::vector<rvk::Output> outputs;
  double elapsed;
};
class InferenceWorker {
public:
  explicit InferenceWorker(rvk::GraphExecutor &engine)
      : engine_(engine), thread_([this] { loop(); }) {}
  ~InferenceWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_one();
    thread_.join();
  }
  std::future<InferenceResult> submit(std::vector<float> input) {
    Job job;
    job.input = std::move(input);
    auto result = job.result.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      check(!pending_ && !stopping_, "Inference worker is busy");
      pending_ = std::move(job);
    }
    ready_.notify_one();
    return result;
  }

private:
  struct Job {
    std::vector<float> input;
    std::promise<InferenceResult> result;
  };
  void loop() {
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [&] { return stopping_ || pending_.has_value(); });
      if (!pending_)
        return;
      Job job = std::move(*pending_);
      pending_.reset();
      lock.unlock();
      try {
        auto started = Clock::now();
        auto outputs = engine_.run(job.input);
        job.result.set_value({std::move(outputs), seconds(started)});
      } catch (...) {
        job.result.set_exception(std::current_exception());
      }
    }
  }
  rvk::GraphExecutor &engine_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::optional<Job> pending_;
  bool stopping_ = false;
  std::thread thread_;
};
int main(int argc, char **argv) {
  try {
    std::map<std::string, std::string> opts;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--help") {
        std::cout
            << "rfdetr-vulkan --model MODEL.rvk [--device N] [--shaders DIR]\n"
               "  --input IMAGE_DIRECTORY --output MASK_DIRECTORY [--threshold "
               "0.30] [--margin 0.05] [--limit N] [--dump-input DIR]\n"
               "  --tensor-input INPUT.f32 --tensor-output DIR [--repeat N]\n"
               "Experimental static FP32 Vulkan executor. White mask "
               "pixels=keep; black=remove.\n";
        return 0;
      }
      check(i + 1 < argc, "Missing argument for " + a);
      check(a == "--model" || a == "--device" || a == "--shaders" ||
                a == "--input" || a == "--output" || a == "--threshold" ||
                a == "--margin" || a == "--limit" || a == "--dump-input" ||
                a == "--tensor-input" || a == "--tensor-output" ||
                a == "--repeat",
            "Unknown option " + a);
      check(opts.emplace(a, argv[++i]).second, "Duplicate option " + a);
    }
    auto option = [&](const std::string &key, std::string fallback = {}) {
      auto i = opts.find(key);
      return i == opts.end() ? fallback : i->second;
    };
    auto integer = [&](const std::string &key, int fallback) {
      auto s = option(key, std::to_string(fallback));
      size_t p = 0;
      int v = std::stoi(s, &p);
      check(p == s.size(), "Invalid integer " + key);
      return v;
    };
    auto real = [&](const std::string &key, float fallback) {
      auto s = option(key, std::to_string(fallback));
      size_t p = 0;
      float v = std::stof(s, &p);
      check(p == s.size() && std::isfinite(v), "Invalid number " + key);
      return v;
    };
    check(opts.count("--model"), "--model is required");
    auto start = Clock::now();
    rvk::GraphExecutor engine(fs::u8path(option("--model")),
                              opts.count("--shaders")
                                  ? fs::u8path(option("--shaders"))
                                  : shader_directory(),
                              integer("--device", -1));
    std::cout << "GPU=" << engine.device_name()
              << " ARENA_BYTES=" << engine.arena_bytes()
              << " MODEL_LOAD_SECONDS=" << seconds(start) << std::endl;
    if (opts.count("--tensor-input")) {
      check(opts.count("--tensor-output") && !opts.count("--input"),
            "Tensor mode requires --tensor-output and no --input");
      int repeats = integer("--repeat", 1);
      check(repeats > 0 && repeats <= 10000, "Invalid repeat count");
      auto x = read_tensor(fs::u8path(option("--tensor-input")),
                           elements(engine.input_shape()));
      start = Clock::now();
      std::vector<rvk::Output> y;
      for (int i = 0; i < repeats; ++i)
        y = engine.run(x);
      dump(y, fs::u8path(option("--tensor-output")));
      std::cout << "RUNS=" << repeats << " SECONDS=" << seconds(start)
                << std::endl;
      return 0;
    }
    check(opts.count("--input") && opts.count("--output"),
          "--input and --output are required");
    auto input = fs::weakly_canonical(fs::u8path(option("--input"))),
         output = fs::weakly_canonical(fs::u8path(option("--output")));
    check(fs::is_directory(input), "Input directory does not exist");
    auto relative = output.lexically_relative(input);
    check(input.root_path() != output.root_path() ||
              (!relative.empty() && *relative.begin() == ".."),
          "Output must be outside the input directory");
    auto shape = engine.input_shape();
    check(shape.size() == 4 && shape[0] == 1 && shape[1] == 3,
          "Image mode needs batch-one NCHW RGB input");
    float threshold = real("--threshold", .30f),
          margin = real("--margin", .05f);
    check(threshold > 0 && threshold < 1 && std::abs(margin) <= 1,
          "Invalid threshold or margin");
    int limit = integer("--limit", INT32_MAX);
    check(limit > 0, "Invalid image limit");
    std::vector<fs::path> files;
    for (const auto &entry : fs::recursive_directory_iterator(input)) {
      if (!entry.is_regular_file())
        continue;
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return char(std::tolower(c)); });
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
          ext == ".tif" || ext == ".tiff")
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    if (files.size() > size_t(limit))
      files.resize(limit);
    check(!files.empty(), "No images found");
    start = Clock::now();
    std::array<double, 5> stage_seconds{};
    struct PreparedFrame {
      rfdetr::RgbImage image;
      std::vector<float> input;
    };
    auto prepare = [&](size_t index) {
      PreparedFrame frame;
      auto stage_start = Clock::now();
      frame.image = rfdetr::read_rgb_image(files[index]);
      stage_seconds[0] += seconds(stage_start);
      stage_start = Clock::now();
      rfdetr::preprocess_rgb(frame.image.pixels.data(), frame.image.width,
                             frame.image.height, shape[3], shape[2],
                             frame.input);
      if (opts.count("--dump-input")) {
        auto path = fs::u8path(option("--dump-input")) /
                    files[index].lexically_relative(input);
        path += ".f32";
        dump_input(frame.input, path);
      }
      stage_seconds[1] += seconds(stage_start);
      return frame;
    };
    InferenceWorker worker(engine);
    std::deque<std::future<void>> writers;
    auto current = prepare(0);
    auto inference = worker.submit(std::move(current.input));
    for (size_t i = 0; i < files.size(); ++i) {
      PreparedFrame next;
      if (i + 1 < files.size())
        next = prepare(i + 1);
      auto result = inference.get();
      stage_seconds[2] += result.elapsed;
      if (i + 1 < files.size())
        inference = worker.submit(std::move(next.input));
      auto stage_start = Clock::now();
      std::vector<nn::TrtTensor> tensors;
      for (auto &o : result.outputs)
        tensors.push_back({o.name, o.shape, std::move(o.data)});
      std::vector<uint8_t> mask(
          size_t(current.image.width) * current.image.height, 0);
      rfdetr::decode_people(tensors, current.image.width, current.image.height,
                            threshold, margin, mask);
      for (auto &p : mask)
        p = p ? 0 : 255;
      stage_seconds[3] += seconds(stage_start);
      stage_start = Clock::now();
      auto target = output / files[i].lexically_relative(input);
      target += ".png";
      fs::create_directories(target.parent_path());
      if (writers.size() >= 2) {
        writers.front().get();
        writers.pop_front();
      }
      writers.push_back(
          std::async(std::launch::async,
                     [target, width = current.image.width,
                      height = current.image.height, mask = std::move(mask)] {
                       auto tmp = target;
                       tmp += ".tmp";
                       rfdetr::write_gray_png(tmp, width, height, mask);
                       check(MoveFileExW(tmp.c_str(), target.c_str(),
                                         MOVEFILE_REPLACE_EXISTING |
                                             MOVEFILE_WRITE_THROUGH) != 0,
                             "Cannot publish mask: " + target.u8string());
                     }));
      stage_seconds[4] += seconds(stage_start);
      std::cout << "MASKS=" << i + 1 << '/' << files.size() << std::endl;
      current = std::move(next);
    }
    for (auto &writer : writers)
      writer.get();
    double elapsed = seconds(start);
    std::cout << "BATCH_SECONDS=" << elapsed << " IMAGES=" << files.size()
              << " IMAGES_PER_SECOND=" << files.size() / elapsed << std::endl;
    std::cout << "STAGE_SECONDS read=" << stage_seconds[0]
              << " preprocess=" << stage_seconds[1]
              << " inference=" << stage_seconds[2]
              << " decode=" << stage_seconds[3] << " write=" << stage_seconds[4]
              << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "RF-DETR Vulkan: " << e.what() << '\n';
    return 1;
  }
}
