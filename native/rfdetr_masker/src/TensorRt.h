// Derived from Spirula Studio's TensorRT inference wrapper.
// Copyright (C) Spirula Studio contributors. Licensed under GPL-3.0;
// see the adjacent LICENSE file. Adapted for a headless standalone batch tool.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace nn {

struct TrtTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> data;
};

// Batch-one float32 I/O; static models may use FP16 internally.
class TensorRt {
public:
    TensorRt();
    ~TensorRt();
    void load(const std::string& path, const std::string& device_uuid,
              const std::string& cache_dir = {}, int dynamic_size = 512);
    int width() const;
    int height() const;
    const std::vector<TrtTensor>& outputs() const;
    const std::vector<TrtTensor>& run(const std::vector<float>& input);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
