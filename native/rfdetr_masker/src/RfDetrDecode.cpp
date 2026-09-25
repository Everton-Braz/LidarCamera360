// Adapted from Spirula Studio's NativeMaskDecode.cpp and MaskDilate.cpp.
// Copyright (C) Spirula Studio contributors. Licensed under GPL-3.0;
// see the adjacent LICENSE file. Vulkan/SAM wrappers were removed.
#include "RfDetrDecode.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <future>

namespace rfdetr {
namespace {
template<class F> void parallel_rows(int count, F fn) {
    const int workers = count >= 128 ? 4 : 1;
    std::vector<std::future<void>> jobs;
    for (int worker = 1; worker < workers; ++worker)
        jobs.push_back(std::async(std::launch::async, fn, count * worker / workers,
                                  count * (worker + 1) / workers));
    fn(0, count / workers);
    for (auto& job : jobs) job.get();
}
struct Mask {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;
};

struct Box {
    float x0, y0, x1, y1;
};

void validate_dimensions(int width, int height, size_t count) {
    if (width <= 0 || height <= 0 || width > 32768 || height > 32768 ||
        size_t(width) * size_t(height) != count)
        throw std::runtime_error("Invalid image or mask dimensions");
}

float bilinear(const float* data, int width, int height, float x, float y) {
    x = std::clamp(x, 0.0f, float(width - 1));
    y = std::clamp(y, 0.0f, float(height - 1));
    const int x0 = int(x), y0 = int(y);
    const int x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    const float fx = x - x0, fy = y - y0;
    return (data[size_t(y0) * width + x0] * (1 - fx) +
            data[size_t(y0) * width + x1] * fx) * (1 - fy) +
           (data[size_t(y1) * width + x0] * (1 - fx) +
            data[size_t(y1) * width + x1] * fx) * fy;
}

const nn::TrtTensor& tensor(const std::vector<nn::TrtTensor>& tensors,
                            const char* name) {
    for (const auto& item : tensors) {
        if (item.name != name) continue;
        size_t count = 1;
        for (int dim : item.shape) {
            if (dim <= 0 || count > (size_t(1) << 28) / size_t(dim))
                throw std::runtime_error("Invalid RF-DETR output shape: " + item.name);
            count *= size_t(dim);
        }
        if (count != item.data.size())
            throw std::runtime_error("RF-DETR output size mismatch: " + item.name);
        for (float value : item.data)
            if (!std::isfinite(value))
                throw std::runtime_error("Non-finite RF-DETR output: " + item.name);
        return item;
    }
    throw std::runtime_error(std::string("Missing RF-DETR output: ") + name);
}

int offset_radius(const Box& box, float ratio) {
    if (ratio == 0.0f) return 0;
    const float box_width = std::max(1.0f, box.x1 - box.x0);
    const float box_height = std::max(1.0f, box.y1 - box.y0);
    int kernel = int(std::fabs(ratio) * 0.5f * (box_width + box_height));
    kernel = std::max(kernel, 3);
    kernel |= 1;
    const int radius = (kernel - 1) / 2;
    return ratio > 0.0f ? radius : -radius;
}

void distance_1d(const float* input, int length, float* output,
                 std::vector<int>& sites, std::vector<double>& boundaries) {
    int k = 0;
    sites[0] = 0;
    boundaries[0] = -std::numeric_limits<double>::infinity();
    boundaries[1] = std::numeric_limits<double>::infinity();
    for (int q = 1; q < length; ++q) {
        double intersection = 0.0;
        for (;;) {
            const int v = sites[k];
            intersection = ((double(input[q]) + double(q) * q) -
                            (double(input[v]) + double(v) * v)) /
                           (2.0 * (q - v));
            if (intersection > boundaries[k] || k == 0) break;
            --k;
        }
        if (intersection <= boundaries[k] && k == 0) {
            // Equal infinite costs mean this line has no target pixels.
            sites[0] = q;
            boundaries[0] = -std::numeric_limits<double>::infinity();
            boundaries[1] = std::numeric_limits<double>::infinity();
            k = 0;
        } else {
            ++k;
            sites[k] = q;
            boundaries[k] = intersection;
            boundaries[k + 1] = std::numeric_limits<double>::infinity();
        }
    }
    k = 0;
    for (int q = 0; q < length; ++q) {
        while (boundaries[k + 1] < q) ++k;
        const float delta = float(q - sites[k]);
        output[q] = delta * delta + input[sites[k]];
    }
}

void distance_to(const std::vector<uint8_t>& mask, int width, int height,
                 uint8_t value, std::vector<float>& squared) {
    const size_t count = size_t(width) * height;
    squared.resize(count);
    const bool has_target = std::find(mask.begin(), mask.end(), value) != mask.end();
    if (!has_target) {
        std::fill(squared.begin(), squared.end(), std::numeric_limits<float>::infinity());
        return;
    }

    const float inf = float(std::numeric_limits<int>::max()) / 4.0f;
    std::vector<float> rows(count);
    parallel_rows(height, [&](int lo, int hi) {
        std::vector<float> input(width), output(width);
        std::vector<int> sites(width);
        std::vector<double> boundaries(size_t(width) + 1);
        for (int y = lo; y < hi; ++y) {
            const uint8_t* src = mask.data() + size_t(y) * width;
            for (int x = 0; x < width; ++x) input[x] = src[x] == value ? 0.0f : inf;
            distance_1d(input.data(), width, output.data(), sites, boundaries);
            std::copy_n(output.data(), width, rows.data() + size_t(y) * width);
        }
    });
    parallel_rows(width, [&](int lo, int hi) {
        std::vector<float> input(height), output(height);
        std::vector<int> sites(height);
        std::vector<double> boundaries(size_t(height) + 1);
        for (int x = lo; x < hi; ++x) {
            for (int y = 0; y < height; ++y) input[y] = rows[size_t(y) * width + x];
            distance_1d(input.data(), height, output.data(), sites, boundaries);
            for (int y = 0; y < height; ++y) squared[size_t(y) * width + x] = output[y];
        }
    });
}

void offset_binary_mask(std::vector<uint8_t>& mask, int width, int height, int radius) {
    if (radius == 0 || mask.empty()) return;
    std::vector<float> to_foreground, to_background;
    distance_to(mask, width, height, 1, to_foreground);
    distance_to(mask, width, height, 0, to_background);
    for (size_t i = 0; i < mask.size(); ++i) {
        const float signed_distance = std::sqrt(to_foreground[i]) - std::sqrt(to_background[i]);
        mask[i] = signed_distance <= float(radius) ? 1 : 0;
    }
}

void merge_region(const Mask& source, int origin_x, int origin_y,
                  int frame_width, int frame_height, int radius,
                  std::vector<uint8_t>& hit) {
    if (source.width <= 0 || source.height <= 0 || source.data.empty() ||
        source.data.size() != size_t(source.width) * source.height ||
        origin_x < 0 || origin_y < 0 || source.width > frame_width - origin_x ||
        source.height > frame_height - origin_y || hit.size() != size_t(frame_width) * frame_height)
        return;

    int min_x = source.width, min_y = source.height, max_x = -1, max_y = -1;
    for (int y = 0; y < source.height; ++y) {
        const uint8_t* row = source.data.data() + size_t(y) * source.width;
        for (int x = 0; x < source.width; ++x) if (row[x] > 127) {
            min_x = std::min(min_x, x); min_y = std::min(min_y, y);
            max_x = std::max(max_x, x); max_y = std::max(max_y, y);
        }
    }
    if (max_x < 0) return;

    const int global_min_x = origin_x + min_x, global_min_y = origin_y + min_y;
    const int global_max_x = origin_x + max_x, global_max_y = origin_y + max_y;
    if (radius == 0) {
        for (int y = min_y; y <= max_y; ++y) {
            const uint8_t* src = source.data.data() + size_t(y) * source.width;
            uint8_t* dst = hit.data() + size_t(origin_y + y) * frame_width + origin_x;
            for (int x = min_x; x <= max_x; ++x) if (src[x] > 127) dst[x] = 1;
        }
        return;
    }

    const int pad = std::abs(radius);
    const int crop_x = radius > 0 ? std::max(0, global_min_x - pad) : global_min_x - pad;
    const int crop_y = radius > 0 ? std::max(0, global_min_y - pad) : global_min_y - pad;
    const int crop_right = radius > 0 ? std::min(frame_width - 1, global_max_x + pad)
                                      : global_max_x + pad;
    const int crop_bottom = radius > 0 ? std::min(frame_height - 1, global_max_y + pad)
                                       : global_max_y + pad;
    const int crop_width = crop_right - crop_x + 1;
    const int crop_height = crop_bottom - crop_y + 1;
    if (crop_width <= 0 || crop_height <= 0) return;

    std::vector<uint8_t> crop(size_t(crop_width) * crop_height, 0);
    for (int y = 0; y < crop_height; ++y) {
        const int source_y = std::clamp(crop_y + y, 0, frame_height - 1) - origin_y;
        if (source_y < 0 || source_y >= source.height) continue;
        const uint8_t* src = source.data.data() + size_t(source_y) * source.width;
        uint8_t* dst = crop.data() + size_t(y) * crop_width;
        for (int x = 0; x < crop_width; ++x) {
            const int source_x = std::clamp(crop_x + x, 0, frame_width - 1) - origin_x;
            if (source_x >= 0 && source_x < source.width) dst[x] = src[source_x] > 127 ? 1 : 0;
        }
    }
    offset_binary_mask(crop, crop_width, crop_height, radius);
    for (int y = 0; y < crop_height; ++y) {
        const int destination_y = crop_y + y;
        if (destination_y < 0 || destination_y >= frame_height) continue;
        const uint8_t* src = crop.data() + size_t(y) * crop_width;
        uint8_t* dst = hit.data() + size_t(destination_y) * frame_width;
        for (int x = 0; x < crop_width; ++x) {
            const int destination_x = crop_x + x;
            if (destination_x >= 0 && destination_x < frame_width && src[x]) dst[destination_x] = 1;
        }
    }
}
}

void preprocess_rgb(const uint8_t* rgb, int source_width, int source_height,
                    int input_width, int input_height, std::vector<float>& nchw) {
    if (!rgb || source_width <= 0 || source_height <= 0 || source_width > 32768 ||
        source_height > 32768 || input_width <= 0 || input_height <= 0 ||
        input_width > 8192 || input_height > 8192)
        throw std::runtime_error("Invalid RGB/model input dimensions");
    const size_t pixels = size_t(source_width) * source_height;
    if (pixels > std::numeric_limits<size_t>::max() / 3 ||
        size_t(input_width) * input_height > (size_t(1) << 27))
        throw std::runtime_error("RGB/model input is too large");
    const size_t output_pixels = size_t(input_width) * input_height;
    nchw.resize(output_pixels * 3);

    const float means[] = {0.485f, 0.456f, 0.406f};
    const float deviations[] = {0.229f, 0.224f, 0.225f};
    const float scale_x = float(source_width) / input_width;
    const float scale_y = float(source_height) / input_height;
    for (int y = 0; y < input_height; ++y) {
        const float source_y = std::clamp((y + 0.5f) * scale_y - 0.5f,
                                          0.0f, float(source_height - 1));
        const int y0 = int(source_y), y1 = std::min(y0 + 1, source_height - 1);
        const float fy = source_y - y0;
        for (int x = 0; x < input_width; ++x) {
            const float source_x = std::clamp((x + 0.5f) * scale_x - 0.5f,
                                              0.0f, float(source_width - 1));
            const int x0 = int(source_x), x1 = std::min(x0 + 1, source_width - 1);
            const float fx = source_x - x0;
            for (int channel = 0; channel < 3; ++channel) {
                const auto sample = [&](int ix, int iy) {
                    return rgb[(size_t(iy) * source_width + ix) * 3 + channel];
                };
                const float value = (sample(x0, y0) * (1 - fx) + sample(x1, y0) * fx) * (1 - fy) +
                                     (sample(x0, y1) * (1 - fx) + sample(x1, y1) * fx) * fy;
                nchw[(size_t(channel) * input_height + y) * input_width + x] =
                    (value / 255.0f - means[channel]) / deviations[channel];
            }
        }
    }
}

void decode_people(const std::vector<nn::TrtTensor>& tensors,
                   int width, int height, float score_threshold,
                   float margin_ratio, std::vector<uint8_t>& hit) {
    validate_dimensions(width, height, hit.size());
    if (!std::isfinite(score_threshold) || score_threshold <= 0.0f || score_threshold >= 1.0f ||
        !std::isfinite(margin_ratio) || std::abs(margin_ratio) > 1.0f)
        throw std::runtime_error("Threshold must be in (0,1) and margin in [-1,1]");

    const auto& boxes = tensor(tensors, "dets");
    const auto& labels = tensor(tensors, "labels");
    const auto& masks = tensor(tensors, "masks");
    if (labels.shape.size() != 3 || labels.shape[0] != 1 || labels.shape[2] != 91)
        throw std::runtime_error("RF-DETR output must use COCO's 91 class slots (person=1)");
    const int query_count = labels.shape[1];
    if (boxes.shape != std::vector<int>{1, query_count, 4} || masks.shape.size() != 4 ||
        masks.shape[0] != 1 || masks.shape[1] != query_count)
        throw std::runtime_error("RF-DETR segmentation output shapes are incompatible");

    const int mask_height = masks.shape[2], mask_width = masks.shape[3];
    const float cutoff = std::log(score_threshold / (1.0f - score_threshold));
    for (int query = 0; query < query_count; ++query) {
        if (labels.data[size_t(query) * 91 + 1] < cutoff) continue;
        const float* plane = masks.data.data() + size_t(query) * mask_width * mask_height;
        int min_x = mask_width, min_y = mask_height, max_x = -1, max_y = -1;
        for (int y = 0; y < mask_height; ++y) for (int x = 0; x < mask_width; ++x) {
            if (plane[size_t(y) * mask_width + x] > 0.0f) {
                min_x = std::min(min_x, x); min_y = std::min(min_y, y);
                max_x = std::max(max_x, x); max_y = std::max(max_y, y);
            }
        }
        if (max_x < 0) continue;

        const int left = std::max(0, int(std::floor((min_x - 0.5f) * width / mask_width - 0.5f)) - 1);
        const int top = std::max(0, int(std::floor((min_y - 0.5f) * height / mask_height - 0.5f)) - 1);
        const int right = std::min(width, int(std::ceil((max_x + 1.5f) * width / mask_width - 0.5f)) + 1);
        const int bottom = std::min(height, int(std::ceil((max_y + 1.5f) * height / mask_height - 0.5f)) + 1);
        if (right <= left || bottom <= top) continue;

        Mask region;
        region.width = right - left;
        region.height = bottom - top;
        region.data.resize(size_t(region.width) * region.height);
        parallel_rows(region.height, [&](int lo, int hi) {
          for (int y = lo; y < hi; ++y) for (int x = 0; x < region.width; ++x) {
            const float low_x = (left + x + 0.5f) * mask_width / width - 0.5f;
            const float low_y = (top + y + 0.5f) * mask_height / height - 0.5f;
            region.data[size_t(y) * region.width + x] =
                bilinear(plane, mask_width, mask_height, low_x, low_y) > 0.0f ? 255 : 0;
          }
        });

        const float* box = boxes.data.data() + size_t(query) * 4;
        const Box source_box{(box[0] - box[2] * 0.5f) * width,
                             (box[1] - box[3] * 0.5f) * height,
                             (box[0] + box[2] * 0.5f) * width,
                             (box[1] + box[3] * 0.5f) * height};
        merge_region(region, left, top, width, height,
                     offset_radius(source_box, margin_ratio), hit);
    }
}

} // namespace rfdetr
