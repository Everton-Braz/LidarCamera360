// Derived from Spirula Studio's RF-DETR decoder and mask dilation code.
// Copyright (C) Spirula Studio contributors. Licensed under GPL-3.0;
// see the adjacent LICENSE file. Adapted to have no Vulkan/SAM dependencies.
#pragma once

#include "TensorRt.h"

#include <cstdint>
#include <vector>

namespace rfdetr {

void preprocess_rgb(const uint8_t* rgb, int source_width, int source_height,
                    int input_width, int input_height, std::vector<float>& nchw);

// Writes 1 for person pixels and 0 for keep pixels into hit.
void decode_people(const std::vector<nn::TrtTensor>& tensors,
                   int width, int height, float score_threshold,
                   float margin_ratio, std::vector<uint8_t>& hit);

} // namespace rfdetr
