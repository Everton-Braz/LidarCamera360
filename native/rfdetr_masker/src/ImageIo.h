// Copyright (C) 2026 LidarCamera360 contributors. Licensed under GPL-3.0;
// see the adjacent LICENSE file.
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace rfdetr {

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

RgbImage read_rgb_image(const std::filesystem::path& path);
void write_gray_png(const std::filesystem::path& path, int width, int height,
                    const std::vector<uint8_t>& pixels);

} // namespace rfdetr
