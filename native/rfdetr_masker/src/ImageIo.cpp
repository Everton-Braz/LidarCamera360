// Copyright (C) 2026 LidarCamera360 contributors. Licensed under GPL-3.0;
// see the adjacent LICENSE file.
#include "ImageIo.h"

#include <Windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace rfdetr {
namespace {
using Microsoft::WRL::ComPtr;

void check_hr(HRESULT hr, const char* operation) {
    if (FAILED(hr)) {
        char code[16]{};
        sprintf_s(code, "0x%08lX", static_cast<unsigned long>(hr));
        throw std::runtime_error(std::string(operation) + " failed (HRESULT " + code + ")");
    }
}

IWICImagingFactory* factory() {
    struct State {
        ComPtr<IWICImagingFactory> instance;
        State() {
            check_hr(CoInitializeEx(nullptr, COINIT_MULTITHREADED), "Initialize WIC thread");
            check_hr(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                      IID_PPV_ARGS(&instance)), "Create WIC imaging factory");
        }
        ~State() { instance.Reset(); CoUninitialize(); }
    };
    thread_local State state;
    return state.instance.Get();
}

size_t checked_pixel_count(int width, int height, size_t channels) {
    if (width <= 0 || height <= 0 || width > 32768 || height > 32768)
        throw std::runtime_error("Image dimensions are outside the supported range");
    const size_t pixels = size_t(width) * size_t(height);
    if (pixels > std::numeric_limits<size_t>::max() / channels)
        throw std::runtime_error("Image size overflow");
    return pixels * channels;
}
}

RgbImage read_rgb_image(const std::filesystem::path& path) {
    ComPtr<IWICBitmapDecoder> decoder;
    check_hr(factory()->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
                                                   WICDecodeMetadataCacheOnLoad, &decoder),
             "Open image");
    ComPtr<IWICBitmapFrameDecode> frame;
    check_hr(decoder->GetFrame(0, &frame), "Read image frame");

    UINT width = 0, height = 0;
    check_hr(frame->GetSize(&width, &height), "Read image dimensions");
    const size_t byte_count = checked_pixel_count(static_cast<int>(width),
                                                   static_cast<int>(height), 3);
    if (byte_count > std::numeric_limits<UINT>::max() || width > std::numeric_limits<UINT>::max() / 3)
        throw std::runtime_error("Image is too large for Windows Imaging Component");

    ComPtr<IWICFormatConverter> converter;
    check_hr(factory()->CreateFormatConverter(&converter), "Create image converter");
    check_hr(converter->Initialize(frame.Get(), GUID_WICPixelFormat24bppRGB,
                                   WICBitmapDitherTypeNone, nullptr, 0.0,
                                   WICBitmapPaletteTypeCustom),
             "Convert image to RGB8");

    RgbImage result;
    result.width = static_cast<int>(width);
    result.height = static_cast<int>(height);
    result.pixels.resize(byte_count);
    const UINT stride = width * 3;
    check_hr(converter->CopyPixels(nullptr, stride, static_cast<UINT>(byte_count),
                                   result.pixels.data()),
             "Decode RGB pixels");
    return result;
}

void write_gray_png(const std::filesystem::path& path, int width, int height,
                    const std::vector<uint8_t>& pixels) {
    const size_t byte_count = checked_pixel_count(width, height, 1);
    if (pixels.size() != byte_count || byte_count > std::numeric_limits<UINT>::max())
        throw std::runtime_error("Invalid grayscale PNG buffer dimensions");

    ComPtr<IWICStream> stream;
    check_hr(factory()->CreateStream(&stream), "Create PNG stream");
    check_hr(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE), "Create PNG file");

    ComPtr<IWICBitmapEncoder> encoder;
    check_hr(factory()->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder),
             "Create PNG encoder");
    check_hr(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), "Initialize PNG encoder");

    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> properties;
    check_hr(encoder->CreateNewFrame(&frame, &properties), "Create PNG frame");
    check_hr(frame->Initialize(properties.Get()), "Initialize PNG frame");
    check_hr(frame->SetSize(static_cast<UINT>(width), static_cast<UINT>(height)),
             "Set PNG dimensions");
    WICPixelFormatGUID format = GUID_WICPixelFormat8bppGray;
    check_hr(frame->SetPixelFormat(&format), "Set grayscale PNG format");
    if (format != GUID_WICPixelFormat8bppGray)
        throw std::runtime_error("WIC PNG encoder did not accept 8-bit grayscale");
    check_hr(frame->WritePixels(static_cast<UINT>(height), static_cast<UINT>(width),
                                static_cast<UINT>(byte_count),
                                const_cast<BYTE*>(pixels.data())),
             "Write grayscale PNG pixels");
    check_hr(frame->Commit(), "Commit PNG frame");
    check_hr(encoder->Commit(), "Commit PNG file");
}

} // namespace rfdetr
