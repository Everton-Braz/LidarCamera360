cmake_minimum_required(VERSION 3.24)
project(rfdetr_masker LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "The current standalone image codec uses Windows Imaging Component (WIC).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(RFDTR_TENSORRT_ROOT "" CACHE PATH "TensorRT SDK or extracted runtime root")
set(RFDTR_TENSORRT_RUNTIME_DIR "" CACHE PATH "Directory containing TensorRT runtime DLLs")
option(RFDTR_COPY_BUILDER_RESOURCE "Copy the >1 GB TensorRT ONNX builder resource DLL" OFF)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS "${RFDTR_TENSORRT_ROOT}/include"
    REQUIRED)
if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvOnnxParser.h")
    message(FATAL_ERROR "NvOnnxParser.h is required to build an engine from ONNX.")
endif()

add_executable(rfdetr-masker
    src/main.cpp
    src/ImageIo.cpp
    src/RfDetrDecode.cpp
    src/TensorRt.cpp)
target_include_directories(rfdetr-masker PRIVATE "${TENSORRT_INCLUDE_DIR}" src)
target_compile_definitions(rfdetr-masker PRIVATE SS_TENSORRT=1 NOMINMAX _CRT_SECURE_NO_WARNINGS)
target_link_libraries(rfdetr-masker PRIVATE CUDA::cudart_static Threads::Threads windowscodecs ole32)
if(MSVC)
    target_compile_options(rfdetr-masker PRIVATE /O2 /EHsc /utf-8 /W4)
endif()

if(RFDTR_TENSORRT_RUNTIME_DIR)
    foreach(dll nvinfer_10.dll nvinfer_plugin_10.dll nvonnxparser_10.dll)
        if(EXISTS "${RFDTR_TENSORRT_RUNTIME_DIR}/${dll}")
            add_custom_command(TARGET rfdetr-masker POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${RFDTR_TENSORRT_RUNTIME_DIR}/${dll}"
                    "$<TARGET_FILE_DIR:rfdetr-masker>/${dll}")
        endif()
    endforeach()
    if(RFDTR_COPY_BUILDER_RESOURCE AND EXISTS "${RFDTR_TENSORRT_RUNTIME_DIR}/nvinfer_builder_resource_10.dll")
        add_custom_command(TARGET rfdetr-masker POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${RFDTR_TENSORRT_RUNTIME_DIR}/nvinfer_builder_resource_10.dll"
                "$<TARGET_FILE_DIR:rfdetr-masker>/nvinfer_builder_resource_10.dll")
    endif()
endif()

install(TARGETS rfdetr-masker RUNTIME DESTINATION bin)

enable_testing()
add_executable(rfdetr-decode-test tests/decode_test.cpp src/RfDetrDecode.cpp)
target_include_directories(rfdetr-decode-test PRIVATE src)
target_link_libraries(rfdetr-decode-test PRIVATE Threads::Threads)
add_test(NAME rfdetr-decode COMMAND rfdetr-decode-test)
