cmake_minimum_required(VERSION 3.24)
project(RavenVulkanRfDetrOps LANGUAGES CXX)

include(CTest)
find_package(Vulkan REQUIRED COMPONENTS glslc)

add_executable(vulkan_rfdetr_ops_test src/main.cpp)
target_compile_features(vulkan_rfdetr_ops_test PRIVATE cxx_std_17)
target_compile_definitions(vulkan_rfdetr_ops_test PRIVATE NOMINMAX)
target_link_libraries(vulkan_rfdetr_ops_test PRIVATE Vulkan::Vulkan)
if(MSVC)
  target_compile_options(vulkan_rfdetr_ops_test PRIVATE /O2 /EHsc /utf-8)
endif()

# The executor has no CUDA/TensorRT link dependency. The existing decoder's
# TrtTensor is only a plain host output struct from a dependency-free header.
add_executable(rfdetr-vulkan src/graph_main.cpp src/GraphExecutor.cpp src/VulkanCompute.cpp
  ../rfdetr_masker/src/ImageIo.cpp ../rfdetr_masker/src/RfDetrDecode.cpp)
target_compile_features(rfdetr-vulkan PRIVATE cxx_std_17)
target_compile_definitions(rfdetr-vulkan PRIVATE NOMINMAX)
target_include_directories(rfdetr-vulkan PRIVATE ../rfdetr_masker/src)
target_link_libraries(rfdetr-vulkan PRIVATE Vulkan::Vulkan windowscodecs ole32)
if(MSVC)
  target_compile_options(rfdetr-vulkan PRIVATE /O2 /EHsc /utf-8 /W4)
endif()
foreach(shader ops matmul reduce)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp")
  set(bin "${CMAKE_CURRENT_BINARY_DIR}/${shader}.comp.spv")
  add_custom_command(OUTPUT "${bin}"
    COMMAND Vulkan::glslc "${src}" -o "${bin}"
    DEPENDS "${src}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/graph_common.glsl" VERBATIM)
  target_sources(rfdetr-vulkan PRIVATE "${bin}")
  add_custom_command(TARGET rfdetr-vulkan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:rfdetr-vulkan>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${bin}" "$<TARGET_FILE_DIR:rfdetr-vulkan>/shaders/${shader}.comp.spv")
endforeach()

set(shader_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/softmax.comp")
set(shader_binary "${CMAKE_CURRENT_BINARY_DIR}/softmax.comp.spv")
add_custom_command(OUTPUT "${shader_binary}"
  COMMAND Vulkan::glslc "${shader_source}" -o "${shader_binary}"
  DEPENDS "${shader_source}" VERBATIM)
target_sources(vulkan_rfdetr_ops_test PRIVATE "${shader_binary}")
add_custom_command(TARGET vulkan_rfdetr_ops_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vulkan_rfdetr_ops_test>/shaders"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${shader_binary}"
          "$<TARGET_FILE_DIR:vulkan_rfdetr_ops_test>/shaders/softmax.comp.spv")

if(BUILD_TESTING)
  add_test(NAME vulkan_rfdetr_softmax_parity COMMAND vulkan_rfdetr_ops_test --self-test)
endif()
