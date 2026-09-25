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
