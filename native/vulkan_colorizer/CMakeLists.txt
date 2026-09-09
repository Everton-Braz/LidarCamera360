cmake_minimum_required(VERSION 3.24)
project(RavenVulkanColorizer LANGUAGES CXX)
find_package(Vulkan REQUIRED COMPONENTS glslc)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgcodecs imgproc)
add_executable(vulkan_colorizer src/main.cpp)
target_compile_features(vulkan_colorizer PRIVATE cxx_std_17)
target_compile_definitions(vulkan_colorizer PRIVATE NOMINMAX)
target_include_directories(vulkan_colorizer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vulkan_colorizer PRIVATE Vulkan::Vulkan ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(vulkan_colorizer PRIVATE /O2 /EHsc /utf-8)
endif()
foreach(shader occlusion_zbuf colorize_consensus resolve_consensus)
  set(out "${CMAKE_CURRENT_BINARY_DIR}/${shader}.spv")
  add_custom_command(OUTPUT "${out}" COMMAND Vulkan::glslc
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp" -o "${out}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp" VERBATIM)
  target_sources(vulkan_colorizer PRIVATE "${out}")
  add_custom_command(TARGET vulkan_colorizer POST_BUILD COMMAND ${CMAKE_COMMAND} -E
    make_directory "$<TARGET_FILE_DIR:vulkan_colorizer>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${out}" "$<TARGET_FILE_DIR:vulkan_colorizer>/shaders/${shader}.spv")
endforeach()
install(TARGETS vulkan_colorizer RUNTIME DESTINATION bin)
