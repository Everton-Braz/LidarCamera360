cmake_minimum_required(VERSION 3.24)
project(MolaTools VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenMP)

add_executable(mola_optimizer mola_optimizer.cpp)
target_include_directories(mola_optimizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mola_optimizer PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(mola_optimizer PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(mola_optimizer PRIVATE NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
if(MSVC)
    target_compile_options(mola_optimizer PRIVATE /O2 /EHsc /utf-8)
else()
    target_compile_options(mola_optimizer PRIVATE -O3)
endif()

install(TARGETS mola_optimizer RUNTIME DESTINATION bin)
