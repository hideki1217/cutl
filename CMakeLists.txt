cmake_minimum_required(VERSION 3.16)
project(cutl_sample VERSION 1.0)

add_subdirectory(cpu)
add_subdirectory(gpu)

include_directories(/usr/local/cuda-11.7/targets/x86_64-linux/include)

add_executable(main main.cpp)
include_directories(main PRIVATE_HEADER cpu gpu)
target_link_libraries(main cpu gpu)
