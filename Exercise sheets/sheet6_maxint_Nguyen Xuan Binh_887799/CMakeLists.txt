## CMake settings
cmake_minimum_required(VERSION 3.18)

## Project settings
project(pps C CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_CUDA_ARCHITECTURES 60 70) # Support for P100 and V100

## Subdirectories
add_subdirectory(src)
