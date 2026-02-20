cmake_minimum_required(VERSION 3.10)
project(cjepa_trt_infer)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

# TensorRT includes and libraries in the nvcr.io container
include_directories(/usr/include/x86_64-linux-gnu ${CUDA_INCLUDE_DIRS})
link_directories(/usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

add_executable(trt_infer trt_infer.cpp)
target_link_libraries(trt_infer nvinfer cudart ${CUDA_LIBRARIES})
