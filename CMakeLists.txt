cmake_minimum_required(VERSION 3.24)
project(cudagrad LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED, ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

add_executable(cudagrad
    src/main.cpp       
    src/launch.cu       
    "include/cudaOps.cuh" "include/cudaTensor.cuh" "include/Node.cuh" "include/cudaLaunch.cuh" "include/AutoGrad.cuh")
target_include_directories(cudagrad PRIVATE include)
target_link_libraries(cudagrad PRIVATE CUDA::cudart CUDA::cublas)

target_compile_options(cudagrad PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)