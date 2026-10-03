# cudagrad benchmark targets (driven by bench.py):
#   cudagrad_train.exe - the native cudagrad trainer: C++ autograd engine + kernels
#   cudagrad_capi.dll  - the same kernels behind a C API, called from PyTorch's autograd
#
#   cmake -S bench -B bench/build
#   cmake --build bench/build --config Release
cmake_minimum_required(VERSION 3.24)
project(cudagrad_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
find_package(CUDAToolkit REQUIRED)

set(CUDA_FLAGS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcompiler=/Zc:preprocessor>)

# cnn.hpp / loader.hpp are copies of the WordHuntAI training setup
add_executable(cudagrad_train train_bench.cpp ../src/launch.cu)
target_include_directories(cudagrad_train PRIVATE . ../include)
target_link_libraries(cudagrad_train PRIVATE CUDA::cudart)
target_compile_options(cudagrad_train PRIVATE ${CUDA_FLAGS})

add_library(cudagrad_capi SHARED cudagrad_capi.cu ../src/launch.cu)
target_include_directories(cudagrad_capi PRIVATE ../include)
# static cudart: no DLL search-path issues; shares the driver's primary context with PyTorch
set_target_properties(cudagrad_capi PROPERTIES CUDA_RUNTIME_LIBRARY Static)
target_compile_options(cudagrad_capi PRIVATE ${CUDA_FLAGS})
