#include "cudaLaunch.cuh"
#include "cudaOps.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>



namespace {
    constexpr int BLOCK = 256;
    inline unsigned grid(std::size_t n) {
        return static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    }
    inline void check(const char* what) {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("launch ") + what + " failed: "
                                     + cudaGetErrorString(e));
    }
}

template<typename T>
void launch_add(const T* a, const T* b, T* out, std::size_t n) {
    add_kernel<T><<<grid(n), BLOCK>>>(a, b, out, n);
    check("add");
}
template<typename T>
void launch_mul(const T* a, const T* b, T* out, std::size_t n) {
    multiply_kernel<T><<<grid(n), BLOCK>>>(a, b, out, n);
    check("mul");
}
template<typename T>
void launch_relu(const T* x, T* out, std::size_t n) {
    relu_kernel<T><<<grid(n), BLOCK>>>(x, out, n);
    check("relu");
}

template<typename T>
void launch_fill(T* out, T val, std::size_t n) {
    fill_kernel<T><<<grid(n), BLOCK>>>(out, val, n);
    check("fill");
}

template<typename T>
void launch_add_backward(const T* gout, T* ga, T* gb, std::size_t n) {
    add_backward_kernel<T><<<grid(n), BLOCK>>>(gout, ga, gb, n);
    check("add_backward");
}
template<typename T>
void launch_mul_backward(const T* gout, const T* a, const T* b, T* ga, T* gb, std::size_t n) {
    mul_backward_kernel<T><<<grid(n), BLOCK>>>(gout, a, b, ga, gb, n);
    check("mul_backward");
}
template<typename T>
void launch_relu_backward(const T* gout, const T* x, T* gx, std::size_t n) {
    relu_backward_kernel<T><<<grid(n), BLOCK>>>(gout, x, gx, n);
    check("relu_backward");
}

namespace {
    //2d
    dim3 grid2d(int cols, int rows) {
        return dim3((cols + 15) / 16, (rows + 15) / 16);
    }
    const dim3 BLOCK2D(16, 16);
}

template<typename T>
void launch_matmul(const T* A, const T* B, T* C, int M, int N, int K) {
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<T, TILE> << <grid, block >> > (A, B, C, M, N, K);
    check("matmul");
}
template<typename T>
void launch_matmul_backward_A(const T* dC, const T* B, T* dA, int M, int N, int K) {
    matmul_backward_A_kernel<T><<<grid2d(K, M), BLOCK2D>>>(dC, B, dA, M, N, K); 
    check("matmul_backward_A");
}
template<typename T>
void launch_matmul_backward_B(const T* A, const T* dC, T* dB, int M, int N, int K) {
    matmul_backward_B_kernel<T><<<grid2d(N, K), BLOCK2D>>>(A, dC, dB, M, N, K); 
    check("matmul_backward_B");
}
template<typename T>
void launch_bias_add(const T* in, const T* b, T* out, int M, int N) {
    bias_add_kernel<T><<<grid(static_cast<std::size_t>(M) * N), BLOCK>>>(in, b, out, M, N);
    check("bias_add");
}
template<typename T>
void launch_accumulate(T* dst, const T* src, std::size_t n) {
    accumulate_kernel<T><<<grid(n), BLOCK>>>(dst, src, n);
    check("accumulate");
}
template<typename T>
void launch_bias_grad(const T* gout, T* db, int M, int N) {
    bias_grad_kernel<T><<<grid(static_cast<std::size_t>(N)), BLOCK>>>(gout, db, M, N);
    check("bias_grad");
}
template<typename T>
void launch_softmax_ce_forward(const T* Z, const int* labels, T* probs, T* lossp, int M, int C) {
    softmax_ce_forward_kernel<T><<<grid(static_cast<std::size_t>(M)), BLOCK>>>(Z, labels, probs, lossp, M, C);
    check("softmax_ce_forward");
}
template<typename T>
void launch_softmax_ce_backward(const T* gscalar, const T* probs, const int* labels, T* dZ, int M, int C) {
    softmax_ce_backward_kernel<T><<<grid(static_cast<std::size_t>(M) * C), BLOCK>>>(gscalar, probs, labels, dZ, M, C);
    check("softmax_ce_backward");
}
template<typename T>
void launch_sgd_update(T* w, const T* g, T lr, std::size_t n) {
    sgd_update_kernel<T><<<grid(n), BLOCK>>>(w, g, lr, n);
    check("sgd_update");
}


template<typename T, int K1, int K2>
void launch_conv2d(const T* mat, const T* kernel, T* out, int N, int M) {
    constexpr int TILES = 16;
    const int outRow = N - K1 + 1;
    const int outCol = M - K2 + 1;
    dim3 block(TILES, TILES);
    dim3 grid((outCol + TILES - 1) / TILES, (outRow + TILES - 1) / TILES);
    convolution_kernel<T, TILES, K1, K2><<<grid, block>>>(mat, kernel, out, N, M);
    check("conv2d");
}

#define INSTANTIATE(T)                                                              \
    template void launch_add<T>(const T*, const T*, T*, std::size_t);               \
    template void launch_mul<T>(const T*, const T*, T*, std::size_t);               \
    template void launch_relu<T>(const T*, T*, std::size_t);                        \
    template void launch_fill<T>(T*, T, std::size_t);                               \
    template void launch_add_backward<T>(const T*, T*, T*, std::size_t);            \
    template void launch_mul_backward<T>(const T*, const T*, const T*, T*, T*, std::size_t); \
    template void launch_relu_backward<T>(const T*, const T*, T*, std::size_t);     \
    template void launch_matmul<T>(const T*, const T*, T*, int, int, int);          \
    template void launch_matmul_backward_A<T>(const T*, const T*, T*, int, int, int); \
    template void launch_matmul_backward_B<T>(const T*, const T*, T*, int, int, int); \
    template void launch_bias_add<T>(const T*, const T*, T*, int, int);             \
    template void launch_accumulate<T>(T*, const T*, std::size_t);                  \
    template void launch_bias_grad<T>(const T*, T*, int, int);                      \
    template void launch_softmax_ce_forward<T>(const T*, const int*, T*, T*, int, int); \
    template void launch_softmax_ce_backward<T>(const T*, const T*, const int*, T*, int, int); \
    template void launch_sgd_update<T>(T*, const T*, T, std::size_t); \

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

// conv2d carries extra compile-time template params (the kernel size), so it is
// instantiated separately — one line per (dtype, K1, K2) the CNN uses.
#define INSTANTIATE_CONV(T, K1, K2) \
    template void launch_conv2d<T, K1, K2>(const T*, const T*, T*, int, int);

INSTANTIATE_CONV(float, 3, 3)
INSTANTIATE_CONV(float, 5, 5)
INSTANTIATE_CONV(double, 3, 3)
INSTANTIATE_CONV(double, 5, 5)
#undef INSTANTIATE_CONV
