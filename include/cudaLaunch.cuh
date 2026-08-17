#pragma once

#include <cstddef>

template<typename T> void launch_add (const T* a, const T* b, T* out, std::size_t n);
template<typename T> void launch_mul (const T* a, const T* b, T* out, std::size_t n);
template<typename T> void launch_relu(const T* x, T* out, std::size_t n);

template<typename T> void launch_fill(T* out, T val, std::size_t n);

template<typename T> void launch_add_backward (const T* gout, T* ga, T* gb, std::size_t n);
template<typename T> void launch_mul_backward (const T* gout, const T* a, const T* b, T* ga, T* gb, std::size_t n);
template<typename T> void launch_relu_backward(const T* gout, const T* x, T* gx, std::size_t n);

template<typename T> void launch_matmul           (const T* A, const T* B, T* C, int M, int N, int K);
template<typename T> void launch_matmul_backward_A(const T* dC, const T* B, T* dA, int M, int N, int K);
template<typename T> void launch_matmul_backward_B(const T* A, const T* dC, T* dB, int M, int N, int K);
template<typename T> void launch_bias_add         (const T* in, const T* b, T* out, int M, int N);
template<typename T> void launch_accumulate       (T* dst, const T* src, std::size_t n);
template<typename T> void launch_bias_grad        (const T* gout, T* db, int M, int N);
template<typename T> void launch_softmax_ce_forward (const T* Z, const int* labels, T* probs, T* lossp, int M, int C);
template<typename T> void launch_softmax_ce_backward(const T* gscalar, const T* probs, const int* labels, T* dZ, int M, int C);
template<typename T> void launch_sgd_update       (T* w, const T* g, T lr, std::size_t n);

template<typename T, int K1, int K2> void launch_conv2d(const T* mat, const T* kernel, T* out, int N, int M);

template<typename T> void launch_conv_dW(const T* in, const T* dOut, T* dW, int W, int oH, int oW, int K1, int K2);

template<typename T> void launch_conv_dIn(const T* mat, const T* kernel, T* dIn, int H, int W, int oH, int oW, int K1, int K2);

template<typename T> void launch_maxpool(const T* mat, T* out, int* argmax_cache, int C, int H, int W, int pool, int oH, int oW);
template<typename T> void launch_backward_maxpool(const T* dOut, T* dIn, const int* argmax_cache, int C, int oH, int oW);