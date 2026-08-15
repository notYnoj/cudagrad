#pragma once

#include <cuda_runtime.h>
#include <concepts>
#include <numeric>



template <typename U>
__global__ void add_kernel(const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}

template <typename U>
__global__ void scalar_add_kernel(const U* __restrict__ x, const U y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y;
    }
}

template <typename U>
__global__ void subtract_kernel(const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] - y[idx];
    }
}

template <typename U>
__global__ void scalar_subtract_kernel(const U* __restrict__ x, const U y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] - y;
    }
}

template <typename U>
__global__ void multiply_kernel(const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] * y[idx];
    }
}

template <typename U>
__global__ void scalar_multiply_kernel(const U* __restrict__ x, const U y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] * y;
    }
}

template <typename U>
__global__ void divide_kernel(const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] / y[idx];
    }
}

template <typename U>
__global__ void scalar_divide_kernel(const U* __restrict__ x, const U y, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] / y;
    }
}

template <typename U>
__global__ void pow_kernel(const U* __restrict__ x, const U power, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<U, float>) out[idx] = powf(x[idx], power);
        else out[idx] = static_cast<U>(pow(x[idx], power));
    }
}

template <typename U>
__global__ void exp_kernel(const U* __restrict__ x, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<U, float>) out[idx] = expf(x[idx]);
        else out[idx] = exp(x[idx]);
    }
}

template <typename U>
__global__ void neg_kernel(const U* __restrict__ x, U* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -x[idx];
    }
}



template<typename U>
__global__ void relu_kernel(const U* __restrict__ x, U* __restrict__ out, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = x[idx] > 0 ? x[idx] : 0;
}




template<typename U>
__global__ void fill_kernel(U* __restrict__ out, const U val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = val;
}

//output = x + y
//L = output
//dL/doutput = gout
//dL/dx = dL/doutput * dOutput/dx
//dOuput/dx = 1
//dL/dx = dL/doutput * 1 (same for y)
//so dx += dldoutput
template<typename U>
__global__ void add_backward_kernel(const U* __restrict__ gout, U* __restrict__ gx, U* __restrict__ gy, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gx[idx] += gout[idx];
        gy[idx] += gout[idx];
    }
}

// z = x * y, dL/dx = dL/dz * dz/dx = y
template<typename U>
__global__ void mul_backward_kernel(const U* __restrict__ gout, const U* __restrict__ x, const U* __restrict__ y, U* __restrict__ gx, U* __restrict__ gy, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gx[idx] += gout[idx] * y[idx];
        gy[idx] += gout[idx] * x[idx];
    }
}

// dx += (x > 0) ? dz same reaosning above : 0
template<typename U>
__global__ void relu_backward_kernel(const U* __restrict__ gout, const U* __restrict__ x, U* __restrict__ gx, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gx[idx] += x[idx] > U(0) ? gout[idx] : U(0);
    }
}


// c = a x b (a is (M, K) b is (K, N) so c is (M,N))
template<typename U>
__global__ void matmul_kernel(const U* __restrict__ A, const U* __restrict__ B, U* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // m
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // n threads are made in mxn
    if (row < M && col < N) {
        U acc = U(0);
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// dA[m,k] += sum_n dC[m,n] * B[k,n]     (dA = dC * B^T)
template<typename U>
__global__ void matmul_backward_A_kernel(const U* __restrict__ dC, const U* __restrict__ B,
                                         U* __restrict__ dA, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && k < K) {
        U acc = U(0);
        for (int n = 0; n < N; ++n) acc += dC[m * N + n] * B[k * N + n];
        dA[m * K + k] += acc;
    }
}

// dB[k,n] += sum_m A[m,k] * dC[m,n]     (dB = A^T * dC)
template<typename U>
__global__ void matmul_backward_B_kernel(const U* __restrict__ A, const U* __restrict__ dC,
                                         U* __restrict__ dB, int M, int N, int K) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K && n < N) {
        U acc = U(0);
        for (int m = 0; m < M; ++m) acc += A[m * K + k] * dC[m * N + n];
        dB[k * N + n] += acc;
    }
}

//add column wise
template<typename U>
__global__ void bias_add_kernel(const U* __restrict__ in, const U* __restrict__ b,
                                U* __restrict__ out, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) out[idx] = in[idx] + b[idx % N];
}

//pass dL/doutput into dL/dX (dL/dx = dL/dOutput * dOutput/dX (1))
template<typename U>
__global__ void accumulate_kernel(U* __restrict__ dst, const U* __restrict__ src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

//bias acts on everything in a col so we just add grad of col
template<typename U>
__global__ void bias_grad_kernel(const U* __restrict__ gout, U* __restrict__ db, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        U acc = U(0);
        for (int m = 0; m < M; ++m) acc += gout[m * N + n];
        db[n] += acc;
    }
}

template<typename U>
__global__ void softmax_ce_forward_kernel(const U* __restrict__ Z, const int* __restrict__ labels, U* __restrict__ probs, U* __restrict__ lossp, int M, int C) {
    int m = blockIdx.x * blockDim.x + threadIdx.x; //what example we are currently looking at
    if (m >= M) return;
    const U* z = Z + m * C; //pointer to where in Z we are, m is our currnet example and C is containers
    U* p = probs + m * C;
    U mx = z[0];
    for (int c = 1; c < C; ++c) mx = z[c] > mx ? z[c] : mx;
    U sum = U(0);
    for (int c = 0; c < C; ++c) { U e = exp(z[c] - mx); p[c] = e; sum += e; }
    for (int c = 0; c < C; ++c) p[c] /= sum;
    lossp[m] = -log(p[labels[m]] + U(1e-9));
}

template<typename U>
__global__ void softmax_ce_backward_kernel(const U* __restrict__ gscalar, const U* __restrict__ probs, const int* __restrict__ labels, U* __restrict__ dZ, int M, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * C) {
        int m = idx / C, c = idx % C;
        U t = (c == labels[m]) ? U(1) : U(0);
        //gscalar[0] = 1 but we can add scaled  (perfect would be probs[idx] = 1 when c==labels[m]
        dZ[idx] += gscalar[0] * (probs[idx] - t) / U(M);
    }
}

// w[i] -= lr * g[i]
template<typename U>
__global__ void sgd_update_kernel(U* __restrict__ w, const U* __restrict__ g, U lr, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}

