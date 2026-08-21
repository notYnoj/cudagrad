/* DOCS:
This is where 
*/

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
__global__ void relu_kernel(const U* __restrict__ x, U* __restrict__ out, size_t n) {
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


//c = a x b (a is (M, K) b is (K, N) so c is (M,N))
template<typename U, int T>
__global__ void matmul_kernel(const U* __restrict__ A, const U* __restrict__ B, U* __restrict__ C, int M, int N, int K) {
    __shared__ U A_cache[T * T];
    __shared__ U B_cache[T * T];
    //tiling helps becuz cache hits are more importnat than operations
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    int num_sections = (K + T - 1) / T;

    U acc = U(0);

    for (int section = 0; section < num_sections; section++) {
        int tCol_A = section * T + threadIdx.x;
        int tRow_B = section * T + threadIdx.y;
        A_cache[(row % T) * T + (col % T)] = (row < M && tCol_A < K ? A[row * K + tCol_A] : 0);
        B_cache[(row % T) * T + (col % T)] = (tRow_B < K && col < N ? B[tRow_B * N + col] : 0);
        __syncthreads();

        for (int i = 0; i < T; i++) {
            acc += A_cache[(row % T) * T + i] * B_cache[i * T + (col % T)];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// dA[m,k] += sum_n dC[m,n] * B[k,n]
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

// dB[k,n] += sum_m A[m,k] * dC[m,n]
template<typename U>
__global__ void matmul_backward_B_kernel(const U* __restrict__ A, const U* __restrict__ dC,
    U* __restrict__ dB, int M, int N, int K) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K && n < N) {
        U acc = U(0);
        //TODO: call from memory is unoptimal
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

// out[n,co,oh,ow] = sum ci,ki,kj in[n,ci,oh+ki,ow+kj] * filt[co,ci,ki,kj]
template<typename U>
__global__ void conv2d_forward_kernel(const U* __restrict__ in, const U* __restrict__ filt,
    U* __restrict__ out,
    int N, int Cin, int H, int W,
    int Cout, int K1, int K2, int oH, int oW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Cout * oH * oW) return;
    int ow = idx % oW;
    int oh = (idx / oW) % oH;
    int co = (idx / (oW * oH)) % Cout;
    int n = idx / (oW * oH * Cout);

    U acc = U(0);
    for (int ci = 0; ci < Cin; ++ci) {
        const U* img = in + ((size_t)n * Cin + ci) * H * W;
        const U* f = filt + ((size_t)co * Cin + ci) * K1 * K2;
        for (int ki = 0; ki < K1; ++ki)
            for (int kj = 0; kj < K2; ++kj)
                acc += img[(oh + ki) * W + (ow + kj)] * f[ki * K2 + kj];
    }
    out[idx] = acc;
}

// dW[co,ci,ki,kj] += sum n,oh,ow in[n,ci,oh+ki,ow+kj] * dOut[n,co,oh,ow]
template<typename U>
__global__ void conv2d_dW_kernel(const U* __restrict__ in, const U* __restrict__ dOut,
    U* __restrict__ dW,
    int N, int Cin, int H, int W,
    int Cout, int K1, int K2, int oH, int oW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Cout * Cin * K1 * K2) return;
    int kj = idx % K2;
    int ki = (idx / K2) % K1;
    int ci = (idx / (K2 * K1)) % Cin;
    int co = idx / (K2 * K1 * Cin);

    U acc = U(0);
    for (int n = 0; n < N; ++n) {
        const U* img = in + ((size_t)n * Cin + ci) * H * W;
        const U* dImg = dOut + ((size_t)n * Cout + co) * oH * oW;
        for (int oh = 0; oh < oH; ++oh)
            for (int ow = 0; ow < oW; ++ow)
                acc += img[(oh + ki) * W + (ow + kj)] * dImg[oh * oW + ow];
    }
    dW[idx] += acc;
}

// dIn[n,ci,h,w] += sum_{co,ki,kj valid} dOut[n,co,h-ki,w-kj] * filt[co,ci,ki,kj]
template<typename U>
__global__ void conv2d_dIn_kernel(const U* __restrict__ dOut, const U* __restrict__ filt,
    U* __restrict__ dIn,
    int N, int Cin, int H, int W,
    int Cout, int K1, int K2, int oH, int oW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Cin * H * W) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int ci = (idx / (W * H)) % Cin;
    int n = idx / (W * H * Cin);

    U acc = U(0);
    for (int co = 0; co < Cout; ++co) {
        const U* dImg = dOut + ((size_t)n * Cout + co) * oH * oW;
        const U* f = filt + ((size_t)co * Cin + ci) * K1 * K2;
        for (int ki = 0; ki < K1; ++ki) {
            int oh = h - ki; if (oh < 0 || oh >= oH) continue;
            for (int kj = 0; kj < K2; ++kj) {
                int ow = w - kj; if (ow < 0 || ow >= oW) continue;
                acc += dImg[oh * oW + ow] * f[ki * K2 + kj];
            }
        }
    }
    dIn[idx] += acc;
}

// each chanel has a bias
template<typename U>
__global__ void conv_bias_forward_kernel(const U* __restrict__ in, const U* __restrict__ bias,
    U* __restrict__ out, int N, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * HW) {
        int c = (idx / HW) % C;
        out[idx] = in[idx] + bias[c];
    }
}


// dBias[c] += sum over all examples and HW (of dOut)
template<typename U>
__global__ void conv_bias_grad_kernel(const U* __restrict__ dOut, U* __restrict__ dBias,
    int N, int C, int HW) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        U acc = U(0);
        for (int n = 0; n < N; ++n) {
            //n*C*HW first stack + then c * HW for prev stacks we start here and then we just add up (0-8) start 9 
            const U* d = dOut + ((size_t)n * C + c) * HW;
            for (int i = 0; i < HW; ++i) acc += d[i];
        }
        dBias[c] += acc;
    }
}



// mat[N,C,H,W] becomesout [N,C,oH,oW]
template<typename U>
__global__ void maxpool_kernel(const U* __restrict__ mat, U* __restrict__ out, int* __restrict__ argmax_cache, int N, int C, int H, int W, int pool, int oH, int oW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * oH * oW) {
        int ow = idx % oW;
        int oh = (idx / oW) % oH;
        int c = (idx / (oW * oH)) % C;
        int n = idx / (oW * oH * C);
        size_t sliceBase = ((size_t)n * C + c) * H * W;
        int base = (oh * pool) * W + (ow * pool);
        U best = mat[sliceBase + base];
        int best_idx = (int)(sliceBase + base);
        for (int dx = 0; dx < pool; dx++) {
            for (int dy = 0; dy < pool; dy++) {
                int cur = base + dx * W + dy;
                if (mat[sliceBase + cur] > best) {
                    best = mat[sliceBase + cur];
                    best_idx = (int)(sliceBase + cur);
                }
            }
        }
        out[idx] = best;
        argmax_cache[idx] = best_idx;
    }
}

template<typename U>
__global__ void maxpool_backward_kernel(const U* __restrict__ dOut, U* __restrict__ dIn, const int* __restrict__ argmax_cache, int N, int C, int oH, int oW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * oH * oW) {
        dIn[argmax_cache[idx]] += dOut[idx];
    }
}

template<typename U>
__global__ void mean_reduce_kernel(const U* __restrict__ in, U* __restrict__ out, int M) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        U s = U(0);
        for (int i = 0; i < M; ++i) s += in[i];
        out[0] = s / U(M);
    }
}

template<typename U>
__global__ void accumulate_loss_correct(U* in, const U* loss, const U* logits, int* labels, int M, int C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < M) {
        atomicAdd(&in[0], loss[0]);
        U mx = logits[idx * C + 0];
        int pred = 0;
        for (int i = 1; i < C; i++) {
            if (mx < logits[idx * C + i]) {
                mx = logits[idx * C + i];
                pred = i;
            }
        }
        if (pred == labels[idx]) atomicAdd(&in[1], (U)1);
    }
}

// w[i] -= lr * g[i]
template<typename U>
__global__ void sgd_update_kernel(U* __restrict__ w, const U* __restrict__ g, U lr, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= lr * g[i];
}
