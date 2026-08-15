#include <cuda_runtime.h>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void launch_vadd(const float* a, const float* b, float* c, int n) {
    vadd<<< (n+255)/256, 256 >>> (a, b, c, n);
}