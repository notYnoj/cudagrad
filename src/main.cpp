#include <iostream>
#include <vector>

#include "tensor.hpp"
#include "cudaTensor.cuh"
#include "Node.cuh"
#include "AutoGrad.cuh"
#include "cudaLaunch.cuh"

int main() {
    Tensor<float> mat({ 5,5 }, { 1,2,3,4,5,  6,7,8,9,10,  11,12,13,14,15,
                           16,17,18,19,20,  21,22,23,24,25 });
    Tensor<float> ker({ 3,3 }, { 1,0,0,  0,0,0,  0,0,0 });

    auto dmat = CudaTensor<float>::from_host(mat, false);
    auto dker = CudaTensor<float>::from_host(ker, false);
    CudaTensor<float> dout(std::vector<long long>{3, 3}, false);

    launch_conv2d<float, 3, 3>(dmat.data, dker.data, dout.data, 5, 5);
    cudaDeviceSynchronize();

    dout.to_host().print();

    return 0;
}
