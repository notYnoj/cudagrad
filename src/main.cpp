#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tensor.hpp"
#include "cudaTensor.cuh"
#include "Node.cuh"
#include "AutoGrad.cuh"
#include "cudaLaunch.cuh"

int main() {
    Tensor<float> inT({ 1,4,4 }, { 1,  5,  2,  6,
                                  8,  3,  4,  7,
                                  9, 13, 10, 14,
                                 16, 12, 11, 15 });

    auto x = leaf(inT);
    auto y = maxpool(x, 2);
    y->backward();                     

    std::cout << "forward (expect 8 7 / 16 15):\n";
    y->value.to_host().print();

    Tensor<float> dIn_analytic = x->value.grad_to_host();
    std::cout << "\ndIn analytic (expect 1s at the max positions):\n";
    dIn_analytic.print();

    auto lossOf = [](Tensor<float> inH) {
        auto xx = leaf(inH);
        auto yy = maxpool(xx, 2);
        Tensor<float> yh = yy->value.to_host();  
        float s = 0.0f;
        for (float v : yh.getData()) s += v;
        return s;
        };

    const float eps = 1e-2f;
    float maxErr = 0.0f;
    for (int i = 0; i < (int)inT.getData().size(); ++i) {
        Tensor<float> ip = inT, im = inT;
        ip.getData()[i] += eps;
        im.getData()[i] -= eps;
        float numeric = (lossOf(ip) - lossOf(im)) / (2.0f * eps);
        maxErr = std::max(maxErr, std::abs(numeric - dIn_analytic.getData()[i]));
    }
    std::cout << "\nmax dIn gradient error = " << maxErr << "\n";
    return 0;
}