#include <iostream>
#include <vector>

#include "tensor.hpp"
#include "cudaTensor.cuh"
#include "Node.cuh"
#include "AutoGrad.cuh"

// End-to-end training demo: a 2-layer MLP learns XOR on the GPU, with
// gradients produced entirely by the autograd graph.
//
//   x [4x2] --Linear(2->8,relu)--> h [4x8] --Linear(8->2)--> logits [4x2]
//   loss = softmax_cross_entropy(logits, labels)
//
// XOR is not linearly separable, so a working hidden layer + backprop is
// required to drive the loss down and classify all four points correctly.
int main() try {
    // XOR dataset (full batch of 4 examples).
    Tensor<float> X(std::vector<long long>{ 4, 2 },
                    std::vector<float>{ 0,0,  0,1,  1,0,  1,1 });
    std::vector<int> labels{ 0, 1, 1, 0 };

    SGD<float> opt(0.5f);
    Linear<float> l1(opt, 2, 8, /*relu=*/true);
    Linear<float> l2(opt, 8, 2, /*relu=*/false);   // logits: no activation

    for (int epoch = 0; epoch <= 2000; ++epoch) {
        opt.zero_grad();

        auto x      = leaf(X);                       // fresh input node each step
        auto h      = l1.forward(x);
        auto logits = l2.forward(h);
        auto loss   = softmax_cross_entropy(logits, labels);

        loss->backward();
        opt.step();

        if (epoch % 200 == 0) {
            float l = loss->value.to_host().getData()[0];
            std::cout << "epoch " << epoch << "\tloss " << l << "\n";
        }
    }

    // Final predictions. Keep the host Tensors alive while we read them
    // (getData() returns a reference into the Tensor).
    auto x         = leaf(X);
    auto logits    = l2.forward(l1.forward(x));
    Tensor<float> zt = logits->value.to_host();
    const std::vector<float>& z  = zt.getData();
    const std::vector<float>& xd = X.getData();

    std::cout << "\npredictions:\n";
    for (int m = 0; m < 4; ++m) {
        int pred = z[m * 2 + 0] > z[m * 2 + 1] ? 0 : 1;
        std::cout << "[" << xd[m * 2] << "," << xd[m * 2 + 1] << "] -> " << pred
                  << "   (label " << labels[m] << ")\n";
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << "\n";
    return 1;
}
