#pragma once

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <stdexcept>

#include "tensor.hpp"
#include "cudaTensor.cuh"
#include "cudaLaunch.cuh"
#include "Node.cuh"

template<typename T>
struct SGD {
    std::vector<NodePtr<T>> params;
    T lr;

    explicit SGD(T lr = T(0.1)) : lr(lr) {}

    void add(NodePtr<T> p) { params.push_back(std::move(p)); }

    void zero_grad() { for (auto& p : params) p->value.zero_grad(); }

    void step() {
        for (auto& p : params)
            launch_sgd_update<T>(p->value.data, p->value.grad, lr, p->value.size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

template<typename T>
struct Linear {
    NodePtr<T> W, b;
    bool use_relu;

    Linear(SGD<T>& opt, int in, int out, bool relu = true) : use_relu(relu) {
        Tensor<T> hw(std::vector<long long>{ (long long)in, (long long)out }, Init::He, in);
        Tensor<T> hb(std::vector<long long>{ (long long)out }, Init::Zero);
        W = leaf(hw);
        b = leaf(hb);
        opt.add(W);
        opt.add(b);
    }

    NodePtr<T> forward(NodePtr<T> x) {
        NodePtr<T> y = bias_add(matmul(x, W), b);
        return use_relu ? relu(y) : y;
    }
};

template <typename T>
struct Conv {
    NodePtr<T> filters; //need [Cout, Cin, kernelSize, kernelSize]
    Conv(SGD<T>& opt, long long Cin, long long Cout, long long kernelSize) {
        Tensor<T> filter(std::vector<long long> {Cout, Cin, kernelSize, kernelSize}, Init::He, Cin*kernelSize*kernelSize);
        filters = leaf(filter);
        opt.add(filters);
    }
    NodePtr<T> forward(NodePtr<T> input) {
        NodePtr<T> y = relu(conv(input, filters));
        return y;
    }
};

//dont need to take in an optimizer because there are no parameters we are optimizing
template<typename T>
struct MaxPool {
    int pool;
    MaxPool(int pool = 2) : pool(pool) {}
    NodePtr<T> forward(NodePtr<T> input) { return maxpool(input, pool); }
};

template<typename T>
struct Flatten {
    NodePtr<T> forward(NodePtr<T> x) { return flatten(x); }
};

//save and load like other one
template<typename T>
void save_params(const std::vector<NodePtr<T>>& params, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_params: cannot open " + path);
    for (const auto& p : params) {
        const std::vector<T>& d = p->value.to_host().getData();
        f.write(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(T));
    }
}

template<typename T>
void load_params(std::vector<NodePtr<T>>& params, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_params: cannot open " + path);
    for (auto& p : params) {
        std::vector<T> buf(p->value.size);
        f.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(T));
        Tensor<T> h(p->value.shape, buf);
        p->value = CudaTensor<T>::from_host(h, /*requires_grad=*/true);
    }
}
