#pragma once

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <functional>
#include "tensor.hpp"
#include "cudaTensor.cuh"
#include "cudaLaunch.cuh"
#include "Node.cuh"
#include "schedulers.hpp"

//Can do everything
template<typename T>
struct Module {
    virtual NodePtr<T> forward(NodePtr<T> x) = 0;
    virtual ~Module() = default;
};

template<typename T>
struct SGD {
    std::vector<NodePtr<T>> params;
    T lr;
    T lrMax, lrMin;
    size_t epoch = 0, totalEpochs = 20;

    std::function<T(std::size_t, std::size_t, T, T)> lr_scheduler;

    explicit SGD(T lrMax = T(0.01), T lrMin = T(1e-6), size_t totalEpochs = 20,
                 std::function<T(std::size_t, std::size_t, T, T)> sched = linear_LR<T>)
        : lr(lrMax), lrMax(lrMax), lrMin(lrMin), totalEpochs(totalEpochs),
          lr_scheduler(std::move(sched)) {}

    void add(NodePtr<T> p) { params.push_back(std::move(p)); }

    void zero_grad() { for (auto& p : params) p->value.zero_grad(); }

    void step() {
        for (auto& p : params)
            launch_sgd_update<T>(p->value.data, p->value.grad, lr, p->value.size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    //update LR
    void stepEpoch() {
        epoch++;
        lr = lr_scheduler(epoch, totalEpochs, lrMax, lrMin);
    }
};

template<typename T>
struct Linear : public Module<T> {
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

    NodePtr<T> forward(NodePtr<T> x) override {
        NodePtr<T> y = bias_add(matmul(x, W), b);
        return use_relu ? relu(y) : y;
    }
};

template <typename T>
struct Conv : public Module<T> {
    NodePtr<T> filters;
    NodePtr<T> bias;
    Conv(SGD<T>& opt, long long Cin, long long Cout, long long kernelSize) {
        Tensor<T> filter(std::vector<long long> {Cout, Cin, kernelSize, kernelSize}, Init::He, Cin*kernelSize*kernelSize);
        filters = leaf(filter);
        Tensor<T> b(std::vector<long long> {Cout}, Init::Zero);
        bias = leaf(b);
        opt.add(filters);
        opt.add(bias);
    }
    NodePtr<T> forward(NodePtr<T> input) override {
        NodePtr<T> y = relu(conv_bias(conv(input, filters), bias));
        return y;
    }
};

//dont need to take in an optimizer because there are no parameters we are optimizing
template<typename T>
struct MaxPool : public Module<T> {
    int pool;
    MaxPool(int pool = 2) : pool(pool) {}
    NodePtr<T> forward(NodePtr<T> input) override { return maxpool(input, pool); }
};

template<typename T>
struct Flatten : public Module<T> {
    NodePtr<T> forward(NodePtr<T> x) override { return flatten(x); }
};

//save and load like other one
template<typename T>
void save_params(const std::vector<NodePtr<T>>& params, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_params: cannot open " + path);
    for (const auto& p : params) {
        const auto dt = p->value.to_host();
        const std::vector<T>& d = dt.getData();
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
        p->value = CudaTensor<T>::from_host(h, true); //if writing we are always going to require a grad so always set to true
    }
}
