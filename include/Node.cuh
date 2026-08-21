#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>

#include "cudaTensor.cuh"
#include "cudaLaunch.cuh"

template<typename T>
struct Node {
    CudaTensor<T> value;
    std::vector<std::shared_ptr<Node<T>>> children;
    std::function<void()> backward_fn;
    const char* op = "leaf";

    Node() = default;

    void backward() {
        std::vector<Node<T>*> topo;
        std::unordered_set<Node<T>*> seen;
        build_topo(this, topo, seen);
        launch_fill<T>(value.grad, T(1), value.size);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            if ((*it)->backward_fn) (*it)->backward_fn();

    }
    void zero_grad() {
        std::vector<Node<T>*> topo;
        std::unordered_set<Node<T>*> seen;
        build_topo(this, topo, seen);
        for (Node<T>* n : topo) n->value.zero_grad();
    }

private:
    static void build_topo(Node<T>* n, std::vector<Node<T>*>& topo,
        std::unordered_set<Node<T>*>& seen) {
        if (seen.count(n)) return;
        seen.insert(n);
        for (auto& c : n->children) build_topo(c.get(), topo, seen);
        topo.push_back(n);
    }
};

template<typename T>
using NodePtr = std::shared_ptr<Node<T>>;

template<typename T>
NodePtr<T> leaf(const Tensor<T>& h) {
    auto n = std::make_shared<Node<T>>();
    n->value = CudaTensor<T>::from_host(h, true);
    return n;
}


template<typename T>
NodePtr<T> add(NodePtr<T> a, NodePtr<T> b) {
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(a->value.shape, true);
    out->op = "add";
    out->children = { a, b };

    const std::size_t n = out->value.size;
    launch_add<T>(a->value.data, b->value.data, out->value.data, n);

    Node<T>* o = out.get(); Node<T>* pa = a.get(); Node<T>* pb = b.get();
    out->backward_fn = [o, pa, pb, n]() {
        launch_add_backward<T>(o->value.grad, pa->value.grad, pb->value.grad, n);
        };
    return out;
}

template<typename T>
NodePtr<T> mul(NodePtr<T> a, NodePtr<T> b) {
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(a->value.shape, true);
    out->op = "mul";
    out->children = { a, b };

    const std::size_t n = out->value.size;
    launch_mul<T>(a->value.data, b->value.data, out->value.data, n);

    Node<T>* o = out.get(); Node<T>* pa = a.get(); Node<T>* pb = b.get();
    out->backward_fn = [o, pa, pb, n]() {
        launch_mul_backward<T>(o->value.grad, pa->value.data, pb->value.data,
            pa->value.grad, pb->value.grad, n);
        };
    return out;
}

template<typename T>
NodePtr<T> relu(NodePtr<T> x) {
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(x->value.shape, true);
    out->op = "relu";
    out->children = { x };

    const std::size_t n = out->value.size;
    launch_relu<T>(x->value.data, out->value.data, n);

    Node<T>* o = out.get(); Node<T>* px = x.get();
    out->backward_fn = [o, px, n]() {
        launch_relu_backward<T>(o->value.grad, px->value.data, px->value.grad, n);
        };
    return out;
}

//A is (M, K) and B is (K,N)
template<typename T>
NodePtr<T> matmul(NodePtr<T> a, NodePtr<T> b) {
    const int M = static_cast<int>(a->value.shape[0]);
    const int K = static_cast<int>(a->value.shape[1]);
    const int N = static_cast<int>(b->value.shape[1]);
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ M, N }, true);
    out->op = "matmul";
    out->children = { a, b };

    launch_matmul<T>(a->value.data, b->value.data, out->value.data, M, N, K);

    Node<T>* o = out.get(); Node<T>* pa = a.get(); Node<T>* pb = b.get();
    out->backward_fn = [o, pa, pb, M, N, K]() {
        launch_matmul_backward_A<T>(o->value.grad, pb->value.data, pa->value.grad, M, N, K);
        launch_matmul_backward_B<T>(pa->value.data, o->value.grad, pb->value.grad, M, N, K);
        };
    return out;
}

template<typename T>
NodePtr<T> bias_add(NodePtr<T> x, NodePtr<T> b) {
    const int M = static_cast<int>(x->value.shape[0]);
    const int N = static_cast<int>(x->value.shape[1]);

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ M, N }, true);
    out->op = "bias_add";
    out->children = { x, b };

    launch_bias_add<T>(x->value.data, b->value.data, out->value.data, M, N);

    Node<T>* o = out.get(); Node<T>* px = x.get(); Node<T>* pb = b.get();
    out->backward_fn = [o, px, pb, M, N]() {
        launch_accumulate<T>(px->value.grad, o->value.grad, static_cast<std::size_t>(M) * N);
        launch_bias_grad<T>(o->value.grad, pb->value.grad, M, N);
        };
    return out;
}


template<typename T>
NodePtr<T> softmax_cross_entropy(NodePtr<T> logits, const std::vector<int>& labels) {
    const int M = static_cast<int>(logits->value.shape[0]);
    const int C = static_cast<int>(logits->value.shape[1]);


    int* dlabels_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&dlabels_raw, sizeof(int) * M));
    CUDA_CHECK(cudaMemcpy(dlabels_raw, labels.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
    std::shared_ptr<int> dlabels(dlabels_raw, [](int* p) { cudaFree(p); });

    auto probs = std::make_shared<CudaTensor<T>>(std::vector<long long>{ M, C }, false);
    T* dlossp_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&dlossp_raw, sizeof(T) * M));
    std::shared_ptr<T> dlossp(dlossp_raw, [](T* p) { cudaFree(p); });

    //convert raw logits into probabiltiies
    launch_softmax_ce_forward<T>(logits->value.data, dlabels.get(), probs->data, dlossp.get(), M, C);
    //dloss IS NOT DERIVATIVE, IT IS DEVICE LOSS, ITS CONFUSING BUT IM LAZY AND I UNDERSTAND IT
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ 1 }, true);
    out->op = "softmax_ce";
    out->children = { logits };
    launch_mean_reduce_kernel<T>(dlossp.get(), out->value.data, M);

    Node<T>* o = out.get(); Node<T>* pl = logits.get();

    out->backward_fn = [o, pl, probs, dlabels, M, C]() {
        launch_softmax_ce_backward<T>(o->value.grad, probs->data, dlabels.get(), pl->value.grad, M, C);
        };
    return out;
}

template<typename T>
NodePtr<T> conv(NodePtr<T> input, NodePtr<T> filters) {
    const int N = (int)input->value.shape[0];
    const int Cin = (int)input->value.shape[1];
    const int H = (int)input->value.shape[2];
    const int W = (int)input->value.shape[3];
    const int Cout = (int)filters->value.shape[0];
    const int K1 = (int)filters->value.shape[2];
    const int K2 = (int)filters->value.shape[3];
    const int oH = H - K1 + 1, oW = W - K2 + 1;

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ N, Cout, oH, oW }, true);
    out->op = "conv";
    out->children = { input, filters };

    launch_conv2d_forward<T>(input->value.data, filters->value.data, out->value.data,
        N, Cin, H, W, Cout, K1, K2, oH, oW);

    Node<T>* o = out.get(); Node<T>* in = input.get(); Node<T>* fi = filters.get();
    out->backward_fn = [o, in, fi, N, Cin, H, W, Cout, K1, K2, oH, oW]() {
        launch_conv_dW<T>(in->value.data, o->value.grad, fi->value.grad,
            N, Cin, H, W, Cout, K1, K2, oH, oW);
        launch_conv_dIn<T>(o->value.grad, fi->value.data, in->value.grad,
            N, Cin, H, W, Cout, K1, K2, oH, oW);
        };
    return out;
}

template<typename T>
NodePtr<T> conv_bias(NodePtr<T> x, NodePtr<T> bias) {
    const int N = static_cast<int>(x->value.shape[0]);
    const int C = static_cast<int>(x->value.shape[1]);
    const int HW = static_cast<int>(x->value.shape[2] * x->value.shape[3]);

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(x->value.shape, true);
    out->op = "conv_bias";
    out->children = { x, bias };

    launch_conv_bias<T>(x->value.data, bias->value.data, out->value.data, N, C, HW);

    Node<T>* o = out.get(); Node<T>* px = x.get(); Node<T>* pb = bias.get();
    out->backward_fn = [o, px, pb, N, C, HW]() {
        launch_accumulate<T>(px->value.grad, o->value.grad, static_cast<std::size_t>(N) * C * HW); // bias passthrough
        launch_conv_bias_grad<T>(o->value.grad, pb->value.grad, N, C, HW);
        };
    return out;
}

template <typename T>
NodePtr<T> maxpool(NodePtr<T> input, int pool) {
    NodePtr<T> out = std::make_shared<Node<T>>();
    const int N = static_cast<int>(input->value.shape[0]);
    const int C = static_cast<int>(input->value.shape[1]);
    const int H = static_cast<int>(input->value.shape[2]);
    const int W = static_cast<int>(input->value.shape[3]);
    const int oH = H / pool, oW = W / pool;

    out->value = CudaTensor<T>(std::vector<long long>{ N, C, oH, oW }, true);
    out->op = "maxpool";
    out->children = { input };

    //kept alive for backwards
    int* argmax_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&argmax_raw, sizeof(int) * (size_t)N * C * oH * oW));
    std::shared_ptr<int> argmax_cache(argmax_raw, [](int* p) { cudaFree(p); });

    launch_maxpool<T>(input->value.data, out->value.data, argmax_cache.get(), N, C, H, W, pool, oH, oW);

    Node<T>* o = out.get(), * in = input.get();
    out->backward_fn = [o, in, N, C, argmax_cache, oH, oW]() {
        launch_backward_maxpool<T>(o->value.grad, in->value.grad, argmax_cache.get(), N, C, oH, oW);
        };
    return out;
}

template<typename T>
NodePtr<T> flatten(NodePtr<T> input) {
    const long long N = input->value.shape[0];
    const std::size_t tot = input->value.size;
    const long long feat = (long long)(tot / (std::size_t)N);

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ N, feat }, true);
    out->op = "flatten";
    out->children = { input };

    CUDA_CHECK(cudaMemcpy(out->value.data, input->value.data,
        tot * sizeof(T), cudaMemcpyDeviceToDevice));

    Node<T>* o = out.get(); Node<T>* px = input.get();
    out->backward_fn = [o, px, tot]() {
        launch_accumulate<T>(px->value.grad, o->value.grad, tot);
        };
    return out;
}