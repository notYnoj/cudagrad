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

        CUDA_CHECK(cudaDeviceSynchronize());
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

    launch_softmax_ce_forward<T>(logits->value.data, dlabels.get(), probs->data, dlossp.get(), M, C);

    std::vector<T> hlp(M);
    CUDA_CHECK(cudaMemcpy(hlp.data(), dlossp.get(), sizeof(T) * M, cudaMemcpyDeviceToHost));
    //total loss
    T total = T(0);
    for (T v : hlp) total += v;
    total /= T(M);

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ 1 }, true);
    out->op = "softmax_ce";
    out->children = { logits };
    CUDA_CHECK(cudaMemcpy(out->value.data, &total, sizeof(T), cudaMemcpyHostToDevice));

    Node<T>* o = out.get(); Node<T>* pl = logits.get();

    out->backward_fn = [o, pl, probs, dlabels, M, C]() {
        launch_softmax_ce_backward<T>(o->value.grad, probs->data, dlabels.get(), pl->value.grad, M, C);
    };
    return out;
}
