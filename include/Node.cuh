#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>

#include "cudaTensor.cuh"
#include "cudaLaunch.cuh"

// ---------------------------------------------------------------------------
// Reverse-mode autograd as a computational GRAPH (PyTorch / micrograd style).
//
// Each Node owns:
//   * value      - a CudaTensor holding this node's data AND its gradient
//   * children   - the input nodes it was produced from (edges to parents)
//   * backward_fn- how to push THIS node's grad into its children's grads
//   * op         - a label, for debugging/printing the graph
//
// Ops (add/mul/relu) are free functions: they run the forward kernel, build a
// new Node, wire up its children, and install its backward_fn. Nodes are held
// by shared_ptr, so the graph keeps its own inputs alive with no external
// bookkeeping — when the final node dies, the whole graph is released.
//
// node->backward() topologically sorts everything reachable from `node`,
// seeds that node's grad with 1, and runs each backward_fn in reverse order.
// This header contains no kernel-launch syntax, so it compiles as plain C++.
// ---------------------------------------------------------------------------

template<typename T>
struct Node {
    CudaTensor<T> value;                                   // data + grad
    std::vector<std::shared_ptr<Node<T>>> children;        // inputs (parents in the graph)
    std::function<void()> backward_fn;                     // push grad -> children
    const char* op = "leaf";

    Node() = default;

    // Reverse-mode backward from this node (treated as scalar loss = sum(value)).
    void backward() {
        // 1. Topologically order every node reachable from here (children first).
        std::vector<Node<T>*> topo;
        std::unordered_set<Node<T>*> seen;
        build_topo(this, topo, seen);

        // 2. Seed dL/d(self) = 1.
        launch_fill<T>(value.grad, T(1), value.size);

        // 3. Replay backward_fns in reverse topo order.
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            if ((*it)->backward_fn) (*it)->backward_fn();

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Zero grads across the whole graph (call before re-running backward).
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
        topo.push_back(n);   // pushed after its children -> children come first
    }
};

template<typename T>
using NodePtr = std::shared_ptr<Node<T>>;

// A leaf: a differentiable input uploaded from a host tensor.
template<typename T>
NodePtr<T> leaf(const Tensor<T>& h) {
    auto n = std::make_shared<Node<T>>();
    n->value = CudaTensor<T>::from_host(h, /*requires_grad=*/true);
    return n;
}

// ---------------------------------------------------------------------------
// Ops. Each: forward kernel -> new Node -> wire children -> install backward.
// backward_fn captures RAW pointers (o/pa/pb), never shared_ptr, so a node's
// closure can't co-own its own children and create a reference cycle. During
// backward() the root shared_ptr keeps every node alive, so raw access is safe.
// ---------------------------------------------------------------------------

template<typename T>
NodePtr<T> add(NodePtr<T> a, NodePtr<T> b) {
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(a->value.shape, /*requires_grad=*/true);
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
    out->value = CudaTensor<T>(a->value.shape, /*requires_grad=*/true);
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
    out->value = CudaTensor<T>(x->value.shape, /*requires_grad=*/true);
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

// out = axb a: (M,K) matrix, b: (K,N) matrix so out is (M, N) matrix
template<typename T>
NodePtr<T> matmul(NodePtr<T> a, NodePtr<T> b) {
    const int M = static_cast<int>(a->value.shape[0]);
    const int K = static_cast<int>(a->value.shape[1]);
    const int N = static_cast<int>(b->value.shape[1]);
    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ M, N }, /*requires_grad=*/true);
    out->op = "matmul";
    out->children = { a, b };

    launch_matmul<T>(a->value.data, b->value.data, out->value.data, M, N, K);

    Node<T>* o = out.get(); Node<T>* pa = a.get(); Node<T>* pb = b.get();
    out->backward_fn = [o, pa, pb, M, N, K]() {
        launch_matmul_backward_A<T>(o->value.grad, pb->value.data, pa->value.grad, M, N, K); // dA = dC @ B^T
        launch_matmul_backward_B<T>(pa->value.data, o->value.grad, pb->value.grad, M, N, K); // dB = A^T @ dC
    };
    return out;
}

// out[m,n] = x[m,n] + b[n]   (x: [M,N], b: [N]  ->  out: [M,N])
template<typename T>
NodePtr<T> bias_add(NodePtr<T> x, NodePtr<T> b) {
    const int M = static_cast<int>(x->value.shape[0]);
    const int N = static_cast<int>(x->value.shape[1]);

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ M, N }, /*requires_grad=*/true);
    out->op = "bias_add";
    out->children = { x, b };

    launch_bias_add<T>(x->value.data, b->value.data, out->value.data, M, N);

    Node<T>* o = out.get(); Node<T>* px = x.get(); Node<T>* pb = b.get();
    out->backward_fn = [o, px, pb, M, N]() {
        launch_accumulate<T>(px->value.grad, o->value.grad, static_cast<std::size_t>(M) * N); // dX += dOut
        launch_bias_grad<T>(o->value.grad, pb->value.grad, M, N);                             // db += colsum(dOut)
    };
    return out;
}

// Softmax + cross-entropy over `logits` [M, C] against integer `labels` [M].
// Returns a scalar loss node (mean over the M examples).
template<typename T>
NodePtr<T> softmax_cross_entropy(NodePtr<T> logits, const std::vector<int>& labels) {
    const int M = static_cast<int>(logits->value.shape[0]);
    const int C = static_cast<int>(logits->value.shape[1]);

    // labels on device (kept alive for backward via shared_ptr with a cudaFree deleter)
    int* dlabels_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&dlabels_raw, sizeof(int) * M));
    CUDA_CHECK(cudaMemcpy(dlabels_raw, labels.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
    std::shared_ptr<int> dlabels(dlabels_raw, [](int* p) { cudaFree(p); });

    // softmax probabilities (needed by backward) + per-row loss buffer
    auto probs = std::make_shared<CudaTensor<T>>(std::vector<long long>{ M, C }, /*requires_grad=*/false);
    T* dlossp_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&dlossp_raw, sizeof(T) * M));
    std::shared_ptr<T> dlossp(dlossp_raw, [](T* p) { cudaFree(p); });

    launch_softmax_ce_forward<T>(logits->value.data, dlabels.get(), probs->data, dlossp.get(), M, C);

    // reduce per-row losses on the host (M is tiny) for the reported scalar
    std::vector<T> hlp(M);
    CUDA_CHECK(cudaMemcpy(hlp.data(), dlossp.get(), sizeof(T) * M, cudaMemcpyDeviceToHost));
    T total = T(0);
    for (T v : hlp) total += v;
    total /= T(M);

    auto out = std::make_shared<Node<T>>();
    out->value = CudaTensor<T>(std::vector<long long>{ 1 }, /*requires_grad=*/true);
    out->op = "softmax_ce";
    out->children = { logits };
    CUDA_CHECK(cudaMemcpy(out->value.data, &total, sizeof(T), cudaMemcpyHostToDevice));

    Node<T>* o = out.get(); Node<T>* pl = logits.get();
    // probs & dlabels captured by value (shared_ptr) so they outlive this call;
    // they aren't Nodes, so this creates no reference cycle.
    out->backward_fn = [o, pl, probs, dlabels, M, C]() {
        launch_softmax_ce_backward<T>(o->value.grad, probs->data, dlabels.get(), pl->value.grad, M, C);
    };
    return out;
}
