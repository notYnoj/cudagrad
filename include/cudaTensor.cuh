#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <string>

#include "tensor.hpp"

//CUDA_CHECK is good practice to ensure that that an expression in CUDA proceeds successfully
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) do {                                              \
    cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess) {                                               \
        throw std::runtime_error(std::string("CUDA error: ")              \
            + cudaGetErrorString(_e) + " at " + __FILE__ + ":"            \
            + std::to_string(__LINE__));                                   \
    }                                                                      \
} while (0)
#endif

template<typename T>
struct CudaTensor {
    T* data = nullptr;
    T* grad = nullptr;
    std::vector<long long> shape;
    std::vector<size_t>    strides;
    size_t size = 0;
    bool requires_grad = false;

    CudaTensor() = default;

    //explicit forces the compiler to not allow nonsensical conversions
    explicit CudaTensor(std::vector<long long> shape_, bool requires_grad_ = false) : shape(std::move(shape_)), requires_grad(requires_grad_) {
        size = 1;
        for (const long long d : shape) size *= static_cast<size_t>(d);
        compute_strides();
        CUDA_CHECK(cudaMalloc(&data, bytes()));
        if (requires_grad) {
            CUDA_CHECK(cudaMalloc(&grad, bytes()));
            zero_grad();
        }
    }

    //We don't need copy for now just move RAII
    CudaTensor(const CudaTensor&) = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;

    CudaTensor(CudaTensor&& o) noexcept { swap(o); }
    CudaTensor& operator=(CudaTensor&& o) noexcept {
        if (this != &o) { free_all(); swap(o); }
        return *this;
    }

    ~CudaTensor() { free_all(); }

    size_t bytes() const { return size * sizeof(T); }

    void zero_grad() {
        if (grad) CUDA_CHECK(cudaMemset(grad, 0, bytes()));
    }

    //converts a host tensor to a device tensor
    static CudaTensor from_host(const Tensor<T>& h, bool requires_grad = false) {
        CudaTensor t(h.getShape(), requires_grad);
        CUDA_CHECK(cudaMemcpy(t.data, h.getData().data(), t.bytes(), cudaMemcpyHostToDevice));
        return t;
    }

    //Converts a cudaTensor into a host tensor
    Tensor<T> to_host() const {
        std::vector<T> buf(size);
        CUDA_CHECK(cudaMemcpy(buf.data(), data, bytes(), cudaMemcpyDeviceToHost));
        return Tensor<T>(shape, buf);
    }

    //Converts grads to host tensor
    Tensor<T> grad_to_host() const {
        if (!grad) throw std::runtime_error("grad_to_host() error: there are no grads");
        std::vector<T> buf(size);
        CUDA_CHECK(cudaMemcpy(buf.data(), grad, bytes(), cudaMemcpyDeviceToHost));
        return Tensor<T>(shape, buf);
    }

private:
    void compute_strides() {
        strides.assign(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * static_cast<size_t>(shape[i + 1]);
    }
    void free_all() {
        if (data) cudaFree(data);
        if (grad) cudaFree(grad);
        data = grad = nullptr;
    }
    void swap(CudaTensor& o) noexcept {
        std::swap(data, o.data);   std::swap(grad, o.grad);
        std::swap(shape, o.shape); std::swap(strides, o.strides);
        std::swap(size, o.size);   std::swap(requires_grad, o.requires_grad);
    }
};
