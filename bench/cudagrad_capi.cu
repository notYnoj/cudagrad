// C API over cudagrad's launch_* wrappers (float only), so Python can load the
// kernels with ctypes and drive them from PyTorch's autograd.
//
// Every export returns 0 on success, -1 on failure; cg_last_error() gives the
// message. (launch.cu throws std::runtime_error, which must not cross into ctypes.)
#include <cstddef>
#include <exception>
#include <string>

#include "cudaLaunch.cuh"

#define CG_API extern "C" __declspec(dllexport)

static thread_local std::string g_err;

template<typename F>
static int guarded(F&& f) {
    try { f(); return 0; }
    catch (const std::exception& e) { g_err = e.what(); return -1; }
    catch (...) { g_err = "unknown error"; return -1; }
}

CG_API const char* cg_last_error() { return g_err.c_str(); }

// ---- elementwise
CG_API int cg_relu(const float* x, float* out, size_t n) { return guarded([&] { launch_relu<float>(x, out, n); }); }
CG_API int cg_relu_backward(const float* gout, const float* x, float* gx, size_t n) { return guarded([&] { launch_relu_backward<float>(gout, x, gx, n); }); }
CG_API int cg_accumulate(float* dst, const float* src, size_t n) { return guarded([&] { launch_accumulate<float>(dst, src, n); }); }

// ---- dense
CG_API int cg_matmul(const float* A, const float* B, float* C, int M, int N, int K) { return guarded([&] { launch_matmul<float>(A, B, C, M, N, K); }); }
CG_API int cg_matmul_backward_A(const float* dC, const float* B, float* dA, int M, int N, int K) { return guarded([&] { launch_matmul_backward_A<float>(dC, B, dA, M, N, K); }); }
CG_API int cg_matmul_backward_B(const float* A, const float* dC, float* dB, int M, int N, int K) { return guarded([&] { launch_matmul_backward_B<float>(A, dC, dB, M, N, K); }); }
CG_API int cg_bias_add(const float* in, const float* b, float* out, int M, int N) { return guarded([&] { launch_bias_add<float>(in, b, out, M, N); }); }
CG_API int cg_bias_grad(const float* gout, float* db, int M, int N) { return guarded([&] { launch_bias_grad<float>(gout, db, M, N); }); }

// ---- conv / pool
CG_API int cg_conv2d_forward(const float* in, const float* filt, float* out, int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW) {
    return guarded([&] { launch_conv2d_forward<float>(in, filt, out, N, Cin, H, W, Cout, K1, K2, oH, oW); });
}
CG_API int cg_conv_dW(const float* in, const float* dOut, float* dW, int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW) {
    return guarded([&] { launch_conv_dW<float>(in, dOut, dW, N, Cin, H, W, Cout, K1, K2, oH, oW); });
}
CG_API int cg_conv_dIn(const float* dOut, const float* filt, float* dIn, int N, int Cin, int H, int W, int Cout, int K1, int K2, int oH, int oW) {
    return guarded([&] { launch_conv_dIn<float>(dOut, filt, dIn, N, Cin, H, W, Cout, K1, K2, oH, oW); });
}
CG_API int cg_conv_bias(const float* in, const float* bias, float* out, int N, int C, int HW) { return guarded([&] { launch_conv_bias<float>(in, bias, out, N, C, HW); }); }
CG_API int cg_conv_bias_grad(const float* dOut, float* dBias, int N, int C, int HW) { return guarded([&] { launch_conv_bias_grad<float>(dOut, dBias, N, C, HW); }); }
CG_API int cg_maxpool(const float* in, float* out, int* argmax, int N, int C, int H, int W, int pool, int oH, int oW) {
    return guarded([&] { launch_maxpool<float>(in, out, argmax, N, C, H, W, pool, oH, oW); });
}
CG_API int cg_maxpool_backward(const float* dOut, float* dIn, const int* argmax, int N, int C, int oH, int oW) {
    return guarded([&] { launch_backward_maxpool<float>(dOut, dIn, argmax, N, C, oH, oW); });
}

// ---- loss / optimizer / metrics
CG_API int cg_softmax_ce_forward(const float* Z, const int* labels, float* probs, float* lossp, int M, int C) { return guarded([&] { launch_softmax_ce_forward<float>(Z, labels, probs, lossp, M, C); }); }
CG_API int cg_softmax_ce_backward(const float* g, const float* probs, const int* labels, float* dZ, int M, int C) { return guarded([&] { launch_softmax_ce_backward<float>(g, probs, labels, dZ, M, C); }); }
CG_API int cg_mean_reduce(const float* in, float* out, int M) { return guarded([&] { launch_mean_reduce_kernel<float>(in, out, M); }); }
CG_API int cg_sgd_update(float* w, const float* g, float lr, size_t n) { return guarded([&] { launch_sgd_update<float>(w, g, lr, n); }); }
CG_API int cg_accumulate_loss_correct(float* acc, const float* loss, const float* logits, int* labels, int M, int C) {
    return guarded([&] { launch_accumulate_loss_correct<float>(acc, loss, logits, labels, M, C); });
}
