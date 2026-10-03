# cudagrad

A reverse-mode autograd engine written from scratch in CUDA C++, with the layers needed to
train a CNN on the GPU. No cuBLAS, cuDNN or deep-learning framework: every forward and backward
pass is a hand-written CUDA kernel.

It trains the letter classifier for WordHunt AI (EMNIwhST Letters, 26 classes). The benchmark below measures it against PyTorch on that exact model.

## Results

The same CNN, data order, batch size (128), optimizer (plain SGD) and learning-rate schedule
(cosine, 0.01 → 1e-6), trained three ways, all in strict FP32:

| Trainer | Autograd engine | Kernels | Train time / epoch | Final val acc |
|---|---|---|---|---|
| `cudagrad` | cudagrad (C++) | cudagrad | **6.15 ± 0.13 s** | 80.6 – 82.9 % |
| `torch+cg-kernels` | PyTorch | cudagrad | 4.53 ± 0.03 s | 80.9 % |
| `torch` | PyTorch | cuDNN / cuBLAS | 1.84 ± 0.04 s | 80.8 – 80.9 % |

**cudagrad is 3.35× slower per epoch than PyTorch**, and that gap splits into two causes:

- **Autograd engine: 1.36×.** Same cudagrad kernels, cudagrad's engine vs PyTorch's (6.15 s vs 4.53 s).
- **Kernels: 2.47×.** Same PyTorch engine, cudagrad's kernels vs cuDNN/cuBLAS (4.53 s vs 1.84 s).

Mean ± standard deviation over 10 runs × epochs 2–3 (n = 20 per trainer; epoch 1 excluded as
warm-up). The run-to-run coefficient of variation is 0.6–2.3%, far smaller than the gaps. All
three reach the same accuracy after 3 epochs.

cudagrad's accuracy range is wider because it draws new random weights every run, while the
PyTorch trainers use a fixed seed.

**Setup:** RTX 4080 (sm_89, 76 SMs) · CUDA 13.3 · PyTorch 2.14.1 + cu132 (cuDNN 9.24) ·
Windows 11, MSVC (VS 2026), Release build · TF32 disabled · EMNIST Letters (124,800 train /
20,800 test, 28×28).

### Where the time goes

Nsight Systems, 300 training steps:

| Kernel | Share of GPU time |
|---|---|
| `conv2d_dW_kernel` (conv weight gradient) | 67.4 % |
| `conv_bias_grad_kernel` (conv bias gradient) | 27.8 % |
| `conv2d_dIn_kernel` | 1.1 % |
| `conv2d_forward_kernel` | 0.8 % |
| all matmul kernels (forward + backward) | 1.2 % |

Two kernels account for **95% of GPU time**. Nsight Compute shows why:
- Each of them has very few outputs: 72 / 1,152 conv weights and 8 / 16 biases.
- They use one thread per output, so they launch only **1–5 thread blocks** on a 76-SM GPU.
- That gives 5.9% achieved occupancy and about 0.2% of peak SM throughput.
- Each thread then serially sums up to 86,528 terms.

The kernels with a thread per output element (`conv2d_forward`, `conv2d_dIn`) reach 70–80% occupancy.

The engine overhead comes mostly from allocation. cudagrad calls `cudaMalloc`/`cudaFree` for every
intermediate tensor on every step: 11,118 `cudaFree` calls in 300 steps, 69% of CUDA API time.
PyTorch reuses memory through its caching allocator.

### Correctness

`python bench/bench.py --check` runs one batch through cudagrad's kernels and through PyTorch with
identical weights:
- The losses match: 4.142801 vs 4.142802.
- Every parameter's gradient agrees within **2.4e-6** max relative error.

The kernels were also verified earlier with finite-difference gradient checks (agreement ~1e-4).

Weights from a full cudagrad training run score **93.1%** test accuracy when loaded into an
equivalent PyTorch model.

## How it works

```cpp
auto logits = net.forward(leaf(batch));               // runs forward kernels, records the graph
auto loss   = softmax_cross_entropy(logits, labels);
loss->backward();                                     // topo-sort, replay backward kernels in reverse
opt.step();                                           // w -= lr * grad, one kernel per parameter
```

The code is a stack; each layer only calls the one below it.

| Layer | File | Role |
|---|---|---|
| Layers + optimizer | `include/AutoGrad.cuh` | `Module`, `Conv`, `MaxPool`, `Flatten`, `Linear`, `SGD` (with LR scheduler) |
| Autograd graph | `include/Node.cuh` | `Node<T>`, the ops, `backward()` |
| Launch bridge | `include/cudaLaunch.cuh`, `src/launch.cu` | `launch_*` wrappers; the only file compiled by nvcc |
| Kernels | `include/cudaOps.cuh` | `__global__` forward and backward kernels |
| Device memory | `include/cudaTensor.cuh` | `CudaTensor<T>`: move-only RAII GPU buffer holding `data` and `grad` |
| Host tensor | `include/tensor.hpp` | CPU `Tensor<T>`, initialisation (He / zero) |
| LR schedules | `include/schedulers.hpp` | linear, cosine annealing, constant |

**Design choices:**
- **Each op is a free function** (`conv`, `maxpool`, `matmul`, `bias_add`, `relu`,
  `softmax_cross_entropy`, …). It allocates its output, launches the forward kernel, records its
  inputs, and stores a lambda that launches the backward kernels.
- **`backward()` topologically sorts the graph**, seeds the loss gradient with 1, and runs each
  node's lambda in reverse order.
- **Backward kernels accumulate (`+=`)**, so a tensor used twice receives the sum of both gradients.
- **Closures capture raw `Node*`, not `shared_ptr`.** Ownership only runs from outputs to inputs,
  so the graph has no reference cycles and frees itself when the loss goes out of scope.
- **Data that backward needs is owned by `shared_ptr`s with a `cudaFree` deleter**, captured in
  the closure. This covers maxpool argmax indices, softmax probabilities and device labels.
- **Launches are quarantined in `launch.cu`.** `<<<>>>` syntax lives only there, with explicit
  `float` and `double` instantiations, so everything else is ordinary C++ that MSVC can compile.

The CNN used in the benchmark (`bench/cnn.hpp`):

```cpp
CNN<float> cnn(0.01f, 1e-6f, epochs, cosine_annealing_LR<float>);
cnn.add(std::make_unique<Conv<float>>(cnn.opt, 1, 8, 3));      // [N,1,28,28] -> [N,8,26,26]
cnn.add(std::make_unique<MaxPool<float>>(2));                  //             -> [N,8,13,13]
cnn.add(std::make_unique<Conv<float>>(cnn.opt, 8, 16, 3));     //             -> [N,16,11,11]
cnn.add(std::make_unique<MaxPool<float>>(2));                  //             -> [N,16,5,5]
cnn.add(std::make_unique<Flatten<float>>());                   //             -> [N,400]
cnn.add(std::make_unique<Linear<float>>(cnn.opt, 400, 128, true));
cnn.add(std::make_unique<Linear<float>>(cnn.opt, 128, 26, false));   // 55,930 parameters
cnn.train(trainData, epochs, "models/", &testData, 128);
```

## Build and run the benchmark

Requirements:
- A CUDA GPU and CUDA toolkit (tested on 13.3)
- CMake ≥ 3.24 and a C++20 compiler
- Python with PyTorch built for CUDA
- The EMNIST Letters dataset in IDX format

`CMAKE_CUDA_ARCHITECTURES` is set to `89` (RTX 40-series); change it for other GPUs.

```bash
cmake -S bench -B bench/build
cmake --build bench/build --config Release
python bench/bench.py --data path/to/emnist --epochs 3 --repeats 10
```

Full documentation, including all options, how the three trainers are built, and how to reproduce
the Nsight profiles, is in [bench/README.md](bench/README.md).

## Roadmap

In order of measured impact:

1. **Split-K reduction for `conv2d_dW` and `conv_bias_grad`.** Spread each output's
   N·oH·oW-term sum across many blocks: a shared-memory tree reduction, then `atomicAdd`. This
   targets 95% of GPU time.
2. **A caching allocator** that reuses device buffers between steps instead of calling
   `cudaMalloc`/`cudaFree`. This targets the 1.36× engine gap.
3. **Skip unneeded gradients.** The first conv's input-image gradient is computed and never used.
4. **A no-grad mode for evaluation.** Evaluation currently builds the full graph, which is why
   cudagrad's eval takes 0.40 s against PyTorch's 0.09 s.
5. **Register-blocked / implicit-GEMM convolution and fused conv + bias + ReLU kernels.**
