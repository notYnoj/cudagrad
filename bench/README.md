# bench — cudagrad vs PyTorch autograd

`bench.py` trains the same CNN three ways and reports the speed gap between cudagrad and PyTorch.
It also separates that gap into the part caused by the autograd engine and the part caused by
the kernels.

| Trainer | Autograd engine | Kernels | How it runs |
|---|---|---|---|
| `cudagrad` | cudagrad (C++) | cudagrad | `cudagrad_train.exe`, started as a subprocess |
| `torch+cg-kernels` | PyTorch | cudagrad | cudagrad's kernels loaded from `cudagrad_capi.dll`, wrapped as `torch.autograd.Function`s |
| `torch` | PyTorch | cuDNN / cuBLAS | stock PyTorch ops (`F.conv2d`, `addmm`, `F.cross_entropy`) |

- `cudagrad` vs `torch+cg-kernels`: same kernels, different engine. The difference is the engine.
- `torch+cg-kernels` vs `torch`: same engine, different kernels. The difference is the kernels.

All three use:
- the same model: conv(1→8, 3×3) → pool → conv(8→16, 3×3) → pool → fc(400→128) → fc(128→26);
- batch 128, plain SGD, and a cosine learning rate from 0.01 to 1e-6;
- He-normal initialisation;
- the same per-batch host gather and copy to the GPU;
- strict FP32 (TF32 off).

## Files

| File | Purpose |
|---|---|
| `bench.py` | Runs the trainers and prints per-epoch results and a summary |
| `train_bench.cpp` | The native cudagrad trainer, the same loop as `CNN::train`, timed per epoch |
| `cnn.hpp`, `loader.hpp` | The CNN container and EMNIST IDX loader (from WordHunt AI) |
| `cudagrad_capi.cu` | `extern "C"` wrappers around cudagrad's `launch_*` functions, so Python can call them |
| `CMakeLists.txt` | Builds `cudagrad_train.exe` and `cudagrad_capi.dll`, both from `../src/launch.cu` |

## Requirements

- An NVIDIA GPU, the CUDA toolkit, CMake ≥ 3.24, and a C++20 compiler. Developed with CUDA 13.3
  and MSVC on Windows; the C API uses `__declspec(dllexport)`.
- `CMAKE_CUDA_ARCHITECTURES` is `89` (RTX 40-series). Change it for your GPU.
- Python with a CUDA build of PyTorch. Pick the install command for your CUDA version at
  pytorch.org.
- EMNIST Letters in IDX format, laid out as `emnist-letters-{train,test}-{images-idx3,labels-idx1}-ubyte/<same name>`.

## Build

```bash
cmake -S bench -B bench/build
cmake --build bench/build --config Release
```

`bench.py` runs these two commands automatically when either output is missing.

## Run

```bash
python bench/bench.py --data path/to/emnist                     # all three trainers, 3 epochs
python bench/bench.py --data path/to/emnist --repeats 10        # 10 runs each, reports mean ± std
python bench/bench.py --data path/to/emnist --only cudagrad torch --epochs 5
python bench/bench.py --data path/to/emnist --check             # gradient check, one batch
```

| Option | Default | Meaning |
|---|---|---|
| `--data` | (a local path) | Folder holding the EMNIST IDX files |
| `--epochs` | 3 | Epochs per run |
| `--repeats` | 1 | Runs per trainer. With more than 1, prints mean ± standard deviation, coefficient of variation, and the range of final accuracy |
| `--only` | all three | Which trainers to run: any of `cudagrad`, `torch+cg-kernels`, `torch` |
| `--check` | off | Compare the cudagrad-kernel loss and gradients against PyTorch's on one batch, then exit |
| `--json` | none | Write every epoch's results to a JSON file |

### Reading the output

Each epoch prints training loss and accuracy, validation accuracy, and **train** and **eval**
times, measured separately.

- Train time covers only the batch loop, with `cudaDeviceSynchronize` / `torch.cuda.synchronize`
  before each clock read, because kernel launches are asynchronous.
- Epoch 1 includes warm-up (CUDA context setup, cuDNN algorithm selection), so the summaries
  leave it out.
- The final two lines give the engine ratio and the kernel ratio.

Measured results are in the [top-level README](../README.md#results).

## How the PyTorch + cudagrad-kernels trainer works

1. `cudagrad_capi.dll` exposes each `launch_*` function as an `extern "C"` function that returns
   0 or -1. The wrapper catches cudagrad's C++ exceptions, which must not cross into Python;
   `cg_last_error()` returns the message.
2. `load_cudagrad()` loads the DLL with `ctypes` and declares every argument type. Pointers are
   64-bit `c_void_p`; without the declarations, ctypes would truncate GPU addresses to 32 bits.
3. `call()` replaces each tensor with `tensor.data_ptr()` (its GPU address) and raises on failure.
4. Each op (`CgConv`, `CgConvBias`, `CgReLU`, `CgMaxPool`, `CgMatmul`, `CgBiasAdd`,
   `CgSoftmaxCE`) is a `torch.autograd.Function`.
   - `forward` launches cudagrad's forward kernel.
   - `backward` launches cudagrad's backward kernels into zeroed buffers, because the kernels
     accumulate with `+=`, and returns the gradients.
   - PyTorch's engine records these ops and decides when to call each backward.
5. Inputs must be contiguous and of the type the kernel expects: `float32`, and `int32` for labels.
   Kernels run on the default CUDA stream, the same one PyTorch uses, so they stay correctly
   ordered with PyTorch's own work.

## Profiling

Profile the native trainer: `cudagrad_train.exe <epochs> <data_dir> [max_batches]`. The
third argument limits batches per epoch, to keep profiles short.

**Nsight Systems** shows each kernel's share of GPU time and counts CUDA API calls:
```bash
nsys profile --trace cuda -o bench/prof/cudagrad_300 bench/build/Release/cudagrad_train.exe 1 path/to/emnist 300
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum bench/prof/cudagrad_300.nsys-rep
```

**Nsight Compute** gives per-kernel occupancy, throughput, stall reasons and per-line source
hotspots (the build uses `-lineinfo`):
```bash
ncu --set full -k "regex:conv2d_dW|conv_bias_grad|conv2d_forward|conv2d_dIn|matmul" --launch-skip 100 --launch-count 18 -o bench/prof/cudagrad_kernels bench/build/Release/cudagrad_train.exe 1 path/to/emnist 12
```

Open the `.ncu-rep` in `ncu-ui` and sort the Summary page by *Runtime Improvement*.

On Windows:
- Nsight Compute needs GPU performance counters enabled: NVIDIA Control Panel → Developer →
  Manage GPU Performance Counters.
- Call `ncu.exe` directly rather than `ncu.bat`, because the batch wrapper splits the regex at `|`.
- Under Nsight Compute the GPU clock is fixed and kernels are replayed, so take wall-clock
  numbers from `bench.py` or Nsight Systems.
