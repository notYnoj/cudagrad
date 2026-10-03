"""
Benchmark cudagrad's autograd against PyTorch's autograd on the WordHunt CNN.

Three trainers, same data / model / batch 128 / plain SGD / cosine LR 0.01 -> 1e-6:

  cudagrad          cudagrad's C++ engine (Node graph + backward()) driving cudagrad's kernels
                    -> bench/build/Release/cudagrad_train.exe, run as a subprocess
  torch+cg-kernels  PyTorch's autograd engine driving the SAME cudagrad kernels
                    (each cudagrad op wrapped in a torch.autograd.Function, kernels via cudagrad_capi.dll)
  torch             PyTorch's autograd engine driving PyTorch's own kernels (cuDNN / cuBLAS)

cudagrad vs torch+cg-kernels -> difference due to the autograd engine (same kernels)
torch+cg-kernels vs torch    -> difference due to the kernels (same engine)

  python bench/bench.py                      # all three, 3 epochs
  python bench/bench.py --epochs 5 --only cudagrad torch
  python bench/bench.py --check              # gradients of the cudagrad kernels vs PyTorch's, one batch
"""
import argparse, ctypes, json, math, re, statistics, struct, subprocess, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import Function

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
EXE = BUILD / "Release" / "cudagrad_train.exe"
DLL = BUILD / "Release" / "cudagrad_capi.dll"
DEFAULT_DATA = Path("C:/Users/Ynoj/Desktop/Portfolio-Dev-Projects-/WordHuntAI/data")
BATCH, LR_MAX, LR_MIN, CLASSES = 128, 0.01, 1e-6, 26
dev = torch.device("cuda")


# BUILD
# We wrap a C api to isolate our kernels and launch them through pytorch to see how our autograd compares
#we can then profile and see how our engine compares to pytorch's engine
#most of the delay came from kernels
def ensure_built():
    if EXE.exists() and DLL.exists():
        return
    print("building cudagrad bench targets ...", flush=True)
    subprocess.run(["cmake", "-S", str(HERE), "-B", str(BUILD)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", str(BUILD), "--config", "Release"], check=True, stdout=subprocess.DEVNULL)


# same thing as loader
#basically:
#reads bytes, reads the number of dimensions
#reads the 4 byte unsigned int (big endian aka left to right) thats what > means
#then it finds dims of the full dataset aka 20800 x 28 x2 three big ones thats why I * ndim
#then return everything past that in that dimension
def read_idx(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    ndim = struct.unpack(">I", raw[:4])[0] & 0xFF
    dims = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    return torch.frombuffer(bytearray(raw[4 + 4 * ndim:]), dtype=torch.uint8).reshape(dims)


def load_emnist(data: Path, split: str):
    base = f"emnist-letters-{split}"
    x = read_idx(data / f"{base}-images-idx3-ubyte" / f"{base}-images-idx3-ubyte")
    y = read_idx(data / f"{base}-labels-idx1-ubyte" / f"{base}-labels-idx1-ubyte")
    #remember emnist is dumb and puts stuff in [c,r] rather than [r,c]
    x = x.transpose(1, 2).contiguous().float().div_(255.0).unsqueeze(1)
    return x, (y.long() - 1)


def cosine_lr(e, total):   # schedulers.hpp cosine_annealing_LR
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * e / total))


def init_params(seed=0):
    """He-normal weights, zero biases, cudagrad layouts: conv [Cout,Cin,k,k], linear W [in,out]."""
    g = torch.Generator().manual_seed(seed)
    def he(shape, fan_in):
        return (torch.randn(shape, generator=g) * math.sqrt(2.0 / fan_in)).to(dev)
    return [he((8, 1, 3, 3), 9), torch.zeros(8, device=dev),
            he((16, 8, 3, 3), 72), torch.zeros(16, device=dev),
            he((400, 128), 400), torch.zeros(128, device=dev),
            he((128, 26), 128), torch.zeros(26, device=dev)]


# ===================================== cudagrad kernels as torch.autograd.Functions =====================================
cg = None
VP, I, SZ = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t

def load_cudagrad():
    global cg
    cg = ctypes.CDLL(str(DLL)) #load the api! ctypes
    sig = {
        "cg_relu": [VP, VP, SZ], "cg_relu_backward": [VP, VP, VP, SZ],
        "cg_matmul": [VP, VP, VP, I, I, I], "cg_matmul_backward_A": [VP, VP, VP, I, I, I],
        "cg_matmul_backward_B": [VP, VP, VP, I, I, I],
        "cg_bias_add": [VP, VP, VP, I, I], "cg_bias_grad": [VP, VP, I, I],
        "cg_conv2d_forward": [VP, VP, VP] + [I] * 9, "cg_conv_dW": [VP, VP, VP] + [I] * 9,
        "cg_conv_dIn": [VP, VP, VP] + [I] * 9,
        "cg_conv_bias": [VP, VP, VP, I, I, I], "cg_conv_bias_grad": [VP, VP, I, I, I],
        "cg_maxpool": [VP, VP, VP] + [I] * 7, "cg_maxpool_backward": [VP, VP, VP, I, I, I, I],
        "cg_softmax_ce_forward": [VP, VP, VP, VP, I, I], "cg_softmax_ce_backward": [VP, VP, VP, VP, I, I],
        "cg_mean_reduce": [VP, VP, I], "cg_sgd_update": [VP, VP, ctypes.c_float, SZ],
        "cg_accumulate_loss_correct": [VP, VP, VP, VP, I, I],
    }
    for name, args in sig.items():
        fn = getattr(cg, name) #same as cg.name
        #fn argtypes are in args
        #restype returns 0 if good -1 if bad
        fn.argtypes, fn.restype = args, ctypes.c_int
    cg.cg_last_error.restype = ctypes.c_char_p


def call(fn, *args):
    args = [a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args] # we need raw pointers for cuda not just torch tensor aka tensor memory address
    if fn(*args) != 0:
        raise RuntimeError(cg.cg_last_error().decode()) #decode error heheh

# cudagrad's backward kernels ACCUMULATE (+=), so every gradient starts from zeros.

class CgConv(Function):
    @staticmethod
    def forward(ctx, x, w):
        N, Cin, H, W = x.shape; Cout, _, K1, K2 = w.shape; oH, oW = H - K1 + 1, W - K2 + 1
        out = torch.empty(N, Cout, oH, oW, device=dev)
        call(cg.cg_conv2d_forward, x, w, out, N, Cin, H, W, Cout, K1, K2, oH, oW)
        ctx.save_for_backward(x, w); ctx.dims = (N, Cin, H, W, Cout, K1, K2, oH, oW)
        return out

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors; g = g.contiguous(); dx = None
        dw = torch.zeros_like(w)
        call(cg.cg_conv_dW, x, g, dw, *ctx.dims)
        if ctx.needs_input_grad[0]:
            dx = torch.zeros_like(x)
            call(cg.cg_conv_dIn, g, w, dx, *ctx.dims)
        return dx, dw


class CgConvBias(Function):
    @staticmethod
    def forward(ctx, x, b):
        N, C, H, W = x.shape; out = torch.empty_like(x)
        call(cg.cg_conv_bias, x, b, out, N, C, H * W); ctx.dims = (N, C, H * W)
        return out

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous(); db = torch.zeros(ctx.dims[1], device=dev)
        call(cg.cg_conv_bias_grad, g, db, *ctx.dims)
        return g, db


class CgReLU(Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.empty_like(x); call(cg.cg_relu, x, out, x.numel())
        ctx.save_for_backward(x); return out

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors; gx = torch.zeros_like(x)
        call(cg.cg_relu_backward, g.contiguous(), x, gx, x.numel()); return gx


class CgMaxPool(Function):
    @staticmethod
    def forward(ctx, x, pool):
        N, C, H, W = x.shape; oH, oW = H // pool, W // pool
        out = torch.empty(N, C, oH, oW, device=dev)
        argmax = torch.empty(N, C, oH, oW, device=dev, dtype=torch.int32)
        call(cg.cg_maxpool, x, out, argmax, N, C, H, W, pool, oH, oW)
        ctx.save_for_backward(argmax); ctx.in_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, g):
        (argmax,) = ctx.saved_tensors; N, C, oH, oW = argmax.shape
        dx = torch.zeros(ctx.in_shape, device=dev)
        call(cg.cg_maxpool_backward, g.contiguous(), dx, argmax, N, C, oH, oW)
        return dx, None


class CgMatmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        M, K = a.shape; N = b.shape[1]; out = torch.empty(M, N, device=dev)
        call(cg.cg_matmul, a, b, out, M, N, K); ctx.save_for_backward(a, b); ctx.dims = (M, N, K)
        return out

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors; g = g.contiguous()
        da, db = torch.zeros_like(a), torch.zeros_like(b)
        call(cg.cg_matmul_backward_A, g, b, da, *ctx.dims)
        call(cg.cg_matmul_backward_B, a, g, db, *ctx.dims)
        return da, db


class CgBiasAdd(Function):
    @staticmethod
    def forward(ctx, x, b):
        M, N = x.shape; out = torch.empty_like(x)
        call(cg.cg_bias_add, x, b, out, M, N); ctx.dims = (M, N)
        return out

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous(); db = torch.zeros(ctx.dims[1], device=dev)
        call(cg.cg_bias_grad, g, db, *ctx.dims)
        return g, db


class CgSoftmaxCE(Function):
    @staticmethod
    def forward(ctx, z, labels_i32):
        M, C = z.shape
        probs = torch.empty(M, C, device=dev); lossp = torch.empty(M, device=dev); out = torch.empty((), device=dev)
        call(cg.cg_softmax_ce_forward, z, labels_i32, probs, lossp, M, C)
        call(cg.cg_mean_reduce, lossp, out, M)
        ctx.save_for_backward(probs, labels_i32)
        return out

    @staticmethod
    def backward(ctx, g):
        probs, labels = ctx.saved_tensors; M, C = probs.shape
        dz = torch.zeros(M, C, device=dev)
        call(cg.cg_softmax_ce_backward, g.contiguous(), probs, labels, dz, M, C)
        return dz, None


# ===================================== the two PyTorch-autograd models =====================================
def forward_cg(p, x):
    x = CgReLU.apply(CgConvBias.apply(CgConv.apply(x, p[0]), p[1]))
    x = CgMaxPool.apply(x, 2)
    x = CgReLU.apply(CgConvBias.apply(CgConv.apply(x, p[2]), p[3]))
    x = CgMaxPool.apply(x, 2)
    x = x.reshape(x.shape[0], -1)
    x = CgReLU.apply(CgBiasAdd.apply(CgMatmul.apply(x, p[4]), p[5]))
    return CgBiasAdd.apply(CgMatmul.apply(x, p[6]), p[7])


def forward_torch(p, x):
    x = F.max_pool2d(F.relu(F.conv2d(x, p[0], p[1])), 2)
    x = F.max_pool2d(F.relu(F.conv2d(x, p[2], p[3])), 2)
    x = x.flatten(1)
    x = F.relu(torch.addmm(p[5], x, p[4]))
    return torch.addmm(p[7], x, p[6])


def run_pytorch_autograd(kind, data, epochs):
    use_cg = kind == "torch+cg-kernels"
    (xtr, ytr), (xte, yte) = data
    params = [t.requires_grad_() for t in init_params()]
    fwd = forward_cg if use_cg else forward_torch
    opt = None if use_cg else torch.optim.SGD(params, lr=LR_MAX, momentum=0.0)
    gen = torch.Generator().manual_seed(0)
    rows = []
    for e in range(epochs):
        lr = cosine_lr(e, epochs)
        if opt:
            for grp in opt.param_groups: grp["lr"] = lr
        perm = torch.randperm(len(xtr), generator=gen)
        acc = torch.zeros(2, device=dev)              # [sum of batch-mean loss * M, correct]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for s in range(0, len(xtr), BATCH):
            idx = perm[s:s + BATCH]
            xb = xtr[idx].to(dev)                     # host gather + H2D, like CNN::toBatch
            if use_cg:
                yb = ytr[idx].to(torch.int32).to(dev)
                for q in params: q.grad = None
                logits = fwd(params, xb)
                loss = CgSoftmaxCE.apply(logits, yb)
                loss.backward()
                for q in params: call(cg.cg_sgd_update, q, q.grad, ctypes.c_float(lr), q.numel())
                call(cg.cg_accumulate_loss_correct, acc, loss.detach(), logits.detach(), yb, len(idx), CLASSES)
            else:
                yb = ytr[idx].to(dev)
                opt.zero_grad()
                logits = fwd(params, xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                acc[0] += loss.detach() * len(idx); acc[1] += (logits.detach().argmax(1) == yb).sum()
        torch.cuda.synchronize(); t_train = time.perf_counter() - t0

        t1 = time.perf_counter(); correct = 0
        with torch.no_grad():
            for s in range(0, len(xte), BATCH):
                correct += (fwd(params, xte[s:s + BATCH].to(dev)).argmax(1).cpu() == yte[s:s + BATCH]).sum().item()
        torch.cuda.synchronize(); t_eval = time.perf_counter() - t1
        tr = acc.tolist()
        rows.append(dict(epoch=e + 1, train_s=t_train, eval_s=t_eval, loss=tr[0] / len(xtr),
                         acc=tr[1] / len(xtr), val_acc=correct / len(xte)))
        print_epoch(kind, rows[-1], epochs)
    return rows


# ===================================== native cudagrad (C++ engine) =====================================
EPOCH_RE = re.compile(r"epoch (\d+)/\d+ \| batches \d+ \| loss ([\d.e+-]+) \| acc ([\d.]+)% \| train ([\d.]+) s"
                      r"(?: \| val acc ([\d.]+)% \| eval ([\d.]+) s)?")

def run_cudagrad_native(data_dir, epochs):
    proc = subprocess.Popen([str(EXE), str(epochs), str(data_dir)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    rows = []
    for line in proc.stdout:
        m = EPOCH_RE.search(line)
        if m:
            rows.append(dict(epoch=int(m[1]), loss=float(m[2]), acc=float(m[3]) / 100, train_s=float(m[4]),
                             val_acc=float(m[5]) / 100, eval_s=float(m[6])))
            print_epoch("cudagrad", rows[-1], epochs)
    if proc.wait() != 0:
        raise RuntimeError(f"cudagrad_train.exe exited with {proc.returncode}")
    return rows


# ===================================== reporting =====================================
def print_epoch(kind, r, epochs):
    print(f"  [{kind:>16}] epoch {r['epoch']}/{epochs} | loss {r['loss']:.4f} | acc {r['acc']*100:5.2f}% "
          f"| val acc {r['val_acc']*100:5.2f}% | train {r['train_s']:.2f} s | eval {r['eval_s']:.2f} s", flush=True)


def summarize(results):
    steady = lambda rows: statistics.median([r["train_s"] for r in (rows[1:] or rows)])
    base = steady(results["torch"]) if "torch" in results else None
    print(f"\n{'trainer':<18}{'engine':<12}{'kernels':<12}{'train s/epoch':>14}{'eval s':>8}{'val acc':>9}{'vs torch':>10}")
    meta = {"cudagrad": ("cudagrad", "cudagrad"), "torch+cg-kernels": ("PyTorch", "cudagrad"), "torch": ("PyTorch", "cuDNN/cuBLAS")}
    for kind, rows in results.items():
        t = steady(rows)
        rel = f"{t / base:.2f}x" if base else "-"
        print(f"{kind:<18}{meta[kind][0]:<12}{meta[kind][1]:<12}{t:>14.3f}{statistics.median(r['eval_s'] for r in rows):>8.2f}"
              f"{rows[-1]['val_acc']*100:>8.2f}%{rel:>10}")
    print("(train s/epoch = median over epochs, excluding epoch 1 warm-up; eval timed separately)")
    if "cudagrad" in results and "torch+cg-kernels" in results:
        a, b = steady(results["cudagrad"]), steady(results["torch+cg-kernels"])
        print(f"\nsame kernels, different engine: cudagrad engine {a:.2f} s vs PyTorch engine {b:.2f} s -> {a / b:.2f}x")
    if "torch+cg-kernels" in results and "torch" in results:
        a, b = steady(results["torch+cg-kernels"]), steady(results["torch"])
        print(f"same engine, different kernels: cudagrad kernels {a:.2f} s vs cuDNN/cuBLAS {b:.2f} s -> {a / b:.2f}x")


def grad_check(data):
    (xtr, ytr), _ = data
    xb, y = xtr[:BATCH].to(dev), ytr[:BATCH].to(dev)
    p_cg = [t.requires_grad_() for t in init_params()]
    p_th = [t.detach().clone().requires_grad_() for t in p_cg]
    l_cg = CgSoftmaxCE.apply(forward_cg(p_cg, xb), y.to(torch.int32)); l_cg.backward()
    l_th = F.cross_entropy(forward_torch(p_th, xb), y); l_th.backward()
    print(f"loss: cudagrad kernels {l_cg.item():.6f} | PyTorch {l_th.item():.6f}")
    names = ["conv1.W", "conv1.b", "conv2.W", "conv2.b", "fc1.W", "fc1.b", "fc2.W", "fc2.b"]
    for n, a, b in zip(names, p_cg, p_th):
        err = ((a.grad - b.grad).abs().max() / b.grad.abs().max().clamp_min(1e-12)).item()
        print(f"  {n:<8} max rel grad diff {err:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--only", nargs="+", choices=["cudagrad", "torch+cg-kernels", "torch"],
                    default=["cudagrad", "torch+cg-kernels", "torch"])
    ap.add_argument("--check", action="store_true", help="compare cudagrad-kernel gradients with PyTorch's on one batch")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--repeats", type=int, default=1, help="run each trainer this many times to measure run-to-run spread")

    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False   # fp32 for cg
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    ensure_built(); load_cudagrad()
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name()} | TF32 off")
    data = (load_emnist(args.data, "train"), load_emnist(args.data, "test"))
    print(f"train {len(data[0][0])} | test {len(data[1][0])} | batch {BATCH} | tests {args.repeats} \n")

    results = {}
    for kind in args.only:
        train_times, eval_times, final_accs = [], [], []
        for rep in range(args.repeats):
            rows = run_cudagrad_native(args.data, args.epochs) if kind == "cudagrad" \
                else run_pytorch_autograd(kind, data, args.epochs)
            for r in (rows[1:] or rows):              # skip epoch 1 (warm-up), like summarize()
                train_times.append(r["train_s"])
                eval_times.append(r["eval_s"])
            final_accs.append(rows[-1]["val_acc"])
        results[kind] = rows                          # last run, for summarize()


        mean = statistics.mean(train_times)
        sd = statistics.stdev(train_times) if len(train_times) > 1 else 0.0
        print(f"{kind}: train {mean:.3f} ± {sd:.3f} s/epoch (cv {100 * sd / mean:.1f}%, n={len(train_times)}) | "
              f"eval {statistics.mean(eval_times):.3f} s | final val acc {min(final_accs)*100:.2f}-{max(final_accs)*100:.2f}%\n")



    summarize(results)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
