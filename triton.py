import time
import numpy as np
import pandas as pd
import os, glob
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import triton
import triton.language as tl

SEED       = 42
N_ROUNDS   = 10
DATA_ROOT  = "data"
MAX_EPOCHS = 1000
LR         = 0.01
PATIENCE   = 30
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True


@triton.jit
def fused_linear_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid       = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m     = pid % num_pid_m
    pid_n     = pid // num_pid_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k  = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_bn[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x     = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        w     = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc  += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias  = tl.load(b_ptr + offs_bn)[None, :]
        acc  += bias

    acc = tl.where(acc > 0, acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    offs_om  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_on  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    tl.store(out_ptrs, acc, mask=(offs_om[:, None] < M) & (offs_on[None, :] < N))


class TritonLinearReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        x_f32 = x.float()
        w_f32 = w.float()
        b_f32 = b.float() if b is not None else None

        M, K  = x_f32.shape
        N, _  = w_f32.shape
        out   = torch.empty((M, N), device=x.device, dtype=torch.float32)

        has_bias = b_f32 is not None
        b_ptr    = b_f32 if has_bias else x_f32

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        fused_linear_relu_kernel[grid](
            x_f32, w_f32, b_ptr, out,
            M, N, K,
            x_f32.stride(0), x_f32.stride(1),
            w_f32.stride(0), w_f32.stride(1),
            out.stride(0),   out.stride(1),
            HAS_BIAS=has_bias,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )
        ctx.save_for_backward(x_f32, w_f32, out)
        ctx.has_bias = has_bias
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, out  = ctx.saved_tensors
        mask        = (out > 0).float()
        grad_out    = grad_output.float() * mask
        grad_x      = grad_out @ w
        grad_w      = grad_out.T @ x
        grad_b      = grad_out.sum(0) if ctx.has_bias else None
        return grad_x.to(x.dtype), grad_w.to(w.dtype), grad_b


class TritonLinearReLULayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias   = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x):
        if x.is_cuda:
            return TritonLinearReLU.apply(x, self.weight, self.bias)
        return torch.relu(x @ self.weight.T + self.bias)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes=(128, 64)):
        super().__init__()
        layers   = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers.append(TritonLinearReLULayer(last_dim, h))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def adj_r2(y_true, y_pred, p):
    n  = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)


def process(csv_path):
    df    = pd.read_csv(csv_path)
    X_raw = df.drop(columns="y").values.astype(np.float32)
    y_raw = df["y"].values.astype(np.float32)
    p     = X_raw.shape[1]

    spl     = os.path.basename(os.path.dirname(csv_path)).split("_")
    tag     = "_".join(spl)
    n_feats = int(spl[0].replace("feats", ""))
    snr     = int(spl[1].replace("snr",   ""))
    N       = int(spl[2].replace("N",     ""))

    rows = []
    for rnd in range(N_ROUNDS):
        seed = SEED + rnd
        torch.manual_seed(seed)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y_raw, test_size=0.5, random_state=seed)

        sc      = StandardScaler()
        X_tr    = torch.from_numpy(sc.fit_transform(X_tr)).to(DEVICE)
        X_te    = torch.from_numpy(sc.transform(X_te)).to(DEVICE)
        y_tr    = torch.from_numpy(y_tr).to(DEVICE)
        y_te    = torch.from_numpy(y_te).to(DEVICE)

        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=0.1, random_state=seed)

        model     = MLP(p).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, fused=True)
        loss_fn   = nn.MSELoss()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        best_val_loss = float('inf')
        stop_cnt      = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_train), y_train)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(model(X_val), y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    stop_cnt      = 0
                else:
                    stop_cnt += 1
                if stop_cnt >= (PATIENCE // 5):
                    break

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1         = time.perf_counter()
            final_pred = model(X_te).cpu().numpy()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            infer_time = time.perf_counter() - t1

        y_te_np = y_te.cpu().numpy()

        rows.append({
            "dataset":               tag,
            "n_feats":               n_feats,
            "snr":                   snr,
            "N":                     N,
            "round":                 rnd,
            "method":                "torch_triton",
            "train_time":            elapsed,
            "infer_time":            infer_time,
            "infer_time_per_sample": infer_time / len(y_te_np),
            "adj_r2":                adj_r2(y_te_np, final_pred, p),
            "rmse":                  float(np.sqrt(mean_squared_error(y_te_np, final_pred))),
            "mae":                   float(mean_absolute_error(y_te_np, final_pred)),
        })
        print(f"  round {rnd}  time={elapsed:.4f}s  infer={infer_time:.4f}s  "
              f"R2={rows[-1]['adj_r2']:.4f}  RMSE={rows[-1]['rmse']:.4f}")

    return rows


if __name__ == "__main__":
    print(f"Using device: {DEVICE} (Triton fused Linear+ReLU)")
    datasets = sorted(glob.glob(os.path.join(DATA_ROOT, "*/data.csv")))
    if not datasets:
        raise FileNotFoundError(f"no data found under {DATA_ROOT}/")

    all_rows = []
    for path in datasets:
        print(f"\n=== {os.path.basename(os.path.dirname(path))} ===")
        all_rows.extend(process(path))

    pd.DataFrame(all_rows).to_csv("results_torch_triton.csv", index=False)
    print(f"\ndone. {len(all_rows)} rows -> results_torch_triton.csv")