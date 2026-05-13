import time
import numpy as np
import pandas as pd
import os, glob
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 42
N_ROUNDS = 10
DATA_ROOT = "data"
MAX_EPOCHS = 1000 
LR = 0.01
PATIENCE = 30 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes=(128, 64)):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def gpu_scale(tr, te):
    mean = tr.mean(dim=0)
    std = tr.std(dim=0)
    std[std == 0] = 1.0 
    return (tr - mean) / std, (te - mean) / std

def adj_r2(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

def process(csv_path):
    df = pd.read_csv(csv_path)
    X_cpu = torch.from_numpy(df.drop(columns="y").values.astype(np.float32)).pin_memory()
    y_cpu = torch.from_numpy(df["y"].values.astype(np.float32)).pin_memory()
    p = X_cpu.shape[1]

    tag = "_".join(os.path.basename(os.path.dirname(csv_path)).split("_"))
    rows = []

    for rnd in range(N_ROUNDS):
        seed = SEED + rnd
        torch.manual_seed(seed)
        
        # CPU 负责逻辑：切分索引
        idx = np.arange(len(X_cpu))
        tr_idx, te_idx = train_test_split(idx, test_size=0.5, random_state=seed)
        
        X_tr = X_cpu[tr_idx].to(DEVICE, non_blocking=True)
        X_te = X_cpu[te_idx].to(DEVICE, non_blocking=True)
        y_tr = y_cpu[tr_idx].to(DEVICE, non_blocking=True)
        y_te = y_cpu[te_idx].to(DEVICE, non_blocking=True)

        X_tr, X_te = gpu_scale(X_tr, X_te)

        val_size = int(len(X_tr) * 0.1)
        X_train, X_val = X_tr[val_size:], X_tr[:val_size]
        y_train, y_val = y_tr[val_size:], y_tr[:val_size]

        model = MLP(p).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, fused=True)
        loss_fn = nn.MSELoss()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        best_val = float('inf')
        wait = 0
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    v_loss = loss_fn(model(X_val), y_val)
                    if v_loss < best_val:
                        best_val = v_loss
                        wait = 0
                    else:
                        wait += 1
                if wait >= (PATIENCE // 5): break
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            final_pred = model(X_te).cpu().numpy()
            y_te_np = y_te.cpu().numpy()

        rows.append({
            "dataset": tag, "round": rnd, "method": "torch_gpu_data_optimized",
            "train_time": elapsed, "adj_r2": adj_r2(y_te_np, final_pred, p),
            "rmse": float(np.sqrt(mean_squared_error(y_te_np, final_pred))),
            "mae": float(mean_absolute_error(y_te_np, final_pred)),
        })
        print(f"  round {rnd}  time={elapsed:.4f}s  R2={rows[-1]['adj_r2']:.4f}")

    return rows

if __name__ == "__main__":
    datasets = sorted(glob.glob(os.path.join(DATA_ROOT, "*/data.csv")))
    all_rows = []
    for path in datasets:
        print(f"\n=== {os.path.basename(os.path.dirname(path))} ===")
        all_rows.extend(process(path))
    pd.DataFrame(all_rows).to_csv("results_torch_GPU_split.csv", index=False)
