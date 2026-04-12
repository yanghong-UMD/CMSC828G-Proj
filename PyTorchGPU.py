import time
import numpy as np
import pandas as pd
import os, glob
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def adj_r2(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

def process(csv_path):
    df = pd.read_csv(csv_path)
    X_raw = df.drop(columns="y").values.astype(np.float32)
    y_raw = df["y"].values.astype(np.float32)
    p = X_raw.shape[1]

    spl = os.path.basename(os.path.dirname(csv_path)).split("_")
    tag = "_".join(spl)
    
    rows = []
    for rnd in range(N_ROUNDS):
        seed = SEED + rnd
        torch.manual_seed(seed)
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.5, random_state=seed)
        
        # pre
        sc = StandardScaler()
        X_tr = torch.from_numpy(sc.fit_transform(X_tr)).to(DEVICE)
        X_te = torch.from_numpy(sc.transform(X_te)).to(DEVICE)
        y_tr = torch.from_numpy(y_tr).to(DEVICE)
        y_te = torch.from_numpy(y_te).to(DEVICE)

        # train & val
        X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=seed)

        model = MLP(p, hidden_sizes=(128, 64)).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # 核心训练循环
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            
            pred = model(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()

            #early stopping
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(model(X_val), y_val)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                
                if epochs_no_improve >= (PATIENCE // 5):
                    break
        
        torch.cuda.synchronize() #sync
        elapsed = time.perf_counter() - t0

        # eval
        model.eval()
        with torch.no_grad():
            final_pred = model(X_te).cpu().numpy()
            y_te_np = y_te.cpu().numpy()

        rows.append({
            "dataset":     tag,
            "round":       rnd,
            "method":      "torch_gpu_optimized",
            "train_time":  elapsed,
            "adj_r2":      adj_r2(y_te_np, final_pred, p),
            "rmse":        float(np.sqrt(mean_squared_error(y_te_np, final_pred))),
            "mae":         float(mean_absolute_error(y_te_np, final_pred)),
        })
        print(f"  round {rnd}  time={elapsed:.4f}s  R2={rows[-1]['adj_r2']:.4f}")

    return rows

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    datasets = sorted(glob.glob(os.path.join(DATA_ROOT, "*/data.csv")))
    if not datasets:
        raise FileNotFoundError(f"no data found under {DATA_ROOT}/")

    all_rows = []
    for path in datasets:
        print(f"\n=== {os.path.basename(os.path.dirname(path))} ===")
        all_rows.extend(process(path))

    pd.DataFrame(all_rows).to_csv("results_torch_GPU.csv", index=False)
    print(f"\ndone. {len(all_rows)} rows -> results_torch_GPU.csv")