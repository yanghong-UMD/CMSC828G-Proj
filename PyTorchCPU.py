import time
import numpy as np
import pandas as pd
import os, glob
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 42
N_ROUNDS = 10
DATA_ROOT = "data"
MAX_EPOCHS = 500 
LR = 0.01 
PATIENCE = 20 

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
        X_tr = torch.from_numpy(sc.fit_transform(X_tr))
        X_te = torch.from_numpy(sc.transform(X_te))
        y_tr = torch.from_numpy(y_tr)
        y_te = torch.from_numpy(y_te)

        # train & val
        X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=seed)

        model = MLP(p, hidden_sizes=(64, 32)) 
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        t0 = time.perf_counter()
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # Seperate data into batches
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=2)   # NOTE Change this

        for epoch in range(MAX_EPOCHS):
            model.train()

						# batch
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()

            # Early Stopping 
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, y_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # 可以在这里保存最佳权重，但为了速度，我们直接继续
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= PATIENCE:
                    break
        
        elapsed = time.perf_counter() - t0

        # 评估
        model.eval()
        with torch.no_grad():
            final_pred = model(X_te).numpy()
            y_te_np = y_te.numpy()

        rows.append({
            "dataset":     tag,
            "round":       rnd,
            "method":      "torch_optimized",
            "train_time":  elapsed,
            "adj_r2":      adj_r2(y_te_np, final_pred, p),
            "rmse":        float(np.sqrt(mean_squared_error(y_te_np, final_pred))),
            "mae":         float(mean_absolute_error(y_te_np, final_pred)),
        })
        print(f"  round {rnd}  time={elapsed:.2f}s  R2={rows[-1]['adj_r2']:.4f}")

    return rows

if __name__ == "__main__":
    datasets = sorted(glob.glob(os.path.join(DATA_ROOT, "*/data.csv")))
    if not datasets:
        raise FileNotFoundError(f"no data found under {DATA_ROOT}/")

    all_rows = []
    for path in datasets:
        print(f"\n=== {os.path.basename(os.path.dirname(path))} ===")
        all_rows.extend(process(path))

    pd.DataFrame(all_rows).to_csv("results_torch.csv", index=False)
    print(f"\ndone. {len(all_rows)} rows -> results_torch.csv")
