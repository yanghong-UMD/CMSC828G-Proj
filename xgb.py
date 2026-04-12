import time
import numpy as np
import pandas as pd
import os, glob
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 42
N_ROUNDS = 10
DATA_ROOT = "data"

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cpu",
    verbosity=0,
)


def adj_r2(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)


def process(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns="y").values
    y = df["y"].values
    p = X.shape[1]

    spl = os.path.basename(os.path.dirname(csv_path)).split("_")
    n_feats = int(spl[0].replace("feats", ""))
    snr     = int(spl[1].replace("snr", ""))
    N       = int(spl[2].replace("N", ""))
    tag     = "_".join(spl)

    rows = []
    for rnd in range(N_ROUNDS):
        seed = SEED + rnd
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=seed)

        model = XGBRegressor(**XGB_PARAMS, random_state=seed)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0

        pred = model.predict(X_te)
        rows.append({
            "dataset":     tag,
            "n_feats":     n_feats,
            "snr":         snr,
            "N":           N,
            "round":       rnd,
            "method":      "xgboost",
            "train_time":  elapsed,
            "adj_r2":      adj_r2(y_te, pred, p),
            "rmse":        float(mean_squared_error(y_te, pred) ** 0.5),
            "mae":         float(mean_absolute_error(y_te, pred)),
            "best_params": "",
        })
        print(f"  round {rnd}  time={elapsed:.2f}s  "
              f"R2={rows[-1]['adj_r2']:.4f}  RMSE={rows[-1]['rmse']:.4f}  MAE={rows[-1]['mae']:.4f}")

    return rows


if __name__ == "__main__":
    datasets = sorted(glob.glob(os.path.join(DATA_ROOT, "*/data.csv")))
    if not datasets:
        raise FileNotFoundError(f"no data found under {DATA_ROOT}/")

    all_rows = []
    for path in datasets:
        print(f"\n=== {os.path.basename(os.path.dirname(path))} ===")
        all_rows.extend(process(path))

    pd.DataFrame(all_rows).to_csv("results_xgb.csv", index=False)
    print(f"\ndone. {len(all_rows)} rows -> results_xgb.csv")