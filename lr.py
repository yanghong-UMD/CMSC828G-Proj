import numpy as np
import pandas as pd
import os, time, glob

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 42
N_ROUNDS = 10
DATA_ROOT = "data"


def adj_r2(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


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
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                               random_state=SEED + rnd)
        t0 = time.perf_counter()
        m = LinearRegression().fit(Xtr, ytr)
        elapsed = time.perf_counter() - t0

        pred = m.predict(Xte)
        rows.append({
            "dataset":    tag,
            "n_feats":    n_feats,
            "snr":        snr,
            "N":          N,
            "round":      rnd,
            "method":     "LR",
            "train_time": elapsed,
            "adj_r2":     adj_r2(yte, pred, p),
            "rmse":       rmse(yte, pred),
            "mae":        mae(yte, pred),
        })
        print(f"  round {rnd}  time={elapsed:.3f}s  "
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

    pd.DataFrame(all_rows).to_csv("results_lr.csv", index=False)
    print(f"\ndone. {len(all_rows)} rows -> results_lr.csv")