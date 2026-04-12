"""
train_sklearn_fixed.py  –  sklearn MLPRegressor, fixed architecture
Same representative config as train_pytorch_cpu.py:
  hidden=(32,32), activation=tanh, lr=1e-3, alpha=1e-4, batch=128
No HP search – pure framework speed baseline.
"""

import numpy as np
import pandas as pd
import os, time, glob

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED      = 42
N_ROUNDS  = 10
DATA_ROOT = "data"

MODEL_PARAMS = dict(
    hidden_layer_sizes=(32, 32),
    activation="tanh",
    solver="adam",
    alpha=1e-4,
    batch_size=128,
    learning_rate_init=1e-3,
    learning_rate="adaptive",   # ReduceLROnPlateau equivalent
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)


def process(csv_path):
    df  = pd.read_csv(csv_path)
    X   = df.drop(columns="y").values
    y   = df["y"].values
    p   = X.shape[1]
    spl = os.path.basename(os.path.dirname(csv_path)).split("_")
    tag = "_".join(spl)
    n_feats = int(spl[0].replace("feats", ""))
    snr     = int(spl[1].replace("snr",   ""))
    N       = int(spl[2].replace("N",     ""))

    rows = []
    for rnd in range(N_ROUNDS):
        seed = SEED + rnd
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5,
                                               random_state=seed)
        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        model = MLPRegressor(**MODEL_PARAMS, random_state=seed)

        t0 = time.perf_counter()
        model.fit(Xtr_s, ytr)
        elapsed = time.perf_counter() - t0

        pred = model.predict(Xte_s)
        n, r2 = len(yte), r2_score(yte, pred)
        rows.append({
            "dataset":    tag,
            "n_feats":    n_feats,
            "snr":        snr,
            "N":          N,
            "round":      rnd,
            "method":     "sklearn_fixed",
            "train_time": elapsed,
            "adj_r2":     1 - (1 - r2) * (n - 1) / max(n - p - 1, 1),
            "rmse":       float(np.sqrt(mean_squared_error(yte, pred))),
            "mae":        float(mean_absolute_error(yte, pred)),
        })
        print(f"  round {rnd}  time={elapsed:.3f}s  "
              f"R2={rows[-1]['adj_r2']:.4f}  RMSE={rows[-1]['rmse']:.4f}")
    return rows


if __name__ == "__main__":
    datasets = sorted(glob.glob(os.path.join(DATA_ROOT, "*/data.csv")))
    if not datasets:
        raise FileNotFoundError(f"no data found under {DATA_ROOT}/")

    all_rows = []
    for path in datasets:
        print(f"\n=== {os.path.basename(os.path.dirname(path))} ===")
        all_rows.extend(process(path))

    pd.DataFrame(all_rows).to_csv("results_sklearn_fixed.csv", index=False)
    print(f"\ndone. {len(all_rows)} rows -> results_sklearn_fixed.csv")