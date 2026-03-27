import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(42)

CONFIGS = []
for n_feats in [5, 10, 20]:
    for snr in [3, 10]:
        for N in [10_000, 100_000, 500_000]:
            CONFIGS.append((n_feats, snr, N))


def make_signal(X, n_feats):
    beta = rng.standard_normal(n_feats)
    y = X @ beta

    # cross terms
    n_pairs = max(1, n_feats // 5)
    for i in range(n_pairs):
        j, k = (i * 3) % n_feats, (i * 3 + 1) % n_feats
        y += 0.4 * X[:, j] * X[:, k]

    # nonlinear terms
    if n_feats >= 10:
        y += 0.5 * np.sin(X[:, 0]) + 0.3 * X[:, 2] ** 2
    if n_feats >= 20:
        y += 0.4 * np.tanh(X[:, 5]) - 0.3 * X[:, 7] * X[:, 3] ** 2

    return y


def add_noise(y, snr):
    noise_std = np.sqrt(np.var(y) / snr)
    return y + rng.standard_normal(len(y)) * noise_std


def make_dataset(n_feats, snr, N):
    X = rng.standard_normal((N, n_feats))
    y_clean = make_signal(X, n_feats)
    y = add_noise(y_clean, snr)

    cols = [f"x{i}" for i in range(n_feats)]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y
    return df

if __name__ == "__main__":
    base = "data"
    for n_feats, snr, N in CONFIGS:
        tag = f"feats{n_feats}_snr{snr}_N{N}"
        out_dir = os.path.join(base, tag)
        os.makedirs(out_dir, exist_ok=True)

        print(f"generating {tag} ...", end=" ", flush=True)
        df = make_dataset(n_feats, snr, N)
        df.to_csv(os.path.join(out_dir, "data.csv"), index=False)

        print(f"done  shape={df.shape}")

    print(f"\ndone. {len(CONFIGS)} datasets under ./{base}/")