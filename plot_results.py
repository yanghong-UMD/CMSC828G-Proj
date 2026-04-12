import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

FILES = {
    "results_lr.csv":            "LR",
    "results_xgb.csv":           "XGBoost",
    "results_sklearn_fixed.csv": "sklearn MLP",
    "results_torch.csv":         "PyTorch (CPU)",
    "results_torch_GPU.csv":     "PyTorch (GPU)",
}

dfs = []
for fname, label in FILES.items():
    df = pd.read_csv(fname)
    if "N" not in df.columns:
        df["n_feats"] = df["dataset"].str.extract(r"feats(\d+)").astype(int)
        df["snr"]     = df["dataset"].str.extract(r"snr(\d+)").astype(int)
        df["N"]       = df["dataset"].str.extract(r"N(\d+)").astype(int)
    df["label"] = label
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

Ns    = [10_000, 100_000, 500_000]
XLABS = ["10K", "100K", "500K"]

COLORS  = {"LR": "#999999", "XGBoost": "#E07B39", "sklearn MLP": "#4878CF",
            "PyTorch (CPU)": "#6ACC65", "PyTorch (GPU)": "#D65F5F"}
MARKERS = {"LR": "s", "XGBoost": "D", "sklearn MLP": "o",
           "PyTorch (CPU)": "^", "PyTorch (GPU)": "v"}

def avg(label, col, N=None, snr=None):
    mask = data["label"] == label
    if N   is not None: mask &= data["N"]   == N
    if snr is not None: mask &= data["snr"] == snr
    return data.loc[mask, col].mean()


# Fig 1 – quality and time: LR / XGBoost / sklearn MLP
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for m in ["LR", "XGBoost", "sklearn MLP"]:
    axes[0].plot(range(3), [avg(m, "adj_r2", N=n) for n in Ns],
                 marker=MARKERS[m], color=COLORS[m], label=m, linewidth=2, markersize=7)
    axes[1].plot(range(3), [avg(m, "train_time", N=n) for n in Ns],
                 marker=MARKERS[m], color=COLORS[m], label=m, linewidth=2, markersize=7)

axes[0].set_ylim(0.76, 0.84)
axes[0].set_ylabel("Mean adjusted $R^2$")
axes[1].set_yscale("log")
axes[1].set_ylabel("Training time (s, log scale)")
for ax in axes:
    ax.set_xticks(range(3)); ax.set_xticklabels(XLABS)
    ax.set_xlabel("Dataset size $N$")
    ax.legend(); ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("fig1.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig1.png")


# Fig 2 – sklearn MLP vs PyTorch CPU: time and speedup
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for m in ["sklearn MLP", "PyTorch (CPU)"]:
    axes[0].plot(range(3), [avg(m, "train_time", N=n) for n in Ns],
                 marker=MARKERS[m], color=COLORS[m], label=m, linewidth=2, markersize=7)

speedups = [avg("sklearn MLP", "train_time", N=n) /
            avg("PyTorch (CPU)", "train_time", N=n) for n in Ns]
bars = axes[1].bar(range(3), speedups, color=COLORS["sklearn MLP"], alpha=0.85, width=0.5)
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="break-even")
for b in bars:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                 f"{b.get_height():.1f}×", ha="center", va="bottom", fontsize=10)

axes[0].set_yscale("log")
axes[0].set_ylabel("Training time (s, log scale)")
axes[1].set_ylabel("Speedup (sklearn time / PyTorch CPU time)")
for ax in axes:
    ax.set_xticks(range(3)); ax.set_xticklabels(XLABS)
    ax.set_xlabel("Dataset size $N$")
    ax.legend(); ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("fig2.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig2.png")


# Fig 3 – PyTorch CPU vs GPU: time and speedup
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for m in ["PyTorch (CPU)", "PyTorch (GPU)"]:
    axes[0].plot(range(3), [avg(m, "train_time", N=n) for n in Ns],
                 marker=MARKERS[m], color=COLORS[m], label=m, linewidth=2, markersize=7)

gpu_speedups = [avg("PyTorch (CPU)", "train_time", N=n) /
                avg("PyTorch (GPU)", "train_time", N=n) for n in Ns]
bar_colors = [COLORS["PyTorch (GPU)"] if s >= 1 else "#cccccc" for s in gpu_speedups]
bars = axes[1].bar(range(3), gpu_speedups, color=bar_colors, alpha=0.9, width=0.5)
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="break-even")
for b, s in zip(bars, gpu_speedups):
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.04,
                 f"{s:.2f}×" if s < 2 else f"{s:.1f}×",
                 ha="center", va="bottom", fontsize=10)

axes[0].set_yscale("log")
axes[0].set_ylabel("Training time (s, log scale)")
axes[1].set_ylabel("Speedup (CPU time / GPU time)")
axes[1].set_title("grey bar = GPU slower than CPU", fontsize=10)
for ax in axes:
    ax.set_xticks(range(3)); ax.set_xticklabels(XLABS)
    ax.set_xlabel("Dataset size $N$")
    ax.legend(); ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("fig3.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig3.png")