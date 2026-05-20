"""
Visualize reconstruction results: Linear Interpolation vs BiLSTM vs Ground Truth.
Saves figures to ./figures/
"""
import os
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import TrajectoryBiLSTM

os.makedirs("figures", exist_ok=True)

COLORS = {
    "gt":     "#2ecc71",
    "linear": "#e74c3c",
    "lstm":   "#3498db",
}


def load_test_data(cfg):
    saved = np.load(cfg["data"]["processed_file"], allow_pickle=True).item()
    X_all = torch.tensor(saved["X"])
    y_all = torch.tensor(saved["y"])
    n = len(X_all)
    n_train = int(n * cfg["train"]["train_ratio"])
    n_val   = int(n * cfg["train"]["val_ratio"])
    n_test  = n - n_train - n_val
    ds = TensorDataset(X_all, y_all)
    _, _, test_ds = random_split(ds, [n_train, n_val, n_test],
                                 generator=torch.Generator().manual_seed(42))
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])
    return loader


def get_predictions(cfg, loader):
    device = torch.device("cpu")
    model = TrajectoryBiLSTM(
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
    )
    model.load_state_dict(torch.load(cfg["output"]["model_path"], map_location=device))
    model.eval()

    lstm_preds, lin_preds, gts, inputs = [], [], [], []
    W = cfg["model"]["window"]
    half = W // 2

    with torch.no_grad():
        for xb, yb in loader:
            lstm_preds.append(model(xb).numpy())
            lin_preds.append(((xb[:, half-1, :] + xb[:, half, :]) / 2).numpy())
            gts.append(yb.numpy())
            inputs.append(xb.numpy())

    return (
        np.concatenate(inputs),
        np.concatenate(gts),
        np.concatenate(lin_preds),
        np.concatenate(lstm_preds),
    )


# ── Figure 1: Metric comparison bar chart ────────────────────────────────────

def plot_metrics(gts, lin_preds, lstm_preds):
    def rmse(p, g): return float(np.sqrt(((p - g) ** 2).sum(axis=1).mean()))
    def mae(p, g):  return float(np.linalg.norm(p - g, axis=1).mean())

    methods = ["Linear\nInterpolation", "LSTM\n(BiLSTM)"]
    rmses   = [rmse(lin_preds, gts), rmse(lstm_preds, gts)]
    maes    = [mae(lin_preds,  gts), mae(lstm_preds,  gts)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    c = [COLORS["linear"], COLORS["lstm"]]

    for ax, vals, title, unit in zip(
        axes,
        [rmses, maes],
        ["RMSE", "Mean Point Distance (Fréchet proxy)"],
        ["m", "m"],
    ):
        bars = ax.bar(methods, vals, color=c, width=0.4, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(f"Error ({unit})")
        ax.set_ylim(0, max(vals) * 1.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Reconstruction Error: Linear Interpolation vs BiLSTM", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/01_metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/01_metric_comparison.png")


# ── Figure 2: Error distribution (histogram) ─────────────────────────────────

def plot_error_dist(gts, lin_preds, lstm_preds):
    lin_err  = np.linalg.norm(lin_preds  - gts, axis=1)
    lstm_err = np.linalg.norm(lstm_preds - gts, axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, np.percentile(np.concatenate([lin_err, lstm_err]), 99), 80)
    ax.hist(lin_err,  bins=bins, alpha=0.6, color=COLORS["linear"], label="Linear Interpolation", density=True)
    ax.hist(lstm_err, bins=bins, alpha=0.6, color=COLORS["lstm"],   label="LSTM (BiLSTM)",        density=True)
    ax.set_xlabel("Positional Error (m)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution on Test Set", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/02_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/02_error_distribution.png")


# ── Figure 3: Sample trajectory reconstruction ───────────────────────────────

def plot_trajectories(inputs, gts, lin_preds, lstm_preds, n_samples=6):
    """
    For each sample, reconstruct a short trajectory segment:
      retained frames (context) + the predicted/GT midpoint.
    """
    W = inputs.shape[1]
    half = W // 2
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(gts), size=n_samples, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for ax, idx in zip(axes, idxs):
        ctx = inputs[idx]          # (W, 2)  retained frames
        gt  = gts[idx]             # (2,)
        lin = lin_preds[idx]       # (2,)
        lstm = lstm_preds[idx]     # (2,)

        # retained frames as trajectory
        ax.plot(ctx[:, 0], ctx[:, 1], "o-", color="gray", lw=1.2,
                ms=4, label="Retained frames", zorder=1)

        # gap position (between frame half-1 and half)
        p_before = ctx[half - 1]
        p_after  = ctx[half]
        ax.plot([p_before[0], p_after[0]], [p_before[1], p_after[1]],
                "--", color="gray", lw=0.8, zorder=1)

        # predictions
        ax.scatter(*gt,   color=COLORS["gt"],     s=100, zorder=5, marker="*",  label="Ground Truth")
        ax.scatter(*lin,  color=COLORS["linear"], s=80,  zorder=4, marker="^",  label="Linear Interp.")
        ax.scatter(*lstm, color=COLORS["lstm"],   s=80,  zorder=4, marker="s",  label="LSTM")

        ax.set_title(f"Sample #{idx}", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.spines[["top", "right"]].set_visible(False)

    # shared legend
    handles = [
        mpatches.Patch(color="gray",          label="Retained frames"),
        mpatches.Patch(color=COLORS["gt"],    label="Ground Truth"),
        mpatches.Patch(color=COLORS["linear"],label="Linear Interp."),
        mpatches.Patch(color=COLORS["lstm"],  label="LSTM (BiLSTM)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Trajectory Segment Reconstruction (6 random test samples)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("figures/03_trajectory_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/03_trajectory_samples.png")


# ── Figure 4: X/Y error separately ───────────────────────────────────────────

def plot_xy_error(gts, lin_preds, lstm_preds):
    lin_ex  = np.abs(lin_preds[:,0]  - gts[:,0]);  lin_ey  = np.abs(lin_preds[:,1]  - gts[:,1])
    lstm_ex = np.abs(lstm_preds[:,0] - gts[:,0]);  lstm_ey = np.abs(lstm_preds[:,1] - gts[:,1])

    labels = ["X-axis error (m)", "Y-axis error (m)"]
    lin_vals  = [lin_ex.mean(),  lin_ey.mean()]
    lstm_vals = [lstm_ex.mean(), lstm_ey.mean()]

    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, lin_vals,  w, label="Linear Interp.", color=COLORS["linear"], edgecolor="white")
    ax.bar(x + w/2, lstm_vals, w, label="LSTM (BiLSTM)", color=COLORS["lstm"],   edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Absolute Error (m)")
    ax.set_title("Per-Axis Reconstruction Error", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    for xi, (lv, sv) in enumerate(zip(lin_vals, lstm_vals)):
        ax.text(xi - w/2, lv + 0.001, f"{lv:.4f}", ha="center", fontsize=9)
        ax.text(xi + w/2, sv + 0.001, f"{sv:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/04_xy_error.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/04_xy_error.png")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Loading test data & running inference...")
    loader = load_test_data(cfg)
    inputs, gts, lin_preds, lstm_preds = get_predictions(cfg, loader)
    print(f"Test samples: {len(gts)}")

    print("\nGenerating figures...")
    plot_metrics(gts, lin_preds, lstm_preds)
    plot_error_dist(gts, lin_preds, lstm_preds)
    plot_trajectories(inputs, gts, lin_preds, lstm_preds)
    plot_xy_error(gts, lin_preds, lstm_preds)

    print("\nDone! All figures saved to ./figures/")
