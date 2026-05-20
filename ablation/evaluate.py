"""
Evaluate linear interpolation and LSTM on the held-out test split.
Metrics: RMSE, mean discrete Fréchet distance.
"""
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import TrajectoryBiLSTM


# ── Fréchet distance (discrete) ──────────────────────────────────────────────

def frechet_distance(P, Q):
    """Discrete Fréchet distance between two (T, 2) trajectories."""
    m, n = len(P), len(Q)
    ca = np.full((m, n), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        d = np.linalg.norm(P[i] - Q[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(c(i-1,j), c(i-1,j-1), c(i,j-1)), d)
        return ca[i, j]

    return c(m - 1, n - 1)


# ── Metrics over predictions ─────────────────────────────────────────────────

def compute_metrics(preds, targets):
    """
    preds, targets: (N, 2) numpy arrays
    Returns mean RMSE and mean Fréchet distance (treating each row as a
    1-frame trajectory for a point-wise Fréchet, or grouping by trajectory).
    Since we predict single intermediate frames, Fréchet == L2 distance here.
    """
    diff = preds - targets
    rmse = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
    frechet = float(np.linalg.norm(diff, axis=1).mean())  # point-wise
    return rmse, frechet


# ── Linear interpolation baseline ───────────────────────────────────────────

def linear_interp(X):
    """
    X: (N, W, 2) — retained frames; predict midpoint between frame W//2-1 and W//2.
    """
    half = X.shape[1] // 2
    return (X[:, half - 1, :] + X[:, half, :]) / 2.0


# ── Main ─────────────────────────────────────────────────────────────────────

def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # ── LSTM ──
    model = TrajectoryBiLSTM(
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
    ).to(device)
    model.load_state_dict(torch.load(cfg["output"]["model_path"], map_location=device))
    model.eval()

    lstm_preds, lin_preds, gts = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            lstm_preds.append(model(xb).cpu().numpy())
            lin_preds.append(linear_interp(xb.cpu().numpy()))
            gts.append(yb.numpy())

    lstm_preds = np.concatenate(lstm_preds)
    lin_preds  = np.concatenate(lin_preds)
    gts        = np.concatenate(gts)

    lstm_rmse, lstm_fd = compute_metrics(lstm_preds, gts)
    lin_rmse,  lin_fd  = compute_metrics(lin_preds,  gts)

    result = (
        f"Test samples: {len(gts)}\n"
        f"\n{'Method':<22} {'RMSE (m)':>10} {'Fréchet (m)':>12}\n"
        f"{'-'*46}\n"
        f"{'Linear Interpolation':<22} {lin_rmse:>10.4f} {lin_fd:>12.4f}\n"
        f"{'LSTM (BiLSTM)':<22} {lstm_rmse:>10.4f} {lstm_fd:>12.4f}\n"
    )
    print(result)

    with open(cfg["output"]["results_path"], "w") as f:
        f.write(result)
    print(f"Results saved → {cfg['output']['results_path']}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg)
