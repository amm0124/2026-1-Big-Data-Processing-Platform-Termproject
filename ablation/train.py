"""
Train the BiLSTM model on downsampled soccer tracking data.
"""
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import TrajectoryBiLSTM


def load_data(path, train_ratio, val_ratio):
    saved = np.load(path, allow_pickle=True).item()
    X = torch.tensor(saved["X"])   # (N, W, 2)
    y = torch.tensor(saved["y"])   # (N, 2)

    n = len(X)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    ds = TensorDataset(X, y)
    return random_split(ds, [n_train, n_val, n_test],
                        generator=torch.Generator().manual_seed(42))


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, _ = load_data(
        cfg["data"]["processed_file"],
        cfg["train"]["train_ratio"],
        cfg["train"]["val_ratio"],
    )

    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs)

    model = TrajectoryBiLSTM(
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg["output"]["model_path"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

    print(f"\nBest val loss: {best_val:.4f} — model saved to {cfg['output']['model_path']}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
