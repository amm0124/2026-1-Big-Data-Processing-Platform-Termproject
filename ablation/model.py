"""
Bidirectional LSTM for offline intermediate-frame reconstruction.
Input : (batch, window, 2)  — retained frames before and after the gap
Output: (batch, 2)          — predicted (x, y) of the missing frame
"""
import torch
import torch.nn as nn


class TrajectoryBiLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        mid = out[:, out.size(1) // 2, :]   # representation at the gap position
        return self.head(mid)
