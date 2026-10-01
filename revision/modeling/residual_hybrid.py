"""Original residual-LSTM training and recursive correction function.

Function body is unchanged from the 1 October AIC experiment.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import model_core as core

def residual_lstm(train: pd.DataFrame, future: pd.DataFrame, base_result,
                  base_prediction: np.ndarray, candidate: dict, seed: int) -> np.ndarray:
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    residuals = train.hsr_pax_total.to_numpy(float) / core.SCALE - np.asarray(base_result.fittedvalues)
    lookback = candidate["lookback"]
    assert len(residuals) > lookback
    mean, std = float(residuals.mean()), float(residuals.std())
    if std <= 1e-12:
        raise ValueError("zero residual variance")
    scaled = ((residuals - mean) / std).astype(np.float32)
    windows = np.asarray([scaled[i-lookback:i] for i in range(lookback, len(scaled))], dtype=np.float32)
    labels = scaled[lookback:]

    class ResidualNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(1, 16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            encoded, _ = self.encoder(x)
            return self.head(encoded[:, -1, :]).squeeze(1)

    net = ResidualNet()
    opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
    x_t = torch.from_numpy(windows).unsqueeze(-1)
    y_t = torch.from_numpy(labels)
    for _ in range(candidate["epochs"]):
        net.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x_t), y_t)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite residual training loss")
        loss.backward()
        opt.step()
    net.eval()
    history = scaled.tolist()
    correction = []
    for _ in range(len(future)):
        x = torch.tensor(history[-lookback:], dtype=torch.float32).reshape(1, lookback, 1)
        with torch.no_grad():
            guess = float(net(x).item())
        correction.append((guess * std + mean) * core.SCALE)
        history.append(guess)
    return np.asarray(base_prediction, float) + np.asarray(correction, float)
