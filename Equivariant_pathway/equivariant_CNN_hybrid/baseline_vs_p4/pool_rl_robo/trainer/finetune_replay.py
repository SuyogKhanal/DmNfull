"""Behavior-cloning trainer (the continuous-control analogue of
pool_x_selector's ``trainer/finetune_replay.py`` replay fine-tuner).

The maze trains a heavy EquivariantCNN via a replay-buffer subprocess; here the
novice is a small MLP, so we train IN-PROCESS (warm-started MSE regression of
the novice action onto the expert action over the aggregated demonstration set
``D``). Called by ``selection/iil_baselines.run_dagger`` each round and by
``selection/uncertainty.Ensemble``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..model import set_dropout_active


def train_bc(model, X: np.ndarray, Y: np.ndarray, *, epochs: int, lr: float,
             batch_size: int, weight_decay: float, device: torch.device,
             seed: int = 0) -> float:
    """Warm-started MSE fit of ``model(X) ≈ Y``. Returns the final epoch loss."""
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    if X.shape[0] == 0:
        return 0.0
    torch.manual_seed(int(seed))
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    n = Xt.shape[0]
    bs = max(1, int(batch_size))
    model.train()
    set_dropout_active(model, True)
    last = 0.0
    for _ep in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        last = ep_loss / max(1, nb)
    model.eval()
    set_dropout_active(model, False)
    return last
