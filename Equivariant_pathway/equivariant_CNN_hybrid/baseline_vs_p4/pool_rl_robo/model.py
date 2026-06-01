"""Continuous-control novice policy (the local analogue of the maze
``Equivariant_pathway.equivariant_CNN_hybrid.model`` that pool_x_selector
borrows from upstream — here there is no upstream policy, so it lives in-suite).

``MLPPolicy`` maps a flat observation to a tanh-squashed continuous action,
rescaled to the env action bounds. Optional dropout layers make MC-dropout
*real* (full-network), a fidelity improvement over the maze fusion-head-only
dropout.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def set_dropout_active(model: nn.Module, active: bool) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(active)


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 n_hidden: int = 2, dropout: float = 0.0,
                 act_low: Optional[np.ndarray] = None,
                 act_high: Optional[np.ndarray] = None):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        layers: List[nn.Module] = []
        d = self.obs_dim
        for _ in range(int(n_hidden)):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            d = hidden
        layers.append(nn.Linear(d, self.act_dim))
        self.net = nn.Sequential(*layers)
        lo = np.full(self.act_dim, -1.0, np.float32) if act_low is None else np.asarray(act_low, np.float32).ravel()
        hi = np.full(self.act_dim, 1.0, np.float32) if act_high is None else np.asarray(act_high, np.float32).ravel()
        self.register_buffer("act_low", torch.from_numpy(lo))
        self.register_buffer("act_high", torch.from_numpy(hi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.tanh(self.net(x))
        return self.act_low + (t + 1.0) * 0.5 * (self.act_high - self.act_low)

    @torch.no_grad()
    def act(self, obs_flat: np.ndarray, device: torch.device) -> np.ndarray:
        was_training = self.training
        self.eval()
        set_dropout_active(self, False)
        x = torch.from_numpy(np.asarray(obs_flat, np.float32)).unsqueeze(0).to(device)
        a = self.forward(x).squeeze(0).cpu().numpy().astype(np.float32)
        if was_training:
            self.train()
        return a
