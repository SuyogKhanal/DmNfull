"""Uncertainty machinery for the IIL baselines (continuous-control port of
pool_x_selector's ``selection/uncertainty.py``).

* ``Ensemble`` — M independently-initialised MLP members (EnsembleDAgger +
  ThriftyDAgger novelty). Diversity from a distinct init seed per member (so
  doubt > 0 from round 1, unlike the maze ensemble that started identical) plus
  a distinct shuffle seed each ``train_all``.
* ``mc_dropout_samples`` — N stochastic forward passes (dropout active) for the
  DropoutDAgger ball/probability test. Here dropout is full-network (a fidelity
  improvement over the maze fusion-head-only dropout).

Action discrepancy ``‖a_nov − a_exp‖₂`` is computed in raw action units by the
caller (cleaner than the discrete maze optimal-mask fraction).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from ..model import MLPPolicy, set_dropout_active
from ..trainer.finetune_replay import train_bc


@torch.no_grad()
def mc_dropout_samples(model: MLPPolicy, obs_flat: np.ndarray, N: int,
                       device: torch.device) -> np.ndarray:
    """N stochastic forward passes with dropout ACTIVE -> (N, act_dim)."""
    model.eval()
    set_dropout_active(model, True)
    x = torch.from_numpy(np.asarray(obs_flat, np.float32)).unsqueeze(0).to(device)
    out = np.stack([model.forward(x).squeeze(0).cpu().numpy().astype(np.float32)
                    for _ in range(int(N))])
    set_dropout_active(model, False)
    return out


class Ensemble:
    def __init__(self, M: int, obs_dim: int, act_dim: int, hidden: int = 256,
                 n_hidden: int = 2, dropout: float = 0.0,
                 act_low: Optional[np.ndarray] = None,
                 act_high: Optional[np.ndarray] = None,
                 base_seed: int = 0, device: Optional[torch.device] = None):
        self.M = int(M)
        self.device = device or torch.device("cpu")
        self.members: List[MLPPolicy] = []
        for m in range(self.M):
            torch.manual_seed(int(base_seed) + 101 * m)
            self.members.append(
                MLPPolicy(obs_dim, act_dim, hidden=hidden, n_hidden=n_hidden,
                          dropout=dropout, act_low=act_low, act_high=act_high).to(self.device))

    def train_all(self, X: np.ndarray, Y: np.ndarray, *, base_seed: int,
                  epochs: int, lr: float, batch_size: int, weight_decay: float) -> List[float]:
        return [train_bc(model, X, Y, epochs=epochs, lr=lr, batch_size=batch_size,
                         weight_decay=weight_decay, device=self.device,
                         seed=int(base_seed) + 101 * m)
                for m, model in enumerate(self.members)]

    @torch.no_grad()
    def member_actions(self, obs_flat: np.ndarray, device: torch.device) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs_flat, np.float32)).unsqueeze(0).to(device)
        acts = []
        for model in self.members:
            model.eval()
            set_dropout_active(model, False)
            acts.append(model.forward(x).squeeze(0).cpu().numpy().astype(np.float32))
        return np.stack(acts)

    def mean_action(self, obs_flat: np.ndarray, device: torch.device) -> np.ndarray:
        return self.member_actions(obs_flat, device).mean(axis=0).astype(np.float32)

    def doubt(self, obs_flat: np.ndarray, device: torch.device) -> float:
        ma = self.member_actions(obs_flat, device)
        return float(ma.var(axis=0).mean())
