"""Goal-conditioned success-Q for ThriftyDAgger risk estimation (continuous).

Port of maze ``selection/success_q.py``. ThriftyDAgger gates queries on **task
risk** = ``1 − Q(s, a)`` where Q predicts ``P(reach goal | s, a)``. We fit a
small MLP on a **Monte-Carlo success-to-go** target

    y_t = 1[success] · gamma^(n−1−t)

over (a) the expert demonstration trajectories (success=1) and (b) this round's
novice rollouts (measured success). Unlike the maze version we already hold the
flat observation vectors from the rollout, so no env replay is needed: the net
input is ``concat(flatten(obs), action_vector)``.

The Q is heuristic (not Bellman-fitted) — it is only used to *rank* candidate
states, so absolute calibration does not matter, only the relative risk order.
For locomotion envs ``success`` is the survival proxy (see ``envs.episode_success``);
on HalfCheetah it is always 1, so risk saturates to 0 there (documented caveat).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SuccessQNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


def build_transitions(
    demo_trajs: Sequence[Dict],
    round_episodes: Sequence[Dict],
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble (X_obs, X_act, y) with MC success-to-go targets from expert
    demos (success=1) + this round's novice rollouts (measured success).
    Each trajectory dict carries ``obs`` (n,obs_dim), ``act`` (n,act_dim),
    ``success`` (bool)."""
    Xo: List[np.ndarray] = []
    Xa: List[np.ndarray] = []
    ys: List[float] = []

    def _add(traj: Dict) -> None:
        obs_raw = traj.get("feat", traj.get("obs"))      # policy features (or legacy obs)
        act_raw = traj.get("act", traj.get("actions"))   # demos use 'act', rollouts 'actions'
        if obs_raw is None or act_raw is None:
            return
        obs = np.asarray(obs_raw, np.float32)
        act = np.asarray(act_raw, np.float32)
        if obs.ndim != 2 or act.ndim != 2 or obs.shape[0] == 0 or act.shape[0] != obs.shape[0]:
            return
        n = obs.shape[0]
        succ = 1.0 if traj.get("success") else 0.0
        for t in range(n):
            Xo.append(obs[t])
            Xa.append(act[t])
            ys.append(succ * float(gamma) ** (n - 1 - t))

    for tr in demo_trajs:
        _add(tr)
    for ep in round_episodes:
        _add(ep)

    if not Xo:
        return (np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32), np.zeros((0,), np.float32))
    return np.stack(Xo).astype(np.float32), np.stack(Xa).astype(np.float32), np.asarray(ys, np.float32)


def train_success_q(qnet: SuccessQNet, demo_trajs: Sequence[Dict],
                    round_episodes: Sequence[Dict], *, gamma: float, epochs: int,
                    lr: float, batch_size: int, device: torch.device,
                    seed: int = 0) -> Optional[SuccessQNet]:
    """Warm-started BCE fit of the success-Q in place. Returns the net, or
    ``None`` if there were no transitions."""
    Xo, Xa, y = build_transitions(demo_trajs, round_episodes, gamma)
    if Xo.shape[0] == 0:
        return None
    torch.manual_seed(int(seed))
    Xot = torch.from_numpy(Xo).to(device)
    Xat = torch.from_numpy(Xa).to(device)
    yt = torch.from_numpy(y).to(device)
    n = Xot.shape[0]
    bs = max(1, int(batch_size))
    opt = torch.optim.AdamW(qnet.parameters(), lr=float(lr), weight_decay=1e-4)
    qnet.train()
    for _ep in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            logit = qnet(Xot[idx], Xat[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    qnet.eval()
    return qnet


@torch.no_grad()
def q_values(qnet: SuccessQNet, obs_arr: np.ndarray, act_arr: np.ndarray,
             device: torch.device) -> List[float]:
    """P(success) in [0,1] for each (obs, action) row."""
    obs_arr = np.asarray(obs_arr, np.float32)
    act_arr = np.asarray(act_arr, np.float32)
    if obs_arr.shape[0] == 0:
        return []
    Xo = torch.from_numpy(obs_arr).to(device)
    Xa = torch.from_numpy(act_arr).to(device)
    probs = torch.sigmoid(qnet(Xo, Xa)).cpu().numpy().reshape(-1)
    return [float(x) for x in probs]
