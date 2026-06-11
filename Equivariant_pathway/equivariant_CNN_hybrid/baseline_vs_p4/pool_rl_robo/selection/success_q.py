"""Goal-conditioned success-Q for ThriftyDAgger risk (continuous, diffusion backbone).

Risk = 1 − Q(s, a) where Q predicts P(reach goal | s, a). We fit a small MLP on a
Monte-Carlo success-to-go target y_t = 1[success]·gamma^(T−1−t) over the aggregated
demonstration dataset. Because only SUCCESSFUL expert demos enter the dataset, every
stored episode has success=1, so y_t = gamma^(steps-to-end).

This is BEST-EFFORT on the fork's TensorDictReplayBuffer: if the buffer can't be
read into (obs, action, y) transitions, ``train_success_q_from_dataset`` returns
None and ThriftyDAgger degrades to a novelty-only switch rule (documented caveat).
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn


class SuccessQNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


def flat_obs(full_obs: dict, obs_keys: Sequence[str]) -> torch.Tensor:
    """Concatenate the expert's obs keys into a flat (D,) vector (batch row 0)."""
    parts = [full_obs[k] for k in sorted(obs_keys)]
    vec = torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=-1)
    return vec[0].float().detach()


@torch.no_grad()
def q_value(qnet: SuccessQNet, obs_vec: torch.Tensor, act_vec: torch.Tensor) -> float:
    dev = next(qnet.parameters()).device
    o = obs_vec.reshape(1, -1).to(dev)
    a = act_vec.reshape(1, -1).to(dev)
    return float(torch.sigmoid(qnet(o, a)).reshape(-1)[0].item())


def _read_all(dataset) -> Optional[Any]:
    """Try to pull the whole replay buffer as one TensorDict; None on failure."""
    rb = dataset.rb
    for getter in (lambda: rb[: len(rb)], lambda: rb[:],
                   lambda: rb.storage._storage):
        try:
            td = getter()
            if td is not None and len(td) > 0:
                return td
        except Exception:
            continue
    return None


def train_success_q_from_dataset(dataset, obs_keys: Sequence[str], *,
                                  num_joints: int, gamma: float = 0.99,
                                  epochs: int = 50, lr: float = 1e-3,
                                  device: str = "cuda") -> Optional[SuccessQNet]:
    try:
        td = _read_all(dataset)
        if td is None:
            return None
        keys = sorted(obs_keys)
        obs = torch.cat([td[k].reshape(len(td[k]), -1).float() for k in keys], dim=-1)
        act = td["action_joint_delta_pos"][:, :num_joints].float()
        ep = td["episode"].reshape(-1).long()
        # success-to-go: per episode, gamma^(steps-to-end). All demos succeed.
        y = torch.zeros(len(ep), dtype=torch.float32)
        for e in torch.unique(ep):
            idx = (ep == e).nonzero(as_tuple=True)[0]
            n = len(idx)
            for j, t in enumerate(idx.tolist()):
                y[t] = gamma ** (n - 1 - j)
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        qnet = SuccessQNet(obs.shape[1], act.shape[1]).to(dev)
        obs, act, y = obs.to(dev), act.to(dev), y.to(dev)
        opt = torch.optim.AdamW(qnet.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        n = obs.shape[0]
        bs = min(256, n)
        for _ in range(int(epochs)):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(qnet(obs[b], act[b]), y[b])
                loss.backward()
                opt.step()
        qnet.eval()
        return qnet
    except Exception as exc:  # pragma: no cover - GPU/buffer-shape dependent
        print(f"[success_q] best-effort train skipped: {type(exc).__name__}: {exc}")
        return None
