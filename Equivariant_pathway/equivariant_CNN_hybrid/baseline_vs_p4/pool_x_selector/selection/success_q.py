"""Goal-conditioned success-Q for ThriftyDAgger risk estimation.

ThriftyDAgger gates expert queries on **task risk** — the probability that
the current policy will fail to reach the goal from a state. We estimate it
with a small goal-conditioned value head that predicts ``P(reach goal | s, a)``
and define ``risk(s, a) = 1 - Q(s, a)``.

This is the **discrete-maze, demonstration-acquisition** adaptation of the
CoRL'21 success-Q:

* Features = the 5-channel grid encoding the equivariant policy already sees
  (``Equivariant_pathway.encoder.encode_from_env``; the goal channel makes it
  goal-conditioned, so no extra goal vector is needed) flattened, concatenated
  with a one-hot of the action.
* Target = a **Monte-Carlo success-to-go** label ``gamma^(remaining-1) * 1[success]``
  computed over (a) the cumulative corrective demos (all reach the goal, so
  success=1) and (b) this round's correction-pool rollouts (success or
  failure). This is a heuristic Q (not fitted by Bellman backup); it is only
  used to **rank** layouts, so absolute calibration does not matter — the
  relative ordering of risk across layouts is what gates the query.

The net is warm-started across rounds and persisted at
``<method_root>/success_q/q_net.pth``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import (  # noqa: E402
    baseline_budget as _bb,  # _build_env, encode_grid_from_env
)

NUM_ACTIONS = 4


class SuccessQNet(nn.Module):
    """MLP predicting logit of P(reach goal | s, a).

    Input = flatten(grid_encoding) ++ one_hot(action). Output = scalar logit
    (apply sigmoid for the probability).
    """

    def __init__(self, grid_flat_dim: int, num_actions: int = NUM_ACTIONS,
                 hidden: int = 128):
        super().__init__()
        self.grid_flat_dim = int(grid_flat_dim)
        self.num_actions = int(num_actions)
        self.net = nn.Sequential(
            nn.Linear(self.grid_flat_dim + self.num_actions, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, grid_feat: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([grid_feat, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# Transition construction (replay envs to recover the exact grid encodings)
# --------------------------------------------------------------------------- #
def _encode_trajectory(layout: Dict, actions: Sequence[int],
                       seed: int) -> List[np.ndarray]:
    """Rebuild the env from ``layout`` and replay ``actions``, returning the
    grid encoding seen BEFORE each executed step (one per action taken).

    Uses the same env-build + encoder primitives as ``_rollout_with_loss`` so
    the features match what the policy/ensemble consumes at decision time.
    """
    env = _bb._build_env(layout, seed=int(seed))
    encs: List[np.ndarray] = []
    terminated = truncated = False
    for a in actions:
        if terminated or truncated:
            break
        encs.append(_bb.encode_grid_from_env(env).astype(np.float32))
        _, _, terminated, truncated, _ = env.step(int(a))
    env.close()
    return encs


def build_transitions(
    demo_dir: Path,
    round_records: Sequence[Dict],
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble (X_grid, X_act_onehot, y) from cumulative demos + this round's
    rollouts.

    * Demos (``demo_dir/**/*.json``) reach the goal → ``success=1``.
    * ``round_records`` are the pool rollouts; each must carry
      ``layout``, ``actions``, ``ep_seed``, ``success``.

    Target per pre-step state ``t`` of a length-``n`` trajectory:
        ``y_t = 1[success] * gamma^(n - 1 - t)``  (1.0 at the last pre-terminal
    state of a successful trajectory, decaying backwards; 0 everywhere for a
    failed trajectory).
    """
    grids: List[np.ndarray] = []
    acts: List[int] = []
    ys: List[float] = []

    def _add(layout: Dict, actions: Sequence[int], seed: int, success: bool) -> None:
        actions = [int(a) for a in (actions or [])]
        if not actions:
            return
        encs = _encode_trajectory(layout, actions, seed)
        n = len(encs)
        if n == 0:
            return
        succ = 1.0 if success else 0.0
        for t in range(n):
            grids.append(encs[t].reshape(-1))
            acts.append(int(actions[t]))
            ys.append(succ * float(gamma) ** (n - 1 - t))

    # (a) cumulative demos
    for p in sorted(Path(demo_dir).rglob("*.json")):
        try:
            payload = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            continue
        layout = {
            "start_pos": payload.get("start_pos"),
            "goal_pos": payload.get("goal_pos"),
            "fire_positions": payload.get("fire_positions") or [],
        }
        if layout["start_pos"] is None or layout["goal_pos"] is None:
            continue
        # Demos store actions as the executed (prefix + BFS) action list.
        _add(layout, payload.get("actions") or [],
             seed=int(payload.get("timestamp", 0)) % 100000,
             success=bool(payload.get("success", True)))

    # (b) this round's pool rollouts
    for rec in round_records:
        _add(rec.get("layout") or {}, rec.get("actions") or [],
             seed=int(rec.get("ep_seed", 0)), success=bool(rec.get("success", False)))

    if not grids:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, NUM_ACTIONS), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))

    X_grid = np.stack(grids).astype(np.float32)
    X_act = np.zeros((len(acts), NUM_ACTIONS), dtype=np.float32)
    for i, a in enumerate(acts):
        if 0 <= a < NUM_ACTIONS:
            X_act[i, a] = 1.0
    y = np.asarray(ys, dtype=np.float32)
    return X_grid, X_act, y


# --------------------------------------------------------------------------- #
# Train / load / score
# --------------------------------------------------------------------------- #
def _ckpt_path(method_root: Path) -> Path:
    return Path(method_root) / "success_q" / "q_net.pth"


def train_success_q(
    method_root: Path,
    demo_dir: Path,
    round_records: Sequence[Dict],
    gamma: float,
    epochs: int,
    lr: float,
    device: torch.device,
    batch_size: int = 256,
    seed: int = 0,
) -> Optional[Path]:
    """Train (warm-started) the success-Q on demo + rollout transitions and
    persist it. Returns the checkpoint path, or ``None`` if no transitions.
    """
    X_grid, X_act, y = build_transitions(demo_dir, round_records, gamma)
    if X_grid.shape[0] == 0:
        return None
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    flat_dim = int(X_grid.shape[1])
    out_path = _ckpt_path(method_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = SuccessQNet(grid_flat_dim=flat_dim, num_actions=NUM_ACTIONS).to(device)
    if out_path.exists():
        try:
            ck = torch.load(str(out_path), map_location=device, weights_only=False)
            if int(ck.get("grid_flat_dim", -1)) == flat_dim:
                model.load_state_dict(ck["model_state_dict"], strict=True)
        except Exception:  # noqa: BLE001 — warm-start is best-effort
            pass

    Xg = torch.from_numpy(X_grid).to(device)
    Xa = torch.from_numpy(X_act).to(device)
    yt = torch.from_numpy(y).to(device)
    n = Xg.shape[0]
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    model.train()
    for _ep in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            logit = model(Xg[idx], Xa[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yt[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()

    torch.save({
        "model_state_dict": model.state_dict(),
        "grid_flat_dim": flat_dim,
        "num_actions": NUM_ACTIONS,
        "gamma": float(gamma),
        "n_transitions": int(n),
    }, out_path)
    with open(out_path.parent / "q_train_log.json", "w") as f:
        json.dump({"n_transitions": int(n), "epochs": int(epochs),
                   "lr": float(lr), "gamma": float(gamma),
                   "grid_flat_dim": flat_dim}, f, indent=2)
    return out_path


def load_q(method_root: Path, device: torch.device) -> Optional[SuccessQNet]:
    p = _ckpt_path(method_root)
    if not p.exists():
        return None
    try:
        ck = torch.load(str(p), map_location=device, weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    model = SuccessQNet(grid_flat_dim=int(ck["grid_flat_dim"]),
                        num_actions=int(ck.get("num_actions", NUM_ACTIONS))).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model


def q_values_for_steps(
    qnet: SuccessQNet,
    grid_encs: Sequence[np.ndarray],
    actions: Sequence[int],
    device: torch.device,
) -> List[float]:
    """Return P(success) in [0,1] for each (grid_enc, action) pair.

    ``grid_encs`` are (C,H,W) arrays (e.g. captured during the ensemble
    rollout); ``actions`` the executed action per step. Used by ThriftyDAgger
    to turn a layout's visited states into a risk score (risk = 1 - Q).
    """
    if not grid_encs:
        return []
    Xg = torch.from_numpy(
        np.stack([g.reshape(-1).astype(np.float32) for g in grid_encs])
    ).to(device)
    Xa = torch.zeros((len(actions), qnet.num_actions), dtype=torch.float32, device=device)
    for i, a in enumerate(actions):
        a = int(a)
        if 0 <= a < qnet.num_actions:
            Xa[i, a] = 1.0
    with torch.no_grad():
        probs = torch.sigmoid(qnet(Xg, Xa))
    return [float(x) for x in probs.detach().cpu().numpy().reshape(-1)]
