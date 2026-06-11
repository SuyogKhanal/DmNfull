"""Uncertainty draws for the IIL baselines on the diffusion backbone.

A diffusion policy is already a stochastic predictor: each ``get_action`` draws a
fresh Gaussian prior and denoises it, so N calls yield N distinct action samples.
That inherent sampling stochasticity is the natural Bayesian-uncertainty signal
for a diffusion policy — we use it for DropoutDAgger's N-sample ball test instead
of MC-dropout on the UNet (the fork ships a dropout layer but disables it in
forward and we never edit the fork). EnsembleDAgger / ThriftyDAgger use M
independently-retrained policies (true ensemble variance over denoised actions).

All draws are reduced to the first executed joint-delta step (rel_joint_pos),
matching how SafeDAgger's discrepancy is computed in selection/iil_baselines.py.

FAITHFULNESS CAVEAT: DropoutDAgger's "MC-dropout" is realised as diffusion-sampling
stochasticity (cleaner + runnable on the shared frozen backbone than monkeypatching
the disabled UNet dropout). EnsembleDAgger/ThriftyDAgger variance is true M-policy
ensemble variance over denoised actions.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch


def seed_all(s: int) -> None:
    import random
    import numpy as np
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _first_delta(action_seq: torch.Tensor, policy, num_joints: int) -> torch.Tensor:
    """First executed step's joint delta (rel_joint_pos), shape (num_joints,)."""
    idx = policy.obs_horizon - 1
    return action_seq[:, idx, :num_joints].reshape(-1).detach()


@torch.no_grad()
def _one_action_seq(policy, obs_seq: Dict[str, Any]) -> torch.Tensor:
    out = policy.get_action(obs_seq, dagger=False, return_dict=False)
    return out["action"] if isinstance(out, dict) else out


@torch.no_grad()
def action_sample_deltas(policy, obs_seq: Dict[str, Any], n: int,
                         num_joints: int) -> torch.Tensor:
    """N stochastic diffusion draws → (N, num_joints) of first-step joint deltas."""
    draws = [_first_delta(_one_action_seq(policy, obs_seq), policy, num_joints)
             for _ in range(max(1, int(n)))]
    return torch.stack(draws, dim=0)


# DropoutDAgger uses diffusion-sampling stochasticity (see module docstring).
mc_dropout_deltas = action_sample_deltas


@torch.no_grad()
def member_deltas(members: List[Any], obs_seq: Dict[str, Any],
                  num_joints: int) -> torch.Tensor:
    """Each ensemble member's first-step joint delta → (M, num_joints)."""
    out = [_first_delta(_one_action_seq(m, obs_seq), m, num_joints) for m in members]
    return torch.stack(out, dim=0)
