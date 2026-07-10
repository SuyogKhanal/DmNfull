"""Honest policy evaluation on a FIXED held-out set (same seeds every round).

Reports success-at-completion via action-chunked rollouts. For Wipe it also
reports marker coverage (proportion_wiped), an informative partial-credit signal
next to the strict all-100-markers binary success.
"""
from collections import deque

import numpy as np
import torch


def make_obs_seq(state_deque, obs_horizon, device):
    """Stack the last obs_horizon states into a (1, obs_horizon, state_dim) tensor."""
    states = list(state_deque)[-obs_horizon:]
    arr = np.stack(states, axis=0)[None]  # (1, oh, Ds)
    return {"state": torch.as_tensor(arr, dtype=torch.float32, device=device)}


@torch.no_grad()
def policy_rollout(policy, env, *, obs_horizon, act_horizon, seed, device, max_steps):
    """Run the (pure) policy with receding-horizon action chunking.

    Returns (success, length, coverage) where coverage is the Wipe marker
    fraction (or None for tasks without it)."""
    s = env.reset(seed=seed)
    dq = deque([s] * obs_horizon, maxlen=obs_horizon)
    success, t = False, 0
    while t < max_steps:
        obs_seq = make_obs_seq(dq, obs_horizon, device)
        actions = policy.get_action(obs_seq)[0].cpu().numpy()  # (act_horizon, act_dim)
        for i in range(act_horizon):
            s, r, done, success, info = env.step(actions[i])
            dq.append(s)
            t += 1
            if done:
                break
        if done:
            break
    return success, t, env.coverage()


def evaluate_policy(policy, env, *, num_episodes, obs_horizon, act_horizon, device,
                    base_seed, max_steps=None, log_fn=None):
    """Evaluate on a FIXED set of `num_episodes` seeds [base_seed, base_seed+N).

    Keeping base_seed constant across DAgger rounds makes the per-round success
    curve directly comparable (and disjoint from the train/rollout seed bands)."""
    max_steps = max_steps or env.horizon
    was_training = policy.training
    policy.eval()
    n_succ, lens, covs = 0, [], []
    for ep in range(num_episodes):
        success, t, cov = policy_rollout(
            policy, env, obs_horizon=obs_horizon, act_horizon=act_horizon,
            seed=base_seed + ep, device=device, max_steps=max_steps,
        )
        n_succ += int(success)
        lens.append(t)
        if cov is not None:
            covs.append(cov)
        if log_fn:
            log_fn(f"    eval ep{ep}: success={success} len={t}"
                   + (f" coverage={cov:.2f}" if cov is not None else ""))
    if was_training:
        policy.train()
    out = dict(success_rate=n_succ / num_episodes, mean_len=float(np.mean(lens)),
               num_episodes=num_episodes)
    if covs:
        out["mean_coverage"] = float(np.mean(covs))
    return out
