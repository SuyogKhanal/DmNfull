"""Failure screening for DISTIL: roll the current policy on screening seeds, score
each step with the Diff-DAgger diffusion loss, and collect the FAILED episodes as
candidates.

DESIGN CHANGE (02_..md #1): the descriptor anchor AND the SELECT takeover step are
now `t_flag` = the FIRST step whose per-step uncertainty crosses the policy's OOD
threshold (with K-patience, exactly like Diff-DAgger's query trigger, Eq 5), NOT the
argmax peak `t_star`. The peak is late in a failing episode (Door median t*/T ~= 0.91,
Lift ~= 192/200), which starves the expert of budget after takeover and inflates the
infeasible rate. t_flag is early -> a less-corrupted state + far more budget. We still
record the peak step (`t_peak`, `peak_loss`) for telemetry / the peak-percentile knob.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..eval import make_obs_seq
from .descriptor import RSFailureDescriptor, snapshot_state


def _first_threshold_crossing(step_losses: List[float], threshold: Optional[float],
                              patience: int) -> tuple:
    """Return (t_flag, source). t_flag = first step index t such that the loss at
    every step in (t-patience, t] exceeds `threshold` (K-patience, Eq 5). Falls back
    to the argmax peak when no threshold is given or nothing crosses."""
    n = len(step_losses)
    if n == 0:
        return 0, "empty"
    peak = int(np.argmax(step_losses))
    if threshold is None or not np.isfinite(threshold):
        return peak, "peak(no_threshold)"
    K = max(0, int(patience))
    run = 0
    for t in range(n):
        if step_losses[t] > threshold:
            run += 1
            if run >= K + 1:            # (t-K, t] all cross
                return t, "flag"
        else:
            run = 0
    return peak, "peak(no_crossing)"    # never crossed -> anchor at the peak


@torch.no_grad()
def screen_failures(policy, env, *, obs_horizon, act_horizon, seeds, device,
                    max_steps, threshold: Optional[float] = None,
                    patience: int = 2) -> List[Dict[str, Any]]:
    """Return one candidate dict per FAILED screening episode:
    {seed, exec_actions, t_star(=t_flag anchor/takeover), t_peak, peak_loss,
     flag_loss, T, flag_source}."""
    if threshold is None:
        threshold = getattr(policy, "loss_threshold", None)
    fails: List[Dict[str, Any]] = []
    for seed in seeds:
        s = env.reset(seed=int(seed))
        dq = deque([s] * obs_horizon, maxlen=obs_horizon)
        exec_actions: List[np.ndarray] = []
        step_losses: List[float] = []
        success, done, t = False, False, 0
        while t < max_steps:
            obs_seq = make_obs_seq(dq, obs_horizon, device)
            action_chunk, naction = policy.plan(obs_seq)          # denoise once
            loss = float(policy.uncertainty(obs_seq, naction))    # diffusion-loss score
            acts = action_chunk[0].cpu().numpy()
            for i in range(act_horizon):
                a = np.asarray(acts[i], dtype=np.float32)
                exec_actions.append(a); step_losses.append(loss)
                s, r, done, success, info = env.step(a); dq.append(s); t += 1
                if done:
                    break
            if done:
                break
        if not success and exec_actions:                          # a genuine failure
            t_flag, source = _first_threshold_crossing(step_losses, threshold, patience)
            t_peak = int(np.argmax(step_losses))
            fails.append({"seed": int(seed), "exec_actions": exec_actions,
                          "t_star": int(t_flag), "t_peak": t_peak,
                          "peak_loss": float(max(step_losses)),
                          "flag_loss": float(step_losses[t_flag]),
                          "T": t, "flag_source": source})
    return fails


def build_descriptor(task: str, env, fail: Dict[str, Any]) -> RSFailureDescriptor | None:
    """Replay a failure's recorded actions to the anchor step (t_flag) and snapshot
    privileged geometry there."""
    try:
        env.reset(seed=fail["seed"])
        for a in fail["exec_actions"][: fail["t_star"]]:
            _, _, done, _, _ = env.step(a)
            if done:
                break
        snap = snapshot_state(task, env.env)
        return RSFailureDescriptor(
            task=task, episode_id=fail["seed"], seed=fail["seed"],
            t_star=int(fail["t_star"]), T=int(fail["T"]), peak_loss=float(fail["peak_loss"]),
            snap=snap, exec_actions=fail["exec_actions"])
    except Exception:
        return None
