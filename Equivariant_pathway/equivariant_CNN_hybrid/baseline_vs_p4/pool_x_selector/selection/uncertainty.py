"""Uncertainty machinery for the IIL baselines.

Two families, both built from scratch (the suite has no pre-existing
ensemble / MC-dropout primitives):

* **Ensembles** (EnsembleDAgger + ThriftyDAgger novelty) — M policy members
  fine-tuned independently each round (the existing replay fine-tuner called
  M times with distinct seeds + per-member checkpoint dirs). The "novice" is
  the ensemble-mean-action policy; doubt = per-step action disagreement.
* **MC-dropout** (DropoutDAgger) — instantiate the shared policy with a >0
  fusion-head dropout and keep the Dropout layer active at inference; N
  stochastic forward passes per step give a sampled action distribution.

All env-walk code mirrors ``baseline_budget._rollout_with_loss`` (same env
build, encoder, success criterion ``reward[-1] > 5.0``) so the numbers line
up with the rest of the suite.

NOTE (faithfulness): the shared checkpoint has a single ``nn.Dropout`` in the
fusion head (``model.py``); the CNN encoder and equivariant backbone have
none. MC-dropout therefore samples stochasticity only from the fusion head —
a known architectural limitation, reported honestly rather than worked around
(retraining with backbone dropout would change the shared checkpoint and break
the apples-to-apples warm-start with P4).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import (  # noqa: E402
    baseline_budget as _bb,  # _build_env, _encode_rgb_from_env, encode_grid_from_env,
                             # _load_model, _eval_heldout, _bce_with_balanced_pos_weight
)
from Equivariant_pathway.expert import (  # noqa: E402
    AStarExpert, build_grid, compute_distance_map, optimal_action_mask,
)
from Equivariant_pathway.equivariant_CNN_hybrid.model import (  # noqa: E402
    EquivariantCNNHybridPolicy,
)

NUM_ACTIONS = 4
CELL_PX = 16


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [uncertainty] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Shared: action discrepancy vs the A* expert (SafeDAgger* oracle)
# --------------------------------------------------------------------------- #
def action_discrepancy(layout: Dict, actions: Sequence[int],
                       trajectory: Sequence) -> float:
    """Fraction of rollout steps whose action is NOT in the expert's optimal
    set (the discrete-action SafeDAgger* discrepancy). A step also counts as a
    discrepancy when the agent is unreachable (empty optimal mask).
    """
    actions = [int(a) for a in (actions or [])]
    if not actions:
        return 0.0
    fires = [tuple(f) for f in layout["fire_positions"]]
    goal = tuple(layout["goal_pos"])
    gs = len(layout.get("grid", [[0] * 5] * 5))
    grid = build_grid(gs, fire_positions=fires, goal_pos=goal)
    expert = AStarExpert(grid, goal)
    n = 0
    off = 0
    for t, a in enumerate(actions):
        if t >= len(trajectory):
            break
        opt = expert.optimal_actions(tuple(trajectory[t]))
        n += 1
        if opt.sum() == 0 or opt[a] < 0.5:
            off += 1
    return (off / n) if n else 0.0


# --------------------------------------------------------------------------- #
# Ensembles
# --------------------------------------------------------------------------- #
def member_dirs(method_root: Path, M: int) -> List[Path]:
    ck = Path(method_root) / "checkpoints"
    return [ck / f"member_{m}" for m in range(int(M))]


def init_ensemble(method_root: Path, shared_ckpt_dir: Path, M: int) -> None:
    """Seed each member's checkpoint dir from the shared bootstrap (once).

    Idempotent: a member dir that already has ``best_hybrid_policy.pth`` is
    left alone (so a re-run warm-starts from that member's own progress).
    """
    for md in member_dirs(method_root, M):
        md.mkdir(parents=True, exist_ok=True)
        if (md / "best_hybrid_policy.pth").exists():
            continue
        for name in ("best_hybrid_policy.pth", "last_hybrid_policy.pth"):
            s = Path(shared_ckpt_dir) / name
            if s.exists():
                shutil.copy2(s, md / name)


def finetune_ensemble(
    cumulative_demo_dir: Path,
    new_demo_dir: Optional[Path],
    method_root: Path,
    M: int,
    base_seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    replay_mix: float,
    replay_mix_floor: float,
) -> List[int]:
    """Fine-tune all M members independently (distinct seed per member) via
    the replay fine-tuner subprocess. Returns per-member return codes.
    """
    rcs: List[int] = []
    for m, md in enumerate(member_dirs(method_root, M)):
        cmd = [
            sys.executable, "-u", "-m",
            "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4."
            "pool_x_selector.trainer.finetune_replay",
            "--demo_dir", str(cumulative_demo_dir),
            "--checkpoint_dir", str(md),
            "--epochs", str(int(epochs)),
            "--batch_size", str(int(batch_size)),
            "--lr", str(float(lr)),
            "--weight_decay", str(float(weight_decay)),
            "--replay_mix", str(float(replay_mix)),
            "--replay_mix_floor", str(float(replay_mix_floor)),
            "--seed", str(int(base_seed) + 101 * int(m)),
            "--resume",
        ]
        if new_demo_dir is not None and Path(new_demo_dir).exists():
            cmd += ["--new_demos_dir", str(new_demo_dir)]
        _info(f"member {m}: $ " + " ".join(cmd))
        rcs.append(subprocess.call(cmd, cwd=str(REPO_ROOT)))
    return rcs


def load_ensemble(method_root: Path, M: int, device: torch.device) -> List:
    """Load all M members (best_hybrid_policy.pth) in eval mode."""
    models = []
    for md in member_dirs(method_root, M):
        models.append(_bb._load_model(md / "best_hybrid_policy.pth", device))
    return models


def _dist_map_for(layout: Dict):
    fires = [tuple(f) for f in layout["fire_positions"]]
    goal = tuple(layout["goal_pos"])
    gs = len(layout.get("grid", [[0] * 5] * 5))
    grid_static = build_grid(gs, fire_positions=fires, goal_pos=goal)
    dist_map = compute_distance_map(grid_static, goal)
    return grid_static, dist_map, goal


def _ensemble_step(models: Sequence, rgb_t: torch.Tensor, grid_t: torch.Tensor):
    """One forward pass through every member. Returns
    (mean_action:int, mean_logits:np.ndarray(4,), per_member_actions:List[int]).
    """
    with torch.no_grad():
        logits = [m(rgb_t, grid_t).squeeze(0) for m in models]  # each (4,)
        stacked = torch.stack(logits, dim=0)                    # (M, 4)
        mean_logits = stacked.mean(dim=0)                       # (4,)
        per_member = [int(l.argmax().item()) for l in logits]
        mean_action = int(mean_logits.argmax().item())
    return mean_action, mean_logits.detach().cpu().numpy(), per_member


def ensemble_rollout(models: Sequence, layout: Dict, seed: int,
                     max_steps: int, device: torch.device) -> Dict:
    """Walk the env under the ensemble-mean-action policy, recording the
    signals EnsembleDAgger (action disagreement) and ThriftyDAgger
    (novelty + per-state Q features) need.

    Returns: success, n_steps, actions (mean), trajectory, policy_loss /
    policy_loss_sum (mean-logit BCE vs expert mask, for prescribed_loss
    parity), per_member_actions (T x M), opt_masks (T x 4), grid_encs (T list
    of (C,H,W) arrays).
    """
    env = _bb._build_env(layout, seed=int(seed))
    grid_static, dist_map, _goal = _dist_map_for(layout)

    actions: List[int] = []
    rewards: List[float] = []
    per_step_loss: List[float] = []
    per_member_actions: List[List[int]] = []
    opt_masks: List[np.ndarray] = []
    grid_encs: List[np.ndarray] = []
    trajectory: List = []

    terminated = truncated = False
    si = 0
    while not (terminated or truncated) and si < int(max_steps):
        x_grid = _bb.encode_grid_from_env(env)
        x_rgb = _bb._encode_rgb_from_env(env, cell_px=CELL_PX)
        rgb_t = torch.from_numpy(x_rgb).float().unsqueeze(0).to(device)
        grid_t = torch.from_numpy(x_grid).float().unsqueeze(0).to(device)
        agent_pos = tuple(env.agent_pos)
        opt_mask = optimal_action_mask(dist_map, agent_pos, grid_static, num_actions=NUM_ACTIONS)

        mean_action, mean_logits, per_member = _ensemble_step(models, rgb_t, grid_t)
        mask_t = torch.from_numpy(opt_mask).float().unsqueeze(0).to(device)
        ml_t = torch.from_numpy(mean_logits).float().unsqueeze(0).to(device)
        step_loss = float(_bb._bce_with_balanced_pos_weight(ml_t, mask_t).item())

        trajectory.append(agent_pos)
        grid_encs.append(np.asarray(x_grid, dtype=np.float32))
        opt_masks.append(np.asarray(opt_mask, dtype=np.float32))
        per_member_actions.append(per_member)
        per_step_loss.append(step_loss)
        actions.append(mean_action)

        _obs, reward, terminated, truncated, _ = env.step(mean_action)
        rewards.append(float(reward))
        si += 1

    success = bool(rewards[-1] > 5.0) if rewards else False
    env.close()
    return {
        "success": success,
        "n_steps": si,
        "policy_loss": float(np.mean(per_step_loss)) if per_step_loss else 0.0,
        "policy_loss_sum": float(np.sum(per_step_loss)) if per_step_loss else 0.0,
        "actions": actions,
        "trajectory": trajectory,
        "per_member_actions": per_member_actions,
        "opt_masks": opt_masks,
        "grid_encs": grid_encs,
    }


def ensemble_doubt(per_member_actions: Sequence[Sequence[int]], M: int) -> float:
    """Mean per-step action disagreement across members:
        mean_t ( 1 - max_a count_t[a] / M ).
    0 = members unanimous every step; →1 = maximal disagreement.
    """
    if not per_member_actions or M <= 1:
        return 0.0
    vals = []
    for votes in per_member_actions:
        if not votes:
            continue
        counts = np.bincount(np.asarray(votes, dtype=int), minlength=NUM_ACTIONS)
        vals.append(1.0 - float(counts.max()) / float(len(votes)))
    return float(np.mean(vals)) if vals else 0.0


def ensemble_mean_discrepancy(per_member_actions: Sequence[Sequence[int]],
                              opt_masks: Sequence[np.ndarray]) -> float:
    """Mean over members of each member's action-discrepancy vs the expert
    mask along the ensemble-mean trajectory.
    """
    if not per_member_actions:
        return 0.0
    n_steps = len(per_member_actions)
    M = len(per_member_actions[0]) if per_member_actions[0] else 0
    if M == 0:
        return 0.0
    member_off = np.zeros(M, dtype=float)
    for t in range(n_steps):
        mask = opt_masks[t]
        for m in range(M):
            a = int(per_member_actions[t][m])
            if mask.sum() == 0 or mask[a] < 0.5:
                member_off[m] += 1.0
    return float((member_off / max(1, n_steps)).mean())


def ensemble_eval_heldout(
    method_root: Path, M: int, heldout_yaml: Path, results_dir: Path,
    tag: str, seed: int, max_steps: int, device: torch.device,
    mode: str = "mean",
) -> Dict:
    """Held-out success rate for the ensemble.

    * ``mode="member0"`` delegates to the upstream subprocess evaluator on
      member 0 (exact parity with single-policy methods).
    * ``mode="mean"`` (default) walks every held-out layout in-process under
      the ensemble-mean-action policy (the faithful "the ensemble is the
      policy" choice) and counts ``reward[-1] > 5.0``.
    Returns {n_episodes, n_successes, success_rate}.
    """
    mds = member_dirs(method_root, M)
    if mode == "member0":
        return _bb._eval_heldout(mds[0], heldout_yaml, results_dir,
                                 tag=tag, seed=seed, max_steps=max_steps)
    layouts = _bb._load_correction_layouts(heldout_yaml)
    if not layouts:
        return {"n_episodes": 0, "n_successes": 0, "success_rate": 0.0}
    models = load_ensemble(method_root, M, device)
    n_succ = 0
    for ep_idx, layout in enumerate(layouts):
        roll = ensemble_rollout(models, layout, seed=int(seed) + ep_idx,
                                max_steps=max_steps, device=device)
        if roll["success"]:
            n_succ += 1
    n = len(layouts)
    return {"n_episodes": n, "n_successes": n_succ,
            "success_rate": (n_succ / n) if n else 0.0}


# --------------------------------------------------------------------------- #
# MC-dropout
# --------------------------------------------------------------------------- #
def _set_dropout_active(model: nn.Module, active: bool) -> None:
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train(active)


def load_dropout_model(checkpoint: Path, device: torch.device,
                       dropout_d: float) -> EquivariantCNNHybridPolicy:
    """Load the shared checkpoint into a policy instantiated with a >0
    fusion-head dropout, with the Dropout layer kept active at inference
    (the rest of the net stays in eval mode). ``nn.Dropout`` has no
    parameters, so ``strict=True`` still matches a checkpoint trained at
    ``head_dropout=0.0``.
    """
    ck = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model = EquivariantCNNHybridPolicy(
        rgb_in_channels=3,
        grid_in_channels=int(ck.get("in_channels", 5)),
        num_actions=int(ck.get("num_actions", NUM_ACTIONS)),
        head_dropout=float(dropout_d),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    _set_dropout_active(model, True)  # MC-dropout: keep Dropout stochastic
    return model


def mc_dropout_rollout(model: EquivariantCNNHybridPolicy, layout: Dict,
                       seed: int, max_steps: int, device: torch.device,
                       N: int) -> Dict:
    """Walk the env under the DETERMINISTIC (dropout-off) policy, and at every
    visited state additionally draw ``N`` stochastic (dropout-on) action
    samples.

    Returns: success, n_steps, actions (clean argmax), trajectory, policy_loss
    / policy_loss_sum (clean-logit BCE), opt_masks (T x 4), per_step_samples
    (T x N argmax draws).
    """
    env = _bb._build_env(layout, seed=int(seed))
    grid_static, dist_map, _goal = _dist_map_for(layout)

    actions: List[int] = []
    rewards: List[float] = []
    per_step_loss: List[float] = []
    per_step_samples: List[List[int]] = []
    opt_masks: List[np.ndarray] = []
    trajectory: List = []

    terminated = truncated = False
    si = 0
    while not (terminated or truncated) and si < int(max_steps):
        x_grid = _bb.encode_grid_from_env(env)
        x_rgb = _bb._encode_rgb_from_env(env, cell_px=CELL_PX)
        rgb_t = torch.from_numpy(x_rgb).float().unsqueeze(0).to(device)
        grid_t = torch.from_numpy(x_grid).float().unsqueeze(0).to(device)
        agent_pos = tuple(env.agent_pos)
        opt_mask = optimal_action_mask(dist_map, agent_pos, grid_static, num_actions=NUM_ACTIONS)

        # Clean (eval) action drives the executed trajectory.
        _set_dropout_active(model, False)
        with torch.no_grad():
            clean_logits = model(rgb_t, grid_t)
            clean_action = int(clean_logits.argmax(dim=-1).item())
            mask_t = torch.from_numpy(opt_mask).float().unsqueeze(0).to(device)
            step_loss = float(_bb._bce_with_balanced_pos_weight(clean_logits, mask_t).item())
        # N stochastic (dropout-on) action samples for the uncertainty signal.
        _set_dropout_active(model, True)
        samples: List[int] = []
        with torch.no_grad():
            for _ in range(int(N)):
                logit = model(rgb_t, grid_t)
                samples.append(int(logit.argmax(dim=-1).item()))

        trajectory.append(agent_pos)
        opt_masks.append(np.asarray(opt_mask, dtype=np.float32))
        per_step_samples.append(samples)
        per_step_loss.append(step_loss)
        actions.append(clean_action)

        _obs, reward, terminated, truncated, _ = env.step(clean_action)
        rewards.append(float(reward))
        si += 1

    success = bool(rewards[-1] > 5.0) if rewards else False
    env.close()
    return {
        "success": success,
        "n_steps": si,
        "policy_loss": float(np.mean(per_step_loss)) if per_step_loss else 0.0,
        "policy_loss_sum": float(np.sum(per_step_loss)) if per_step_loss else 0.0,
        "actions": actions,
        "trajectory": trajectory,
        "opt_masks": opt_masks,
        "per_step_samples": per_step_samples,
    }


def dropout_uncertainty(per_step_samples: Sequence[Sequence[int]],
                        opt_masks: Sequence[np.ndarray],
                        p_thresh: float) -> Dict[str, float]:
    """Aggregate MC-dropout samples vs the expert ball into per-layout scores.

    p_hat(t) = fraction of the step's N samples whose action ∈ expert optimal
    mask. Returns:
      * mean_disagreement = mean_t (1 - p_hat(t))
      * frac_low          = fraction of steps with p_hat(t) < p_thresh
    """
    if not per_step_samples:
        return {"mean_disagreement": 0.0, "frac_low": 0.0}
    disagree = []
    n_low = 0
    for t, samples in enumerate(per_step_samples):
        if not samples:
            continue
        mask = opt_masks[t]
        in_ball = sum(1 for a in samples if mask.sum() > 0 and mask[int(a)] >= 0.5)
        p_hat = in_ball / len(samples)
        disagree.append(1.0 - p_hat)
        if p_hat < float(p_thresh):
            n_low += 1
    return {
        "mean_disagreement": float(np.mean(disagree)) if disagree else 0.0,
        "frac_low": (n_low / len(per_step_samples)) if per_step_samples else 0.0,
    }
