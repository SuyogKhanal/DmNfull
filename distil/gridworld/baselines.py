"""When-to-query baselines on GridWorld (SafeDAgger*, Stagger).

Same skeleton as gridworld/loop.py — train-from-scratch each round -> frozen held-out
eval -> roll the policy over the SAME screening pool DISTIL uses (identical seed band
screen_seed_base + rnd*n_screen + i) -> gate -> ONE expert (BFS) demo -> repeat until
the budget of successful demos is spent. The ONLY thing that differs from DISTIL is
WHICH failed layout (and which takeover cell) gets the demo: no descriptor, no
clustering, no Eq-9 memory, no VLM, no reasoning LLM.

Gates ported verbatim from the GridWorld original,
pool_x_selector/selection/iil_baselines.py:

  safe (SafeDAgger*)  score = discrepancy = fraction of rollout steps whose action is
                      off the A* expert's optimal-action set; queryable when
                      score > tau (tau = 0.10, config_baselines.yaml `safe.tau`).
                      Pick argmax score among queryable (fallback: top-scoring), and
                      the expert takes over at the FIRST off-optimal step.
  stagger             one uniformly-random failure; takeover at a uniformly-random step.

Demo primitive is IDENTICAL to DISTIL's SELECT (collect_select_gw): BFS-optimal path
from the takeover cell to the goal. So the arms differ only in the ALLOCATION decision,
which is exactly the claim under test.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from ..compute_log import log_round, round_row
from .collect import collect_select_gw
from .descriptor import GWFailureDescriptor
from .env import make_maze, reset_to, scene_of
from .expert import build_grid, compute_distance_map, optimal_action_mask
from .layouts import sample_layout, sample_layouts
from .loop import _eval_sr, bootstrap_demos
from .policy import GridWorldPolicy

TAU_SAFE = 0.10          # config_baselines.yaml baselines.safe.tau
GW_BASELINE_ARMS = ("safe", "stagger")


def _rollout_scored(policy, layout: Dict, max_steps: int) -> Dict[str, Any]:
    """One rollout with the per-step A* optimal-action mask -> SafeDAgger discrepancy."""
    env = make_maze()
    reset_to(env, layout)
    grid_arr = build_grid(len(layout["grid"]), layout["fires"], layout["goal"],
                          wall_template=np.asarray(layout["grid"], dtype=np.int32))
    dmap = compute_distance_map(grid_arr, tuple(layout["goal"]))

    cells = [tuple(env.agent_pos)]        # cell BEFORE each action
    actions: List[int] = []
    off: List[bool] = []                  # step action off the optimal set?
    success, t = False, 0
    while t < max_steps:
        grid, agent, goal, fires = scene_of(env)
        a = policy.act(grid, agent, goal, fires)
        mask = optimal_action_mask(dmap, tuple(agent), grid_arr)
        off.append(bool(mask.sum() > 0 and mask[a] <= 0))
        actions.append(int(a))
        _, _, term, trunc, info = env.step(a)
        success = bool(info["success"])
        t += 1
        cells.append(tuple(env.agent_pos))
        if term or trunc:
            break
    env.close()
    disc = float(np.mean(off)) if off else 0.0
    first_off = next((i for i, o in enumerate(off) if o), 0)
    return {"layout": layout, "success": success, "n_steps": t, "actions": actions,
            "cells": cells, "discrepancy": disc, "first_off": int(first_off)}


def _score(kind: str, recs: List[Dict], rng: np.random.Generator) -> None:
    """Set score + queryable in place (kind-specific gate). Verbatim port."""
    if kind == "safe":
        for r in recs:
            r["score"] = float(r["discrepancy"])
            r["queryable"] = r["discrepancy"] > TAU_SAFE
            r["takeover_idx"] = r["first_off"]
    elif kind == "stagger":
        for r in recs:
            r["score"] = float(rng.random()) if not r["success"] else 0.0
            r["queryable"] = not r["success"]
            n = max(1, r["n_steps"])
            r["takeover_idx"] = int(rng.integers(0, n))
    else:
        raise ValueError(f"unknown GridWorld baseline arm {kind!r}; "
                         f"known: {sorted(GW_BASELINE_ARMS)}")


def _rank(recs: List[Dict]) -> List[Dict]:
    """Queryable records first (the gate), then argmax score; ties by longer rollout.
    Nothing queryable -> fall back to the top-scoring record so a round with any signal
    is never wasted (== the original's fallback)."""
    q = [r for r in recs if r.get("queryable")]
    pool = q if q else recs
    return sorted(pool, key=lambda r: (r["score"], r["n_steps"]), reverse=True)


def _as_desc(rec: Dict) -> GWFailureDescriptor:
    """Wrap a scored rollout as the descriptor collect_select_gw() consumes, so the demo
    primitive is byte-for-byte the one DISTIL uses (only agent_cell differs)."""
    idx = min(int(rec["takeover_idx"]), len(rec["cells"]) - 1)
    return GWFailureDescriptor(
        episode_id=int(rec["layout"]["seed"]), seed=int(rec["layout"]["seed"]),
        t_star=idx, T=int(rec["n_steps"]), peak_loss=float(rec["discrepancy"]),
        layout=rec["layout"], agent_cell=list(rec["cells"][idx]),
        exec_actions=rec["actions"])


def run_baseline_gridworld(cfg, arm: str, device, log_fn=print, init_demos=None,
                           work_dir="."):
    """SafeDAgger*/Stagger on GridWorld. Same stop rule, same eval, same demo primitive
    as run_distil_gridworld — only the allocation decision differs."""
    if arm not in GW_BASELINE_ARMS:
        raise NotImplementedError(
            f"GridWorld baseline arm '{arm}' is not implemented. Implemented: "
            f"{sorted(GW_BASELINE_ARMS)}. The dropout/ensemble/thrifty/diffdagger gates "
            f"need MC-dropout / M-member ensembles / a success-Q head on the classifier "
            f"(see pool_x_selector/selection/uncertainty.py) — not ported.")

    p4 = dict(cfg.get("p4", {}) or {})
    budget = int(cfg["budget"]) if cfg.get("budget") is not None else int(cfg["final_demos"])
    max_steps = int(cfg["env_horizon"])
    eval_ms = int(cfg.get("eval_max_steps", max_steps))
    n_screen = int(p4.get("screen_episodes", 20))
    screen_base = int(p4.get("screen_seed_base", 3_000_000))
    infeas = int(p4.get("infeasible_attempts", 4))
    patience = int(cfg.get("patience_window", 2))
    rng = np.random.default_rng(int(cfg["seed"]))

    heldout = sample_layouts(int(cfg.get("eval_seed_base", 5_000_000)),
                             int(cfg.get("eval_episodes", 200)))
    if init_demos is None:
        init_demos = bootstrap_demos(int(cfg["num_init_demos"]), log_fn)
    demos = list(init_demos)
    final_demos = len(init_demos) + budget
    log_fn(f"[init] {len(demos)} bootstrap demos; arm={arm}; budget={budget}; "
           f"heldout={len(heldout)}")

    def new_policy():
        return GridWorldPolicy(modality=cfg.get("modality", "state"),
                               alpha=cfg["alpha"], patience_window=patience,
                               patience=cfg.get("patience", 2), device=device)

    history: List[Dict[str, Any]] = []
    for rnd in range(int(cfg["max_rounds"])):
        t0 = time.time()
        sec_train = sec_eval = sec_screen = sec_prescribe = 0.0
        steps = cfg["initial_train_steps"] if rnd == 0 else cfg["round_train_steps"]
        log_fn(f"\n===== GW[{arm}] Round {rnd} | demos={len(demos)} =====")
        policy = new_policy()
        _t = time.time()
        policy.train_from_scratch(demos, steps=steps, batch_size=cfg["batch_size"],
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                  seed=cfg["seed"], log_fn=log_fn)
        sec_train = time.time() - _t
        _t = time.time()
        sr = _eval_sr(policy, heldout, eval_ms)
        sec_eval = time.time() - _t
        log_fn(f"  [eval] heldout SR={sr:.3f} ({len(heldout)} layouts)")
        n_at_eval = len(demos)

        if len(demos) >= final_demos:
            log_fn(f"[stop] reached final_demos={final_demos}")
            history.append(_rec(rnd, sr, n_at_eval, 0, 0, None, 0,
                                round(time.time() - t0, 1)))
            log_round(work_dir, round_row(rnd, arm, n_demos_at_eval=n_at_eval,
                                          sec_total=time.time() - t0, sec_train=sec_train,
                                          sec_eval=sec_eval))
            break

        # SAME screening pool as DISTIL (identical seed band) — the gate is the only diff.
        _t = time.time()
        layouts = [sample_layout(screen_base + rnd * n_screen + i) for i in range(n_screen)]
        recs = [_rollout_scored(policy, lay, max_steps) for lay in layouts]
        fails = [r for r in recs if not r["success"]]
        _score(arm, recs, rng)
        sec_screen = time.time() - _t
        n_q = sum(1 for r in recs if r.get("queryable"))
        gate = f"tau={TAU_SAFE}" if arm == "safe" else "random-failure"
        log_fn(f"  [gate:{arm}] {len(fails)}/{n_screen} failures, "
               f"{n_q} queryable ({gate})")
        if not fails and n_q == 0:
            history.append(_rec(rnd, sr, n_at_eval, 0, 0, None, 0,
                                round(time.time() - t0, 1)))
            log_round(work_dir, round_row(rnd, arm, n_demos_at_eval=n_at_eval,
                                          sec_total=time.time() - t0, sec_train=sec_train,
                                          sec_eval=sec_eval, sec_screen=sec_screen))
            continue

        _t = time.time()
        ranked = _rank(recs)
        picked, n_infeas = None, 0
        for attempt in range(min(infeas, len(ranked))):
            rec = ranked[attempt]
            demo, ok, meta = collect_select_gw(_as_desc(rec))
            log_fn(f"  [collect a{attempt}] ep={rec['layout']['seed']} "
                   f"score={rec['score']:.3f} takeover_t={rec['takeover_idx']} "
                   f"ok={ok} len={meta.get('len')}")
            if ok:
                demos.append(demo)
                picked = rec
                break
            n_infeas += 1
        sec_prescribe = time.time() - _t

        history.append(_rec(rnd, sr, n_at_eval, len(fails), n_q,
                            (int(picked["layout"]["seed"]) if picked else None),
                            n_infeas, round(time.time() - t0, 1)))
        log_round(work_dir, round_row(rnd, arm, n_demos_at_eval=n_at_eval,
                                      sec_total=time.time() - t0, sec_train=sec_train,
                                      sec_eval=sec_eval, sec_screen=sec_screen,
                                      sec_prescribe=sec_prescribe))

    result = {
        "history": history,
        "final_success": next((h["eval_success"] for h in reversed(history)
                               if h["eval_success"] is not None), None),
        "n_demos": len(demos), "budget": budget, "n_init_demos": len(init_demos),
        "final_demos": final_demos, "arm": arm,
    }
    log_fn(f"\n[FINAL] GridWorld[{arm}] SR={result['final_success']} | demos={len(demos)}")
    return result


def _rec(rnd, sr, n_demos_at_eval, n_fail, n_queryable, picked_ep,
         n_infeasible_attempts, sec):
    """Mirrors gridworld/loop.py::_rec so build_d5_compute.py reads both arms the same
    way (`sec` per round; `tokens` empty => the token columns skip this arm)."""
    return {"round": rnd, "n_demos_at_eval": n_demos_at_eval, "eval_success": sr,
            "n_screen_failures": n_fail, "n_queryable": n_queryable,
            "picked_ep": picked_ep, "mode": "select",
            "n_infeasible_attempts": n_infeasible_attempts, "sec": sec, "tokens": {}}
