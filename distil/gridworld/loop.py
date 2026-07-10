"""run_distil_gridworld — the DISTIL loop on GridWorld 5x5.

Same skeleton as distil/p4/loop.py (robot): train-from-scratch -> frozen held-out eval
-> screen failures (entropy self-uncertainty, t_flag anchor) -> silhouette-k cluster ->
Eq-9 memory rotation -> LLM SELECT/BRIDGE + CONFIDENCE -> collect ONE successful demo ->
repeat until the budget of successful demos is spent. Policy = the equivariant classifier
(GridWorldPolicy); demos = BFS-expert paths; eval = held-out-layout success rate.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import numpy as np

from .collect import _demo  # noqa: F401  (kept for symmetry / potential reuse)
from .env import bfs_path, make_maze, render_frames, reset_to, scene_of
from .layouts import sample_layout, sample_layouts
from .planner import GridWorldHybridPlanner
from .policy import GridWorldPolicy
from .screen import screen_failures_gw


def _eval_sr(policy, layouts: List[Dict], max_steps: int) -> float:
    env = make_maze()
    succ = 0
    for lay in layouts:
        reset_to(env, lay)
        done = False
        t = 0
        s = False
        while not done and t < max_steps:
            grid, agent, goal, fires = scene_of(env)
            a = policy.act(grid, agent, goal, fires)
            _, _, term, trunc, info = env.step(a)
            s = bool(info["success"]); done = term or trunc; t += 1
        succ += int(s)
    env.close()
    return succ / max(1, len(layouts))


def _replay_end_cell(layout: Dict, exec_actions: List[int]) -> List[int]:
    env = make_maze(); reset_to(env, layout)
    for a in exec_actions:
        _, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            break
    cell = list(env.agent_pos); env.close()
    return cell


def run_distil_gridworld(cfg, device, log_fn=print, init_demos=None, work_dir="."):
    p4 = dict(cfg.get("p4", {}) or {})
    p4["_seed"] = cfg["seed"]                 # for the planner's random-allocation rng
    budget = int(cfg["budget"]) if cfg.get("budget") is not None else int(cfg["final_demos"])
    max_steps = int(cfg["env_horizon"])
    eval_ms = int(cfg.get("eval_max_steps", max_steps))
    n_screen = int(p4.get("screen_episodes", 20))
    screen_base = int(p4.get("screen_seed_base", 3_000_000))
    infeas = int(p4.get("infeasible_attempts", 4))
    patience = int(cfg.get("patience_window", 2))

    # frozen held-out set (disjoint seed band) + shared bootstrap demos
    heldout = sample_layouts(int(cfg.get("eval_seed_base", 5_000_000)),
                             int(cfg.get("eval_episodes", 200)))
    if init_demos is None:
        init_demos = bootstrap_demos(int(cfg["num_init_demos"]), log_fn)
    demos = list(init_demos)
    log_fn(f"[init] {len(demos)} bootstrap demos; budget={budget}; heldout={len(heldout)}")

    def new_policy():
        return GridWorldPolicy(modality=cfg.get("modality", "state"),
                               alpha=cfg["alpha"], patience_window=patience,
                               patience=cfg.get("patience", 2), device=device)

    planner = GridWorldHybridPlanner(work_dir, p4)
    from ..p4.llm import make_llm
    llm = None
    use_llm = ((p4.get("decision") == "llm") and not p4.get("fallback_only")
               and p4.get("allocation") != "random")
    if use_llm and os.environ.get("OPENAI_BASE_URL"):
        llm = make_llm(use_vlm=bool(p4.get("vlm", True)), use_kag=bool(p4.get("kag", True)))
    log_fn(f"[llm] " + ("READY" if llm else "geometric fallback"))

    policy = new_policy()
    history: List[Dict[str, Any]] = []
    final_demos = len(init_demos) + budget

    for rnd in range(int(cfg["max_rounds"])):
        t0 = time.time()
        steps = cfg["initial_train_steps"] if rnd == 0 else cfg["round_train_steps"]
        log_fn(f"\n===== GW Round {rnd} | demos={len(demos)} =====")
        policy = new_policy()
        policy.train_from_scratch(demos, steps=steps, batch_size=cfg["batch_size"],
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                  seed=cfg["seed"], log_fn=log_fn)
        sr = _eval_sr(policy, heldout, eval_ms)
        log_fn(f"  [eval] heldout SR={sr:.3f} ({len(heldout)} layouts)")
        n_at_eval = len(demos)

        if len(demos) >= final_demos:
            log_fn(f"[stop] reached final_demos={final_demos}")
            history.append(_rec(rnd, sr, n_at_eval, 0, None, None, None, None, None, 0,
                                round(time.time() - t0, 1)))
            break

        layouts = [sample_layout(screen_base + rnd * n_screen + i) for i in range(n_screen)]
        fails = screen_failures_gw(policy, layouts, max_steps=max_steps, patience=patience)
        log_fn(f"  [screen] {len(fails)}/{n_screen} failures")
        if not fails:
            history.append(_rec(rnd, sr, n_at_eval, 0, None, None, None, None, None, 0,
                                round(time.time() - t0, 1)))
            continue

        planner.set_round(rnd, fails)
        label, confidence, conf_rat, tokens = None, None, "", {}
        if llm is not None:
            try:
                fdir = os.path.join(work_dir, "frames", f"round_{rnd:04d}")
                pdir = os.path.join(work_dir, "prompts", f"round_{rnd:04d}")
                failures = []
                for d in planner.target_descs():
                    end_cell = _replay_end_cell(d.layout, d.exec_actions)
                    frames = render_frames(d.layout,
                                           {"start": d.layout["start"], "high_loss": d.agent_cell,
                                            "end": end_cell}, fdir, d.episode_id)
                    failures.append({"ep_id": d.episode_id, "frames": frames,
                                     "ox": float(d.agent_cell[0]), "oy": float(d.agent_cell[1]),
                                     "t_star": int(d.t_star), "T": int(d.T),
                                     "peak_loss": float(d.peak_loss)})
                res = llm.decide("GridWorld", failures, planner.bridge_supported,
                                 log_fn=log_fn, prompt_dir=pdir)
                label, confidence = res["label"], res["confidence"]
                conf_rat, tokens = res["confidence_rationale"], res["tokens"]
                planner.tele.event(rnd, "llm_decision", {
                    "raw_decision": res["raw_decision"][:800], "confidence": confidence,
                    "members": res["members"], "tokens": tokens, "models": res["models"]})
                log_fn(f"  [gw-llm] decision={str(label).splitlines()[-1][:70] if label else '?'} "
                       f"conf={confidence} tokens={tokens.get('total')}")
            except Exception as e:
                log_fn(f"  [gw-llm] failed ({e}); geometric fallback")

        mode, n_infeas = None, 0
        for attempt in range(infeas):
            spec = planner.decide(label if attempt == 0 else None, attempt)
            demo, ok, meta = planner.collect(spec)
            planner.note_collect(spec, ok, meta)
            log_fn(f"  [collect a{attempt}] mode={spec.mode} choice={spec.choice} "
                   f"cited={spec.cited_ids} ok={ok} len={meta.get('len')}")
            if ok:
                demos.append(demo); mode = spec.mode; break
            n_infeas += 1
        history.append(_rec(rnd, sr, n_at_eval, len(fails), mode, confidence, conf_rat,
                            planner.k_star(), planner.target_label(), n_infeas,
                            round(time.time() - t0, 1), tokens=tokens,
                            cluster_method=planner.cluster_method()))

    result = {
        "history": history,
        "final_success": next((h["eval_success"] for h in reversed(history)
                               if h["eval_success"] is not None), None),
        "n_demos": len(demos), "budget": budget, "n_init_demos": len(init_demos),
        "final_demos": final_demos,
        "n_bridge_used": sum(1 for h in history if h.get("mode") == "bridge"),
        "n_select_used": sum(1 for h in history if h.get("mode") == "select"),
    }
    log_fn(f"\n[FINAL] GridWorld SR={result['final_success']} | demos={len(demos)}")
    return result


def bootstrap_demos(n: int, log_fn=print, seed_base: int = 0) -> List[Dict]:
    """Byte-identical shared bootstrap: n BFS-expert demos on sampled layouts."""
    demos = []
    for i in range(n):
        lay = sample_layout(seed_base + i)
        path = bfs_path(lay["grid"], lay["start"], lay["goal"])
        if len(path) >= 2 and tuple(path[-1]) == tuple(lay["goal"]):
            demos.append({"grid": lay["grid"], "start": lay["start"], "goal": lay["goal"],
                          "fires": lay["fires"], "cells": [list(c) for c in path]})
    log_fn(f"[bootstrap] {len(demos)}/{n} BFS demos")
    return demos


def _rec(rnd, sr, n_demos_at_eval, n_fail, mode, confidence, conf_rat, k_star,
         cluster_label_target, n_infeasible_attempts, sec, tokens=None, cluster_method=None):
    return {"round": rnd, "n_demos_at_eval": n_demos_at_eval, "eval_success": sr,
            "n_screen_failures": n_fail, "mode": mode, "confidence": confidence,
            "confidence_rationale": conf_rat, "k_star": k_star,
            "cluster_method": cluster_method, "cluster_label_target": cluster_label_target,
            "n_infeasible_attempts": n_infeasible_attempts, "sec": sec, "tokens": tokens or {}}
