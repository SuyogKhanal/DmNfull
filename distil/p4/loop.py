"""run_distil — the DISTIL interactive loop on robosuite (Lift / Door / Wipe).

Same skeleton and eval schema as diffdagger.run_diffdagger (retrain-from-scratch
each round + fixed held-out eval), so DISTIL and every baseline/ablation are directly
comparable. The one difference is the DATA-ACQUISITION step: each round SCREENS the
policy for failures, CLUSTERS them into modes, rotates the target via cluster memory,
and the reasoning LLM (or a geometric fallback) decides SELECT one real failure or
BRIDGE several — collecting ONE successful demo per round (budget = successful demos).

This run's changes (02_..md): take over at t_flag (first threshold crossing, screen.py),
require a CONFIDENCE score per round, dump the exact prompts + KAG into the output dir
(golden rule 3), and log confidence/tokens/k*/target-label/infeasible-attempts per round
(09_..md history schema).
"""
from __future__ import annotations

import gc
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..collect import collect_demos
from ..diffdagger import train_and_calibrate
from ..eval import evaluate_policy
from .planner import RobosuiteHybridPlanner
from .render import render_failure_frames
from .screen import build_descriptor, screen_failures


def run_distil(cfg, make_env_fn, make_expert_fn, device, log_fn=print,
               ckpt_path=None, init_trajs=None, work_dir=".", llm=None):
    task = cfg["task"]
    env_h = cfg["env_horizon"]
    obs_h, act_h = cfg["obs_horizon"], cfg["act_horizon"]
    eval_seed_base, eval_eps = cfg["eval_seed_base"], cfg["eval_episodes"]
    patience = int(cfg.get("patience_window", 2))
    p4 = dict(cfg.get("p4", {}) or {})
    p4["_seed"] = cfg["seed"]                 # for the planner's random-allocation rng
    n_screen = int(p4.get("screen_episodes", 40))
    screen_base = int(p4.get("screen_seed_base", 3_000_000))
    infeas = int(p4.get("infeasible_attempts", 4))
    rng = np.random.default_rng(cfg["seed"])

    env = make_env_fn(env_h)            # screening + descriptor replay + collection
    eval_env = make_env_fn(env_h)       # frozen held-out (disjoint seeds)
    expert = make_expert_fn()

    if init_trajs is not None:
        trajs = list(init_trajs)
        log_fn(f"[init] using {len(trajs)} bootstrap demos (shared with baselines)")
    else:
        trajs = collect_demos(env, expert, cfg["num_init_demos"],
                              seed_base=int(rng.integers(0, 1_000_000)),
                              max_steps=env_h, log_fn=log_fn)

    planner = RobosuiteHybridPlanner(work_dir, task, p4)
    history: List[Dict[str, Any]] = []
    # Saturation early-stop: if the policy yields NO usable failures for this many
    # consecutive rounds, the budget can't be spent further (nothing to correct) and
    # retrain+screen just burns compute. Stop. (Lift is near-saturated -> 0 failures.)
    sat_patience = int(p4.get("saturation_patience", 4))
    consec_nofail = 0

    for rnd in range(cfg["max_rounds"]):
        t_round = time.time()
        train_steps = cfg["initial_train_steps"] if rnd == 0 else cfg["round_train_steps"]
        log_fn(f"\n===== DISTIL Round {rnd} | dataset={len(trajs)} trajs =====")
        policy, dataset = train_and_calibrate(cfg, trajs, device, train_steps, log_fn)

        ev = evaluate_policy(policy, eval_env, num_episodes=eval_eps, obs_horizon=obs_h,
                             act_horizon=act_h, device=device, base_seed=eval_seed_base,
                             max_steps=env_h)
        cov = f" coverage={ev['mean_coverage']:.3f}" if "mean_coverage" in ev else ""
        log_fn(f"  [eval] pure-policy success={ev['success_rate']:.3f} "
               f"mean_len={ev['mean_len']:.0f}{cov} (fixed {eval_eps}-ep set)")
        n_at_eval = len(trajs)   # dataset size THIS eval was computed on (curve x-axis)

        if ckpt_path is not None:
            torch.save({"policy_state_dict": policy.state_dict(), "cfg": cfg, "round": rnd,
                        "normalization": dataset.get_normalization(),
                        "loss_threshold": float(policy.loss_threshold), "alpha": cfg["alpha"]},
                       ckpt_path)

        # Stop rule = budget of *successful* demos (09_..md fix 1): run until the
        # dataset reaches final_demos (= n_init + budget), NOT an sr>=target early-stop.
        if len(trajs) >= cfg["final_demos"]:
            log_fn(f"[stop] reached final_demos={cfg['final_demos']} ({len(trajs)} trajs), "
                   f"sr={ev['success_rate']:.3f}")
            history.append(_record(rnd, ev, n_at_eval, 0, None, None, None, None, None, 0,
                                   round(time.time() - t_round, 1)))
            break

        # ── screen the current policy for failures (t_flag anchor, 02_..md #1) ──
        seeds = list(range(screen_base + rnd * n_screen, screen_base + (rnd + 1) * n_screen))
        fails = screen_failures(policy, env, obs_horizon=obs_h, act_horizon=act_h,
                                seeds=seeds, device=device, max_steps=env_h,
                                threshold=float(policy.loss_threshold), patience=patience)
        log_fn(f"  [screen] {len(fails)}/{n_screen} failures found")
        descs = [d for d in (build_descriptor(task, env, f) for f in fails) if d is not None]
        if not descs:
            consec_nofail += 1
            history.append(_record(rnd, ev, n_at_eval, len(fails), None, None, None, None,
                                   None, 0, round(time.time() - t_round, 1)))
            del policy; gc.collect(); torch.cuda.is_available() and torch.cuda.empty_cache()
            if consec_nofail >= sat_patience:
                log_fn(f"  [screen] no usable failures for {consec_nofail} consecutive rounds "
                       f"-> saturated (sr={ev['success_rate']:.3f}); stopping early")
                break
            log_fn(f"  [screen] no usable failures — budget-free, next round "
                   f"({consec_nofail}/{sat_patience} toward saturation stop)")
            continue
        consec_nofail = 0

        planner.set_round(rnd, descs)
        label, confidence, conf_rat, tokens = None, None, "", {}
        if llm is not None:
            try:
                fdir = os.path.join(work_dir, "frames", f"round_{rnd:04d}")
                pdir = os.path.join(work_dir, "prompts", f"round_{rnd:04d}")
                failures: List[Dict[str, Any]] = []
                for d in planner.target_descs():
                    frames = render_failure_frames(task, env_h, d.seed, d.exec_actions,
                                                   d.t_star, fdir, d.episode_id)
                    failures.append({"ep_id": d.episode_id, "frames": frames,
                                     "ox": float(d.snap["obj_xyz"][0]),
                                     "oy": float(d.snap["obj_xyz"][1]),
                                     "t_star": int(d.t_star), "T": int(d.T),
                                     "peak_loss": float(d.peak_loss)})
                res = llm.decide(task, failures, planner.bridge_supported, log_fn=log_fn,
                                 prompt_dir=pdir)
                label = res["label"]; confidence = res["confidence"]
                conf_rat = res["confidence_rationale"]; tokens = res["tokens"]
                planner.tele.event(rnd, "llm_decision", {
                    "raw_decision": res["raw_decision"][:800], "confidence": confidence,
                    "confidence_rationale": conf_rat, "members": res["members"],
                    "tokens": tokens, "models": res["models"]})
                log_fn(f"  [distil-llm] {len(failures)} failures analysed -> "
                       f"decision={str(label).splitlines()[-1][:80] if label else '?'} "
                       f"confidence={confidence} tokens={tokens.get('total')}")
            except Exception as e:
                log_fn(f"  [distil-llm] failed ({e}); geometric fallback")

        # ── collect ONE successful demo (in-round escalation on infeasible) ──
        got, mode, choice, n_infeas = False, None, None, 0
        for attempt in range(infeas):
            spec = planner.decide(label if attempt == 0 else None, attempt)
            demo, ok, meta = planner.collect(spec, env, expert, cfg)
            planner.note_collect(spec, ok, meta)
            log_fn(f"  [collect a{attempt}] mode={spec.mode} choice={spec.choice} "
                   f"cited={spec.cited} success={ok} len={meta.get('len')}")
            if ok:
                trajs.append(demo); got = True; mode, choice = spec.mode, spec.choice
                break
            n_infeas += 1
        history.append(_record(rnd, ev, n_at_eval, len(fails), mode, confidence, conf_rat,
                               planner.k_star(), planner.target_label(), n_infeas,
                               round(time.time() - t_round, 1), tokens=tokens,
                               cluster_method=planner.cluster_method()))
        del policy; gc.collect(); torch.cuda.is_available() and torch.cuda.empty_cache()

    final = history[-1] if history else {}
    result = {
        "history": history,
        "final_success": next((h["eval_success"] for h in reversed(history)
                               if h["eval_success"] is not None), None),
        "n_demos": len(trajs),
        "budget": cfg.get("budget"),
        "n_init_demos": len(init_trajs) if init_trajs is not None else cfg["num_init_demos"],
        "final_demos": cfg["final_demos"],
        "n_bridge_used": sum(1 for h in history if h.get("mode") == "bridge"),
        "n_select_used": sum(1 for h in history if h.get("mode") == "select"),
    }
    log_fn(f"\n[FINAL] DISTIL pure-policy success={result['final_success']} "
           f"| demos={len(trajs)} (last frozen held-out eval)")
    for e in (env, eval_env):
        e.close()
    return result


def _record(rnd, ev, n_demos_at_eval, n_fail, mode, confidence, conf_rat,
            k_star, cluster_label_target, n_infeasible_attempts, sec, tokens=None,
            cluster_method=None) -> Dict[str, Any]:
    return {
        "round": rnd,
        "n_demos_at_eval": n_demos_at_eval,
        "eval_success": ev["success_rate"],
        "eval_coverage": ev.get("mean_coverage"),
        "n_screen_failures": n_fail,
        "mode": mode,                       # 'select' | 'bridge' | None
        "confidence": confidence,           # 0-100 | None
        "confidence_rationale": conf_rat,
        "k_star": k_star,
        "cluster_method": cluster_method,
        "cluster_label_target": cluster_label_target,
        "n_infeasible_attempts": n_infeasible_attempts,
        "sec": sec,
        "tokens": tokens or {},
    }


# Backward-compat alias for the legacy engine CLI (distil.engine_cli).
run_p4_hybrid = run_distil
