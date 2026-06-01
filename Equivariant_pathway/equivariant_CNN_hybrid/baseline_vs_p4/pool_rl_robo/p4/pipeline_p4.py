"""Unified P4-LLM pipeline.

* GOAL-conditioned Fetch envs -> ``run_full_fetch``: the full prescribe-and-load
  P4 on the R3M-image **diffusion** backbone. Each round: roll out the novice
  (several episodes) -> top-K FAILURES -> render START/FIRST-MISTAKE/END frames ->
  VLM ANALYSIS pass -> text PRESCRIPTION pass (compress K failures into ONE config)
  -> STRICT loadable+solvable check (load config, roll EXPERT, require is_success)
  -> the expert trajectory is the demo. Infeasible -> fall back to a real
  (env-sampled, expert-solvable) failure config.
* MuJoCo locomotion -> ``iil_baselines.run_dagger`` (MLP backbone) with the LLM as
  a state SELECTOR (no config space to prescribe there).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..envs import env_setup as E
from ..envs import fetch_config as FC
from ..logging_ext.compression_log import CompressionLog
from ..logging_ext.training_log import TrainingLog
from ..policies import factory as PF
from ..selection import iil_baselines, rollout
from . import analysis as AN
from . import frames as FR
from . import kag
from . import prescribe as PR
from .runner import QwenClient


def _info(env_id: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [p4-full|{env_id}] {msg}", flush=True)


def make_client() -> Optional[QwenClient]:
    import os
    return QwenClient() if os.environ.get("OPENAI_BASE_URL") else None


def run(*, requested_env_id: str, resolved_env_id: str, expert, cfg: Dict, seed: int,
        budget: int, device: torch.device, method_name: str = "p4_llm",
        method_root: Optional[Path] = None, llm_client: Optional[QwenClient] = None) -> Dict:
    if llm_client is None:
        llm_client = make_client()
    kag.set_kag_dir((cfg.get("p4") or {}).get("kag_dir"))
    if E.is_goal_env(resolved_env_id):
        return run_full_fetch(requested_env_id=requested_env_id, resolved_env_id=resolved_env_id,
                              expert=expert, cfg=cfg, seed=seed, budget=budget, device=device,
                              method_name=method_name, method_root=method_root, llm_client=llm_client)
    return iil_baselines.run_dagger(
        requested_env_id=requested_env_id, resolved_env_id=resolved_env_id, expert=expert,
        method_name=method_name, method_kind="p4_llm", cfg=cfg, seed=seed, budget=budget,
        device=device, method_root=method_root, llm_client=llm_client)


def run_full_fetch(*, requested_env_id: str, resolved_env_id: str, expert, cfg: Dict, seed: int,
                   budget: int, device: torch.device, method_name: str = "p4_llm",
                   method_root: Optional[Path] = None, llm_client: Optional[QwenClient] = None) -> Dict:
    p4cfg = cfg.get("p4", {}) or {}
    max_steps = int(cfg["max_steps"])
    n_eval = int(cfg["n_eval_episodes"])
    n_roll = int(p4cfg.get("n_failure_rollouts", 6))
    top_k = int(p4cfg.get("top_k_failures", 3))
    max_consec = int(cfg.get("max_consecutive_empty", 8))
    # Fetch P4 always uses the diffusion backbone -> heavier diffusion schedule.
    _diff = cfg.get("diffusion", {}) or {}
    n_seed = int(_diff.get("n_seed_demos", cfg["n_seed_demos"]))
    initial_epochs = int(_diff.get("initial_epochs", cfg.get("initial_epochs", 300)))
    round_epochs = int(_diff.get("round_epochs", cfg.get("round_epochs", 150)))
    is_pick = requested_env_id.startswith("FetchPickAndPlace")

    results_dir = tlog = clog = None
    if method_root is not None:
        results_dir = Path(method_root) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        tlog = TrainingLog(results_dir / "training_log.csv")
        clog = CompressionLog(results_dir / "compression_log.csv")

    env = E.make_env(resolved_env_id, render=True)
    policy = PF.build_policy(resolved_env_id, "p4_llm", cfg, device, env)
    bounds = FC.get_bounds(env, resolved_env_id)
    render_env = None
    rng = random.Random(0xC0FFEE + seed)

    def _train(epochs):
        X, Y = rollout.assemble(demo_trajs)
        if X.shape[0]:
            policy.fit(X, Y, epochs=epochs)

    def _eval(ts):
        return rollout.eval_policy(env, resolved_env_id, policy, n_eval, ts, max_steps)

    def _persist(history, final, stopped):
        if results_dir is None:
            return
        with open(results_dir / "learning_curve.json", "w") as f:
            json.dump({"method": method_name, "env": resolved_env_id, "requested_env": requested_env_id,
                       "seed": seed, "budget": budget, "backbone": "diffusion",
                       "p4_mode": "full_prescribe_and_load", "history": history,
                       "final_performance": final, "stopped_reason": stopped}, f, indent=2, default=str)

    demo_trajs = rollout.collect_expert_demos(env, resolved_env_id, expert, policy, n_seed, seed, max_steps)
    _info(resolved_env_id, f"seed demos={len(demo_trajs)}; initial BC ({initial_epochs} ep); "
                           f"LLM={'on' if llm_client else 'OFF (fallback configs only)'}")
    _train(initial_epochs)
    ev0 = _eval(seed)
    history: List[Dict] = [{"round": 0, "n_queries": 0, "n_demos": len(demo_trajs),
                            "mean_reward": ev0["mean_reward"], "std_reward": ev0["std_reward"],
                            "success_rate": ev0["success_rate"], "n_failures": 0,
                            "prescribed_solved": None, "infeasible": None, "rollout_success_rate": None}]
    _persist(history, dict(history[-1]), "running")
    if tlog:
        tlog.write_row(0, 0, len(demo_trajs), ev0["mean_reward"], ev0["std_reward"], ev0["success_rate"], initial_epochs, 0.0)
    _info(resolved_env_id, f"round 0: sr={ev0['success_rate']} mean_reward={ev0['mean_reward']:.1f}")

    n_queries = 0
    consec = 0
    stopped = "budget_exhausted"
    for rnd in range(1, int(budget) + 1):
        records, episodes = rollout.rollout_candidates(env, resolved_env_id, policy, expert,
                                                       seed + rnd * 1000, n_roll, max_steps,
                                                       need_samples=False, k=1)
        by_ep: Dict[int, List[Dict]] = {}
        for r in records:
            by_ep.setdefault(r["ep"], []).append(r)
        roll_sr = float(np.mean([ep["success"] for ep in episodes])) if episodes else 0.0
        fails = [i for i, ep in enumerate(episodes) if not ep["success"]]

        def _final_dist(i):
            gds = [rr.get("goal_dist") for rr in by_ep.get(i, []) if rr.get("goal_dist") is not None]
            return gds[-1] if gds else 1e9
        fails_sorted = sorted(fails, key=lambda i: -_final_dist(i))[:max(1, top_k)] if fails else []

        analyses: List[str] = []
        demo = None
        solved = False
        infeasible = False
        pres = None
        if fails_sorted and llm_client is not None:
            if render_env is None:
                render_env = FR.make_render_env(resolved_env_id)
            for i in fails_sorted:
                ep = episodes[i]
                recs = by_ep.get(i, [])
                mistake = max(recs, key=lambda rr: rr.get("discrepancy", 0.0))["t"] if recs else 0
                summary = (f"goal={np.round(ep.get('goal'), 3) if ep.get('goal') is not None else '?'}, "
                           f"final object/gripper-to-goal distance={_final_dist(i):.3f}, "
                           f"return={ep['ep_return']:.1f}, length={len(ep['actions'])}, success=False.")
                frs = FR.render_episode_frames(render_env, ep["ep_seed"], ep["actions"], [0, mistake])
                analyses.append(AN.analyze_failure(requested_env_id, summary, frs, llm_client))
            try:
                pres = PR.prescribe_config(requested_env_id, analyses, bounds, llm_client, resolved_env_id)
                demo, solved, _g, _o = rollout.expert_demo_from_config(
                    env, resolved_env_id, policy, pres.get("goal"),
                    (pres.get("object") if is_pick else None), expert, max_steps)
            except Exception as e:  # noqa: BLE001
                _info(resolved_env_id, f"round {rnd}: prescription/strict-check error: {e}")
                demo, solved = None, False

        if demo is None:
            infeasible = bool(fails_sorted)
            cand = fails_sorted[0] if fails_sorted else (rng.randrange(len(episodes)) if episodes else None)
            if cand is not None:
                ep = episodes[cand]
                demo, solved, _g, _o = rollout.expert_demo_from_config(
                    env, resolved_env_id, policy, ep.get("goal"),
                    (ep.get("object") if is_pick else None), expert, max_steps)

        saved = 0
        if demo is not None and len(demo["feat"]) > 0:
            demo_trajs.append(demo)
            saved = 1
            n_queries += 1
        if saved > 0:
            _train(round_epochs)
        ev = _eval(seed + rnd)
        history.append({"round": rnd, "n_queries": n_queries, "n_demos": len(demo_trajs),
                        "mean_reward": ev["mean_reward"], "std_reward": ev["std_reward"],
                        "success_rate": ev["success_rate"], "n_failures": len(fails),
                        "prescribed_solved": (None if pres is None else bool(solved and not infeasible)),
                        "infeasible": infeasible, "rollout_success_rate": roll_sr})
        _persist(history, dict(history[-1]), "running")
        if tlog:
            tlog.write_row(rnd, n_queries, len(demo_trajs), ev["mean_reward"], ev["std_reward"],
                           ev["success_rate"], (round_epochs if saved else 0), 0.0)
        if clog:
            clog.write_row(rnd, len(fails), saved, (1.0 if (solved and not infeasible) else 0.0), roll_sr, ev["mean_reward"])
        _info(resolved_env_id, f"round {rnd}: fails={len(fails)} prescribed={'Y' if pres else 'N'} "
                               f"solved={solved} infeasible={infeasible} q={n_queries}/{budget} "
                               f"sr={ev['success_rate']} mean_reward={ev['mean_reward']:.1f}")
        if saved == 0:
            consec += 1
            if consec >= max_consec:
                stopped = "no_progress"
                break
        else:
            consec = 0
        if n_queries >= int(budget):
            stopped = "budget_exhausted"
            break
    else:
        stopped = "max_rounds"

    env.close()
    if render_env is not None:
        render_env.close()
    last = history[-1]
    final = {"mean_reward": last["mean_reward"], "std_reward": last["std_reward"],
             "success_rate": last["success_rate"], "n_queries": n_queries, "n_demos": last["n_demos"]}
    _persist(history, final, stopped)
    return {"method": method_name, "env": resolved_env_id, "history": history,
            "final_performance": final, "stopped_reason": stopped}
