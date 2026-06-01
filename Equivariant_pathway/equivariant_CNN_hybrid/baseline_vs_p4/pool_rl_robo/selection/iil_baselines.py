"""The 5 IIL decision rules + the unified DAgger loop, backbone-agnostic.

Each method builds its env's backbone via ``policies.factory.build_policy`` (MLP
for locomotion, R3M-image Diffusion for Fetch) and flows through the same
``rollout`` helpers. ``finalize`` sets ``score``/``queryable`` per state using
``record["samples"]`` (MC-dropout / ensemble members / diffusion denoise draws —
whichever the backbone produced); ``pick_one`` selects ONE state to query; the
expert demonstrates from there. 1 query/round across all methods.
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
from ..logging_ext.compression_log import CompressionLog
from ..logging_ext.training_log import TrainingLog
from ..policies import factory as PF
from . import rollout, success_q

KIND_OF = {
    "safe_dagger": "safe", "dropout_dagger": "dropout", "ensemble_dagger": "ensemble",
    "thrifty_dagger": "thrifty", "stagger": "stagger", "p4_llm": "p4_llm",
}
KINDS = ("safe", "dropout", "ensemble", "thrifty", "stagger")


def _info(method: str, env_id: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{method}|{env_id}] {msg}", flush=True)


def _norm(vals):
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    return [0.0 for _ in vals] if hi - lo < 1e-12 else [(v - lo) / (hi - lo) for v in vals]


def _quantile(vals, q):
    return float(np.quantile(np.asarray(vals, float), float(q))) if vals else 0.0


def finalize(kind: str, records: List[Dict], hp: Dict, rng: random.Random, ctx: Dict) -> None:
    if not records:
        return
    if kind == "safe":
        tau = float(hp.get("tau", 0.5))
        for r in records:
            r["score"] = float(r["discrepancy"])
            r["queryable"] = r["discrepancy"] > tau
    elif kind == "dropout":
        tau, p = float(hp.get("tau", 0.5)), float(hp.get("p", 0.5))
        for r in records:
            S = np.asarray(r["samples"], np.float32)
            exp = np.asarray(r["expert_action"], np.float32)
            d = np.linalg.norm(S - exp[None, :], axis=1) if S.size else np.array([1e9])
            p_hat = float(np.mean(d <= tau))
            r["score"] = float(1.0 - p_hat)
            r["queryable"] = p_hat < p
    elif kind == "ensemble":
        tau, sigma = float(hp.get("tau", 0.5)), float(hp.get("sigma", 0.02))
        for r in records:
            S = np.asarray(r["samples"], np.float32)
            exp = np.asarray(r["expert_action"], np.float32)
            doubt = float(S.var(axis=0).mean()) if S.size else 0.0
            mean_disc = float(np.mean(np.linalg.norm(S - exp[None, :], axis=1))) if S.size else 0.0
            r["doubt"], r["mean_discrepancy"] = doubt, mean_disc
            r["score"] = doubt
            r["queryable"] = (mean_disc > tau) or (doubt > sigma)
    elif kind == "thrifty":
        gamma, alpha_h = float(hp.get("gamma", 0.98)), float(hp.get("alpha_h", 0.10))
        novelties = []
        for r in records:
            S = np.asarray(r["samples"], np.float32)
            r["novelty"] = float(S.var(axis=0).mean()) if S.size else 0.0
            novelties.append(r["novelty"])
        qnet = ctx.get("qnet")
        risks = [0.0] * len(records)
        if qnet is not None:
            success_q.train_success_q(qnet, ctx.get("demo_trajs", []), ctx.get("round_episodes", []),
                                      gamma=gamma, epochs=int(hp.get("q_epochs", 50)),
                                      lr=float(hp.get("q_lr", 1e-3)), batch_size=int(ctx.get("batch_size", 256)),
                                      device=ctx["device"], seed=int(ctx.get("seed", 0)))
            feat = np.stack([np.asarray(r["feat"], np.float32) for r in records])
            act = np.stack([np.asarray(r["novice_action"], np.float32) for r in records])
            qs = success_q.q_values(qnet, feat, act, ctx["device"])
            if len(qs) == len(records):
                risks = [1.0 - q for q in qs]
        delta_h, beta_h = _quantile(novelties, 1 - alpha_h), _quantile(risks, 1 - alpha_h)
        nov_n, risk_n = _norm(novelties), _norm(risks)
        for i, r in enumerate(records):
            r["risk"] = float(risks[i])
            r["score"] = float(nov_n[i] + risk_n[i])
            r["queryable"] = (r["novelty"] > delta_h) or (r["risk"] > beta_h)
    elif kind == "stagger":
        for r in records:
            r["score"] = rng.random()
            r["queryable"] = True
    elif kind == "p4_llm":
        for r in records:
            r["score"] = float(r["discrepancy"])
            r["queryable"] = True
    else:
        raise ValueError(f"unknown selector kind {kind!r}")


def pick_one(records: List[Dict], rng: random.Random) -> Optional[Dict]:
    cands = [r for r in records if r.get("queryable") or float(r.get("score", 0.0)) > 0.0]
    if not cands:
        return None
    queryable = [r for r in cands if r.get("queryable")]
    pool = queryable if queryable else cands
    for r in pool:
        r.setdefault("_jitter", rng.random())
    pool.sort(key=lambda r: (-float(r.get("score", 0.0)), -float(r.get("discrepancy", 0.0)), r["_jitter"]))
    return pool[0]


def run_dagger(*, requested_env_id: str, resolved_env_id: str, expert, method_name: str,
               method_kind: str, cfg: Dict, seed: int, budget: int, device: torch.device,
               method_root: Optional[Path] = None, llm_client=None) -> Dict:
    kind = method_kind
    bl = cfg.get("baselines", {}) or {}
    need_samples = kind in rollout.SAMPLE_KINDS
    max_steps = int(cfg["max_steps"])
    n_eval = int(cfg["n_eval_episodes"])
    n_roll = int(cfg.get("n_rollout_episodes", 1))
    max_consec = int(cfg.get("max_consecutive_empty", 8))
    # diffusion (Fetch) is far more data/epoch-hungry than the locomotion MLP, so it
    # gets its own heavier seed-demo/epoch schedule (cfg.diffusion overrides).
    _diff = cfg.get("diffusion", {}) or {}
    _is_diff = PF.backbone_for(resolved_env_id) == "diffusion"
    n_seed = int(_diff.get("n_seed_demos", cfg["n_seed_demos"]) if _is_diff else cfg["n_seed_demos"])
    initial_epochs = int(_diff.get("initial_epochs", cfg.get("initial_epochs", 300)) if _is_diff else cfg.get("initial_epochs", 300))
    round_epochs = int(_diff.get("round_epochs", cfg.get("round_epochs", 150)) if _is_diff else cfg.get("round_epochs", 150))

    env = E.make_env(resolved_env_id, render=PF.needs_render(resolved_env_id))
    policy = PF.build_policy(resolved_env_id, kind, cfg, device, env)
    if PF.backbone_for(resolved_env_id) == "diffusion":
        k = int((cfg.get("diffusion", {}) or {}).get("n_uncertainty_samples", 8))
    elif kind == "dropout":
        k = int((bl.get("dropout", {}) or {}).get("N", 16))
    else:
        k = int((bl.get(kind, {}) or {}).get("M", 5)) if kind in ("ensemble", "thrifty") else 1

    qnet = success_q.SuccessQNet(policy.feature_dim, policy.act_dim).to(device) if kind == "thrifty" else None
    rng = random.Random(0xC0FFEE + 1009 * (abs(hash(method_name)) % 997) + seed)
    batch_size = int(cfg.get("batch_size", 256))

    results_dir = tlog = clog = None
    if method_root is not None:
        results_dir = Path(method_root) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        tlog = TrainingLog(results_dir / "training_log.csv")
        if kind == "p4_llm":
            clog = CompressionLog(results_dir / "compression_log.csv")

    def _train(epochs: int) -> None:
        X, Y = rollout.assemble(demo_trajs)
        if X.shape[0]:
            policy.fit(X, Y, epochs=epochs)

    def _eval(ts: int) -> Dict:
        return rollout.eval_policy(env, resolved_env_id, policy, n_eval, ts, max_steps)

    def _persist(history, final, stopped):
        if results_dir is None:
            return
        with open(results_dir / "learning_curve.json", "w") as f:
            json.dump({"method": method_name, "env": resolved_env_id, "requested_env": requested_env_id,
                       "seed": seed, "budget": budget, "backbone": PF.backbone_for(resolved_env_id),
                       "history": history, "final_performance": final, "stopped_reason": stopped},
                      f, indent=2, default=str)

    demo_trajs: List[Dict] = rollout.collect_expert_demos(env, resolved_env_id, expert, policy, n_seed, seed, max_steps)
    _info(method_name, resolved_env_id, f"backbone={PF.backbone_for(resolved_env_id)} seed demos={len(demo_trajs)}; initial BC ({initial_epochs} ep)")
    _train(initial_epochs)
    ev0 = _eval(seed)
    history: List[Dict] = [{"round": 0, "n_queries": 0, "n_demos": len(demo_trajs),
                            "mean_reward": ev0["mean_reward"], "std_reward": ev0["std_reward"],
                            "success_rate": ev0["success_rate"], "picked_score": None,
                            "n_candidates": 0, "rollout_success_rate": None}]
    _persist(history, dict(history[-1]), "running")
    if tlog:
        tlog.write_row(0, 0, len(demo_trajs), ev0["mean_reward"], ev0["std_reward"], ev0["success_rate"], initial_epochs, 0.0)
    _info(method_name, resolved_env_id, f"round 0: mean_reward={ev0['mean_reward']:.1f} sr={ev0['success_rate']}")

    n_queries = 0
    consec = 0
    stopped = "budget_exhausted"
    for rnd in range(1, int(budget) + 1):
        records, episodes = rollout.rollout_candidates(env, resolved_env_id, policy, expert,
                                                       seed + rnd * 1000, n_roll, max_steps, need_samples, k)
        roll_sr = float(np.mean([ep["success"] for ep in episodes])) if episodes else 0.0
        ctx = {"device": device, "qnet": qnet, "demo_trajs": demo_trajs,
               "round_episodes": episodes, "batch_size": batch_size, "seed": seed}
        if kind == "p4_llm":
            from ..p4 import selector as p4_selector
            finalize("p4_llm", records, {}, rng, ctx)
            idx = p4_selector.select(requested_env_id, records, llm_client, rng, resolved_env_id=resolved_env_id)
            picked = records[idx] if (0 <= idx < len(records)) else pick_one(records, rng)
        else:
            finalize(kind, records, bl.get(kind, {}), rng, ctx)
            picked = pick_one(records, rng)

        saved = 0
        picked_score = None
        if picked is not None and records:
            ep = episodes[picked["ep"]]
            demo = rollout.expert_demo_from_state(env, resolved_env_id, policy, ep["ep_seed"],
                                                  ep["actions"][:picked["t"]], expert, max_steps)
            if demo is not None and len(demo["feat"]) > 0:
                demo_trajs.append(demo)
                saved = 1
                n_queries += 1
                picked_score = float(picked.get("score", 0.0))
        if saved > 0:
            _train(round_epochs)
        ev = _eval(seed + rnd)
        history.append({"round": rnd, "n_queries": n_queries, "n_demos": len(demo_trajs),
                        "mean_reward": ev["mean_reward"], "std_reward": ev["std_reward"],
                        "success_rate": ev["success_rate"], "picked_score": picked_score,
                        "n_candidates": len(records), "rollout_success_rate": roll_sr})
        _persist(history, dict(history[-1]), "running")
        if tlog:
            tlog.write_row(rnd, n_queries, len(demo_trajs), ev["mean_reward"], ev["std_reward"],
                           ev["success_rate"], (round_epochs if saved else 0), 0.0)
        if clog:
            clog.write_row(rnd, len(records), saved, picked_score, roll_sr, ev["mean_reward"])
        _info(method_name, resolved_env_id,
              f"round {rnd}: cand={len(records)} saved={saved} q={n_queries}/{budget} "
              f"mean_reward={ev['mean_reward']:.1f} sr={ev['success_rate']}")
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
    last = history[-1]
    final = {"mean_reward": last["mean_reward"], "std_reward": last["std_reward"],
             "success_rate": last["success_rate"], "n_queries": n_queries, "n_demos": last["n_demos"]}
    _persist(history, final, stopped)
    return {"method": method_name, "env": resolved_env_id, "history": history,
            "final_performance": final, "stopped_reason": stopped}


def run(**kwargs) -> Dict:
    return run_dagger(**kwargs)
