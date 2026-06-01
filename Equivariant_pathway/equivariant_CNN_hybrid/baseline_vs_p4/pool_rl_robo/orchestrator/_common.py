"""Shared driver helpers for run_one (P4) and run_baselines (5 IIL).

Both drivers are env-indexed (one env per SLURM array task) and run their
method(s) sequentially into ``results/{ENV}/run_{id}/{method}/``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from ..envs import env_setup as E
from ..envs import experts as X
from ..p4 import pipeline_p4
from ..selection import iil_baselines
from . import bootstrap as bootstrap_mod
from .workspace import workspace_for


def _info(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [orchestrator] {msg}", flush=True)


def load_cfg(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_smoke(cfg: Dict) -> Dict:
    cfg = dict(cfg)
    cfg["budget"] = 1
    cfg["n_eval_episodes"] = 2
    cfg["n_seed_demos"] = 2
    cfg["initial_epochs"] = 20
    cfg["round_epochs"] = 10
    cfg["max_steps"] = min(int(cfg.get("max_steps", 1000)), 200)
    cfg["n_rollout_episodes"] = 1
    cfg.setdefault("baselines", {})
    cfg["baselines"] = dict(cfg["baselines"])
    for k in ("ensemble", "thrifty"):
        cfg["baselines"][k] = {**(cfg["baselines"].get(k, {}) or {}), "M": 2}
    cfg["baselines"]["thrifty"]["q_epochs"] = 10
    # shrink the diffusion + P4 knobs so the Fetch path is fast in smoke
    cfg["diffusion"] = {**(cfg.get("diffusion", {}) or {}),
                        "T": 10, "n_uncertainty_samples": 3, "batch_size": 64}
    cfg["p4"] = {**(cfg.get("p4", {}) or {}), "n_failure_rollouts": 2, "top_k_failures": 2}
    return cfg


def run_method(method: str, requested_env: str, resolved_env: str, expert,
               cfg: Dict, seed: int, budget: int, device: torch.device,
               method_root: Path, llm_client=None) -> Dict:
    kind = iil_baselines.KIND_OF[method]
    if kind == "p4_llm":
        return pipeline_p4.run(
            requested_env_id=requested_env, resolved_env_id=resolved_env,
            expert=expert, cfg=cfg, seed=seed, budget=budget, device=device,
            method_name=method, method_root=method_root, llm_client=llm_client)
    return iil_baselines.run_dagger(
        requested_env_id=requested_env, resolved_env_id=resolved_env,
        expert=expert, method_name=method, method_kind=kind, cfg=cfg, seed=seed,
        budget=budget, device=device, method_root=method_root, llm_client=None)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--env", required=True)
    ap.add_argument("--run_id", type=int, default=0)
    ap.add_argument("--methods", type=str, default=None,
                    help="comma-sep subset; default = config methods")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_llm", action="store_true")


def run_main(args: argparse.Namespace, default_methods: List[str],
             summary_filename: str) -> int:
    cfg = load_cfg(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.budget is not None:
        cfg["budget"] = int(args.budget)
    if args.smoke:
        cfg = apply_smoke(cfg)
    seed = int(cfg.get("seed", 42))
    budget = int(cfg.get("budget", 15))

    methods = ([m.strip() for m in args.methods.split(",") if m.strip()]
               if args.methods else list(cfg.get("methods", default_methods) or default_methods))

    E.register_robotics()
    resolved, sub = E.resolve_env_id(args.env)
    if sub:
        _info(f"NOTE substituting env {args.env} -> {resolved}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _info(f"env={args.env} resolved={resolved} device={device} seed={seed} "
          f"budget={budget} methods={methods}")

    ws = bootstrap_mod.ensure_bootstrap(args.env, args.run_id)
    expert = X.load_expert(args.env, resolved_env_id=resolved, device="cpu")

    llm_client = None
    if (not args.no_llm) and ("p4_llm" in methods):
        llm_client = pipeline_p4.make_client()
        _info("P4 LLM client: " + ("connected" if llm_client else
              "OPENAI_BASE_URL unset -> P4 falls back to highest-discrepancy"))

    results: Dict[str, Dict] = {}
    for m in methods:
        if m not in iil_baselines.KIND_OF:
            _info(f"skipping unknown method {m!r}")
            continue
        t0 = time.time()
        res = run_method(m, args.env, resolved, expert, cfg, seed, budget, device,
                         ws.method_root(m), llm_client=llm_client)
        wall = round(time.time() - t0, 1)
        fp = res["final_performance"]
        results[m] = {"final_performance": fp, "stopped_reason": res["stopped_reason"],
                      "wall_sec": wall}
        _info(f"DONE {m}: mean_reward={fp['mean_reward']:.1f} sr={fp['success_rate']} "
              f"queries={fp['n_queries']} ({wall}s)")

    summary = {"env": args.env, "resolved_env": resolved, "run_id": args.run_id,
               "seed": seed, "budget": budget, "methods": methods, "results": results}
    out = ws.root / summary_filename
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _info(f"wrote {out}")
    return 0
