#!/usr/bin/env python
"""Convenience single-env driver: run P4-LLM + the 5 IIL baselines for ONE
``--env`` and write the combined ``results/{ENV}_run{id}.json`` (the schema the
original spec asked for), plus per-method dirs under
``results/{ENV}/run_{id}/{method}/``. Equivalent to running orchestrator.run_one
then orchestrator.run_baselines for the env, then aggregation for that env.

    python run_experiment.py --env HalfCheetah-v4 --seed 42
    python run_experiment.py --env HalfCheetah-v4 --smoke      # fast wiring check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.envs import (  # noqa: E402
    env_setup as E, experts as X,
)
from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.orchestrator import (  # noqa: E402
    _common,
)
from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.orchestrator.workspace import (  # noqa: E402
    workspace_for,
)
from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.p4 import (  # noqa: E402
    pipeline_p4,
)

SUITE_ROOT = Path(__file__).resolve().parent
ALL_METHODS = ["p4_llm", "safe_dagger", "dropout_dagger", "ensemble_dagger",
               "thrifty_dagger", "stagger"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--run_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=ALL_METHODS)
    ap.add_argument("--config", type=str, default=str(SUITE_ROOT / "config.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_llm", action="store_true")
    args = ap.parse_args()

    cfg = _common.load_cfg(args.config)
    cfg["seed"] = int(args.seed)
    if args.budget is not None:
        cfg["budget"] = int(args.budget)
    if args.smoke:
        cfg = _common.apply_smoke(cfg)
    seed = int(cfg.get("seed", 42))
    budget = int(cfg.get("budget", 15))

    E.register_robotics()
    resolved, sub = E.resolve_env_id(args.env)
    if sub:
        print(f"[run] NOTE substituting {args.env} -> {resolved}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] env={args.env} resolved={resolved} device={device} seed={seed} budget={budget}", flush=True)

    ws = workspace_for(args.env, args.run_id)
    expert = X.load_expert(args.env, resolved_env_id=resolved, device="cpu")
    llm_client = None
    if (not args.no_llm) and ("p4_llm" in args.methods):
        llm_client = pipeline_p4.make_client()
        print("[run] P4 LLM client: " + ("connected" if llm_client else
              "OPENAI_BASE_URL unset -> P4 falls back to highest-discrepancy"), flush=True)

    methods_out = {}
    for m in args.methods:
        t0 = time.time()
        res = _common.run_method(m, args.env, resolved, expert, cfg, seed, budget,
                                 device, ws.method_root(m), llm_client=llm_client)
        wall = round(time.time() - t0, 1)
        methods_out[m] = {"history": res["history"], "final_performance": res["final_performance"],
                          "stopped_reason": res["stopped_reason"], "wall_sec": wall}
        fp = res["final_performance"]
        print(f"[run] DONE {m}: mean_reward={fp['mean_reward']:.1f} sr={fp['success_rate']} "
              f"queries={fp['n_queries']} ({wall}s)", flush=True)

    out = {"env": args.env, "resolved_env": resolved, "seed": seed, "budget": budget,
           "config": cfg, "methods": methods_out}
    out_path = E.RESULTS_ROOT / f"{args.env}_run{args.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[run] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
