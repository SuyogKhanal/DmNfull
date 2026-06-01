"""Cross-environment aggregation (continuous-control analogue of
pool_x_selector's ``aggregation/aggregate.py``). Walks every
``results/{ENV}/run_{id}/{method}/results/learning_curve.json``, writes a
convenience combined ``results/{ENV}_run{id}.json`` per env (all methods'
curves + final performance, the schema the original spec asked for), and a
cross-env ``results/aggregate/summary.json`` (per-(env, method) final reward /
success rate).

    python -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.aggregation.aggregate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

from ..envs.env_setup import RESULTS_ROOT
from ..orchestrator.workspace import METHOD_DIR_NAMES, aggregate_dir


def _load(p: Path) -> Dict:
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return {}


def aggregate() -> Dict:
    import statistics
    if not RESULTS_ROOT.exists():
        print(f"no results at {RESULTS_ROOT}")
        return {}
    # env -> method -> {"reward": [per-seed], "sr": [per-seed], "seeds": [...]}
    acc: Dict[str, Dict[str, Dict[str, list]]] = {}
    for env_dir in sorted(RESULTS_ROOT.glob("*")):
        if not env_dir.is_dir() or env_dir.name == "aggregate":
            continue
        env = env_dir.name
        for run_dir in sorted(env_dir.glob("run_*")):
            run_id = run_dir.name.split("_")[-1]
            combined = {"env": env, "run_id": run_id, "methods": {}}
            for m in METHOD_DIR_NAMES:
                lc = run_dir / m / "results" / "learning_curve.json"
                if not lc.exists():
                    continue
                d = _load(lc)
                if not d:
                    continue
                combined["methods"][m] = {
                    "history": d.get("history", []),
                    "final_performance": d.get("final_performance", {}),
                    "stopped_reason": d.get("stopped_reason"),
                }
                combined.setdefault("resolved_env", d.get("env"))
                combined.setdefault("seed", d.get("seed"))
                fp = d.get("final_performance", {})
                slot = acc.setdefault(env, {}).setdefault(m, {"reward": [], "sr": [], "seeds": []})
                if fp.get("mean_reward") is not None:
                    slot["reward"].append(float(fp["mean_reward"]))
                if fp.get("success_rate") is not None:
                    slot["sr"].append(float(fp["success_rate"]))
                slot["seeds"].append(d.get("seed"))
            if combined["methods"]:
                out = RESULTS_ROOT / f"{env}_run{run_id}.json"
                with open(out, "w") as f:
                    json.dump(combined, f, indent=2, default=str)
                print(f"wrote {out}  ({len(combined['methods'])} methods)")

    def _stats(xs):
        if not xs:
            return {"mean": None, "std": None, "n": 0, "per_seed": []}
        return {"mean": round(statistics.mean(xs), 4),
                "std": round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0,
                "n": len(xs), "per_seed": [round(x, 4) for x in xs]}

    # Cross-seed summary: per-(env, method) mean ± std over all seeds/runs.
    summary: Dict[str, Dict] = {}
    for env, ms in acc.items():
        for m, slot in ms.items():
            summary.setdefault(env, {})[m] = {
                "mean_reward": _stats(slot["reward"]),
                "success_rate": _stats(slot["sr"]),
                "n_seeds": len(slot["seeds"]),
                "seeds": slot["seeds"],
            }
    agg = aggregate_dir() / "summary.json"
    with open(agg, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {agg}  ({sum(len(v) for v in summary.values())} env×method cells)")
    return summary


if __name__ == "__main__":
    aggregate()
    sys.exit(0)
