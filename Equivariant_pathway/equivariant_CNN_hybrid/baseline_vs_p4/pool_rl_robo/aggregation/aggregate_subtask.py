"""Aggregate p4_subtask vs diff_dagger on PushT-v1.

Headline metrics (same as the rest of the suite):
  * queries-to-threshold (demos to reach a held-out SR threshold, default 0.90),
  * final SR at budget,
both off the FROZEN held-out curve that is identical for both arms. Also emits a
coverage summary from each p4_subtask seed's centroid_memory.json (the prescribed
sub-task coverage points), so the "rotate coverage" effect is auditable.

Usage:
  python -m ...pool_rl_robo.aggregation.aggregate_subtask \
      --env PushT-v1 --seeds 1 2 --threshold 0.90
Reads results/<env>/run_<seed>/{p4_subtask,diff_dagger}/results/learning_curve.json
and writes results/aggregate/subtask/<env>_subtask_vs_diffdagger.json.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional

SUITE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = SUITE_ROOT / "results"


def _load_curve(env: str, seed: int, method: str) -> Optional[Dict[str, Any]]:
    p = RESULTS / env / f"run_{seed}" / method / "results" / "learning_curve.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _queries_to_threshold(curve: Dict[str, Any], thr: float) -> Optional[float]:
    """First n_queries at which success_rate >= thr; None if never reached."""
    for r in curve.get("history", []):
        sr = r.get("success_rate")
        nq = r.get("n_queries")
        if sr is not None and nq is not None and float(sr) >= thr:
            return float(nq)
    return None


def _final_sr(curve: Dict[str, Any]) -> Optional[float]:
    fp = curve.get("final_performance") or {}
    if fp.get("success_rate") is not None:
        return float(fp["success_rate"])
    hist = curve.get("history") or []
    for r in reversed(hist):
        if r.get("success_rate") is not None:
            return float(r["success_rate"])
    return None


def _coverage(env: str, seed: int) -> Optional[Dict[str, Any]]:
    p = (RESULTS / env / f"run_{seed}" / "p4_subtask" / "results"
         / "telemetry" / "centroid_memory.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        pts = [e.get("centroid_xyz") for e in data.get("entries", [])
               if e.get("centroid_xyz")]
        if not pts:
            return {"n_points": 0}
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        return {"n_points": len(pts),
                "x_range": [round(min(xs), 4), round(max(xs), 4)],
                "y_range": [round(min(ys), 4), round(max(ys), 4)],
                "x_spread": round(max(xs) - min(xs), 4),
                "y_spread": round(max(ys) - min(ys), 4)}
    except Exception:
        return None


def _agg(values: List[float]) -> Dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "values": values}
    return {"n": len(vals),
            "mean": round(st.mean(vals), 4),
            "std": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "values": values}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PushT-v1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    methods = ["p4_subtask", "diff_dagger"]
    out: Dict[str, Any] = {"env": args.env, "seeds": args.seeds,
                           "threshold": args.threshold, "per_seed": {}, "summary": {}}
    q_thr = {m: [] for m in methods}
    f_sr = {m: [] for m in methods}
    cov = []

    for seed in args.seeds:
        row: Dict[str, Any] = {}
        for m in methods:
            c = _load_curve(args.env, seed, m)
            if c is None:
                row[m] = {"missing": True}
                q_thr[m].append(None)
                f_sr[m].append(None)
                continue
            qt = _queries_to_threshold(c, args.threshold)
            fs = _final_sr(c)
            row[m] = {"queries_to_threshold": qt, "final_sr": fs,
                      "stopped_reason": c.get("stopped_reason")}
            q_thr[m].append(qt)
            f_sr[m].append(fs)
        cv = _coverage(args.env, seed)
        if cv is not None:
            row["p4_subtask_coverage"] = cv
            cov.append(cv)
        out["per_seed"][str(seed)] = row

    for m in methods:
        out["summary"][m] = {
            f"queries_to_{args.threshold:g}": _agg(q_thr[m]),
            "final_sr": _agg(f_sr[m])}
    if cov:
        out["summary"]["p4_subtask_coverage_points_mean"] = round(
            st.mean([c.get("n_points", 0) for c in cov]), 2)

    outdir = RESULTS / "aggregate" / "subtask"
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"{args.env}_subtask_vs_diffdagger.json"
    outp.write_text(json.dumps(out, indent=2, default=str))

    # console table
    print(f"\n=== p4_subtask vs diff_dagger — {args.env} (threshold {args.threshold}) ===")
    for m in methods:
        s = out["summary"][m]
        print(f"  {m:12s}  queries→{args.threshold:g}: "
              f"{s[f'queries_to_{args.threshold:g}']['mean']} "
              f"(±{s[f'queries_to_{args.threshold:g}']['std']})  "
              f"final_SR: {s['final_sr']['mean']} (±{s['final_sr']['std']})")
    print(f"  wrote {outp}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
