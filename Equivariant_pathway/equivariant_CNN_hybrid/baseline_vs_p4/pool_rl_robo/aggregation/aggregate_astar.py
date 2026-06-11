"""A* study aggregation — P4-LLM-select vs Diff-DAgger across 5 seeds.

Walks ``results/{ENV}/run_{1..5}/{p4_select,diff_dagger}/results/learning_curve.json``
(one isolated seed per run; both methods in a run share the SAME bootstrap), and
produces the publishable comparison:

  * mean ± std success-rate curve vs cumulative demonstrations (step-interpolated
    per seed onto a shared demo grid, then averaged across seeds present),
  * queries-to-0.90 per method (first demo count whose held-out SR >= 0.90),
    reported as mean ± std and #seeds-reached,
  * a figure with the two mean curves + std bands + the 0.90 line.

    python -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.aggregation.aggregate_astar
    # options: --env PushT-v1  --runs 1,2,3,4,5  --threshold 0.90

Robust to partial / in-flight runs: a method/seed with no curve is skipped and
noted; the script never crashes on missing or half-written files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..envs.env_setup import RESULTS_ROOT
from ..orchestrator.workspace import aggregate_dir

METHODS = ["p4_select", "diff_dagger"]
LABELS = {"p4_select": "P4-LLM-select (ours)", "diff_dagger": "Diff-DAgger"}
COLORS = {"p4_select": "#d62728", "diff_dagger": "#1f77b4"}


def _load(p: Path) -> dict:
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return {}


def _curve_xy(d: dict) -> List[Tuple[float, float]]:
    """(expert_queries, success_rate) points, sorted by queries, monotone-x deduped.

    x = ``n_queries`` = cumulative expert interventions = demonstrations ADDED via
    DAgger (the budget axis, directly comparable across methods and exactly the
    'queries-to-threshold' quantity). NOT ``n_demos`` (= len(replay buffer) =
    transitions, ~200/episode) which would inflate counts ~200x."""
    pts: List[Tuple[float, float]] = []
    for h in d.get("history", []):
        sr = h.get("success_rate")
        x = h.get("n_queries")
        if x is None:                       # fallback only if queries were unavailable
            x = h.get("n_demos")
        if sr is None or x is None:
            continue
        pts.append((float(x), float(sr)))
    pts.sort(key=lambda t: t[0])
    out: List[Tuple[float, float]] = []
    for x, y in pts:                       # keep last sr seen at each x
        if out and out[-1][0] == x:
            out[-1] = (x, y)
        else:
            out.append((x, y))
    return out


def _step_interp(pts: List[Tuple[float, float]], grid: List[float]) -> List[Optional[float]]:
    """Step (zero-order-hold) interpolation WITHIN a seed's recorded range only.
    None before the seed's first point AND None beyond its last point — we do NOT
    forward-fill past the end, so a seed that finished early (e.g. reached 100% and
    stopped) does not extrapolate a flat line across the rest of the x-axis."""
    if not pts:
        return [None] * len(grid)
    last_x = pts[-1][0]
    out: List[Optional[float]] = []
    i = 0
    cur: Optional[float] = None
    for g in grid:
        while i < len(pts) and pts[i][0] <= g:
            cur = pts[i][1]
            i += 1
        out.append(cur if g <= last_x else None)   # truncate, don't extrapolate
    return out


def _truncate_at_full(pts: List[Tuple[float, float]], full: float = 1.0
                      ) -> List[Tuple[float, float]]:
    """Keep points up to AND INCLUDING the first time SR reaches `full` (100%);
    drop everything after — the method has solved the task, no point extending it."""
    out: List[Tuple[float, float]] = []
    for x, y in pts:
        out.append((x, y))
        if y >= full:
            break
    return out


def _queries_to_threshold(pts: List[Tuple[float, float]], thr: float) -> Optional[float]:
    for x, y in pts:
        if y >= thr:
            return x
    return None


def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _final_expert_calls(d: dict):
    """Last non-null total_expert_calls in the history (secondary cost axis)."""
    tec = None
    for h in d.get("history", []):
        if h.get("total_expert_calls") is not None:
            tec = h.get("total_expert_calls")
    return tec


def _write_per_round_csv(env: str, runs: List[int], out_dir: Path) -> Optional[Path]:
    """Long-format CSV: one row per (run, method, round) with the success rate and the
    other per-round fields. round == cumulative demonstrations (nd_retrain=1); round 0 is
    the shared bootstrap. Sort by (run, round) to read p4_select vs diff_dagger side by
    side. Regenerated for whatever env/runs are passed (works for StackCube etc.)."""
    env_dir = RESULTS_ROOT / env
    rows: List[dict] = []
    for r in runs:
        for m in METHODS:
            d = _load(env_dir / f"run_{r}" / m / "results" / "learning_curve.json")
            if not d:
                continue
            seed = d.get("seed")
            stop = d.get("stopped_reason")
            hist = d.get("history", [])
            for i, h in enumerate(hist):
                rows.append({
                    "env": env, "run": r, "seed": seed, "method": m,
                    "round": h.get("round"),
                    "cumulative_demos": h.get("n_queries"),   # == round at nd_retrain=1
                    "success_rate": h.get("success_rate"),
                    "demos_added": h.get("demos_added"),
                    "total_expert_calls": h.get("total_expert_calls"),
                    # the final row of each curve carries the run-level stop reason
                    "stop_reason": stop if i == len(hist) - 1 else h.get("stop_reason"),
                })
    if not rows:
        return None
    rows.sort(key=lambda x: (x["run"], x["round"] if x["round"] is not None else -1,
                             0 if x["method"] == "p4_select" else 1))
    cols = ["env", "run", "seed", "method", "round", "cumulative_demos",
            "success_rate", "demos_added", "total_expert_calls", "stop_reason"]
    p = out_dir / f"astar_{env}_per_round.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[astar] wrote {p}  ({len(rows)} rows)")
    return p


def aggregate(env: str, runs: List[int], thr: float) -> dict:
    env_dir = RESULTS_ROOT / env
    # method -> seed -> [(demos, sr)]
    curves: Dict[str, Dict[int, List[Tuple[float, float]]]] = {m: {} for m in METHODS}
    # method -> seed -> {stopped_reason, final_q, final_expert_calls}
    meta: Dict[str, Dict[int, dict]] = {m: {} for m in METHODS}
    for r in runs:
        for m in METHODS:
            lc = env_dir / f"run_{r}" / m / "results" / "learning_curve.json"
            d = _load(lc)
            xy = _curve_xy(d) if d else []
            if xy:
                curves[m][r] = xy
                meta[m][r] = {
                    "stopped_reason": d.get("stopped_reason"),
                    "final_q": xy[-1][0],
                    "final_sr": xy[-1][1],
                    "final_expert_calls": _final_expert_calls(d),
                }

    # shared demo grid across all seeds/methods present
    all_x = [x for m in METHODS for s in curves[m] for x, _ in curves[m][s]]
    summary: dict = {"env": env, "threshold": thr, "runs": runs, "methods": {}, "grid": []}
    if not all_x:
        print(f"[astar] no curves yet under {env_dir} (runs {runs}) — nothing to aggregate.")
        return summary
    lo, hi = int(min(all_x)), int(max(all_x))
    grid = list(range(lo, hi + 1))
    summary["grid"] = grid

    for m in METHODS:
        per_seed = curves[m]
        interp = {s: _step_interp(pts, grid) for s, pts in per_seed.items()}
        mean_curve, std_curve, n_curve = [], [], []
        for gi in range(len(grid)):
            vals = [interp[s][gi] for s in per_seed if interp[s][gi] is not None]
            mu, sd = _mean_std(vals)
            mean_curve.append(mu)
            std_curve.append(sd if sd is not None else 0.0)
            n_curve.append(len(vals))
        q90 = {s: _queries_to_threshold(pts, thr) for s, pts in per_seed.items()}
        reached = [v for v in q90.values() if v is not None]
        q_mu, q_sd = _mean_std(reached)
        finals = [pts[-1][1] for pts in per_seed.values() if pts]
        f_mu, f_sd = _mean_std(finals)
        # ---- honest censoring bookkeeping (do NOT silently drop non-reachers) ----
        mseed = meta[m]
        CAP_REASONS = {"max_episodes", "max_rounds"}
        n_hit_cap = sum(1 for s in mseed if mseed[s].get("stopped_reason") in CAP_REASONS)
        stop_dist: Dict[str, int] = {}
        for s in mseed:
            sr_reason = mseed[s].get("stopped_reason") or "unknown"
            stop_dist[sr_reason] = stop_dist.get(sr_reason, 0) + 1
        # seeds that never reached the threshold → right-censored at their final demo count
        censored = {str(s): mseed[s].get("final_q") for s in per_seed if q90[s] is None}
        tec_vals = [mseed[s].get("final_expert_calls") for s in mseed
                    if mseed[s].get("final_expert_calls") is not None]
        tec_mu, tec_sd = _mean_std(tec_vals)
        summary["methods"][m] = {
            "label": LABELS[m],
            "n_seeds": len(per_seed),
            "seeds": sorted(per_seed),
            "mean_curve": mean_curve, "std_curve": std_curve, "n_per_grid": n_curve,
            "queries_to_threshold": {str(s): q90[s] for s in q90},
            "queries_to_threshold_mean": q_mu, "queries_to_threshold_std": q_sd,
            "n_reached_threshold": len(reached),
            "censored_seeds": censored,            # never-reached, right-censored at final_q
            "n_hit_cap": n_hit_cap,                # stopped on the episode backstop
            "stopped_reason_dist": stop_dist,
            "final_sr_mean": f_mu, "final_sr_std": f_sd,
            "final_expert_calls_mean": tec_mu, "final_expert_calls_std": tec_sd,
        }
        qstr = f"{q_mu:.1f}±{q_sd:.1f}" if q_mu is not None else "—"
        fstr = f"{f_mu:.3f}±{f_sd:.3f}" if f_mu is not None else "—"
        tstr = f"{tec_mu:.0f}±{tec_sd:.0f}" if tec_mu is not None else "—"
        cens = f" CENSORED={list(censored.keys())}" if censored else ""
        cap = f" hit_cap={n_hit_cap}" if n_hit_cap else ""
        print(f"[astar] {m:11s} seeds={len(per_seed)} reached{thr:.2f}={len(reached)}/{len(per_seed)} "
              f"q->{thr:.2f}={qstr} demos | final_sr={fstr} | expert_calls={tstr}{cap}{cens}")

    out_dir = aggregate_dir() / "astar"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"astar_{env}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_per_round_csv(env, runs, out_dir)
    _plot(summary, env, thr, out_dir)                 # mean±std across seeds
    _plot_per_seed(env, runs, thr, out_dir)           # one figure per seed (truncated at 100%)
    print(f"[astar] wrote {out_dir}/astar_{env}_summary.json")
    return summary


def _plot_per_seed(env: str, runs: List[int], thr: float, out_dir: Path) -> None:
    """One figure per run: p4_select vs diff_dagger raw success-rate curves, each
    TRUNCATED at the first round it reaches 100% (no extrapolation past it)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[astar] matplotlib unavailable ({e}); skipping per-seed figures")
        return
    env_dir = RESULTS_ROOT / env
    for r in runs:
        series = {}
        for m in METHODS:
            d = _load(env_dir / f"run_{r}" / m / "results" / "learning_curve.json")
            if not d:
                continue
            pts = _truncate_at_full(_curve_xy(d), 1.0)   # cut at first 100%
            if pts:
                series[m] = (pts, d.get("stopped_reason"))
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for m, (pts, stop) in series.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            c = COLORS[m]
            reached = ys[-1] >= 1.0
            lab = LABELS[m]
            if reached:
                lab += f" → 100% at {int(xs[-1])} demos"
            else:
                lab += f" → {ys[-1]:.2f} at {int(xs[-1])} demos ({stop})"
            ax.plot(xs, ys, color=c, lw=2.2, marker="o", ms=3, label=lab, zorder=3)
            if reached:                                  # mark the 100% point
                ax.scatter([xs[-1]], [1.0], color=c, s=70, marker="*", zorder=4)
        ax.axhline(thr, ls="--", color="gray", lw=1.0, zorder=1)
        ax.axhline(1.0, ls=":", color="green", lw=1.0, zorder=1)
        ax.text(0, thr + 0.01, f"{thr:.0%}", color="gray", fontsize=8, va="bottom")
        ax.set_xlabel("Expert demonstrations added")
        ax.set_ylabel("Held-out success rate")
        ax.set_title(f"{env} — run_{r} (seed {r}): P4-LLM-select vs Diff-DAgger\n"
                     f"shared bootstrap, retrain every demo, truncated at 100%")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        p = out_dir / f"astar_{env}_run{r}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[astar] wrote {p}")


def _plot(summary: dict, env: str, thr: float, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[astar] matplotlib unavailable ({e}); skipping figure")
        return
    grid = summary["grid"]
    if not grid:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for m in METHODS:
        md = summary["methods"].get(m)
        if not md:
            continue
        xs, mu, sd = [], [], []
        for gi, g in enumerate(grid):
            if md["mean_curve"][gi] is None:
                continue
            xs.append(g); mu.append(md["mean_curve"][gi]); sd.append(md["std_curve"][gi])
        if not xs:
            continue
        c = COLORS[m]
        lab = md["label"] + f" (n={md['n_seeds']})"
        if md.get("queries_to_threshold_mean") is not None:
            nr, ns = md.get("n_reached_threshold", 0), md.get("n_seeds", 0)
            lab += (f", q→{thr:.0%}={md['queries_to_threshold_mean']:.0f}"
                    f"±{md['queries_to_threshold_std']:.0f}")
            if nr < ns:                      # be explicit when not all seeds reached
                lab += f" ({nr}/{ns} reached)"
        ax.plot(xs, mu, color=c, lw=2.2, label=lab, zorder=3)
        lo = [max(0.0, a - b) for a, b in zip(mu, sd)]
        hi = [min(1.0, a + b) for a, b in zip(mu, sd)]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.18, zorder=2)
    ax.axhline(thr, ls="--", color="gray", lw=1.0, zorder=1)
    ax.text(grid[0], thr + 0.01, f"{thr:.0%}", color="gray", fontsize=9, va="bottom")
    ax.set_xlabel("Expert queries (demonstrations added)")
    ax.set_ylabel("Held-out success rate")
    ax.set_title(f"A* study — {env}: P4-LLM-select vs Diff-DAgger\n"
                 f"5 seeds, shared bootstrap, retrain every demo (mean ± std)")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = out_dir / f"astar_{env}_p4select_vs_diffdagger.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[astar] wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PushT-v1")
    ap.add_argument("--runs", default="1,2,3,4,5")
    ap.add_argument("--threshold", type=float, default=0.90)
    a = ap.parse_args()
    runs = [int(x) for x in str(a.runs).split(",") if x.strip()]
    aggregate(a.env, runs, a.threshold)


if __name__ == "__main__":
    main()
