"""Aggregate baseline-vs-P4 across N repetitions and plot mean +/- std.

For each repetition we read both methods' ``learning_curve.json`` and
re-express the trajectory as ``heldout_sr`` vs ``extra_demos`` (i.e.
demos on top of the initial 20). To make the per-run trajectories
directly comparable we sample-and-hold each curve onto a common
x-grid ``[0, 1, ..., budget]``: at every grid point we take the
most-recent (smaller-or-equal) measurement from that run, falling
back to the round-0 heldout SR if no extras have been added yet.

Outputs (under ``--out_dir``):
  * ``baseline_vs_p4_mean_std.png`` — the main figure (mean + 1 std
    band for both methods + the target line + per-run thin lines).
  * ``baseline_vs_p4_clean.png``    — same but without the thin
    per-run overlays; cleaner for slides.
  * ``aggregate_data.json``         — the raw arrays (x, per-run y,
    mean, std, target, per-method metadata) so the figure can be
    regenerated with a different title / styling without rerunning.
  * ``summary.json``                — small per-run + per-method
    stats (extras-to-target, final SR, runs hitting target).

Usage:
  python -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.aggregate_chart \
      --runs_dir Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/runs \
      --out_dir  Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/figures \
      --budget 15 \
      --title "Hybrid: baseline_only vs p4_only (10 runs, K=15)"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE_PATH = Path(__file__).resolve().parent / "paper.mplstyle"
if STYLE_PATH.exists():
    plt.style.use(str(STYLE_PATH))

METHOD_COLORS = {"baseline_only": "tab:orange", "p4_only": "tab:purple"}
METHOD_LABELS = {"baseline_only": "Baseline-DAgger (top-K by loss)",
                 "p4_only":       "P4 LLM (budgeted prescription)"}


def _load_curve(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[aggregate] failed to read {path}: {e}")
        return None


def _curve_xy(history: List[Dict]) -> Tuple[List[int], List[float]]:
    """Return per-history-row (extra_demos, heldout_sr).

    Both baseline_budget and p4_budget write ``extra_demos`` and
    ``heldout_sr`` for every round, including round 0 (extras=0).
    """
    xs, ys = [], []
    for row in history:
        xs.append(int(row.get("extra_demos", 0) or 0))
        ys.append(float(row.get("heldout_sr", 0.0) or 0.0))
    return xs, ys


def _sample_hold(xs: List[int], ys: List[float], grid: np.ndarray) -> np.ndarray:
    """Sample-and-hold ``ys`` (indexed by ``xs``) onto ``grid``.

    Algorithm: for each g in grid, return the y associated with the
    largest x <= g. If no such x exists (the curve doesn't start at 0)
    we return ys[0] for all g < xs[0] — that's the round-0 SR.
    """
    if not xs:
        return np.full_like(grid, fill_value=np.nan, dtype=float)
    order = np.argsort(xs)
    xs_arr = np.asarray([xs[i] for i in order], dtype=float)
    ys_arr = np.asarray([ys[i] for i in order], dtype=float)
    out = np.empty_like(grid, dtype=float)
    for i, g in enumerate(grid):
        if g < xs_arr[0]:
            out[i] = ys_arr[0]
            continue
        idx = np.searchsorted(xs_arr, g, side="right") - 1
        idx = int(max(0, min(idx, len(xs_arr) - 1)))
        out[i] = ys_arr[idx]
    return out


def _gather_method(runs_dir: Path, method: str, budget: int,
                   ) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Return (per_run_metadata, x_grid, Y) where Y[i, j] is run i's
    heldout SR at grid x_grid[j]. Missing runs are skipped with a
    warning.
    """
    x_grid = np.arange(0, budget + 1, dtype=int)
    per_run: List[Dict] = []
    rows: List[np.ndarray] = []
    for run_path in sorted(runs_dir.glob("run_*")):
        curve_path = run_path / method / "results" / "learning_curve.json"
        data = _load_curve(curve_path)
        if data is None:
            print(f"[aggregate] missing {curve_path}; skipping run")
            continue
        history = data.get("history", []) or []
        if not history:
            print(f"[aggregate] empty history in {curve_path}; skipping run")
            continue
        xs, ys = _curve_xy(history)
        y_on_grid = _sample_hold(xs, ys, x_grid)
        per_run.append({
            "run": run_path.name,
            "raw_xs": xs, "raw_ys": ys,
            "final_extra": int(xs[-1]) if xs else 0,
            "final_sr": float(ys[-1]) if ys else None,
            "stopped_reason": data.get("history", [{}])[-1].get("stopped_reason"),
        })
        rows.append(y_on_grid)
    if not rows:
        return per_run, x_grid, np.zeros((0, len(x_grid)), dtype=float)
    Y = np.vstack(rows)
    return per_run, x_grid, Y


def _first_cross(mean_curve: np.ndarray, x_grid: np.ndarray,
                 target: float) -> Optional[int]:
    above = np.where(mean_curve >= target)[0]
    if above.size == 0:
        return None
    return int(x_grid[int(above[0])])


def _per_run_extras_to_target(per_run: List[Dict], target: float) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for r in per_run:
        crossed: Optional[int] = None
        for x, y in zip(r["raw_xs"], r["raw_ys"]):
            if y >= target:
                crossed = int(x)
                break
        out.append(crossed)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, required=True)
    p.add_argument("--out_dir",  type=str, required=True)
    p.add_argument("--budget",   type=int, required=True)
    p.add_argument("--target_sr", type=float, default=0.90)
    p.add_argument("--title",    type=str, default=None)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or (
        f"Hybrid: baseline_only vs p4_only "
        f"(K={args.budget}, target={args.target_sr:.2f})"
    )

    aggregate_payload: Dict = {
        "runs_dir": str(runs_dir), "budget": int(args.budget),
        "target_sr": float(args.target_sr), "methods": {},
    }
    summary_rows: List[Dict] = []
    method_stats: Dict[str, Dict] = {}

    for method in ("baseline_only", "p4_only"):
        per_run, x_grid, Y = _gather_method(runs_dir, method, args.budget)
        n_runs = Y.shape[0]
        if n_runs == 0:
            print(f"[aggregate] no runs found for {method}; skipping in plot.")
            mean_curve = np.full_like(x_grid, np.nan, dtype=float)
            std_curve  = np.full_like(x_grid, np.nan, dtype=float)
        else:
            mean_curve = Y.mean(axis=0)
            std_curve  = Y.std(axis=0, ddof=0)

        extras_to_target = _per_run_extras_to_target(per_run, args.target_sr)
        n_target = sum(1 for x in extras_to_target if x is not None)

        method_stats[method] = {
            "n_runs": int(n_runs),
            "n_runs_hit_target": int(n_target),
            "mean_extras_to_target": (
                float(np.mean([x for x in extras_to_target if x is not None]))
                if n_target > 0 else None
            ),
            "mean_final_sr_at_budget": float(mean_curve[-1]) if n_runs > 0 else None,
            "std_final_sr_at_budget":  float(std_curve[-1])  if n_runs > 0 else None,
        }

        aggregate_payload["methods"][method] = {
            "x_grid": x_grid.tolist(),
            "Y_per_run": Y.tolist(),
            "mean": mean_curve.tolist(),
            "std":  std_curve.tolist(),
            "per_run": per_run,
            "extras_to_target": extras_to_target,
        }

        for i, r in enumerate(per_run):
            summary_rows.append({
                "method": method, "run": r["run"],
                "final_extra": r["final_extra"], "final_sr": r["final_sr"],
                "extras_to_target": extras_to_target[i],
                "stopped_reason": r.get("stopped_reason"),
            })

    aggregate_payload["method_stats"] = method_stats

    # ------------------------------------------------------------------ #
    # Plot 1: thin per-run lines + bold mean + std band                  #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in ("baseline_only", "p4_only"):
        block = aggregate_payload["methods"][method]
        x_grid = np.asarray(block["x_grid"])
        Y      = np.asarray(block["Y_per_run"])
        mean_c = np.asarray(block["mean"])
        std_c  = np.asarray(block["std"])
        color  = METHOD_COLORS[method]
        label  = METHOD_LABELS[method]
        for row in Y:
            ax.plot(x_grid, row, color=color, alpha=0.18, linewidth=0.8)
        if Y.shape[0] > 0:
            ax.plot(x_grid, mean_c, color=color, marker="o", markersize=4,
                    linewidth=1.8, label=f"{label}  (mean, n={Y.shape[0]})")
            ax.fill_between(x_grid, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.18, linewidth=0)
    ax.axhline(args.target_sr, ls="--", color="tab:red", alpha=0.7,
               label=f"target = {args.target_sr:.2f}")
    ax.set_xlabel("extra demonstrations (on top of initial 20)")
    ax.set_ylabel("heldout success rate (200 layouts)")
    ax.set_xlim(0, args.budget)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_vs_p4_mean_std.png")
    plt.close(fig)
    print(f"[aggregate] wrote {out_dir / 'baseline_vs_p4_mean_std.png'}")

    # ------------------------------------------------------------------ #
    # Plot 2: clean version (no thin per-run lines)                      #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in ("baseline_only", "p4_only"):
        block = aggregate_payload["methods"][method]
        x_grid = np.asarray(block["x_grid"])
        Y      = np.asarray(block["Y_per_run"])
        mean_c = np.asarray(block["mean"])
        std_c  = np.asarray(block["std"])
        color  = METHOD_COLORS[method]
        label  = METHOD_LABELS[method]
        if Y.shape[0] > 0:
            ax.plot(x_grid, mean_c, color=color, marker="o", markersize=4,
                    linewidth=2.0, label=f"{label}  (n={Y.shape[0]})")
            ax.fill_between(x_grid, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.22, linewidth=0,
                            label=f"{label}  ±1 std")
    ax.axhline(args.target_sr, ls="--", color="tab:red", alpha=0.7,
               label=f"target = {args.target_sr:.2f}")
    ax.set_xlabel("extra demonstrations (on top of initial 20)")
    ax.set_ylabel("heldout success rate (200 layouts)")
    ax.set_xlim(0, args.budget)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_vs_p4_clean.png")
    plt.close(fig)
    print(f"[aggregate] wrote {out_dir / 'baseline_vs_p4_clean.png'}")

    # ------------------------------------------------------------------ #
    # Serialise raw arrays + per-run + per-method stats                  #
    # ------------------------------------------------------------------ #
    with open(out_dir / "aggregate_data.json", "w") as f:
        json.dump(aggregate_payload, f, indent=2, default=str)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "method_stats": method_stats,
            "per_run_per_method": summary_rows,
            "target_sr": args.target_sr,
            "budget": int(args.budget),
        }, f, indent=2, default=str)
    print(f"[aggregate] wrote {out_dir / 'aggregate_data.json'}")
    print(f"[aggregate] wrote {out_dir / 'summary.json'}")
    print(json.dumps(method_stats, indent=2))


if __name__ == "__main__":
    main()
