"""Cross-run aggregation, summary, and figures for the suite.

Order of operations:

  1. Cross-run contamination check (write
     ``results/aggregate/contamination_report.json`` and print OK / VIOLATION).
  2. For each method, load every run's ``learning_curve.json`` and
     ``training_log.csv``; build mean±std curves on two x-axes:
        - extra demos (legacy)
        - training rounds (new)
  3. For the P4 methods, load every run's ``compression_log.csv`` and
     compute compression-ratio stats (mean/std/min/max/n_infeasible).
  4. Write ``summary.json`` + ``compression_summary.csv`` + three figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ..layouts.contamination import (
    cross_run_check, print_summary as print_contam_summary, write_report,
)
from ..layouts.layout_setup import RESULTS_ROOT
from ..orchestrator.workspace import aggregate_dir, all_run_ids

# Reuse upstream's curve helpers (read-only import).
from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import (  # noqa: E402
    aggregate_chart as _upstream_chart,
)


# The comparison set: P4-LLM (the already-run p4_top3_rotate, relabeled) vs the
# 5 rotated-only IIL baselines. The old fixed/vague-rotate methods are dropped
# from the new comparison (their results stay on disk; restore a key here to
# re-include one). Dir name == key; p4_top3_rotate keeps its on-disk dir name.
METHOD_DIRS = {
    "safe_dagger": "safe_dagger",
    "dropout_dagger": "dropout_dagger",
    "ensemble_dagger": "ensemble_dagger",
    "thrifty_dagger": "thrifty_dagger",
    "stagger": "stagger",
    "p4_top3_rotate": "p4_top3_rotate",
}

# Methods with compression_log.csv + LLM prescriptions (only P4-LLM). The 5 IIL
# baselines produce no compression log, so they're correctly excluded.
P4_METHODS = ("p4_top3_rotate",)

METHOD_COLORS = {
    "safe_dagger": "tab:blue",
    "dropout_dagger": "tab:orange",
    "ensemble_dagger": "tab:green",
    "thrifty_dagger": "tab:purple",
    "stagger": "tab:gray",
    "p4_top3_rotate": "tab:red",
}

METHOD_LABELS = {
    "safe_dagger": "SafeDAgger*",
    "dropout_dagger": "DropoutDAgger",
    "ensemble_dagger": "EnsembleDAgger",
    "thrifty_dagger": "ThriftyDAgger",
    "stagger": "Stagger",
    "p4_top3_rotate": "P4-LLM",
}

STYLE_PATH = REPO_ROOT / "Equivariant_pathway" / "equivariant_CNN_hybrid" / "baseline_vs_p4" / "paper.mplstyle"


def _use_style() -> None:
    try:
        if STYLE_PATH.exists():
            plt.style.use(str(STYLE_PATH))
    except Exception:
        pass


def _gather_demos_axis(
    results_root: Path, method_dir: str, budget: int,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Stack per-run heldout_sr onto a [0..budget] x-grid."""
    x_grid = np.arange(0, budget + 1, dtype=int)
    per_run: List[Dict] = []
    rows: List[np.ndarray] = []
    for run_id in all_run_ids():
        curve_path = results_root / f"run_{run_id}" / method_dir / "results" / "learning_curve.json"
        data = _upstream_chart._load_curve(curve_path)
        if data is None:
            continue
        history = data.get("history", []) or []
        if not history:
            continue
        xs, ys = _upstream_chart._curve_xy(history)
        y = _upstream_chart._sample_hold(xs, ys, x_grid)
        per_run.append({
            "run_id": int(run_id),
            "final_extra": int(xs[-1]) if xs else 0,
            "final_sr": float(ys[-1]) if ys else None,
            "stopped_reason": (history[-1].get("stopped_reason") if history else None),
        })
        rows.append(y)
    Y = np.vstack(rows) if rows else np.zeros((0, len(x_grid)), dtype=float)
    return per_run, x_grid, Y


def _gather_training_axis(
    results_root: Path, method_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack per-run heldout_sr onto a [0..max_training_rounds] x-grid
    derived from each run's training_log.csv.
    """
    per_run_curves: List[Tuple[List[int], List[float]]] = []
    max_tr = 0
    for run_id in all_run_ids():
        log_path = results_root / f"run_{run_id}" / method_dir / "results" / "training_log.csv"
        if not log_path.exists():
            continue
        xs_tr: List[int] = []
        ys_sr: List[float] = []
        try:
            with open(log_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    xs_tr.append(int(row["training_rounds_total"]))
                    ys_sr.append(float(row["heldout_sr"]))
        except (OSError, KeyError, ValueError):
            continue
        if xs_tr:
            per_run_curves.append((xs_tr, ys_sr))
            max_tr = max(max_tr, xs_tr[-1])

    if not per_run_curves:
        return np.arange(0, 1), np.zeros((0, 1), dtype=float)

    grid = np.arange(0, max_tr + 1, dtype=int)
    rows = [_upstream_chart._sample_hold(xs, ys, grid) for xs, ys in per_run_curves]
    return grid, np.vstack(rows)


def _plot_overlay(
    methods: List[str], x_grid: np.ndarray, Ys: Dict[str, np.ndarray],
    title: str, xlabel: str, target_sr: Optional[float], out_path: Path,
) -> None:
    _use_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in methods:
        Y = Ys.get(m)
        if Y is None or Y.size == 0:
            continue
        color = METHOD_COLORS.get(m, None)
        mean = Y.mean(0)
        std = Y.std(0)
        ax.plot(
            x_grid, mean, marker="o", markersize=4, linewidth=1.8,
            color=color,
            label=f"{METHOD_LABELS.get(m, m)} (n={Y.shape[0]})",
        )
        ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    if target_sr is not None:
        ax.axhline(target_sr, ls="--", color="tab:red", alpha=0.7, label=f"target = {target_sr:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("heldout success rate")
    if x_grid.size:
        ax.set_xlim(int(x_grid[0]), int(x_grid[-1]))
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _gather_compression(results_root: Path, method_dir: str) -> Dict:
    """Pool all per-round compression ratios for one P4 method across runs."""
    ratios: List[float] = []
    n_infeasible_total = 0
    n_rows = 0
    for run_id in all_run_ids():
        log_path = results_root / f"run_{run_id}" / method_dir / "results" / "compression_log.csv"
        if not log_path.exists():
            continue
        try:
            with open(log_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    n_rows += 1
                    r = row.get("compression_ratio")
                    if r is not None:
                        try:
                            ratios.append(float(r))
                        except ValueError:
                            pass
                    n_infeasible_total += int(row.get("n_corridor_infeasible") or 0)
        except OSError:
            continue
    if not ratios:
        return {
            "n_rows": n_rows,
            "mean": None, "std": None, "min": None, "max": None,
            "n_corridor_infeasible_total": n_infeasible_total,
        }
    arr = np.asarray(ratios, dtype=float)
    return {
        "n_rows": n_rows,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_corridor_infeasible_total": n_infeasible_total,
    }


def _plot_compression(
    compression: Dict[str, Dict], out_path: Path,
) -> None:
    _use_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars_x = []
    bars_y = []
    bars_err = []
    bars_color = []
    for m in P4_METHODS:
        stats = compression.get(m) or {}
        if stats.get("mean") is None:
            continue
        bars_x.append(METHOD_LABELS.get(m, m))
        bars_y.append(stats["mean"])
        bars_err.append(stats.get("std") or 0.0)
        bars_color.append(METHOD_COLORS.get(m, "gray"))
    if not bars_x:
        # No compression data — emit an empty placeholder so callers can
        # see the figure was attempted.
        ax.text(0.5, 0.5, "no compression data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
    else:
        ax.bar(bars_x, bars_y, yerr=bars_err, color=bars_color, alpha=0.7, capsize=4)
        ax.set_ylabel("compression ratio  (n_failures / n_layouts_prescribed)")
        ax.set_title("Compression ratio (mean ± std across all rounds, all runs)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--target_sr", type=float, default=0.90)
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = aggregate_dir()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Cross-run contamination check ------------------------------------
    contam_report = cross_run_check(results_root=results_root)
    write_report(contam_report, out_dir / "contamination_report.json")
    print_contam_summary(contam_report)

    # ---- Per-method curves on both x-axes ---------------------------------
    summary: Dict = {
        "results_root": str(results_root),
        "n_runs": len(list(all_run_ids())),
        "budget": args.budget,
        "target_sr": args.target_sr,
        "methods": {},
    }
    Ys_demos: Dict[str, np.ndarray] = {}
    Ys_tr: Dict[str, np.ndarray] = {}
    x_grid_demos = np.arange(0, args.budget + 1, dtype=int)

    methods_with_data: List[str] = []
    for m, mdir in METHOD_DIRS.items():
        per_run, x_grid, Y = _gather_demos_axis(results_root, mdir, args.budget)
        Ys_demos[m] = Y
        x_tr_grid, Y_tr = _gather_training_axis(results_root, mdir)
        Ys_tr[m] = Y_tr
        if Y.shape[0] > 0:
            methods_with_data.append(m)
        summary["methods"][m] = {
            "n_runs_with_curve": int(Y.shape[0]),
            "final_sr_mean": float(Y[:, -1].mean()) if Y.size else None,
            "final_sr_std": float(Y[:, -1].std()) if Y.size else None,
            "per_run": per_run,
        }

    # Pick the longest training-axis grid as the common one.
    max_len = max((Ys_tr[m].shape[1] for m in METHOD_DIRS if Ys_tr.get(m) is not None and Ys_tr[m].size), default=1)
    x_grid_tr = np.arange(0, max_len, dtype=int)
    # Pad each method's training-axis curve to the common length with its final value.
    for m in list(Ys_tr.keys()):
        Y = Ys_tr[m]
        if Y.size == 0:
            continue
        if Y.shape[1] < max_len:
            pad = np.repeat(Y[:, -1:], max_len - Y.shape[1], axis=1)
            Ys_tr[m] = np.concatenate([Y, pad], axis=1)

    _plot_overlay(
        methods=methods_with_data, x_grid=x_grid_demos, Ys=Ys_demos,
        title="Held-out SR vs extra demonstrations added",
        xlabel="extra demonstrations (on top of initial set)",
        target_sr=args.target_sr, out_path=fig_dir / "sr_vs_demos_added.png",
    )
    _plot_overlay(
        methods=methods_with_data, x_grid=x_grid_tr, Ys=Ys_tr,
        title="Held-out SR vs training rounds (fine-tune invocations)",
        xlabel="cumulative training rounds (fine-tune calls)",
        target_sr=args.target_sr, out_path=fig_dir / "sr_vs_training_rounds.png",
    )

    # ---- Compression summary ----------------------------------------------
    compression: Dict[str, Dict] = {}
    for m in P4_METHODS:
        compression[m] = _gather_compression(results_root, METHOD_DIRS[m])
    with open(out_dir / "compression_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_rows", "mean", "std", "min", "max", "n_corridor_infeasible_total"])
        for m, stats in compression.items():
            w.writerow([
                m, stats.get("n_rows", 0),
                stats.get("mean"), stats.get("std"),
                stats.get("min"), stats.get("max"),
                stats.get("n_corridor_infeasible_total", 0),
            ])
    summary["compression"] = compression
    _plot_compression(compression, fig_dir / "compression_dist.png")

    # ---- summary.json -----------------------------------------------------
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[aggregate] wrote {out_dir}/summary.json")
    print(f"[aggregate] wrote {fig_dir}/sr_vs_demos_added.png")
    print(f"[aggregate] wrote {fig_dir}/sr_vs_training_rounds.png")
    print(f"[aggregate] wrote {fig_dir}/compression_dist.png")


if __name__ == "__main__":
    main()
