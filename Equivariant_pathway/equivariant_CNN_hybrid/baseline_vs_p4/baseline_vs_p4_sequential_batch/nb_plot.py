"""Plot helpers for inspect_results.ipynb.

Self-contained loading + plotting of the baseline_vs_p4_sequential_batch
sweep results. The notebook imports this module and calls one function per
cell, so each cell is independent (no shared setup cell required).

Data source: results/run_<id>/<method>/results/learning_curve.json, the
same files aggregate.py reads. Each history row has:
    round, extra_demos (demos added on top of the initial set), heldout_sr,
    n_new_demos, correction_sr, ...

Methods: baseline, p4_sequential, p4_batch.

Typical use inside a notebook cell::

    import sys, pathlib, importlib
    sys.path.insert(0, str(pathlib.Path.cwd()))
    import nb_plot; importlib.reload(nb_plot)
    nb_plot.single_run(2)                    # one run, all methods
    nb_plot.compare("baseline", "p4_sequential")   # aggregate mean +/- std
    nb_plot.compare_all()                    # all three
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Resolve results/ relative to THIS file so the notebook works regardless
# of the kernel's working directory.
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

METHODS: Tuple[str, ...] = ("baseline", "p4_sequential")
LABELS: Dict[str, str] = {
    "baseline": "baseline (DAgger)",
    "p4_sequential": "P4 sequential",
    "p4_batch": "P4 batch",
}
COLORS: Dict[str, str] = {
    "baseline": "tab:blue",
    "p4_sequential": "tab:orange",
    "p4_batch": "tab:green",
}
DEFAULT_BUDGET = 15
DEFAULT_TARGET = 0.90


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def available_runs() -> List[int]:
    """Run ids that have at least one method curve on disk, sorted."""
    ids = []
    for d in RESULTS_ROOT.glob("run_*"):
        if not d.is_dir():
            continue
        try:
            rid = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if any((d / m / "results" / "learning_curve.json").exists() for m in METHODS):
            ids.append(rid)
    return sorted(ids)


def _curve_path(run_id: int, method: str) -> Path:
    return RESULTS_ROOT / f"run_{run_id}" / method / "results" / "learning_curve.json"


def load_curve(run_id: int, method: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (extra_demos, heldout_sr) arrays for one run+method, or None.

    Rows are sorted by extra_demos and de-duplicated keeping the last value
    at each demo count (a round that added 0 demos but re-evaluated heldout
    overwrites the prior point at the same x).
    """
    p = _curve_path(run_id, method)
    if not p.exists():
        return None
    try:
        hist = (json.loads(p.read_text()).get("history") or [])
    except Exception:
        return None
    pts: Dict[int, float] = {}
    for h in hist:
        sr = h.get("heldout_sr")
        if sr is None:
            continue
        x = h.get("extra_demos")
        if x is None:
            x = int(h.get("cum_demos", 0)) - 20  # fallback: cum minus initial
        pts[int(x)] = float(sr)
    if not pts:
        return None
    xs = np.array(sorted(pts), dtype=float)
    ys = np.array([pts[int(x)] for x in xs], dtype=float)
    return xs, ys


def _sample_hold(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Step-sample a curve onto grid: value at g = y of the largest x <= g."""
    out = np.empty(len(grid), dtype=float)
    for i, g in enumerate(grid):
        mask = xs <= g
        out[i] = ys[mask][-1] if mask.any() else ys[0]
    return out


def final_sr_table(runs: Optional[Sequence[int]] = None):
    """Return a list of dicts: per-run final heldout SR for each method."""
    runs = list(runs) if runs is not None else available_runs()
    rows = []
    for rid in runs:
        row = {"run": rid}
        for m in METHODS:
            c = load_curve(rid, m)
            row[m] = round(float(c[1][-1]), 3) if c else None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _style(ax, target: float, title: str, xlabel: str):
    ax.axhline(target, ls="--", lw=1, color="grey", alpha=0.7,
               label=f"target {target:g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("held-out success rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")


def single_run(run_id: int, methods: Sequence[str] = METHODS,
               target: float = DEFAULT_TARGET, figsize=(7, 4.5)):
    """Plot held-out SR vs extra demos for ONE run, one line per method."""
    fig, ax = plt.subplots(figsize=figsize)
    plotted = False
    for m in methods:
        c = load_curve(run_id, m)
        if c is None:
            continue
        xs, ys = c
        ax.plot(xs, ys, marker="o", ms=4, color=COLORS[m], label=LABELS[m])
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, f"no curves for run {run_id}", ha="center", va="center")
    _style(ax, target, f"Run {run_id}: held-out SR vs extra demonstrations",
           "extra demonstrations (on top of initial 20)")
    fig.tight_layout()
    plt.show()
    return fig


def all_runs(method: str, target: float = DEFAULT_TARGET,
             runs: Optional[Sequence[int]] = None, figsize=(7, 4.5)):
    """Overlay every run's curve for ONE method (thin lines + mean band)."""
    runs = list(runs) if runs is not None else available_runs()
    budget = DEFAULT_BUDGET
    grid = np.arange(0, budget + 1, dtype=int)
    fig, ax = plt.subplots(figsize=figsize)
    rows = []
    for rid in runs:
        c = load_curve(rid, method)
        if c is None:
            continue
        xs, ys = c
        ax.plot(xs, ys, lw=1, alpha=0.35, color=COLORS[method])
        rows.append(_sample_hold(xs, ys, grid))
    if rows:
        Y = np.vstack(rows)
        mean = Y.mean(0)
        ax.plot(grid, mean, lw=2.5, color=COLORS[method],
                label=f"{LABELS[method]} mean (n={Y.shape[0]})")
    _style(ax, target, f"{LABELS[method]}: all runs",
           "extra demonstrations (on top of initial 20)")
    fig.tight_layout()
    plt.show()
    return fig


def compare(*methods: str, target: float = DEFAULT_TARGET,
            runs: Optional[Sequence[int]] = None, show_runs: bool = True,
            figsize=(7.5, 5)):
    """Aggregate mean +/- std across runs for the given methods.

    Pass any subset of method names, e.g.::

        compare("baseline", "p4_sequential")
        compare("p4_sequential", "p4_batch")
        compare("baseline", "p4_batch")
        compare("baseline", "p4_sequential", "p4_batch")
    """
    methods = methods or METHODS
    runs = list(runs) if runs is not None else available_runs()
    budget = DEFAULT_BUDGET
    grid = np.arange(0, budget + 1, dtype=int)
    fig, ax = plt.subplots(figsize=figsize)
    n_used = 0
    for m in methods:
        rows = []
        for rid in runs:
            c = load_curve(rid, m)
            if c is None:
                continue
            rows.append(_sample_hold(c[0], c[1], grid))
        if not rows:
            continue
        Y = np.vstack(rows)
        mean, std = Y.mean(0), Y.std(0)
        if show_runs:
            for r in Y:
                ax.plot(grid, r, lw=0.8, alpha=0.18, color=COLORS[m])
        ax.plot(grid, mean, lw=2.5, color=COLORS[m],
                label=f"{LABELS[m]} (n={Y.shape[0]})")
        ax.fill_between(grid, mean - std, mean + std, color=COLORS[m], alpha=0.15)
        n_used = max(n_used, Y.shape[0])
    title = "  vs  ".join(LABELS[m] for m in methods)
    _style(ax, target, f"{title}\nmean ± std over {n_used} runs",
           "extra demonstrations (on top of initial 20)")
    fig.tight_layout()
    plt.show()
    return fig


# Convenience wrappers for the exact comparisons requested.
def baseline_vs_seq(**kw):
    return compare("baseline", "p4_sequential", **kw)


def seq_vs_batch(**kw):
    return compare("p4_sequential", "p4_batch", **kw)


def baseline_vs_batch(**kw):
    return compare("baseline", "p4_batch", **kw)


def compare_all(**kw):
    return compare("baseline", "p4_sequential", "p4_batch", **kw)
