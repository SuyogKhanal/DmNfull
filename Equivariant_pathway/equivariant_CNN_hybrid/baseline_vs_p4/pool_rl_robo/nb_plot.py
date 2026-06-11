"""Paper-grade plotting for the 7-method ManiSkill comparison.

LOCKSTEP method list (keep in sync with config.yaml::methods,
orchestrator/run_one.py::METHOD_SPEC, orchestrator/workspace.py::METHOD_DIR_NAMES,
aggregation/aggregate.py). Primary plot: held-out success rate vs cumulative
expert queries (the sample-efficiency yardstick).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .envs.env_setup import RESULTS_ROOT

METHODS = ["p4_top3", "diff_dagger", "safe_dagger", "dropout_dagger",
           "ensemble_dagger", "thrifty_dagger", "stagger", "p4_select"]

LABELS = {
    "p4_top3":         "P4-LLM (ours)",
    "diff_dagger":     "Diff-DAgger",
    "safe_dagger":     "SafeDAgger",
    "dropout_dagger":  "DropoutDAgger",
    "ensemble_dagger": "EnsembleDAgger",
    "thrifty_dagger":  "ThriftyDAgger",
    "stagger":         "Stagger",
    "p4_select":       "P4-LLM-select (on SafeDAgger)",
}

COLORS = {
    "p4_top3":         "tab:red",
    "diff_dagger":     "tab:blue",
    "safe_dagger":     "tab:cyan",
    "dropout_dagger":  "tab:orange",
    "ensemble_dagger": "tab:green",
    "thrifty_dagger":  "tab:purple",
    "stagger":         "tab:brown",
    "p4_select":       "tab:red",
}


def load_curve(env: str, method: str, run_id: int = 0) -> Optional[dict]:
    p = RESULTS_ROOT / str(env) / f"run_{run_id}" / method / "results" / "learning_curve.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _xy(curve: dict):
    pts = [(r.get("n_queries"), r.get("success_rate")) for r in curve.get("history", [])]
    pts = [(x, y) for x, y in pts if x is not None and y is not None]
    pts.sort(key=lambda t: t[0])
    return [x for x, _ in pts], [y for _, y in pts]


def compare(env: str, methods: Optional[List[str]] = None, run_id: int = 0,
            target_sr: float = 0.90, ax=None):
    """SR vs cumulative expert queries for each method on one env."""
    import matplotlib.pyplot as plt
    methods = methods or METHODS
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        c = load_curve(env, m, run_id)
        if not c:
            continue
        x, y = _xy(c)
        if x:
            ax.plot(x, y, "-o", label=LABELS.get(m, m), color=COLORS.get(m))
    ax.axhline(target_sr, ls="--", color="gray", lw=1, label=f"target {target_sr:.0%}")
    ax.set_xlabel("Cumulative expert queries")
    ax.set_ylabel("Held-out success rate")
    ax.set_title(f"Sample efficiency — {env}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    return ax
