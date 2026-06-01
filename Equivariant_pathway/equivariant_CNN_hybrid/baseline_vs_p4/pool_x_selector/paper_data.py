"""Data loaders + figure helpers for paper_figures.ipynb.

Each function makes one thing easy: a tidy iteration over an on-disk
artifact, a derived feature, or a reusable render. Notebook cells stay
short and the data-source paths are explicit at each call site.

Reused upstream:
  - ``Equivariant_pathway.expert.build_grid`` + ``compute_distance_map``
    for the BFS optimal-path-length toughness feature.
  - ``Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.nb_helpers._layout_rgb``
    for matplotlib-friendly 5x5 maze rendering (start/goal/fires colored).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Heavy upstream imports are DEFERRED into the functions that actually use
# them. This module imports fast (matplotlib + numpy + pandas + yaml only);
# `expert` only loads when compute_toughness() is first called, and
# `nb_helpers._layout_rgb` only loads when render_gallery() is first called.
# Python's module cache handles subsequent calls. Crucially, importlib.reload
# on this module no longer re-traverses the upstream chain unless those
# functions are re-invoked.

SUITE_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = SUITE_ROOT / "results"
PAPER_FIG_ROOT = RESULTS_ROOT / "aggregate" / "figures" / "paper"

# The P4-LLM-vs-IIL-baselines comparison set (rotated-only). p4_top3_rotate
# keeps its on-disk dir name and is relabeled "P4-LLM" in figures (see nb_plot).
METHODS: Tuple[str, ...] = (
    "safe_dagger", "dropout_dagger", "ensemble_dagger",
    "thrifty_dagger", "stagger",
    "p4_top3_rotate",
)
# Only P4-LLM produces compression_log.csv / recommended_layouts.json; the 5
# IIL baselines do not (those P4-only figures stay P4-only).
P4_METHODS: Tuple[str, ...] = ("p4_top3_rotate",)
BASELINE_METHODS: Tuple[str, ...] = tuple(m for m in METHODS if m not in P4_METHODS)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
def available_runs() -> List[int]:
    if not RESULTS_ROOT.exists():
        return []
    out: List[int] = []
    for d in sorted(RESULTS_ROOT.glob("run_*")):
        if not d.is_dir():
            continue
        try:
            out.append(int(d.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return out


def _runs(runs: Optional[Sequence[int]]) -> List[int]:
    return list(runs) if runs is not None else available_runs()


# ---------------------------------------------------------------------------
# Per-round iterators (uniform schemas)
# ---------------------------------------------------------------------------
def iter_prescribed_losses(method: str,
                            runs: Optional[Sequence[int]] = None) -> Iterator[Dict]:
    """Yields one row per picked/prescribed demo:
       {run, round, method, layout, policy_loss, policy_loss_sum,
        n_steps, ep_seed, success}.
    Source: ``results/run_*/<method>/results/round_*/prescribed_loss.json``
    """
    for rid in _runs(runs):
        method_root = RESULTS_ROOT / f"run_{rid}" / method / "results"
        for rd in sorted(method_root.glob("round_*")):
            p = rd / "prescribed_loss.json"
            if not p.exists():
                continue
            try:
                obj = json.load(open(p))
            except (OSError, json.JSONDecodeError):
                continue
            for pick in obj.get("picks", []) or []:
                yield {
                    "run": rid, "round": obj.get("round"), "method": method,
                    "layout": pick.get("layout"),
                    "policy_loss": float(pick.get("policy_loss", 0.0)),
                    "policy_loss_sum": float(pick.get("policy_loss_sum", 0.0)),
                    "n_steps": int(pick.get("n_steps", 0)),
                    "ep_seed": pick.get("ep_seed"),
                    "success": bool(pick.get("success", False)),
                }


def iter_p4_prescriptions(method: str,
                          runs: Optional[Sequence[int]] = None) -> Iterator[Dict]:
    """Yields each LLM-prescribed layout for a P4 method:
       {run, round, method, layout: {start_pos, goal_pos, fire_positions, steps, rationale, ...}}
    Source: ``results/run_*/<method>/results/round_*/p4_analysis/recommended_layouts.json``
    """
    for rid in _runs(runs):
        method_root = RESULTS_ROOT / f"run_{rid}" / method / "results"
        for rd in sorted(method_root.glob("round_*")):
            p = rd / "p4_analysis" / "recommended_layouts.json"
            if not p.exists():
                continue
            try:
                obj = json.load(open(p))
            except (OSError, json.JSONDecodeError):
                continue
            try:
                round_idx = int(rd.name.split("_")[-1])
            except ValueError:
                continue
            for layout in obj.get("layouts", []) or []:
                yield {"run": rid, "round": round_idx, "method": method,
                       "layout": layout}


def iter_correction_pool_layouts(runs: Optional[Sequence[int]] = None) -> Iterator[Dict]:
    """Walks every shared correction-pool yaml in the suite. Yields:
       {run, pool_kind: 'fixed'|'rotate', round: Optional[int], layouts: List}.
    Sources:
      - ``results/run_*/shared/fixed_pool/correction_layouts.yaml`` (fixed)
      - ``results/run_*/shared/round_*/correction_layouts.yaml``   (rotate)
    The same yaml is shared across all methods that use that pool, so this
    iter is method-agnostic.
    """
    for rid in _runs(runs):
        shared = RESULTS_ROOT / f"run_{rid}" / "shared"
        fp = shared / "fixed_pool" / "correction_layouts.yaml"
        if fp.exists():
            yield {"run": rid, "pool_kind": "fixed", "round": None,
                   "layouts": _read_layouts_yaml(fp)}
        for rd in sorted(shared.glob("round_*")):
            pp = rd / "correction_layouts.yaml"
            if not pp.exists():
                continue
            try:
                round_idx = int(rd.name.split("_")[-1])
            except ValueError:
                continue
            yield {"run": rid, "pool_kind": "rotate", "round": round_idx,
                   "layouts": _read_layouts_yaml(pp)}


def _read_layouts_yaml(path: Path) -> List[Dict]:
    try:
        obj = yaml.safe_load(open(path)) or {}
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(obj, dict):
        return []
    for k in ("layouts", "test_layouts", "training_layouts",
              "heldout_test_layouts"):
        v = obj.get(k)
        if isinstance(v, list):
            return v
    return []


# ---------------------------------------------------------------------------
# Toughness features (derived in notebook from raw layouts)
# ---------------------------------------------------------------------------
def _quadrant(pos) -> str:
    try:
        r, c = int(pos[0]), int(pos[1])
    except (TypeError, ValueError, IndexError):
        return "?"
    rb = "top" if r < 2 else ("mid" if r == 2 else "bot")
    cb = "left" if c < 2 else ("mid" if c == 2 else "right")
    return f"{rb}-{cb}"


def compute_toughness(layout: Dict) -> Dict:
    """Cheap per-layout features:
      n_fires, bfs_opt_len (BFS optimal path length around fires; -1 if
      unreachable), start_quadrant, goal_quadrant, manhattan, start_pos,
      goal_pos. Uses upstream ``expert.build_grid`` +
      ``compute_distance_map`` (imported lazily on first call).
    """
    # Deferred import: loaded once on first call, cached thereafter.
    from Equivariant_pathway.expert import build_grid, compute_distance_map

    start = tuple(layout.get("start_pos") or [0, 0])
    goal = tuple(layout.get("goal_pos") or [0, 0])
    fires = [tuple(f) for f in (layout.get("fire_positions") or [])]
    grid_field = layout.get("grid") or [[0] * 5] * 5
    gs = len(grid_field) if grid_field else 5
    bfs_len = -1
    try:
        grid = build_grid(gs, fire_positions=fires, goal_pos=goal)
        dist_map = compute_distance_map(grid, goal)
        val = float(dist_map[start[0], start[1]])
        if np.isfinite(val) and 0 <= val <= gs * gs:
            bfs_len = int(val)
    except Exception:
        bfs_len = -1
    return {
        "n_fires": len(fires),
        "bfs_opt_len": bfs_len,
        "start_quadrant": _quadrant(start),
        "goal_quadrant": _quadrant(goal),
        "manhattan": int(abs(start[0] - goal[0]) + abs(start[1] - goal[1])),
        "start_pos": list(start),
        "goal_pos": list(goal),
    }


# ---------------------------------------------------------------------------
# Tabular frames for tables / cross-method aggregates
# ---------------------------------------------------------------------------
def final_sr_dataframe(runs: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """One row per (run, method): final_sr, final_extras, extras_to_target,
    stopped_reason.
    Sources:
      - last entry of ``results/run_*/<method>/results/learning_curve.json``
      - ``results/run_*/run_summary.json`` for stopped_reason.
    """
    rows: List[Dict] = []
    for rid in _runs(runs):
        for method in METHODS:
            p = RESULTS_ROOT / f"run_{rid}" / method / "results" / "learning_curve.json"
            if not p.exists():
                continue
            try:
                obj = json.load(open(p))
            except (OSError, json.JSONDecodeError):
                continue
            history = obj.get("history") or []
            if not history:
                continue
            last = history[-1]
            target_sr = float(obj.get("target_sr") or 0.9)
            extras_to_target: Optional[int] = None
            for h in history:
                if (h.get("heldout_sr") or 0.0) >= target_sr:
                    extras_to_target = h.get("extra_demos")
                    break
            rows.append({
                "run": rid, "method": method,
                "final_sr": float(last.get("heldout_sr") or 0.0),
                "final_extras": int(last.get("extra_demos") or 0),
                "extras_to_target": extras_to_target,
                "stopped_reason": _stopped_reason(rid, method),
            })
    return pd.DataFrame(rows)


def _stopped_reason(run_id: int, method: str) -> Optional[str]:
    # P4 + the 8-method grid write run_summary.json; the 5 IIL baselines write
    # run_summary_baselines.json (separate, to avoid clobbering). Probe both.
    for fname in ("run_summary.json", "run_summary_baselines.json"):
        p = RESULTS_ROOT / f"run_{run_id}" / fname
        if not p.exists():
            continue
        try:
            obj = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            continue
        reason = ((obj.get("results") or {}).get(method) or {}).get("stopped_reason")
        if reason is not None:
            return reason
    return None


def compression_dataframe(method: Optional[str] = None,
                          runs: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Flatten compression_log.csv across runs (and optionally methods).
    Source: ``results/run_*/<P4-method>/results/compression_log.csv``
    Columns: run, method, round, n_failures_observed, n_layouts_prescribed,
    compression_ratio, n_demos_collected, n_corridor_infeasible,
    prescribed_loss_mean.
    """
    methods = (method,) if method is not None else P4_METHODS
    rows: List[Dict] = []
    for rid in _runs(runs):
        for m in methods:
            p = RESULTS_ROOT / f"run_{rid}" / m / "results" / "compression_log.csv"
            if not p.exists():
                continue
            try:
                with open(p, "r") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        loss_raw = r.get("prescribed_loss_mean")
                        loss_val: Optional[float] = None
                        if loss_raw not in (None, "", "None"):
                            try:
                                loss_val = float(loss_raw)
                            except (TypeError, ValueError):
                                loss_val = None
                        rows.append({
                            "run": rid, "method": m,
                            "round": int(r.get("round") or 0),
                            "n_failures_observed": int(r.get("n_failures_observed") or 0),
                            "n_layouts_prescribed": int(r.get("n_layouts_prescribed") or 0),
                            "compression_ratio": float(r.get("compression_ratio") or 0.0),
                            "n_demos_collected": int(r.get("n_demos_collected") or 0),
                            "n_corridor_infeasible": int(r.get("n_corridor_infeasible") or 0),
                            "prescribed_loss_mean": loss_val,
                        })
            except (OSError, ValueError):
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Corridor parsing + gallery rendering
# ---------------------------------------------------------------------------
_STEPS_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_corridor_steps(steps_str: Optional[str]) -> List[Tuple[int, int]]:
    """Parse '(r,c)->(r,c)->...' into a list of (r, c) tuples."""
    if not steps_str:
        return []
    return [(int(m.group(1)), int(m.group(2)))
            for m in _STEPS_RE.finditer(steps_str)]


def render_gallery(items: Sequence[Dict], ncols: int = 4,
                   suptitle: Optional[str] = None,
                   panel_size: float = 2.4):
    """Grid of maze panels. Each ``item``:
        {"layout": Dict, "title": str (optional),
         "corridor_steps": str (optional; otherwise read layout['steps'])}
    Renders start/goal/fires (via ``_layout_rgb``) + an orange polyline for
    the prescribed corridor when present. Returns the matplotlib Figure.
    The upstream ``_layout_rgb`` is imported lazily on first call.
    """
    # Deferred import: loaded once on first call, cached thereafter.
    from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.nb_helpers import (
        _layout_rgb,
    )

    n = len(items)
    if n == 0:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.text(0.5, 0.5, "no items", ha="center", va="center")
        ax.set_axis_off()
        return fig
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * panel_size, nrows * panel_size),
                             squeeze=False)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if k >= n:
            ax.axis("off"); continue
        it = items[k]
        lay = it.get("layout") or {}
        rgb = _layout_rgb(lay)
        g = rgb.shape[0]
        ax.imshow(rgb, interpolation="nearest", origin="upper",
                  extent=[-0.5, g - 0.5, g - 0.5, -0.5])
        pts = parse_corridor_steps(it.get("corridor_steps") or lay.get("steps") or "")
        if len(pts) >= 2:
            xs = [c for r, c in pts]; ys = [r for r, c in pts]
            ax.plot(xs, ys, "-", lw=2.0, color="#ff9100", alpha=0.9)
            ax.plot(xs, ys, "o", ms=3.5, color="#ff9100", alpha=0.9)
        for i in range(1, g):
            ax.axhline(i - 0.5, color="#cccccc", lw=0.4)
            ax.axvline(i - 0.5, color="#cccccc", lw=0.4)
        title = it.get("title") or ""
        if title:
            ax.set_title(title, fontsize=7, pad=2)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96 if suptitle else 1.0])
    return fig


# ---------------------------------------------------------------------------
# Output paths (consistent location for paper artefacts)
# ---------------------------------------------------------------------------
def ensure_paper_dir() -> Path:
    PAPER_FIG_ROOT.mkdir(parents=True, exist_ok=True)
    return PAPER_FIG_ROOT


def save_figure(fig, name: str, dpi: int = 150) -> Path:
    out = ensure_paper_dir() / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Tabulate-free markdown table formatter (the `tabulate` package isn't
    installed in the maze env, so pandas' df.to_markdown() won't work)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}" if np.isfinite(v) else "")
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str, caption: str = "") -> Tuple[Path, Path]:
    """Write df as markdown (.md) + LaTeX tabular (.tex). Returns both paths."""
    out_md = ensure_paper_dir() / f"{name}.md"
    out_tex = ensure_paper_dir() / f"{name}.tex"
    with open(out_md, "w") as f:
        if caption:
            f.write(f"**{caption}**\n\n")
        f.write(_df_to_markdown(df))
    with open(out_tex, "w") as f:
        if caption:
            f.write(f"% {caption}\n")
        f.write(df.to_latex(index=False, escape=False))
    return out_md, out_tex
