"""Thin helpers for ``baseline_vs_p4_notebook.ipynb``.

Keeps the notebook itself short: correction-pool rendering, per-layout
success-map derivation, the single-run learning-curve plot, and the
self-contained ``notebook_runs.json`` accumulator used by the REPLOT cell.

Nothing here re-implements experiment logic — the budget loops, layout
sampling and bootstrap all stay in ``baseline_budget`` / ``p4_budget`` /
``layout_setup`` / ``run_one``. We only *read* their on-disk artifacts and
draw. The plot/curve helpers reuse ``aggregate_chart`` so the notebook and
``swatch.sh``'s aggregate chart never diverge.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.baseline_budget import (
    _load_correction_layouts,
)
from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import aggregate_chart

STYLE_PATH = Path(__file__).resolve().parent / "paper.mplstyle"

# Cell colours for the correction-pool thumbnails.
_C_FREE = (1.00, 1.00, 1.00)
_C_WALL = (0.25, 0.25, 0.25)
_C_FIRE = (0.85, 0.15, 0.15)
_C_START = (0.15, 0.35, 0.85)
_C_GOAL = (0.15, 0.65, 0.25)

# Per-thumbnail tint / border when a success map is supplied.
_TINT_OK = (0.90, 0.97, 0.90)
_TINT_BAD = (0.99, 0.91, 0.91)
_TINT_NA = (0.96, 0.96, 0.96)
_BORDER_OK = (0.15, 0.60, 0.20)
_BORDER_BAD = (0.80, 0.15, 0.15)
_BORDER_NA = (0.70, 0.70, 0.70)


# --------------------------------------------------------------------------- #
# Correction pool loading + rendering
# --------------------------------------------------------------------------- #
def load_pool(correction_yaml) -> List[Dict]:
    """Layout list for a run's correction pool (wraps the budget loader)."""
    return _load_correction_layouts(Path(correction_yaml))


def count_layouts(yaml_path) -> int:
    """Number of layouts in a layout YAML (training / heldout / correction)."""
    return len(_load_correction_layouts(Path(yaml_path)))


# --------------------------------------------------------------------------- #
# Notebook-owned bootstrap (a freshly trained initial model, isolated from the
# Slurm `shared/init_*` cache so the notebook's config actually takes effect)
# --------------------------------------------------------------------------- #
def notebook_bootstrap_dir(runs_nb_dir, initial_demos: int,
                           initial_epochs: int, seed: int) -> Path:
    """Config-keyed bootstrap dir under ``runs_nb/_bootstrap/``.

    The (demos, epochs, seed) signature is in the path, so changing any of
    them yields a *new* dir and forces a fresh train instead of silently
    reusing a stale checkpoint.
    """
    return (Path(runs_nb_dir) / "_bootstrap" /
            f"demos{int(initial_demos)}_ep{int(initial_epochs)}_seed{int(seed)}")


def seed_run_from_bootstrap(boot_dir, run_shared_dir, initial_demos: int):
    """Copy the notebook's freshly-trained initial demos + checkpoint into a
    run's ``shared/`` dir (mirrors run_one.py:104-112, but from the notebook
    bootstrap — never the Slurm ``shared/init_*``).

    Returns ``(shared_demo_dir, shared_ckpt_dir)`` for the run. Raises a
    clear error if SETUP has not produced the bootstrap yet.
    """
    boot_dir = Path(boot_dir)
    boot_demos = boot_dir / "init_demos"
    boot_ckpts = boot_dir / "init_checkpoints"
    n_demos = len(list(boot_demos.glob("*.json"))) if boot_demos.exists() else 0
    best = boot_ckpts / "best_hybrid_policy.pth"
    if n_demos < int(initial_demos) or not best.exists():
        raise RuntimeError(
            f"notebook bootstrap missing/incomplete at {boot_dir} "
            f"(found {n_demos} demos, best_hybrid_policy.pth="
            f"{best.exists()}). Run the SETUP cell first — it trains the "
            f"notebook's own initial model.")

    run_shared_dir = Path(run_shared_dir)
    shared_demo_dir = run_shared_dir / "init_demos"
    shared_ckpt_dir = run_shared_dir / "init_checkpoints"
    # RESET both dirs first. Demo filenames are timestamp-unique, so a plain
    # copy-if-absent into a pre-existing dir (e.g. left by an earlier run, or
    # by the old prefer_global code that copied the Slurm shared/init_demos)
    # silently ACCUMULATES stale demos -> the run trains on more demos than
    # the budget accounts for. Clearing guarantees the run sees EXACTLY the
    # notebook bootstrap's initial set and nothing else.
    shutil.rmtree(shared_demo_dir, ignore_errors=True)
    shutil.rmtree(shared_ckpt_dir, ignore_errors=True)
    shared_demo_dir.mkdir(parents=True, exist_ok=True)
    shared_ckpt_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(boot_demos.glob("*.json")):
        shutil.copy2(src, shared_demo_dir / src.name)
    for name in ("best_hybrid_policy.pth", "last_hybrid_policy.pth"):
        src = boot_ckpts / name
        if src.exists():
            shutil.copy2(src, shared_ckpt_dir / name)
    n_seeded = len(list(shared_demo_dir.glob("*.json")))
    if n_seeded != n_demos:
        raise RuntimeError(
            f"seed mismatch: bootstrap has {n_demos} demos but "
            f"{n_seeded} landed in {shared_demo_dir}")
    print(f"[nb] seeded run from bootstrap: {n_seeded} init demos + "
          f"checkpoint -> {run_shared_dir}")
    return shared_demo_dir, shared_ckpt_dir


def _coerce_rc(p):
    try:
        return int(p[0]), int(p[1])
    except (TypeError, ValueError, IndexError):
        return None


def _layout_rgb(layout: Dict) -> np.ndarray:
    grid = np.asarray(layout.get("grid") or [[0] * 5] * 5)
    g = grid.shape[0] if grid.ndim == 2 and grid.size else 5
    img = np.ones((g, g, 3), dtype=float)
    for r in range(g):
        for c in range(g):
            if grid.ndim == 2 and int(grid[r, c]) == 1:
                img[r, c] = _C_WALL
    for fp in layout.get("fire_positions") or []:
        rc = _coerce_rc(fp)
        if rc and 0 <= rc[0] < g and 0 <= rc[1] < g:
            img[rc[0], rc[1]] = _C_FIRE
    s = _coerce_rc(layout.get("start_pos"))
    if s and 0 <= s[0] < g and 0 <= s[1] < g:
        img[s[0], s[1]] = _C_START
    go = _coerce_rc(layout.get("goal_pos"))
    if go and 0 <= go[0] < g and 0 <= go[1] < g:
        img[go[0], go[1]] = _C_GOAL
    return img


def render_correction_pool(
    layouts: List[Dict],
    success_map: Optional[Dict[str, bool]] = None,
    out_png=None,
    title: Optional[str] = None,
    ncols: int = 8,
    cell_px: int = 40,
):
    """Grid of 5x5 correction-pool thumbnails.

    blue=start, green=goal, red=fire, grey=wall, white=free. When
    ``success_map`` is given each thumbnail is tinted/bordered green
    (solved), red (failed) or grey (unknown). Returns the Figure (not
    closed, so it renders inline); also writes ``out_png`` if given.
    """
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    n = len(layouts)
    ncols = max(1, min(ncols, n)) if n else 1
    nrows = max(1, math.ceil(n / ncols))
    scale = max(0.9, cell_px / 36.0)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 1.35 * scale, nrows * 1.5 * scale),
        squeeze=False,
    )
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if k >= n:
            ax.axis("off")
            continue
        lay = layouts[k]
        img = _layout_rgb(lay)
        g = img.shape[0]
        ax.imshow(img, interpolation="nearest", origin="upper",
                  extent=[-0.5, g - 0.5, g - 0.5, -0.5])
        for i in range(1, g):
            ax.axhline(i - 0.5, color="#cccccc", lw=0.4)
            ax.axvline(i - 0.5, color="#cccccc", lw=0.4)
        name = str(lay.get("name", f"#{k}"))
        ax.set_title(name, fontsize=6, pad=2)
        if success_map is not None:
            ok = success_map.get(name)
            tint = _TINT_OK if ok else (_TINT_NA if ok is None else _TINT_BAD)
            border = _BORDER_OK if ok else (_BORDER_NA if ok is None else _BORDER_BAD)
            ax.set_facecolor(tint)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(border)
                sp.set_linewidth(2.0)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97 if title else 1.0])
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
        print(f"[nb] wrote {out_png}")
    return fig


# --------------------------------------------------------------------------- #
# Per-layout success maps + per-round helpers
# --------------------------------------------------------------------------- #
_ROUND_RE = re.compile(r"^round_(\d+)$")


def latest_round_dir(results_dir) -> Optional[Path]:
    """Newest ``round_NNN`` dir (skips ``round_000*`` heldout-only dirs)."""
    results_dir = Path(results_dir)
    best_n, best = -1, None
    if not results_dir.exists():
        return None
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        m = _ROUND_RE.match(d.name)
        if not m:
            continue
        n = int(m.group(1))
        if n <= 0:
            continue
        if n > best_n:
            best_n, best = n, d
    return best


def p4_success_map(round_dir) -> Dict[str, bool]:
    """{layout_name: success} from a p4 round's correction rollout."""
    fo = Path(round_dir) / "correction_rollout" / "full_output.json"
    if not fo.exists():
        return {}
    data = json.load(open(fo))
    out: Dict[str, bool] = {}
    for r in (data.get("phase_a", {}) or {}).get("all_rollouts", []) or []:
        nm = r.get("maze_name")
        if nm is not None:
            out[str(nm)] = bool(r.get("success", False))
    return out


def baseline_success_map(round_dir, pool_names: List[str]) -> Dict[str, bool]:
    """{layout_name: success} for baseline: ranking.json lists *failures*
    only, so success = pool names not present in that file."""
    rk = Path(round_dir) / "ranking.json"
    failed: set = set()
    if rk.exists():
        for row in json.load(open(rk)) or []:
            nm = row.get("layout_name")
            if nm is not None:
                failed.add(str(nm))
    return {str(nm): (str(nm) not in failed) for nm in pool_names}


def read_learning_curve(results_dir) -> Optional[Dict]:
    p = Path(results_dir) / "learning_curve.json"
    return aggregate_chart._load_curve(p)


def _per_round_correction_sr(curve: Optional[Dict]) -> List[Dict]:
    rows = []
    for h in ((curve or {}).get("history") or []):
        if int(h.get("round", 0) or 0) <= 0:
            continue
        rows.append({"round": int(h.get("round")),
                     "correction_sr": h.get("correction_sr")})
    return rows


def correction_pool_performance(
    run_index: int, budget: int, target_sr: float,
    pool_layouts: List[Dict], bo_root, p4_root, methods: List[str],
) -> Dict:
    """Assemble the ``correction_pool_performance.json`` payload."""
    pool_names = [str(l.get("name")) for l in pool_layouts]
    out: Dict = {
        "run_index": int(run_index),
        "budget": int(budget),
        "target_sr": float(target_sr),
        "pool_names": pool_names,
    }
    if "baseline" in methods:
        rd = Path(bo_root) / "results"
        last = latest_round_dir(rd)
        out["baseline"] = {
            "per_round_correction_sr": _per_round_correction_sr(read_learning_curve(rd)),
            "final_round": int(_ROUND_RE.match(last.name).group(1)) if last else None,
            "final_success_map": baseline_success_map(last, pool_names) if last else {},
        }
    if "p4" in methods:
        rd = Path(p4_root) / "results"
        last = latest_round_dir(rd)
        out["p4"] = {
            "per_round_correction_sr": _per_round_correction_sr(read_learning_curve(rd)),
            "final_round": int(_ROUND_RE.match(last.name).group(1)) if last else None,
            "final_success_map": p4_success_map(last) if last else {},
        }
    return out


# --------------------------------------------------------------------------- #
# Single-run learning curve plot
# --------------------------------------------------------------------------- #
def plot_single_run(bo_curve: Optional[Dict], p4_curve: Optional[Dict],
                     target_sr: float, out_png=None, title: Optional[str] = None):
    """extra_demos vs heldout_sr for baseline and p4 on one axes."""
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, curve in (("baseline_only", bo_curve), ("p4_only", p4_curve)):
        if not curve:
            continue
        xs, ys = aggregate_chart._curve_xy(curve.get("history", []) or [])
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8,
                color=aggregate_chart.METHOD_COLORS[key],
                label=aggregate_chart.METHOD_LABELS[key])
    ax.axhline(target_sr, ls="--", color="tab:red", alpha=0.7,
               label=f"target = {target_sr:.2f}")
    ax.set_xlabel("extra demonstrations (on top of initial 20)")
    ax.set_ylabel("heldout success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "single run: baseline vs p4")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
        print(f"[nb] wrote {out_png}")
    return fig


# --------------------------------------------------------------------------- #
# Self-contained accumulating JSON (replot without re-running)
# --------------------------------------------------------------------------- #
def _method_block(curve: Optional[Dict]) -> Optional[Dict]:
    if not curve:
        return None
    history = curve.get("history", []) or []
    xs, ys = aggregate_chart._curve_xy(history)
    return {
        "xs": [int(x) for x in xs],
        "ys": [float(y) for y in ys],
        "final_extra": int(xs[-1]) if xs else 0,
        "final_sr": float(ys[-1]) if ys else None,
        "stopped_reason": (history[-1].get("stopped_reason") if history else None),
        "correction_sr": [h.get("correction_sr") for h in history],
    }


def load_notebook_json(json_path) -> Dict:
    p = Path(json_path)
    if not p.exists():
        return {"budget": None, "target_sr": None,
                "created": datetime.now().isoformat(timespec="seconds"),
                "runs": []}
    with open(p) as f:
        return json.load(f)


def append_run_to_notebook_json(json_path, run_index: int, budget: int,
                                target_sr: float, methods: List[str],
                                bo_root, p4_root) -> Dict:
    """Upsert this run's plottable data (dedupe by run_index, atomic write).

    The stored ``xs/ys`` are exactly what ``aggregate_chart._sample_hold``
    consumes, so the REPLOT cell can rebuild mean/std for any run count
    and any budget without touching ``runs_nb/``.
    """
    json_path = Path(json_path)
    doc = load_notebook_json(json_path)
    doc["budget"] = int(budget)
    doc["target_sr"] = float(target_sr)
    doc.setdefault("created", datetime.now().isoformat(timespec="seconds"))

    rec: Dict = {
        "run_index": int(run_index),
        "budget": int(budget),
        "target_sr": float(target_sr),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if "baseline" in methods:
        rec["baseline_only"] = _method_block(
            read_learning_curve(Path(bo_root) / "results"))
    if "p4" in methods:
        rec["p4_only"] = _method_block(
            read_learning_curve(Path(p4_root) / "results"))

    runs = [r for r in (doc.get("runs") or [])
            if int(r.get("run_index", -1)) != int(run_index)]
    runs.append(rec)
    runs.sort(key=lambda r: int(r.get("run_index", 0)))
    doc["runs"] = runs

    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    os.replace(tmp, json_path)
    print(f"[nb] notebook_runs.json now has {len(runs)} run(s) -> {json_path}")
    return doc
