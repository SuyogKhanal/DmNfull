"""Rank sweep configs by per-demo curve dominance over the fixed baselines.

For every config in the sweep, build P4's mean SR-vs-demos curve (over the
sweep runs) and compare it, at EVERY demo count, against the per-demo envelope
(max) of the 5 fixed production baselines on the SAME runs. A config "dominates"
iff P4 is >= the envelope at every demo count; configs are ranked by the
worst-case margin ``min_d (P4_mean(d) - envelope(d))`` — i.e. how far P4 stays
above ALL baselines across the whole curve (the user's "above all baselines at
5 demos, at 10, ..."). Reference rows for P4@500 and P4@90 are included.

Reads sweep P4 curves from ``<out_root>/<tag>/run_<R>/<dest_method>/results/`` and
the fixed baseline curves from the main ``results/`` tree. Writes
``<out_root>/leaderboard.{json,md}`` and ``<out_root>/leaderboard_top.png``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ..layouts.layout_setup import RESULTS_ROOT  # noqa: E402

BASELINES = ("safe_dagger", "dropout_dagger", "ensemble_dagger",
             "thrifty_dagger", "stagger")
BASELINE_LABEL = {"safe_dagger": "SafeDAgger*", "dropout_dagger": "DropoutDAgger",
                  "ensemble_dagger": "EnsembleDAgger", "thrifty_dagger": "ThriftyDAgger",
                  "stagger": "Stagger"}


def _load_pts(curve_path: Path) -> Optional[Dict[int, float]]:
    if not curve_path.exists():
        return None
    try:
        hist = (json.loads(curve_path.read_text()).get("history") or [])
    except (OSError, json.JSONDecodeError):
        return None
    pts: Dict[int, float] = {}
    for h in hist:
        sr = h.get("heldout_sr")
        if sr is None:
            continue
        x = h.get("extra_demos")
        if x is None:
            x = int(h.get("cum_demos", 0)) - 20
        pts[int(x)] = float(sr)
    return pts or None


def _sample_hold(pts: Dict[int, float], grid: np.ndarray) -> Optional[np.ndarray]:
    if not pts:
        return None
    xs = sorted(pts)
    out = np.empty(len(grid), dtype=float)
    for i, g in enumerate(grid):
        le = [x for x in xs if x <= g]
        out[i] = pts[le[-1]] if le else pts[xs[0]]
    return out


def _mean_curve(curve_paths: List[Path], grid: np.ndarray) -> Optional[np.ndarray]:
    rows = []
    for p in curve_paths:
        pts = _load_pts(p)
        s = _sample_hold(pts, grid) if pts else None
        if s is not None:
            rows.append(s)
    if not rows:
        return None
    return np.vstack(rows).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--runs", type=str, default="1 6")
    ap.add_argument("--dest_method", type=str, default="p4_top3_rotate")
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    grid_cfgs: List[Dict] = json.load(open(args.grid))
    out_root = Path(args.out_root)
    runs = [int(x) for x in args.runs.split()]
    D = int(args.budget)
    g = np.arange(0, D + 1, dtype=int)
    demo_axis = list(range(1, D + 1))  # exclude d=0 (shared init)

    # Fixed baseline envelope on the SAME runs (from the main results tree).
    base_curves = {}
    for m in BASELINES:
        c = _mean_curve([RESULTS_ROOT / f"run_{r}" / m / "results" / "learning_curve.json"
                         for r in runs], g)
        if c is not None:
            base_curves[m] = c
    if not base_curves:
        print("[sweep_rank] ERROR: no baseline curves found on the sweep runs", flush=True)
        return
    envelope = np.vstack(list(base_curves.values())).max(0)

    # Reference P4 curves (main tree).
    refs = {}
    for ref_method, ref_label in (("p4_top3_rotate", "P4@500"),
                                  ("p4_top3_rotate_e90", "P4@90")):
        c = _mean_curve([RESULTS_ROOT / f"run_{r}" / ref_method / "results" / "learning_curve.json"
                         for r in runs], g)
        if c is not None:
            refs[ref_label] = c

    rows: List[Dict] = []
    for cfg in grid_cfgs:
        tag = cfg["tag"]
        p4 = _mean_curve([out_root / tag / f"run_{r}" / args.dest_method / "results" / "learning_curve.json"
                          for r in runs], g)
        if p4 is None:
            rows.append({"tag": tag, **{k: cfg.get(k) for k in
                        ("epochs", "lr", "replay_mix", "head_dropout")},
                        "status": "no_data"})
            continue
        margin = p4 - envelope
        m_axis = margin[demo_axis]  # margins at d=1..D
        rows.append({
            "tag": tag,
            "epochs": cfg.get("epochs"), "lr": cfg.get("lr"),
            "replay_mix": cfg.get("replay_mix"), "head_dropout": cfg.get("head_dropout"),
            "dominates_all_demos": bool(np.all(m_axis >= 0)),
            "min_margin": float(m_axis.min()),
            "margin@5": float(margin[5]) if D >= 5 else None,
            "margin@10": float(margin[10]) if D >= 10 else None,
            "margin@15": float(margin[15]) if D >= 15 else None,
            "p4_final_sr": float(p4[-1]),
            "envelope_final_sr": float(envelope[-1]),
        })

    ranked = sorted([r for r in rows if r.get("status") != "no_data"],
                    key=lambda r: r["min_margin"], reverse=True)
    no_data = [r for r in rows if r.get("status") == "no_data"]

    # ---- write JSON + markdown leaderboard ----
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {"runs": runs, "budget": D, "baselines": list(base_curves.keys()),
               "envelope_per_demo": [float(x) for x in envelope],
               "references": {k: [float(x) for x in v] for k, v in refs.items()},
               "n_configs": len(grid_cfgs), "n_with_data": len(ranked),
               "n_dominating": sum(1 for r in ranked if r["dominates_all_demos"]),
               "leaderboard": ranked, "no_data": [r["tag"] for r in no_data]}
    (out_root / "leaderboard.json").write_text(json.dumps(payload, indent=2, default=str))

    lines = [f"# Sweep leaderboard (runs={runs}, budget={D})",
             f"baselines envelope (max over {', '.join(base_curves.keys())}); "
             f"dominates = P4 >= envelope at every demo 1..{D}",
             f"configs with data: {len(ranked)}/{len(grid_cfgs)}  |  "
             f"dominating all demos: {payload['n_dominating']}", "",
             "| rank | tag | epochs | lr | replay_mix | head_dropout | dominates | min_margin | m@5 | m@10 | m@15 | P4_final | env_final |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['tag']} | {r['epochs']} | {r['lr']} | {r['replay_mix']} | "
            f"{r['head_dropout']} | {'YES' if r['dominates_all_demos'] else 'no'} | "
            f"{r['min_margin']:+.3f} | {r['margin@5']:+.3f} | {r['margin@10']:+.3f} | "
            f"{r['margin@15']:+.3f} | {r['p4_final_sr']:.3f} | {r['envelope_final_sr']:.3f} |")
    if no_data:
        lines.append(f"\n_no data yet: {', '.join(r['tag'] for r in no_data)}_")
    (out_root / "leaderboard.md").write_text("\n".join(lines) + "\n")

    # ---- plot top-k P4 curves vs baseline envelope + references ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m, c in base_curves.items():
        ax.plot(g, c, lw=1, alpha=0.35, color="gray")
    ax.plot(g, envelope, lw=2.5, color="black", label="baseline envelope (max)")
    for lab, c in refs.items():
        ax.plot(g, c, lw=1.6, ls="--", alpha=0.8, label=lab)
    for r in ranked[:args.topk]:
        tag = r["tag"]
        p4 = _mean_curve([out_root / tag / f"run_{rr}" / args.dest_method / "results" / "learning_curve.json"
                          for rr in runs], g)
        if p4 is not None:
            mark = " *DOM*" if r["dominates_all_demos"] else ""
            ax.plot(g, p4, lw=2.0, marker="o", ms=3,
                    label=f"{tag} (min {r['min_margin']:+.3f}){mark}")
    ax.axhline(0.0, color="none")
    ax.set_xlabel("extra demonstrations"); ax.set_ylabel("held-out success rate")
    ax.set_xlim(0, D); ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.25)
    ax.set_title(f"P4 sweep: top-{args.topk} configs vs baseline envelope (runs {runs})")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_root / "leaderboard_top.png", dpi=150)

    print(f"[sweep_rank] {len(ranked)}/{len(grid_cfgs)} configs with data; "
          f"{payload['n_dominating']} dominate all demos.")
    for i, r in enumerate(ranked[:args.topk], 1):
        print(f"  #{i} {r['tag']:30s} min_margin={r['min_margin']:+.3f} "
              f"m@5={r['margin@5']:+.3f} m@10={r['margin@10']:+.3f} "
              f"m@15={r['margin@15']:+.3f} dom={r['dominates_all_demos']}")
    print(f"[sweep_rank] wrote {out_root}/leaderboard.md, .json, leaderboard_top.png")


if __name__ == "__main__":
    main()
