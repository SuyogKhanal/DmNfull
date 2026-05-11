"""Hybrid p5_only vs hybrid baseline_only chart."""
from __future__ import annotations

import json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_ROOT = Path(__file__).resolve().parent
_P5_OVERRIDE = os.environ.get("P5_ONLY_ROOT")
_BO_OVERRIDE = os.environ.get("BASELINE_ONLY_ROOT")
CURVE_ROOT = Path(_P5_OVERRIDE).resolve() if _P5_OVERRIDE else CODE_ROOT
BO_CURVE_ROOT = (Path(_BO_OVERRIDE).resolve() if _BO_OVERRIDE
                 else CODE_ROOT.parent / "baseline_only")
RESULTS_DIR = CURVE_ROOT / "results"
CURVE_PATH = RESULTS_DIR / "learning_curve.json"
CHART_PATH = RESULTS_DIR / "success_rate_vs_demos.png"
BO_CURVE_PATH = BO_CURVE_ROOT / "results" / "learning_curve.json"


def _load(p):
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    if not CURVE_PATH.exists():
        return 1
    data = _load(CURVE_PATH) or {}
    history = data.get("history", []) or []
    target = float(data.get("target_sr", 0.90))
    if not history:
        return 1
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = [int(r["cum_demos"]) for r in history]
    ys = [float(r["heldout_sr"]) for r in history]
    ax.plot(xs, ys, marker="o", color="tab:cyan", label="p5_only_hybrid SR")
    for r, x, y in zip(history, xs, ys):
        ax.annotate(f"r{r['round']}", (x, y),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    bo_data = _load(BO_CURVE_PATH)
    if bo_data:
        bo_h = bo_data.get("history", []) or []
        if bo_h:
            bxs = [int(r["cum_demos"]) for r in bo_h]
            bys = [float(r["heldout_sr"]) for r in bo_h]
            ax.plot(bxs, bys, marker="s", linestyle="--", color="tab:orange",
                    label="baseline_only_hybrid SR", alpha=0.85)
    ax.axhline(target, ls="--", color="tab:red", label=f"target = {target:.2f}")
    ax.set_xlabel("cumulative demos (initial 20 + LLM-prescribed)")
    ax.set_ylabel("heldout success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Hybrid pathway: p5_only vs baseline_only (CNN+Equivariant)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=140)
    plt.close(fig)
    print(f"[chart] wrote {CHART_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
