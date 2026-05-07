"""Plot heldout success rate vs cumulative demos for the p4_only loop.

Reads <p4_only>/results/learning_curve.json and writes
success_rate_vs_demos.png alongside it. If
<baseline_only>/results/learning_curve.json also exists, both curves
are overlaid so the sample-efficiency comparison is one glance away.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CURVE_PATH = RESULTS_DIR / "learning_curve.json"
CHART_PATH = RESULTS_DIR / "success_rate_vs_demos.png"

BO_CURVE_PATH = ROOT.parent / "baseline_only" / "results" / "learning_curve.json"


def _load(p: Path):
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def main() -> int:
    if not CURVE_PATH.exists():
        print(f"[chart] no learning_curve.json at {CURVE_PATH}; nothing to plot.")
        return 1
    data = _load(CURVE_PATH) or {}
    history = data.get("history", []) or []
    target = float(data.get("target_sr", 0.90))
    if not history:
        print("[chart] history is empty; nothing to plot.")
        return 1

    fig, ax = plt.subplots(figsize=(8, 4.8))

    xs = [int(r["cum_demos"]) for r in history]
    ys = [float(r["heldout_sr"]) for r in history]
    ax.plot(xs, ys, marker="o", color="tab:green", label="p4_only heldout success rate")
    for r, x, y in zip(history, xs, ys):
        ax.annotate(f"r{r['round']}", (x, y),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)

    bo_data = _load(BO_CURVE_PATH)
    if bo_data:
        bo_h = bo_data.get("history", []) or []
        if bo_h:
            bxs = [int(r["cum_demos"]) for r in bo_h]
            bys = [float(r["heldout_sr"]) for r in bo_h]
            ax.plot(bxs, bys, marker="s", linestyle="--", color="tab:blue",
                    label="baseline_only heldout success rate", alpha=0.85)

    ax.axhline(target, ls="--", color="tab:red", label=f"target = {target:.2f}")
    ax.set_xlabel("cumulative demos (initial 20 + LLM-prescribed)")
    ax.set_ylabel("heldout success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("p4_only vs baseline_only: heldout success rate vs demos collected")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=140)
    plt.close(fig)
    print(f"[chart] wrote {CHART_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
