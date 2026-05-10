"""Plot heldout SR vs cumulative demos for the CNN_pathway baseline."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_ROOT = Path(__file__).resolve().parent
_BO_OVERRIDE = os.environ.get("BASELINE_ONLY_ROOT")
CURVE_ROOT = Path(_BO_OVERRIDE).resolve() if _BO_OVERRIDE else CODE_ROOT
RESULTS_DIR = CURVE_ROOT / "results"
CURVE_PATH = RESULTS_DIR / "learning_curve.json"
CHART_PATH = RESULTS_DIR / "success_rate_vs_demos.png"


def main() -> int:
    if not CURVE_PATH.exists():
        print(f"[chart] no learning_curve.json at {CURVE_PATH}")
        return 1
    data = json.load(open(CURVE_PATH))
    history = data.get("history", []) or []
    target = float(data.get("target_sr", 0.90))
    if not history:
        return 1
    xs = [int(r["cum_demos"]) for r in history]
    ys = [float(r["heldout_sr"]) for r in history]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, ys, marker="o", color="tab:blue",
            label="heldout success rate")
    ax.axhline(target, ls="--", color="tab:red", label=f"target = {target:.2f}")
    for r, x, y in zip(history, xs, ys):
        ax.annotate(f"r{r['round']}", (x, y),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("cumulative demos (initial 20 + corrective)")
    ax.set_ylabel("heldout success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("CNN baseline (RGBCNNPolicy): SR vs demos")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=140)
    plt.close(fig)
    print(f"[chart] wrote {CHART_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
