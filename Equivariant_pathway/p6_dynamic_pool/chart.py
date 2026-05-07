"""Plot the dynamic-pool experiment.

Top axes: heldout SR vs cumulative demos (with baseline_only / p4_only /
p6_only curves overlaid when present).
Bottom axes: pool size growth + pool SR vs round.

The dynamic-pool plot is the headline figure for the
"LLM-guided dynamic pool expansion overcomes correction-pool saturation"
claim — both the demo cost (top) and the pool growth + saturation
trajectory (bottom) need to be inspectable in one figure.
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
CHART_PATH = RESULTS_DIR / "dynamic_pool_curves.png"

PEERS = {
    "baseline_only": ROOT.parent / "baseline_only" / "results" / "learning_curve.json",
    "p4_only":      ROOT.parent / "p4_only"      / "results" / "learning_curve.json",
    "p6_only":      ROOT.parent / "p6_only"      / "results" / "learning_curve.json",
}

PEER_STYLE = {
    "baseline_only": {"marker": "s", "color": "tab:blue",   "linestyle": "--"},
    "p4_only":      {"marker": "^", "color": "tab:green",  "linestyle": "--"},
    "p6_only":      {"marker": "D", "color": "tab:purple", "linestyle": "--"},
}


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

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7.2),
                                         gridspec_kw={"height_ratios": [3, 2]})

    # --- Top: heldout SR vs cumulative demos ---
    xs = [int(r["cum_demos"]) for r in history]
    ys = [float(r["heldout_sr"]) for r in history]
    ax_top.plot(xs, ys, marker="o", color="tab:red", linewidth=2,
                label="p6_dynamic_pool (this run)")
    for r, x, y in zip(history, xs, ys):
        ax_top.annotate(f"r{r['round']}", (x, y),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)

    for name, path in PEERS.items():
        d = _load(path)
        if not d:
            continue
        h = d.get("history", []) or []
        if not h:
            continue
        ax_top.plot([int(r["cum_demos"]) for r in h],
                    [float(r["heldout_sr"]) for r in h],
                    label=f"{name} heldout SR", alpha=0.85,
                    **PEER_STYLE.get(name, {}))

    ax_top.axhline(target, ls="--", color="black", alpha=0.6,
                   label=f"target = {target:.2f}")
    ax_top.set_xlabel("cumulative demos")
    ax_top.set_ylabel("heldout success rate")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_title("LLM-guided dynamic pool expansion: heldout SR vs demos")
    ax_top.grid(True, ls=":", alpha=0.5)
    ax_top.legend(loc="lower right", fontsize=8)

    # --- Bottom: pool growth + pool SR ---
    rounds = [int(r["round"]) for r in history]
    pool_sizes = [int(r.get("pool_size") or 0) for r in history]
    pool_srs   = [r.get("pool_sr") for r in history]
    pool_srs_plot = [float(s) if s is not None else None for s in pool_srs]

    ax_bot.plot(rounds, pool_sizes, marker="o", color="tab:orange",
                label="dynamic pool size")
    ax_bot.set_xlabel("round")
    ax_bot.set_ylabel("dynamic pool size", color="tab:orange")
    ax_bot.tick_params(axis="y", labelcolor="tab:orange")
    ax_bot.grid(True, ls=":", alpha=0.5)

    ax_bot2 = ax_bot.twinx()
    valid_pts = [(r, s) for r, s in zip(rounds, pool_srs_plot) if s is not None]
    if valid_pts:
        rs, ss = zip(*valid_pts)
        ax_bot2.plot(rs, ss, marker="s", color="tab:cyan",
                     label="pool SR (saturation indicator)")
    ax_bot2.set_ylabel("pool success rate", color="tab:cyan")
    ax_bot2.set_ylim(0, 1.05)
    ax_bot2.tick_params(axis="y", labelcolor="tab:cyan")

    # Annotate rounds where the LLM fired vs skipped.
    for r in history:
        if r["round"] == 0:
            continue
        if r.get("llm_fired"):
            ax_bot.annotate("LLM", (int(r["round"]), int(r["pool_size"])),
                            textcoords="offset points", xytext=(0, 6),
                            fontsize=7, color="tab:purple")

    ax_bot.set_title("Dynamic correction pool: size growth and saturation curve")

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=140)
    plt.close(fig)
    print(f"[chart] wrote {CHART_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
