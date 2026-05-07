"""Plot the random-vs-LLM dynamic-pool comparison.

Top axes: heldout SR vs cumulative demos with all four method curves
overlaid (baseline_only fixed pool, p4_only fixed pool, p6_only fixed
pool, p6_dynamic_pool LLM expansion, baseline_dynamic_pool random
expansion).

Bottom axes: dynamic pool size + per-round expansion budget for the two
expansion methods (this run + p6_dynamic_pool when present). Lets you
see at a glance whether random spent more layouts than P6 to reach the
same heldout SR.
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
CHART_PATH = RESULTS_DIR / "random_vs_llm_dynamic_pool.png"

PEERS = {
    "baseline_only":    ROOT.parent / "baseline_only"    / "results" / "learning_curve.json",
    "p4_only":         ROOT.parent / "p4_only"          / "results" / "learning_curve.json",
    "p6_only":         ROOT.parent / "p6_only"          / "results" / "learning_curve.json",
    "p6_dynamic_pool": ROOT.parent / "p6_dynamic_pool"  / "results" / "learning_curve.json",
}

PEER_STYLE = {
    "baseline_only":    {"marker": "s", "color": "tab:blue",   "linestyle": "--"},
    "p4_only":         {"marker": "^", "color": "tab:green",  "linestyle": "--"},
    "p6_only":         {"marker": "D", "color": "tab:purple", "linestyle": "--"},
    "p6_dynamic_pool": {"marker": "*", "color": "tab:red",    "linestyle": "-"},
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

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.5, 7.5),
                                         gridspec_kw={"height_ratios": [3, 2]})

    # --- Top: heldout SR vs cumulative demos ---
    xs = [int(r["cum_demos"]) for r in history]
    ys = [float(r["heldout_sr"]) for r in history]
    ax_top.plot(xs, ys, marker="o", color="tab:orange", linewidth=2,
                label="baseline_dynamic_pool (random, this run)")
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
    ax_top.set_title("LLM-guided vs random dynamic-pool expansion: heldout SR vs demos")
    ax_top.grid(True, ls=":", alpha=0.5)
    ax_top.legend(loc="lower right", fontsize=8)

    # --- Bottom: pool growth and per-round budget ---
    rounds = [int(r["round"]) for r in history]
    pool_sizes = [int(r.get("pool_size") or 0) for r in history]
    budgets = [int(r["expansion_budget"]) if r.get("expansion_budget") is not None else 0
               for r in history]

    ax_bot.plot(rounds, pool_sizes, marker="o", color="tab:orange",
                label="random pool size")
    ax_bot.bar(rounds, budgets, alpha=0.25, color="tab:orange",
               label="random expansion budget per round")

    p6dp = _load(PEERS["p6_dynamic_pool"])
    if p6dp:
        h = p6dp.get("history", []) or []
        if h:
            ax_bot.plot([int(r["round"]) for r in h],
                        [int(r.get("pool_size") or 0) for r in h],
                        marker="*", color="tab:red",
                        label="p6_dynamic_pool pool size")

    ax_bot.set_xlabel("round")
    ax_bot.set_ylabel("layouts")
    ax_bot.grid(True, ls=":", alpha=0.5)
    ax_bot.legend(loc="best", fontsize=8)
    ax_bot.set_title("Dynamic pool growth — random vs LLM")

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=140)
    plt.close(fig)
    print(f"[chart] wrote {CHART_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
