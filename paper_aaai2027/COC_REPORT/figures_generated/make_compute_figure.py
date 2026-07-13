#!/usr/bin/env python3
"""F16_compute_cost — per-round wall-clock decomposition, DISEIL vs SafeDAgger.

Every number below is transcribed from
    paper_aaai2027/COC_REPORT/build/d5_compute.md
(Protocol P1: the first round of each run, the only protocol available for all
five settings, because Push-T and GridWorld were run at BUDGET=1).
Nothing here is estimated. Source SLURM jobs: 110355/110356 (Door state),
110357/110358 (Door image), 110359/110360 (Wipe image), 110375/110376 (Push-T
image), 110384/110385 (GridWorld image). Seed 1 throughout.

Usage:  python3 make_compute_figure.py
Writes: F16_compute_cost.pdf and F16_compute_cost.png next to this script.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- data, Protocol P1 (first round), seconds -------------------------------
# baseline_total / diseil_total are the measured per-round wall-clock of each arm.
# gate      = the baseline's query-gate rollout.
# screen    = DISEIL's failure-screening rollout.
# analysis  = DISEIL's clustering + VLM + reasoning LLM + prescription + feasibility.
# shared    = DISEIL's from-scratch policy retrain + 100-episode held-out eval.
# The baseline's own retrain+eval is its total minus its gate rollout.
SETTINGS = [
    # name,                baseline_total, gate, shared, screen, analysis, overhead, addon
    ("Wipe (image)",        1468.0, 4.0, 1491.0, 431.0, 273.0, 1.50,  700.0),
    ("Door (image)",        1247.0, 1.0, 1180.0, 201.0,  93.0, 1.18,  293.0),
    ("Door (state)",         737.0, 1.0,  783.0, 205.0,  66.0, 1.43,  270.0),
    ("Push-T (image)",       688.0, 6.3,  652.5, 746.8, 491.6, 2.75, 1232.1),
    ("GridWorld\n(image)",     54.6, 2.9,   51.1,   5.1,  60.4, 2.16,   62.6),
]

# Okabe-Ito, colourblind-safe.
C_SHARED = "#BBBBBB"   # retrain + eval, paid by both arms
C_GATE = "#56B4E9"     # baseline query-gate rollout
C_SCREEN = "#E69F00"   # DISEIL failure screening
C_ANALYSIS = "#D55E00" # DISEIL analysis + prescription

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

big = SETTINGS[:4]
small = SETTINGS[4:]

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [2.7, 1.0], "wspace": 0.26}
)

BAR_H = 0.32
OFFSET = 0.19


def draw(ax, rows):
    ticks, labels = [], []
    for i, (name, base_tot, gate, shared, screen, analysis, ratio, addon) in enumerate(rows):
        y = len(rows) - 1 - i
        base_train = base_tot - gate

        # baseline arm (lower bar)
        yb = y - OFFSET
        ax.barh(yb, base_train, BAR_H, color=C_SHARED, edgecolor="white", linewidth=0.6)
        ax.barh(yb, gate, BAR_H, left=base_train, color=C_GATE,
                edgecolor="white", linewidth=0.6)

        # DISEIL arm (upper bar)
        yd = y + OFFSET
        ax.barh(yd, shared, BAR_H, color=C_SHARED, edgecolor="white", linewidth=0.6)
        ax.barh(yd, screen, BAR_H, left=shared, color=C_SCREEN,
                edgecolor="white", linewidth=0.6)
        ax.barh(yd, analysis, BAR_H, left=shared + screen, color=C_ANALYSIS,
                edgecolor="white", linewidth=0.6)

        total = shared + screen + analysis
        pad = 0.015 * ax_limit(rows)
        ax.text(total + pad, yd, f"{ratio:.2f}×  +{addon:,.0f} s",
                va="center", ha="left", fontsize=8.5)
        ax.text(base_tot + pad, yb, "SafeDAgger", va="center", ha="left",
                fontsize=8, color="#555555")

        ticks.append(y)
        labels.append(name)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.62, len(rows) - 0.38)
    ax.set_xlabel("Wall-clock per round (s)")
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)


def ax_limit(rows):
    return max(s + sc + an for _, _, _, s, sc, an, _, _ in rows)


draw(axL, big)
axL.set_xlim(0, 3050)
axL.set_title("Robot tasks and Push-T", loc="left", pad=6)

draw(axR, small)
axR.set_xlim(0, 300)
axR.set_xticks([0, 100, 200, 300])
axR.set_title("GridWorld", loc="left", pad=6)

handles = [
    Patch(facecolor=C_SHARED, label="Policy retrain + held-out eval (paid by both arms)"),
    Patch(facecolor=C_GATE, label="SafeDAgger query-gate rollout"),
    Patch(facecolor=C_SCREEN, label="DISEIL failure screening"),
    Patch(facecolor=C_ANALYSIS, label="DISEIL analysis + prescription"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02), handlelength=1.5, columnspacing=1.6)

fig.subplots_adjust(left=0.155, right=0.985, top=0.90, bottom=0.29)

out = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out, "F16_compute_cost.pdf"))
fig.savefig(os.path.join(out, "F16_compute_cost.png"), dpi=300)
print("wrote F16_compute_cost.pdf / .png")
