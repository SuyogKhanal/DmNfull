#!/usr/bin/env python
"""
Publication figures for the CoC report (method name: DISEIL).

Source of truth: ablation_se.json (loaded through ablation_data.py), the tidy
per-method table the DATA agent built from ablations_results/sheets/*.csv (the
NEW run). Every number plotted is read from that file at run time; nothing is
hand-entered here. The method is DISEIL on every axis, legend and label.

Dispersion rule (the crux): the CSVs store mean +/- sample standard deviation.
The report reports mean +/- STANDARD ERROR, SE = std / sqrt(n), with n = 9 for
GridWorld 5x5 (9 seeds) and n = 5 for the robot tasks. The JSON already carries
the precomputed `se` for every method row, so every error bar drawn here is SE
(small), never std/variance. The A6 (F5) ablated bars now carry their SE bar,
which the previous build was missing.

Outputs: <fig>.pdf (vector, for LaTeX) and <fig>.png (preview) per figure.

Rules enforced by this script (supervisor revision spec, section D):
  D-G1  no orange ink of any kind (neither VERM #D55E00 nor ORANGE #E69F00);
  D-G2  no prose, verdict or interpretation inside the artwork. Numeric data
        labels and axis/legend keys stay; sentences do not;
  D-G3  Lift appears in no ablation figure, and no figure says that it was left out;
  E1    the ablations are drawn on three settings only: GridWorld (image),
        Push-T (state), Door (image).

Two figures have been retired and their generating code removed: the memory-
constant sweep and the paired-margin forest. The round-2 deletions
(memory-constants, aggregate-significance, cluster-purity, compute-cost) are not
resurrected.

Typeface: Liberation Serif, metrically identical to Times New Roman and carrying
the Greek glyphs the report needs. Palette: Okabe-Ito, minus both orange hues.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ablation_data import load

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "ablations_results" / "sheets"
OUT = ROOT / "figures_generated"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- style
# The report is A4 with 2.4 cm margins, so the text block is 16.2 cm = 6.38 in, and
# pandoc scales any wider figure down to fit. Every figure is therefore BUILT at the
# printed width: a point in the artwork is a point on the page, and the 9 pt floor
# for figure type is a real 9 pt.
FIGW = 6.38

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,      # embed TrueType, not Type-3: required by most venues
    "ps.fonttype": 42,
})

# Okabe-Ito, with both orange hues withdrawn from the working set (D-G1).
BLUE, SKY, GREEN, PURPLE = "#0072B2", "#56B4E9", "#009E73", "#CC79A7"
GREY, INK, MUTED, DARK = "#999999", "#222222", "#666666", "#404040"

# Sequential single-hue ramp for ORDERED categories (a ladder is magnitude, not identity).
SEQ = ["#08306B", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1", "#C6DBEF"]

SHORT = {
    ("GridWorld 5x5", "state"): "GridWorld state", ("GridWorld 5x5", "image"): "GridWorld image",
    ("Push-T", "state"): "Push-T state", ("Push-T", "image"): "Push-T image",
    ("Lift", "state"): "Lift state", ("Lift", "image"): "Lift image",
    ("Wipe", "state"): "Wipe state", ("Wipe", "image"): "Wipe image",
    ("Door", "state"): "Door state", ("Door", "image"): "Door image",
}
# The three ablation settings (E1). Every ablation figure iterates this list.
PRIMARY = [("GridWorld 5x5", "image"), ("Push-T", "state"), ("Door", "image")]
# Colour by setting, for the figures whose series ARE the settings.
SETTING_COL = {PRIMARY[0]: BLUE, PRIMARY[1]: GREEN, PRIMARY[2]: PURPLE}


def save(fig, name: str) -> str:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")
    return name


# ----------------------------------------------------------------- data adapters
def _rmap(sheet):
    """(task, obs, method) -> row dict for a JSON sheet."""
    return {(r["task"], r["obs"], r["method"]): r for r in load(sheet)}


def ms(rmap, task, obs, method):
    """Return (mean, se) for one cell; se may be nan where the source has none."""
    r = rmap[(task, obs, method)]
    se = r.get("se")
    return (r["mean"], np.nan if se is None else se)


# Ground-truth success rate, keyed for quick lookup.
_GT = _rmap("GT_SR")


def diseil_full(s):
    """Full DISEIL final SR (mean, SE) for a setting, from GT_SR."""
    return ms(_GT, s[0], s[1], "DISEIL")


def best_baseline(s):
    """Best-baseline mean for a setting, read from the knockout sheets' ref column."""
    r = _rmap("A3_Clustering_Off")[(s[0], s[1], "Best baseline (ref)")]
    return r["mean"]


def a6_fallback():
    """Fallback rate (% of rounds) with the knowledge graph off, from A6_KAG_Off.csv.

    The JSON tidy table carries only the three success-rate arms; the fallback
    column lives in the source CSV, so it is read straight from there. Column
    layout: Task, Obs, Full ref, Best ref, mean +/- std, delta, margin, Fallback%.
    """
    out = {}
    tasks = {"GridWorld 5x5", "Push-T", "Lift", "Wipe", "Door"}
    with open(SHEETS / "A6_KAG_Off.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 8 and row[0] in tasks and row[1] in ("state", "image"):
                out[(row[0], row[1])] = float(row[7])
    return out


# --------------------------------------------------------------- shared bar-panel style
def bar_panel(ax, labels, means, ses, colors, baseline=None, fmt="{:.1f}",
              floor=2.0, width=0.68, tick_fs=9, legend_loc="upper right",
              headroom_frac=0.30, rotation=28):
    """
    The grouped-bar panel used by F1, F2, F4 and F5.

    labels        x tick labels, one per arm
    means/ses     arm values; se may be nan, in which case no error bar is drawn
    baseline      (mean, name) of the best baseline, drawn as a dashed reference
                  line and named in a legend key
    legend_loc    the corner of the panel that the arm ordering leaves empty
    headroom_frac top margin, as a fraction of the plotted range, reserved for the
                  value labels and the legend key

    Error bars are STANDARD ERROR (SE = std / sqrt(n)); no bar carries std.
    """
    x = np.arange(len(labels))
    lo_c = [m - (0.0 if np.isnan(se) else se) for m, se in zip(means, ses)]
    hi_c = [m + (0.0 if np.isnan(se) else se) for m, se in zip(means, ses)]
    if baseline is not None:
        lo_c.append(baseline[0])
    span = max(max(hi_c) - min(lo_c), 1e-6)
    lo = min(lo_c) - floor
    hi = max(hi_c) + headroom_frac * span
    ax.set_ylim(lo, hi)

    for xi, m, se, c in zip(x, means, ses, colors):
        ax.bar(xi, m, width=width, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        if not np.isnan(se):
            ax.errorbar(xi, m, yerr=se, fmt="none", ecolor=INK, elinewidth=1.0,
                        capsize=3, zorder=4)
        off = (0.0 if np.isnan(se) else se) + 0.012 * (hi - lo)
        ax.text(xi, m + off, fmt.format(m), ha="center", va="bottom",
                fontsize=9, color=INK, zorder=5)
    if baseline is not None:
        bm = baseline[0]
        ax.axhline(bm, color=INK, ls="--", lw=1.1, zorder=2)
        # The reference value is carried by a legend key, not by a floating label:
        # a label pinned to the line collides with the bar values beneath it. The
        # NAME of the best baseline in each setting is a caption matter, not artwork.
        ax.legend(handles=[Line2D([], [], color=INK, ls="--", lw=1.1,
                                  label=f"best baseline {bm:.1f}")],
                  loc=legend_loc, fontsize=9, handlelength=1.6, borderaxespad=0.2,
                  handletextpad=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fs, rotation=rotation,
                       ha="right" if rotation else "center",
                       rotation_mode="anchor" if rotation else None)
    ax.set_xlim(-0.72, len(labels) - 0.28)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


# ============================================================ F1 — allocation ladder
def fig_allocation_ladder():
    a2 = _rmap("A2_RandomAlloc_Robots")

    def random_alloc(s):
        # A2 supplies the uniform-random-allocation (Stagger) control for the robot
        # tasks; GridWorld's random-allocation control is Stagger in GT_SR.
        if s[0].startswith("GridWorld"):
            return ms(_GT, s[0], s[1], "Stagger")
        return ms(a2, s[0], s[1], "Stagger (random alloc)")

    a8 = _rmap("A8_Fallback_Only")
    a3 = _rmap("A3_Clustering_Off")

    labels = ["Random alloc. (A2)", "Fallback only (A8)", "Clustering off (A3)",
              "DISEIL (full)"]
    colors = [GREY, GREEN, SKY, BLUE]

    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 3.5), gridspec_kw=dict(wspace=0.34))
    for ax, s in zip(axes, PRIMARY):
        vals = [random_alloc(s), ms(a8, s[0], s[1], "DISEIL (ablated)"),
                ms(a3, s[0], s[1], "DISEIL (ablated)"), diseil_full(s)]
        # the ladder rises left to right, so the upper-left corner is the empty one
        bar_panel(ax, labels, [m for m, _ in vals], [se for _, se in vals], colors,
                  baseline=(best_baseline(s),), legend_loc="upper left")
        ax.set_title(SHORT[s], pad=6)
    axes[0].set_ylabel("Final success rate (%), mean ± 1 s.e. over seeds")
    return save(fig, "F1_allocation_ladder")


# ================================================== F2 — gain without allocation (A3)
def fig_gain_vs_success():
    """Grouped bars in the F1 style: success rate with the partition removed.

    A single panel row. The per-demonstration information gain is reported in the
    table and in the text, not here.
    """
    a3 = _rmap("A3_Clustering_Off")
    labels = ["DISEIL (full)", "Clustering off (A3)"]
    colors = [BLUE, SKY]

    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 3.5), gridspec_kw=dict(wspace=0.34))
    for ax, s in zip(axes, PRIMARY):
        # DISEIL is the tallest bar and sits on the left, so the legend goes right
        vals = [diseil_full(s), ms(a3, s[0], s[1], "DISEIL (ablated)")]
        bar_panel(ax, labels, [m for m, _ in vals], [se for _, se in vals], colors,
                  baseline=(best_baseline(s),), legend_loc="upper right")
        ax.set_title(SHORT[s], pad=6)
    axes[0].set_ylabel("Final success rate (%), mean ± 1 s.e. over seeds")
    return save(fig, "F2_gain_without_allocation")


# ================================================== F3 — knockout summary (heatmap)
def fig_knockout_heatmap():
    order = ["A3", "A8", "A6", "A7", "A4", "A5", "A1"]   # ranked by mean damage
    sheetof = {
        "A1": "A1_Memory_Off", "A3": "A3_Clustering_Off", "A4": "A4_LLM_vs_Heuristic",
        "A5": "A5_VLM_Off", "A6": "A6_KAG_Off", "A7": "A7_Bridging_Off_",
        "A8": "A8_Fallback_Only",
    }
    names = {
        "A3": "A3  clustering off", "A8": "A8  fallback only", "A6": "A6  knowledge graph off",
        "A7": "A7  bridging off", "A4": "A4  reasoning model → heuristic",
        "A5": "A5  vision-language model off", "A1": "A1  cluster memory off",
    }
    RM = {k: _rmap(sheetof[k]) for k in order}

    def cell(k, s):
        m = RM[k]
        full = m[(s[0], s[1], "Full DISEIL (ref)")]["mean"]
        base = m[(s[0], s[1], "Best baseline (ref)")]["mean"]
        mean = m[(s[0], s[1], "DISEIL (ablated)")]["mean"]
        delta = mean - full
        margin = np.nan if full == base else 100.0 * (mean - base) / (full - base)
        return margin, delta

    M = np.array([[cell(k, s)[0] for s in PRIMARY] for k in order])
    D = np.array([[cell(k, s)[1] for s in PRIMARY] for k in order])

    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    cmap = plt.get_cmap("Blues_r")
    im = ax.imshow(np.clip(M, 0, 100), cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v, d = M[i, j], D[i, j]
            if np.isnan(v):
                v = 0.0
            if v < 0:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc=DARK,
                                           ec="white", lw=1.4, zorder=2))
                col = "white"
            else:
                col = "white" if v < 45 else INK
            ax.text(j, i - 0.08, f"{v:.0f}%", ha="center", va="center", fontsize=9.5,
                    color=col, zorder=3)
            ax.text(j, i + 0.28, f"{d:+.1f}", ha="center", va="center", fontsize=8,
                    color=("white" if (v < 45) else "#444444"), zorder=3)

    ax.set_xticks(range(len(PRIMARY)))
    ax.set_xticklabels([SHORT[s] for s in PRIMARY], rotation=20, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in order])
    ax.set_xticks(np.arange(-.5, len(PRIMARY), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("Margin over the best baseline retained (%)", fontsize=9)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)
    ax.legend(handles=[Patch(fc=DARK, label="margin retained < 0")],
              loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=1, fontsize=9)
    return save(fig, "F3_knockout_summary")


# ========================================== F4 — reasoning and vision (A4/A5)
def fig_a4_a5_dotplot():
    """Redrawn as grouped bars in the F1 style."""
    a4 = _rmap("A4_LLM_vs_Heuristic")
    a5 = _rmap("A5_VLM_Off")
    labels = ["DISEIL (full)", "Reasoning → heuristic (A4)",
              "Vision-language model off (A5)"]
    colors = [BLUE, SKY, GREEN]

    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 4.0), gridspec_kw=dict(wspace=0.34))
    for ax, s in zip(axes, PRIMARY):
        vals = [diseil_full(s), ms(a4, s[0], s[1], "DISEIL (ablated)"),
                ms(a5, s[0], s[1], "DISEIL (ablated)")]
        bar_panel(ax, labels, [m for m, _ in vals], [se for _, se in vals], colors,
                  baseline=(best_baseline(s),), legend_loc="upper right")
        ax.set_title(SHORT[s], pad=6)
    axes[0].set_ylabel("Final success rate (%), mean ± 1 s.e. over seeds")
    return save(fig, "F4_reasoning_and_vision_small")


# ============================================ F5 — grounding and feasibility (A6)
def fig_kag_fallback():
    """Bars in the F1 style.

    Top: success rate with the knowledge graph removed, against full DISEIL and the
    best baseline. Bottom: the rate at which the feasibility screen falls back to
    the deterministic rule once the graph is gone.

    The A6 ablated bars now carry their SE error bar (SE = std / sqrt(n) from the
    A6_KAG_Off sheet); the previous build drew them without one.
    """
    a6 = _rmap("A6_KAG_Off")
    fb = a6_fallback()

    labels = ["DISEIL (full)", "Knowledge graph off (A6)"]
    colors = [BLUE, SKY]

    fig = plt.figure(figsize=(FIGW, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.95, wspace=0.42)
    for j, s in enumerate(PRIMARY):
        ax = fig.add_subplot(gs[0, j])
        vals = [diseil_full(s), ms(a6, s[0], s[1], "DISEIL (ablated)")]
        bar_panel(ax, labels, [m for m, _ in vals], [se for _, se in vals], colors,
                  baseline=(best_baseline(s),), legend_loc="upper right")
        ax.set_title(SHORT[s], pad=6)
        if j == 0:
            ax.set_ylabel("Final success rate (%),\nmean ± 1 s.e. over seeds")

    axb = fig.add_subplot(gs[1, :])
    x = np.arange(len(PRIMARY))
    for xi, s in zip(x, PRIMARY):
        v = fb[s]
        axb.bar(xi, v, width=0.42, color=SETTING_COL[s], edgecolor="white",
                linewidth=1.2, zorder=3)
        axb.text(xi, v + 0.7, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    axb.set_xticks(x)
    axb.set_xticklabels([SHORT[s] for s in PRIMARY], fontsize=9)
    axb.set_xlim(-0.6, len(PRIMARY) - 0.4)
    axb.set_ylim(0, max(fb[s] for s in PRIMARY) + 8)
    axb.set_ylabel("Fallback rate with the\nknowledge graph off (% of rounds)")
    axb.yaxis.grid(True, zorder=0)
    axb.set_axisbelow(True)
    return save(fig, "F5_grounding_and_feasibility")


# ================================================== F6 — bridging (A7 + D3)
def fig_bridging():
    d3 = _rmap("D3_Bridge_Split")

    def split(s):
        t = d3[(s[0], s[1], "Targeted %")]["mean"]
        b = d3[(s[0], s[1], "Bridge %")]["mean"]
        return t, b

    fig, ax = plt.subplots(figsize=(FIGW, 2.5))
    y = np.arange(len(PRIMARY))[::-1]
    for s, yi in zip(PRIMARY, y):
        t, b = split(s)
        ax.barh(yi, t, height=0.6, color=SKY, edgecolor="white", linewidth=1.4, zorder=3)
        ax.barh(yi, b, left=t, height=0.6, color=BLUE, edgecolor="white", linewidth=1.4,
                zorder=3)
        ax.text(t / 2, yi, f"{t:.0f}", ha="center", va="center", fontsize=9, color=INK)
        ax.text(t + b / 2, yi, f"{b:.0f}", ha="center", va="center", fontsize=9,
                color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[s] for s in PRIMARY])
    ax.set_ylim(-0.6, len(PRIMARY) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of accepted prescriptions (%)")
    ax.legend(handles=[Patch(fc=SKY, label="targeted in-place correction"),
                       Patch(fc=BLUE, label="bridged placement")],
              loc="lower left", bbox_to_anchor=(0.0, -0.52), ncol=2)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    return save(fig, "F6_bridging")


# =============================== F7 — descriptor dimensionality by silhouette (A10)
def fig_descriptor_dims():
    a10 = _rmap("A10")
    # arm labels in the A10 sheet, in ascending dimensionality
    arms = [
        "2-D  — position only [p_x, p_y]",
        "4-D  — + orientation (sinθ, cosθ)",
        "5-D  — + task progress ρ",
        "6-D  — FULL φ  (paper)",
        "8-D  — + end-effector velocity (v_x, v_y)",
        "10-D — + gripper state, z-height",
        "12-D — + joint-angle summary",
    ]
    dims = [2, 4, 5, 6, 8, 10, 12]
    fig, ax = plt.subplots(figsize=(FIGW, 4.0))
    M = []
    for s in PRIMARY:
        ys = [a10[(s[0], s[1], a)]["mean"] for a in arms]
        M.append(ys)
        ax.plot(dims, ys, "-", lw=1.1, color=SKY, alpha=0.9, zorder=2)
    M = np.array(M)
    mean = M.mean(axis=0)
    ax.plot(dims, mean, "-o", lw=2.6, ms=6, color=BLUE, zorder=4)
    ax.axvline(6, color=INK, ls="--", lw=1.2, zorder=1)
    for d, v in zip(dims, mean):
        ax.text(d, v - 0.018, f"{v:.3f}", ha="center", va="top", fontsize=9, color=BLUE,
                zorder=5, bbox=dict(fc="white", ec="none", pad=0.8))
    ax.set_xticks(dims)
    ax.set_xlabel("Descriptor dimensionality (number of features in φ)")
    ax.set_ylabel("Mean silhouette of the resulting failure clusters")
    ax.set_ylim(0.29, 0.70)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.6, marker="o", label="mean over the three settings"),
        Line2D([], [], color=SKY, lw=1.1, label="individual setting"),
    ], loc="upper right", ncol=1, fontsize=9)
    return save(fig, "F7_descriptor_dimensionality")


# ================================================================ F8 — budget sweep
def fig_budget():
    a11 = _rmap("A11")
    B = [10, 20, 40]

    def series(s):
        dis = [a11[(s[0], s[1], f"DISEIL B={b}")]["mean"] for b in B]
        base = [a11[(s[0], s[1], f"Best baseline B={b}")]["mean"] for b in B]
        return dis, base

    order = [("Push-T", "state"), ("GridWorld 5x5", "image"), ("Door", "image")]
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 3.3), gridspec_kw=dict(wspace=0.40))
    for ax, s in zip(axes, order):
        dis, base = series(s)
        ax.plot(B, dis, "-o", ms=6, color=BLUE, label="DISEIL")
        ax.plot(B, base, "-s", ms=6, color=MUTED, label="best DAgger-family baseline")
        for b, a_, b_ in zip(B, dis, base):
            ax.annotate("", xy=(b, a_), xytext=(b, b_),
                        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
            ax.text(b * 1.06, (a_ + b_) / 2, f"+{a_ - b_:.1f}", fontsize=9, color=INK,
                    va="center")
        ax.set_xscale("log")
        ax.set_xticks(B)
        ax.set_xticklabels([str(b) for b in B])
        ax.minorticks_off()
        ax.set_xlim(8.8, 50)
        ax.set_xlabel("Budget B (demonstrations)")
        lo = min(base) - 4.0
        hi = max(dis) + 3.0
        ax.set_ylim(lo, min(hi, 102.5))
        ax.set_title(SHORT[s], pad=6)
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Final success rate (%)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=9)
    return save(fig, "F8_budget_sweep")


# ============================ F11 — context set, cluster count, cited episodes (A9/A14/A15)
def fig_context_and_selection():
    a9 = _rmap("A9")
    a14 = _rmap("A14")
    a15 = _rmap("A15")

    panels = [
        ("A9  context-set composition", a9,
         ["Full S (rep + worst-loss + FPS)", "− worst-loss seed", "− FPS (random fill)",
          "− forced representative", "Random 3 from cluster"],
         ["Full S", "− worst-loss seed", "− FPS (random fill)", "− forced rep.",
          "Random 3"]),
        ("A13  cluster-count selection", a14,
         ["Silhouette (adaptive, Eq 8)", "fixed k = 3", "fixed k = 4", "fixed k = 2",
          "fixed k = 5"],
         ["silhouette (adaptive)", "fixed k = 3", "fixed k = 4", "fixed k = 2", "fixed k = 5"]),
        ("A14  number of cited episodes", a15,
         ["Full S — rep + worst-loss + FPS (current)", "Top-3 by peak loss",
          "Top-2 by peak loss", "Random 3 from target cluster"],
         ["Full S (κ = 3)", "Top-3 by loss", "Top-2 by loss", "Random 3"]),
    ]
    # global y-floor so the deepest arm (Door image) is not clipped
    lo_all = min(a[(s[0], s[1], arm)]["mean"]
                 for _, a, arms, _ in panels for arm in arms for s in PRIMARY)

    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 4.2), gridspec_kw=dict(wspace=0.34))
    for ax, (title, rows, arms, labels) in zip(axes, panels):
        n = len(arms)
        w = 0.26
        x = np.arange(n)
        for i, s in enumerate(PRIMARY):
            cells = [rows[(s[0], s[1], a)] for a in arms]
            means = [c["mean"] for c in cells]
            ses = [(np.nan if c.get("se") is None else c["se"]) for c in cells]
            ax.bar(x + (i - 1) * w, means, width=w * 0.92,
                   color=SETTING_COL[s], edgecolor="white", linewidth=0.8, zorder=3,
                   label=SHORT[s])
            ax.errorbar(x + (i - 1) * w, means, yerr=ses, fmt="none", ecolor=INK,
                        elinewidth=0.8, capsize=2, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=38, ha="right", rotation_mode="anchor")
        ax.set_ylim(np.floor(lo_all - 2), 105)
        ax.set_title(title, fontsize=9.5, pad=6)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.axvline(0.5, color="#CCCCCC", lw=0.8, ls=":")
    axes[0].set_ylabel("Final success rate (%), mean ± 1 s.e.")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3)
    return save(fig, "F11_context_and_selection")


# ================================================== F12 — chosen cluster count (D2)
def fig_kstar():
    d2 = _rmap("D2")
    rows = {}
    for s in PRIMARY:
        ks = [d2[(s[0], s[1], f"k={k}")]["mean"] for k in range(2, 7)]
        rejected = d2[(s[0], s[1], "N<=3_rejected")]["mean"]
        tot = sum(ks) + rejected          # clustered rounds + skipped rounds
        rows[s] = (ks, rejected, tot)

    # narrower canvas: the trailing round counts otherwise push the tight bounding
    # box past the text block, and the figure would be scaled down on the page
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    y = np.arange(len(PRIMARY))[::-1]
    cols = SEQ[:5]
    for s, yi in zip(PRIMARY, y):
        ks, skipped, tot = rows[s]
        left = 0.0
        for k, (v, c) in enumerate(zip(ks, cols), start=2):
            pct = 100 * v / tot
            ax.barh(yi, pct, height=0.6, left=left, color=c, edgecolor="white",
                    linewidth=1.2, zorder=3)
            if pct > 6:
                ax.text(left + pct / 2, yi, f"{k}", ha="center", va="center", fontsize=9,
                        color="white" if k <= 3 else INK)
            left += pct
        pct = 100 * skipped / tot
        ax.barh(yi, pct, height=0.6, left=left, color="#E8E8E8", edgecolor="white",
                linewidth=1.2, hatch="///", zorder=3)
        ax.text(left + pct / 2, yi, f"{pct:.0f}%", ha="center", va="center", fontsize=9,
                color=INK)
        ax.text(102, yi, f"{int(tot)} rounds", fontsize=9, color=MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[s] for s in PRIMARY])
    ax.set_ylim(-0.6, len(PRIMARY) - 0.4)
    ax.set_xlim(0, 114)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Share of all rounds (%)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(fc=c, label=f"k = {k}") for k, c in zip(range(2, 7), cols)]
              + [Patch(fc="#E8E8E8", hatch="///", ec="white",
                       label="clustering skipped (N ≤ 3)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.72), ncol=6, fontsize=9)
    return save(fig, "F12_cluster_count_distribution")


# ================================================== F13 — failures over the budget (D4)
def fig_failure_count():
    d4 = load("D4_FailureCount")
    d4 = sorted(d4, key=lambda r: int(r["method"].split("_")[1]))
    rounds = [int(r["method"].split("_")[1]) for r in d4]
    N = [r["mean"] for r in d4]
    skipped = [bool(r.get("skipped")) for r in d4]

    fig, ax = plt.subplots(figsize=(FIGW, 3.4))
    ax.axhspan(0, 3.5, color="#F2F2F2", zorder=1)
    ax.plot(rounds, N, "-o", ms=5, color=BLUE, zorder=3)
    for r, n, sk in zip(rounds, N, skipped):
        if sk:
            ax.plot(r, n, "o", ms=9, mfc="none", mec=INK, mew=1.4, zorder=4)
    ax.set_xticks(range(2, 21, 2))
    ax.set_xlim(0.4, 20.6)
    ax.set_ylim(0, 46)
    ax.set_xlabel("Round r  (budget B = 20)")
    ax.set_ylabel("Mean number of recorded failures N")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ms=5, label="mean recorded failures"),
        Line2D([], [], color=INK, ls="", marker="o", ms=9, mfc="none", mew=1.4,
               label="clustering sweep skipped (N ≤ 3)"),
    ], loc="upper right", fontsize=9)
    return save(fig, "F13_failures_over_budget")


if __name__ == "__main__":
    print("source of truth: ablation_se.json (new run, DISEIL, SE = std/sqrt(n))")
    made = [
        fig_allocation_ladder(), fig_gain_vs_success(), fig_knockout_heatmap(),
        fig_a4_a5_dotplot(), fig_kag_fallback(), fig_bridging(), fig_descriptor_dims(),
        fig_budget(), fig_context_and_selection(),
        fig_kstar(), fig_failure_count(),
    ]
    print(f"\n{len(made)} figures written to {OUT}")
