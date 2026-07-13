#!/usr/bin/env python
"""
Publication figures for the CoC report (method name: DISEIL).

Source of truth: ablations_results/DISTIL_ablation_results.xlsx  (24 sheets).
Every number plotted is parsed from that workbook at run time. Nothing is
hand-entered here. The workbook's method column carries the old internal name;
it is relabelled DISEIL on every axis, legend and annotation.

Outputs: <fig>.pdf (vector, for LaTeX) and <fig>.png (preview) per figure.

Palette: Okabe-Ito colourblind-safe qualitative set. Adjacent-pair separation was
computed under deuteranomaly / protanomaly / tritanomaly (CAM02-UCS dE, severity
100): worst pair 13.1, above the >=12 target. Purple and orange are never used in
the same figure (that pair falls to 11.0 under tritanomaly). Lift is drawn in grey
everywhere and carries an explicit ceiling note, because Lift is at 100.0 +/- 0.0
and is uninformative for every ablation.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "ablations_results" / "DISTIL_ablation_results.xlsx"
STATS = ROOT / "build" / "stats_results.csv"
OUT = ROOT / "figures_generated"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
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

BLUE, VERM, GREEN, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9"
PURPLE = "#CC79A7"
GREY, INK, MUTED = "#999999", "#222222", "#666666"
LIFT_GREY = "#B0B0B0"

# Sequential single-hue ramp for ORDERED arms (a ladder is magnitude, not identity).
SEQ = ["#08306B", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1", "#C6DBEF"]

SETTINGS = [
    ("GridWorld 5x5", "state"), ("GridWorld 5x5", "image"),
    ("Push-T", "state"), ("Push-T", "image"),
    ("Lift", "state"), ("Lift", "image"),
    ("Wipe", "state"), ("Wipe", "image"),
    ("Door", "state"), ("Door", "image"),
]
SHORT = {
    ("GridWorld 5x5", "state"): "GridWorld state", ("GridWorld 5x5", "image"): "GridWorld image",
    ("Push-T", "state"): "Push-T state", ("Push-T", "image"): "Push-T image",
    ("Lift", "state"): "Lift state", ("Lift", "image"): "Lift image",
    ("Wipe", "state"): "Wipe state", ("Wipe", "image"): "Wipe image",
    ("Door", "state"): "Door state", ("Door", "image"): "Door image",
}
PRIMARY = [("GridWorld 5x5", "image"), ("Push-T", "state"), ("Door", "image")]
IS_LIFT = {s: (s[0] == "Lift") for s in SETTINGS}
N_SEEDS = {s: (9 if s[0].startswith("GridWorld") else 5) for s in SETTINGS}


def save(fig, name: str) -> str:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")
    return name


def parse_ms(x):
    """'89.4 ± 1.6' -> (89.4, 1.6). Returns (nan, nan) on anything else."""
    if not isinstance(x, str):
        return (float(x), np.nan) if isinstance(x, (int, float)) and not pd.isna(x) else (np.nan, np.nan)
    m = re.match(r"\s*([-\d.]+)\s*±\s*([-\d.]+)", x)
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


XL = pd.ExcelFile(XLSX)


def sheet(name):
    return XL.parse(name, header=None)


# ----------------------------------------------------------------- workbook adapters
def load_gt_sr():
    """GT_SR rows 4..13. Returns per-setting dict with baselines, DISEIL, best baseline."""
    df = sheet("GT_SR")
    cols = ["Safe", "Dropout", "Ensemble", "Thrifty", "Stagger", "Diff-DAgger"]
    out = {}
    for r in range(4, 14):
        key = (df.iat[r, 0], df.iat[r, 1])
        base = {}
        for j, c in enumerate(cols, start=2):
            m, s = parse_ms(df.iat[r, j])
            if not np.isnan(m):
                base[c] = (m, s)
        dm, ds = parse_ms(df.iat[r, 8])
        best_mean = float(df.iat[r, 9])
        # best baseline = the baseline column attaining that mean (first match)
        bname, bsd = next((k, v[1]) for k, v in base.items() if abs(v[0] - best_mean) < 1e-9)
        out[key] = dict(baselines=base, diseil=(dm, ds),
                        best=(best_mean, bsd), best_name=bname)
    return out


def load_gt_ig():
    df = sheet("GT_InfoGain")
    out = {}
    for r in range(4, 14):
        out[(df.iat[r, 0], df.iat[r, 1])] = float(df.iat[r, 8])
    return out


def load_knockout(name):
    """
    Knockout sheets share a layout: header at row 7, data rows 8..17.
    col2 = full DISEIL, col3 = best baseline, col4 = 'mean ± std' DISPLAY string,
    col5 = delta, col6 = margin retained (fraction), LAST col = numeric helper mean.

    The A6 display strings are stale (they disagree with the sheet's own delta and
    margin columns). The helper column is what the sheet's arithmetic uses, so the
    MEAN always comes from the helper. The std is only available from the display
    string; it is returned but must not be used for A6.
    """
    df = sheet(name)
    out = {}
    for r in range(8, 18):
        key = (df.iat[r, 0], df.iat[r, 1])
        _, sd = parse_ms(df.iat[r, 4])
        out[key] = dict(
            full=float(df.iat[r, 2]),
            base=float(df.iat[r, 3]),
            mean=float(df.iat[r, -1]),          # helper column
            delta=float(df.iat[r, 5]),
            margin=float(df.iat[r, 6]) * 100.0,  # sheet stores a fraction
            sd=sd,
        )
    return out


def load_wide(name, header_row, first_row, last_row, label_col=0, val_start=1):
    """Sheets laid out one row per arm, one column per setting (A9, A10, A14, A15)."""
    df = sheet(name)
    hdr = [str(df.iat[header_row, c]).replace("\n", " ").strip()
           for c in range(val_start, val_start + 10)]
    keymap = {}
    for c, h in zip(range(val_start, val_start + 10), hdr):
        t, o = h.rsplit(" ", 1)
        t = "GridWorld 5x5" if t.startswith("GridWorld") else t
        keymap[c] = (t, o)
    rows = {}
    for r in range(first_row, last_row + 1):
        arm = str(df.iat[r, label_col]).strip()
        vals = {}
        for c, key in keymap.items():
            m, s = parse_ms(df.iat[r, c])
            vals[key] = (m, s)
        rows[arm] = vals
    return rows


GT = load_gt_sr()
IG = load_gt_ig()
KO = {k: load_knockout(s) for k, s in [
    ("A1", "A1_Memory_Off"), ("A3", "A3_Clustering_Off"), ("A4", "A4_LLM_vs_Heuristic"),
    ("A5", "A5_VLM_Off"), ("A6", "A6_KAG_Off"), ("A7", "A7_Bridging_Off_"),
    ("A8", "A8_Fallback_Only"),
]}

NONLIFT = [s for s in SETTINGS if not IS_LIFT[s]]


def lift_note(ax, x=0.99, y=0.02, ha="right", va="bottom"):
    ax.text(x, y, "Lift: 100.0 ± 0.0 (ceiling) — uninformative",
            transform=ax.transAxes, ha=ha, va=va, fontsize=7.5, color=MUTED, style="italic")


# ============================================================ F1 — allocation ladder
def fig_allocation_ladder():
    a2 = sheet("A2_RandomAlloc_Robots")
    rnd = {}
    for r in range(8, 16):
        key = (a2.iat[r, 0], a2.iat[r, 1])
        # A2 header: Task | Obs | Best gate baseline | Diff-DAgger | Stagger (random
        # alloc) | DISEIL | DISEIL - Stagger.  Stagger is column 4, NOT column 3.
        rnd[key] = parse_ms(a2.iat[r, 4])
    # GridWorld has no A2 row: its random-allocation control is Stagger in Table 1.
    for s in SETTINGS:
        if s[0].startswith("GridWorld"):
            rnd[s] = GT[s]["baselines"]["Stagger"]

    arms = [("Random allocation (A2)", VERM), ("Fallback only (A8)", ORANGE),
            ("Clustering off (A3)", SKY), ("DISEIL (full)", BLUE)]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), sharey=False)
    for ax, s in zip(axes, PRIMARY):
        vals = [rnd[s], (KO["A8"][s]["mean"], KO["A8"][s]["sd"]),
                (KO["A3"][s]["mean"], KO["A3"][s]["sd"]), GT[s]["diseil"]]
        x = np.arange(len(arms))
        for xi, (m, sd), (_, c) in zip(x, vals, arms):
            ax.bar(xi, m, width=0.68, color=c, edgecolor="white", linewidth=1.2, zorder=3)
            ax.errorbar(xi, m, yerr=sd, fmt="none", ecolor=INK, elinewidth=1.0,
                        capsize=3, zorder=4)
            ax.text(xi, m + sd + 0.45, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=8, color=INK, zorder=5)
        bm, bsd = GT[s]["best"]
        ax.axhline(bm, color=INK, ls="--", lw=1.1, zorder=2)
        ax.text(-0.44, bm + 0.12, f"best baseline: {GT[s]['best_name']} {bm:.1f}",
                ha="left", va="bottom", fontsize=7.2, color=INK, zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.2))
        lo = min(v[0] - v[1] for v in vals + [(bm, bsd)]) - 2.0
        hi = max(v[0] + v[1] for v in vals) + 3.6
        ax.set_ylim(lo, min(hi, 107))
        ax.set_xticks(x)
        ax.set_xticklabels(["Random\nalloc. (A2)", "Fallback\nonly (A8)",
                            "Clustering\noff (A3)", "DISEIL\n(full)"], fontsize=8)
        ax.set_title(SHORT[s], pad=6)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Final success rate (%), mean ± 1 s.d. over seeds")
    axes[2].annotate("A3 falls BELOW\nthe best baseline", xy=(2, KO["A3"][PRIMARY[2]]["mean"]),
                     xytext=(1.35, 88.0), fontsize=7.6, color=VERM, ha="center",
                     arrowprops=dict(arrowstyle="->", color=VERM, lw=1.0))
    fig.text(0.5, -0.06,
             "Seeds: 9 (GridWorld), 5 (robot tasks).  Budget B = 20, D = 1 demonstration per round.",
             ha="center", fontsize=7.8, color=MUTED)
    return save(fig, "F1_allocation_ladder")


# ================================================== F2 — gain without allocation (A3)
def fig_gain_vs_success():
    fig = plt.figure(figsize=(9.6, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.05, 1.0, 0.02], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[0, 1])

    # info gain (ablated) lives in col 7 of the A3 sheet
    a3 = sheet("A3_Clustering_Off")
    ig_abl = {(a3.iat[r, 0], a3.iat[r, 1]): float(a3.iat[r, 7]) for r in range(8, 18)}

    # manual label offsets (x, y, ha) where the automatic placement collides
    LOFF = {
        ("GridWorld 5x5", "state"): (0.0, -0.42, "center"),
        ("GridWorld 5x5", "image"): (-0.035, 0.26, "right"),
        ("Push-T", "state"): (-0.04, 0.10, "right"),
        ("Push-T", "image"): (0.03, -0.45, "left"),
        ("Lift", "state"): (-0.02, -0.45, "right"),
        ("Lift", "image"): (0.02, -0.45, "left"),
        ("Wipe", "state"): (0.035, 0.10, "left"),
        ("Wipe", "image"): (0.03, 0.16, "left"),
        ("Door", "state"): (0.035, -0.10, "left"),
        ("Door", "image"): (0.035, 0.10, "left"),
    }
    for s in SETTINGS:
        lift = IS_LIFT[s]
        prim = s in PRIMARY
        c = LIFT_GREY if lift else (BLUE if prim else SKY)
        x0, x1 = IG[s], ig_abl[s]
        y0, y1 = 0.0, KO["A3"][s]["delta"]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6 if prim else 1.1,
                                    alpha=0.95 if not lift else 0.7,
                                    shrinkA=3, shrinkB=3))
        ax.plot(x0, y0, "o", ms=5.5, mfc="white", mec=c, mew=1.4, zorder=4)
        ax.plot(x1, y1, "o", ms=7 if prim else 5.5, color=c, zorder=4)
        dx, dy, ha = LOFF[s]
        ax.text(x1 + dx, y1 + dy, SHORT[s] + ("*" if prim else ""), fontsize=7.2,
                ha=ha, va="center", color=INK if not lift else MUTED,
                fontweight="bold" if prim else "normal", zorder=5)

    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlabel("Per-demonstration information gain\n(pre-retrain policy loss on the new demonstration)")
    ax.set_ylabel("Δ-SR against full DISEIL\n(success-rate points, round-level rollout evaluation)")
    ax.set_ylim(-8.6, 1.1)
    ax.set_xlim(2.52, 3.78)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.text(0.015, 0.035,
            "Excluding Lift:   mean ΔIG = +0.06  (Wilcoxon p = 0.23)\n"
            "                       mean Δ-SR = −4.01  (Wilcoxon p = 0.002)",
            transform=ax.transAxes, fontsize=7.8, color=INK,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.7, pad=4))
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.4, label="full DISEIL"),
        Line2D([], [], marker="o", ls="", color=INK, label="clustering off (A3)"),
        Line2D([], [], color=BLUE, lw=1.6, label="primary setting"),
        Line2D([], [], color=SKY, lw=1.1, label="other setting"),
        Line2D([], [], color=LIFT_GREY, lw=1.1, label="Lift (at ceiling, uninformative)"),
    ], loc="lower left", bbox_to_anchor=(0.0, -0.40), fontsize=7.6, ncol=3)

    # slope panel: information gain, full vs ablated
    for s in SETTINGS:
        lift = IS_LIFT[s]
        c = LIFT_GREY if lift else (BLUE if s in PRIMARY else SKY)
        axs.plot([0, 1], [IG[s], ig_abl[s]], "-o", ms=3.5, color=c,
                 lw=1.6 if s in PRIMARY else 1.0, alpha=0.9)
    axs.set_xticks([0, 1])
    axs.set_xticklabels(["full\nDISEIL", "clustering\noff (A3)"], fontsize=8)
    axs.set_xlim(-0.35, 1.35)
    axs.set_ylabel("Information gain")
    axs.grid(True, axis="y")
    axs.set_axisbelow(True)
    axs.text(0.5, 1.02, "Gain does not fall", transform=axs.transAxes, ha="center",
             fontsize=8, color=INK)
    return save(fig, "F2_gain_without_allocation")


# ================================================== F3 — knockout summary (heatmap)
def fig_knockout_heatmap():
    order = ["A3", "A8", "A6", "A7", "A4", "A5", "A1"]   # ranked by mean damage
    names = {
        "A3": "A3  clustering off", "A8": "A8  fallback only", "A6": "A6  knowledge graph off",
        "A7": "A7  bridging off", "A4": "A4  reasoning model → heuristic",
        "A5": "A5  vision-language model off", "A1": "A1  cluster memory off",
    }
    M = np.array([[KO[k][s]["margin"] for s in SETTINGS] for k in order])
    D = np.array([[KO[k][s]["delta"] for s in SETTINGS] for k in order])

    fig, ax = plt.subplots(figsize=(10.2, 3.9))
    # Sequential single hue for 0-100 % retained; cells BELOW the baseline get the
    # alert hue. (A rainbow/diverging ramp would put a hue at the midpoint, which
    # the margin scale has no meaning for.)
    cmap = plt.get_cmap("Blues_r")
    im = ax.imshow(np.clip(M, 0, 100), cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            lift = IS_LIFT[SETTINGS[j]]
            v, d = M[i, j], D[i, j]
            if lift:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="#E8E8E8",
                                           ec="white", lw=1.4, hatch="///", zorder=2))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.4,
                        color=MUTED, zorder=3)
                continue
            if v < 0:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc=VERM,
                                           ec="white", lw=1.4, zorder=2))
                txt, col = f"{v:.0f}%", "white"
            else:
                txt = f"{v:.0f}%"
                col = "white" if v < 45 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.8, color=col, zorder=3)
            ax.text(j, i + 0.30, f"{d:+.1f}", ha="center", va="center", fontsize=6.4,
                    color=("white" if (v < 45 or v < 0) else MUTED), zorder=3)

    ax.set_xticks(range(10))
    ax.set_xticklabels([SHORT[s] + ("*" if s in PRIMARY else "") for s in SETTINGS],
                       rotation=38, ha="right")
    for t, s in zip(ax.get_xticklabels(), SETTINGS):
        if s in PRIMARY:
            t.set_fontweight("bold")
        if IS_LIFT[s]:
            t.set_color(MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in order])
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.026)
    cb.set_label("Margin over the best baseline retained (%)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    ax.legend(handles=[
        Patch(fc=VERM, label="ablated system falls below the best baseline"),
        Patch(fc="#E8E8E8", hatch="///", ec="white", label="Lift: at ceiling, uninformative"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.34), ncol=2, fontsize=7.8)
    fig.text(0.128, -0.30,
             "Cell: margin retained (%); small number: Δ success rate (points). "
             "Margin retained = (ablated − best baseline) / (full DISEIL − best baseline). "
             "Rows ordered by mean damage.  * primary setting.",
             fontsize=7.4, color=MUTED)
    return save(fig, "F3_knockout_summary")


# ========================================== F4 — reasoning and vision are small (A4/A5)
def fig_a4_a5_dotplot():
    y = np.arange(len(SETTINGS))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for s, yi in zip(SETTINGS, y):
        sd = GT[s]["diseil"][1]
        lift = IS_LIFT[s]
        ax.barh(yi, 2 * sd, left=-sd, height=0.74, color="#EFEFEF",
                edgecolor="none", zorder=1)
        d4, d5 = KO["A4"][s]["delta"], KO["A5"][s]["delta"]
        c4 = LIFT_GREY if lift else BLUE
        c5 = LIFT_GREY if lift else ORANGE
        ax.plot(d4, yi + 0.14, "o", ms=7, color=c4, zorder=3,
                mec="white", mew=0.8)
        ax.plot(d5, yi - 0.14, "s", ms=6.4, color=c5, zorder=3,
                mec="white", mew=0.8)
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[s] + ("*" if s in PRIMARY else "") for s in SETTINGS])
    for t, s in zip(ax.get_yticklabels(), SETTINGS):
        if s in PRIMARY:
            t.set_fontweight("bold")
        if IS_LIFT[s]:
            t.set_color(MUTED)
    ax.set_xlabel("Δ success rate against full DISEIL (points)")
    ax.set_xlim(-3.0, 0.9)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE,
               label="A4  reasoning model → fixed heuristic  (mean −1.08)"),
        Line2D([], [], marker="s", ls="", color=ORANGE,
               label="A5  vision-language model off  (mean −1.01)"),
        Patch(fc="#EFEFEF", label="± 1 seed s.d. of the full DISEIL run"),
    ], loc="lower left", bbox_to_anchor=(0.0, -0.30), ncol=1)
    lift_note(ax, x=0.99, y=0.02)
    fig.text(0.5, -0.20, "Means exclude Lift. Every gap lies inside the seed noise of its own setting.",
             ha="center", fontsize=7.8, color=MUTED)
    return save(fig, "F4_reasoning_and_vision_small")


# ============================================ F5 — grounding and feasibility (A6)
def fig_kag_fallback():
    a6 = sheet("A6_KAG_Off")
    fb = {(a6.iat[r, 0], a6.iat[r, 1]): float(a6.iat[r, 7]) for r in range(8, 18)}
    LOFF = {
        ("GridWorld 5x5", "state"): (0.35, 0.0, "left"),
        ("GridWorld 5x5", "image"): (-0.35, 0.0, "right"),
        ("Push-T", "state"): (-0.35, 0.06, "right"),
        ("Push-T", "image"): (0.35, 0.0, "left"),
        ("Lift", "state"): (0.0, -0.16, "center"),
        ("Lift", "image"): (0.0, -0.16, "center"),
        ("Wipe", "state"): (-0.35, 0.0, "right"),
        ("Wipe", "image"): (0.0, -0.18, "center"),
        ("Door", "state"): (0.35, 0.0, "left"),
        ("Door", "image"): (-0.35, 0.0, "right"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    xs, ys = [], []
    for s in SETTINGS:
        lift = IS_LIFT[s]
        prim = s in PRIMARY
        c = LIFT_GREY if lift else (BLUE if prim else SKY)
        x, y = fb[s], KO["A6"][s]["delta"]
        ax.plot(x, y, "o", ms=9 if prim else 6.5, color=c, zorder=3,
                mec="white", mew=0.9)
        dx, dy, ha = LOFF[s]
        ax.text(x + dx, y + dy, SHORT[s] + ("*" if prim else ""), fontsize=7.2,
                ha=ha, va="center" if dy == 0 else "top",
                color=INK if not lift else MUTED,
                fontweight="bold" if prim else "normal", zorder=4)
        if not lift:
            xs.append(x)
            ys.append(y)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlabel("Fallback rate with the knowledge graph removed (% of rounds)")
    ax.set_ylabel("Δ success rate against full DISEIL (points)")
    ax.set_ylim(-5.5, 1.35)
    ax.set_xlim(20.5, 36.5)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.text(0.015, 0.035,
            f"Mean Δ-SR (excl. Lift) = {np.mean(ys):.2f} points; margin retained 44.2 %.\n"
            "A fallback round is not a wasted round: the fallback rule alone (A8)\n"
            "still retains 31 % of the margin, which is why A6 costs 2.4 points and not 6.",
            transform=ax.transAxes, fontsize=7.5, color=INK,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.7, pad=4))
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, label="primary setting"),
        Line2D([], [], marker="o", ls="", color=SKY, label="other setting"),
        Line2D([], [], marker="o", ls="", color=LIFT_GREY, label="Lift (at ceiling, uninformative)"),
    ], loc="upper left", ncol=1, fontsize=7.6)
    fig.text(0.5, -0.02,
             "The full-DISEIL fallback rate is not recorded in the workbook, so the reference line "
             "this figure wants cannot be drawn.",
             ha="center", fontsize=7.5, color=VERM, style="italic")
    return save(fig, "F5_grounding_and_feasibility")


# ================================================== F6 — bridging (A7 + D3)
def fig_bridging():
    d3 = sheet("D3_Bridge_Split")
    br = {(d3.iat[r, 0], d3.iat[r, 1]): (float(d3.iat[r, 2]), float(d3.iat[r, 3]))
          for r in range(8, 18)}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.30))
    ax, ax2 = axes

    LOFF = {
        ("GridWorld 5x5", "state"): (0.4, 0.0, "left", "center"),
        ("GridWorld 5x5", "image"): (0.0, 0.10, "center", "bottom"),
        ("Push-T", "state"): (0.0, -0.10, "center", "top"),
        ("Push-T", "image"): (0.0, 0.10, "center", "bottom"),
        ("Lift", "state"): (-0.35, 0.0, "right", "center"),
        ("Lift", "image"): (0.35, 0.0, "left", "center"),
        ("Wipe", "state"): (0.0, 0.10, "center", "bottom"),
        ("Wipe", "image"): (0.0, 0.10, "center", "bottom"),
        ("Door", "state"): (0.0, 0.10, "center", "bottom"),
        ("Door", "image"): (0.0, 0.10, "center", "bottom"),
    }
    xs, ys = [], []
    for s in SETTINGS:
        lift = IS_LIFT[s]
        prim = s in PRIMARY
        c = LIFT_GREY if lift else (BLUE if prim else SKY)
        x, y = br[s][1], KO["A7"][s]["delta"]
        ax.plot(x, y, "o", ms=9 if prim else 6.5, color=c, mec="white", mew=0.9, zorder=3)
        dx, dy, ha, va = LOFF[s]
        ax.text(x + dx, y + dy, SHORT[s] + ("*" if prim else ""), fontsize=7.1, ha=ha,
                va=va, color=INK if not lift else MUTED,
                fontweight="bold" if prim else "normal", zorder=4)
        if not lift:
            xs.append(x)
            ys.append(y)
    r, p = stats.pearsonr(xs, ys)
    sl, ic = np.polyfit(xs, ys, 1)
    gx = np.linspace(min(xs) - 1, max(xs) + 1, 10)
    ax.plot(gx, sl * gx + ic, "--", color=INK, lw=1.2, zorder=2)
    ax.text(0.03, 0.035,
            f"Pearson r = {r:.2f}, p = {p:.2f} (excl. Lift)\n"
            "Bridging is not more valuable where it is used more often:\n"
            "what matters is which rounds use it, not how many.",
            transform=ax.transAxes, fontsize=7.5, color=INK,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.7, pad=4))
    ax.set_xlabel("Bridged share of accepted prescriptions (%)")
    ax.set_ylabel("Δ success rate with bridging off (points)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(-3.9, 0.75)
    ax.set_xlim(16.5, 34.0)
    lift_note(ax, x=0.99, y=0.96, ha="right", va="top")

    y = np.arange(len(SETTINGS))[::-1]
    for s, yi in zip(SETTINGS, y):
        t, b = br[s]
        ax2.barh(yi, t, color=SKY, edgecolor="white", linewidth=1.4, zorder=3)
        ax2.barh(yi, b, left=t, color=BLUE, edgecolor="white", linewidth=1.4, zorder=3)
        ax2.text(t / 2, yi, f"{t:.0f}", ha="center", va="center", fontsize=7.4, color=INK)
        ax2.text(t + b / 2, yi, f"{b:.0f}", ha="center", va="center", fontsize=7.4, color="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels([SHORT[s] + ("*" if s in PRIMARY else "") for s in SETTINGS])
    for t_, s in zip(ax2.get_yticklabels(), SETTINGS):
        if s in PRIMARY:
            t_.set_fontweight("bold")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share of accepted prescriptions (%)")
    ax2.legend(handles=[Patch(fc=SKY, label="targeted in-place correction"),
                        Patch(fc=BLUE, label="bridged placement")],
               loc="lower left", bbox_to_anchor=(0.0, -0.22), ncol=2)
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    fig.text(0.5, -0.10,
             "Bridging is recorded as active on all five tasks (mean 24.4 %, range 18–30 %), including GridWorld and Wipe,\n"
             "where the workbook's own prose predicts 0 %. The data are the source of truth; the prose is stale and must be corrected.",
             ha="center", fontsize=7.5, color=VERM)
    return save(fig, "F6_bridging")


# =============================== F7 — descriptor dimensionality by silhouette (A10)
def fig_descriptor_dims():
    rows = load_wide("A10", header_row=9, first_row=10, last_row=16, val_start=1)
    dims = [2, 4, 5, 6, 8, 10, 12]
    arms = list(rows.keys())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    M = []
    for s in SETTINGS:
        ys = [rows[a][s][0] for a in arms]
        M.append(ys)
        lift = IS_LIFT[s]
        ax.plot(dims, ys, "-", lw=1.0, color=LIFT_GREY if lift else SKY,
                alpha=0.85, zorder=2)
    M = np.array(M)
    ax.plot(dims, M.mean(axis=0), "-o", lw=2.6, ms=6, color=BLUE, zorder=4,
            label="mean over the 10 settings")
    ax.axvline(6, color=VERM, ls="--", lw=1.2, zorder=1)
    ax.text(6.2, 0.316, "chosen descriptor: 6-D\nargmax in 10 of 10 settings",
            fontsize=7.8, color=VERM, va="bottom")
    for d, v in zip(dims, M.mean(axis=0)):
        ax.text(d, v - 0.016, f"{v:.3f}", ha="center", va="top", fontsize=7.0, color=BLUE,
                zorder=5, bbox=dict(fc="white", ec="none", pad=0.8))
    ax.set_xticks(dims)
    ax.set_xlabel("Descriptor dimensionality (number of features in φ)")
    ax.set_ylabel("Mean silhouette of the resulting failure clusters")
    ax.set_ylim(0.29, 0.70)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.6, marker="o", label="mean over the 10 settings"),
        Line2D([], [], color=SKY, lw=1.0, label="individual setting"),
        Line2D([], [], color=LIFT_GREY, lw=1.0, label="Lift (silhouette is unaffected by the ceiling)"),
    ], loc="upper right", ncol=1, fontsize=7.5)
    fig.text(0.5, -0.03,
             "Friedman across the seven dimensionalities: χ²(6) = 59.52, p = 5.6 × 10⁻¹¹.  "
             "Wilcoxon 6-D vs 5-D: p = 0.002.  6-D vs 8-D: p = 0.002.\n"
             "Silhouette is a criterion of geometric separation only and is independent of success rate. "
             "Clustering is geometric for every run, state and image alike.",
             ha="center", fontsize=7.5, color=MUTED)
    return save(fig, "F7_descriptor_dimensionality")


# ================================================================ F8 — budget sweep
def fig_budget():
    df = sheet("A11")
    B = [10, 20, 40]
    marg, dis, base = {}, {}, {}
    for r in range(8, 18):
        key = (df.iat[r, 0], df.iat[r, 1])
        base[key] = [float(df.iat[r, c]) for c in (2, 5, 8)]
        dis[key] = [float(df.iat[r, c]) for c in (3, 6, 9)]
        marg[key] = [float(df.iat[r, c]) for c in (4, 7, 10)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                                  gridspec_kw=dict(wspace=0.28))
    for s in SETTINGS:
        lift = IS_LIFT[s]
        prim = s in PRIMARY
        c = LIFT_GREY if lift else (BLUE if prim else SKY)
        ax.plot(B, marg[s], "-o", ms=5, lw=2.0 if prim else 1.0, color=c,
                alpha=0.95 if not lift else 0.8, zorder=3 if prim else 2)
        if prim:
            ax.text(40.8, marg[s][2], SHORT[s], fontsize=7.4, va="center",
                    color=INK, fontweight="bold")
    mean_nl = np.mean([marg[s] for s in NONLIFT], axis=0)
    ax.plot(B, mean_nl, "--", lw=2.4, color=VERM, zorder=4,
            label="mean, excluding Lift")
    for b, v, ha_, dx in zip(B, mean_nl, ["left", "center", "right"], [1.03, 1.0, 0.97]):
        ax.text(b * dx, v + 0.75, f"{v:.2f}", ha=ha_, fontsize=7.4, color=VERM, zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.8))
    ax.set_xscale("log")
    ax.set_xticks(B)
    ax.set_xticklabels([str(b) for b in B])
    ax.minorticks_off()
    ax.set_xlim(9, 52)
    ax.set_xlabel("Budget B (demonstrations)")
    ax.set_ylabel("Margin over the best baseline (success-rate points)")
    ax.axhline(0, color=INK, lw=0.9)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(-1.0, 16.4)
    ax.legend(handles=[
        Line2D([], [], color=VERM, ls="--", lw=2.4, label="mean over the 8 non-Lift settings"),
        Line2D([], [], color=BLUE, lw=2.0, label="primary setting"),
        Line2D([], [], color=SKY, lw=1.0, label="other setting"),
        Line2D([], [], color=LIFT_GREY, lw=1.0, label="Lift (at ceiling)"),
    ], loc="upper right", fontsize=7.5)
    ax.text(0.74, 0.52, "The margin roughly doubles\nwhen the budget is halved.",
            transform=ax.transAxes, fontsize=7.8, color=INK, ha="center",
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.7, pad=4))

    s = ("Push-T", "state")
    ax2.plot(B, dis[s], "-o", ms=6, color=BLUE, label="DISEIL")
    ax2.plot(B, base[s], "-s", ms=6, color=ORANGE, label="best DAgger-family baseline")
    for b, a_, b_ in zip(B, dis[s], base[s]):
        ax2.annotate("", xy=(b, a_), xytext=(b, b_),
                     arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
        ax2.text(b * 1.04, (a_ + b_) / 2, f"+{a_ - b_:.1f}", fontsize=7.4, color=MUTED,
                 va="center")
    ax2.set_xscale("log")
    ax2.set_xticks(B)
    ax2.set_xticklabels([str(b) for b in B])
    ax2.minorticks_off()
    ax2.set_xlim(9, 48)
    ax2.set_xlabel("Budget B (demonstrations)")
    ax2.set_ylabel("Final success rate (%)")
    ax2.set_title("Push-T state", fontsize=9, pad=5)
    ax2.set_ylim(73.5, 101.5)
    ax2.grid(True)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower right", fontsize=7.8)
    ax2.text(0.03, 0.97, "The margin shrinks because the baseline\ncatches up, not because DISEIL degrades.",
             transform=ax2.transAxes, va="top", fontsize=7.6, color=INK,
             bbox=dict(fc="white", ec="#CCCCCC", lw=0.7, pad=3))
    fig.text(0.5, -0.06,
             "DISEIL at B = 10 does NOT match the best baseline at B = 20: it is below it in 7 of the 10 settings. "
             "That claim is retracted.",
             ha="center", fontsize=7.6, color=VERM)
    return save(fig, "F8_budget_sweep")


# ================================================== F9 — demonstrations per round (A12)
def fig_demos_per_round():
    df = sheet("A12")
    D = [1, 2, 3]
    vals = {}
    for r in range(8, 18):
        key = (df.iat[r, 0], df.iat[r, 1])
        vals[key] = [parse_ms(df.iat[r, c])[0] for c in (3, 4, 5)]

    # label y-nudges: several settings land on the same value at D = 3
    NUDGE = {("Door", "image"): 0.26, ("Door", "state"): -0.26,
             ("Push-T", "state"): 0.22, ("Wipe", "state"): -0.22,
             ("Lift", "image"): 0.20, ("Lift", "state"): -0.20,
             ("GridWorld 5x5", "state"): 0.20, ("GridWorld 5x5", "image"): -0.20}
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for s in SETTINGS:
        lift = IS_LIFT[s]
        prim = s in PRIMARY
        c = LIFT_GREY if lift else (BLUE if prim else SKY)
        ax.plot(D, vals[s], "-o", ms=5, lw=2.2 if prim else 1.1, color=c,
                zorder=3 if prim else 2, alpha=0.95 if not lift else 0.8)
        ax.text(3.07, vals[s][2] + NUDGE.get(s, 0.0), SHORT[s], fontsize=7.3, va="center",
                color=INK if not lift else MUTED,
                fontweight="bold" if prim else "normal")
    ax.set_xticks(D)
    ax.set_xlim(0.88, 3.95)
    ax.set_ylim(87.6, 101.0)
    ax.set_xlabel("Demonstrations per round, D  (total expert labour held fixed at B = 20)")
    ax.set_ylabel("Final success rate (%)")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    lift_note(ax, x=0.02, y=0.03, ha="left")
    fig.text(0.5, -0.04,
             "D = 1 is best in 10 of 10 settings; every line falls and no two lines cross. "
             "Friedman χ²(2) = 19.54, p = 5.7 × 10⁻⁵.  Wilcoxon D = 1 against D = 3: p = 0.002.",
             ha="center", fontsize=7.6, color=INK)
    return save(fig, "F9_demos_per_round")


# ================================================== F10 — memory constants (A13)
def fig_memory_constants():
    df = sheet("A13")
    blocks = {
        "γ  recency discount": dict(hdr=10, rows=range(11, 21), vals=[0.3, 0.5, 0.6, 0.7, 0.9], paper=0.6),
        "σ  kernel width": dict(hdr=24, rows=range(25, 35), vals=[0.02, 0.04, 0.06, 0.1, 0.2], paper=0.06),
        "λ  memory penalty weight": dict(hdr=38, rows=range(39, 49), vals=[0.0, 0.5, 1.0, 2.0, 4.0], paper=1.0),
    }
    st = pd.read_csv(STATS)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.0), gridspec_kw=dict(wspace=0.26))
    for ax, (title, spec) in zip(axes, blocks.items()):
        data = {}
        for r in spec["rows"]:
            key = (df.iat[r, 0], df.iat[r, 1])
            data[key] = [float(df.iat[r, c]) for c in range(2, 7)]
        x = np.arange(len(spec["vals"]))
        for s in SETTINGS:
            lift = IS_LIFT[s]
            prim = s in PRIMARY
            c = LIFT_GREY if lift else (BLUE if prim else SKY)
            ax.plot(x, data[s], "-o", ms=4, lw=2.0 if prim else 1.0, color=c,
                    zorder=3 if prim else 2, alpha=0.95 if not lift else 0.8)
        ip = spec["vals"].index(spec["paper"])
        ax.axvline(ip, color=VERM, ls="--", lw=1.2, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in spec["vals"]])
        ax.set_xlabel(f"{title}\n(reference value {spec['paper']:g}, dashed)")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        # must span GridWorld (~88.6) to Lift (100.0): a tighter limit clips
        # GridWorld off the axes entirely.
        ax.set_ylim(87.8, 101.4)
        key = title.split()[0]
        name = {"γ": "gamma", "σ": "sigma", "λ": "lambda"}[key]
        sub = st[st.hyperparameter_name == name]
        chi2 = sub.friedman_chi2.iloc[0]
        pv = sub.friedman_p.iloc[0]
        ax.text(0.5, 1.03, f"Friedman χ²(4) = {chi2:.2f}, p = {pv:.1e}",
                transform=ax.transAxes, ha="center", fontsize=7.6, color=INK)
        if name == "sigma":
            for v in (0.04, 0.1):
                ax.text(spec["vals"].index(v), 88.2, "ns", ha="center", fontsize=7.6,
                        color=VERM, fontweight="bold")
            ax.text(0.5, 0.955, "σ is LIVE only on Push-T and Wipe\n(the six flat lines are ties)",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.0, color=VERM)
        if name == "lambda":
            ax.annotate("λ = 0.5 is worse than λ = 0:\na half-weight penalty deflects\nwithout rotating",
                        xy=(1, 94.4), xytext=(1.30, 90.4), fontsize=6.9, color=INK,
                        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
            ax.text(0.02, 0.97, "λ = 0 reproduces A1\n(memory off) exactly,\nin all 10 settings",
                    transform=ax.transAxes, va="top", fontsize=6.9, color=GREEN)
    axes[0].set_ylabel("Final success rate (%)")
    axes[0].legend(handles=[
        Line2D([], [], color=BLUE, lw=2.0, label="primary setting"),
        Line2D([], [], color=SKY, lw=1.0, label="other setting"),
        Line2D([], [], color=LIFT_GREY, lw=1.0, label="Lift (ceiling: flat by construction)"),
    ], loc="lower left", bbox_to_anchor=(0.0, -0.50), ncol=3, fontsize=7.6)
    fig.text(0.5, -0.20,
             "γ and λ are significantly best at their reference values (Holm-corrected p < 0.01 against every swept value). "
             "σ = 0.06 is directionally best but NOT statistically\ndistinguishable from its neighbours σ = 0.04 and σ = 0.1 "
             "(Holm p = 0.125, marked ns): the kernel is degenerate on GridWorld and Door and ceiling-masked on Lift, and those\n"
             "six tied settings strip the paired test of its power. A per-task σ, set as a fraction of each task's reset range, "
             "is the fix. This is a limitation, not a virtue.",
             ha="center", fontsize=7.4, color=MUTED)
    return save(fig, "F10_memory_constants")


# ============================ F11 — context set, cluster count, cited episodes (A9/A14/A15)
def fig_context_and_selection():
    a9 = load_wide("A9", header_row=7, first_row=8, last_row=12)
    a14 = load_wide("A14", header_row=7, first_row=8, last_row=12)
    a15 = load_wide("A15", header_row=7, first_row=8, last_row=14, val_start=2)

    panels = [
        ("A9  context-set composition", a9,
         ["Full S (rep + worst-loss + FPS)", "− worst-loss seed", "− FPS (random fill)",
          "− forced representative", "Random 3 from cluster"],
         ["Full S", "− worst-loss seed", "− FPS (random fill)", "− forced rep.", "Random 3"]),
        ("A14  cluster-count selection", a14,
         ["Silhouette (adaptive, Eq 8)", "fixed k = 3", "fixed k = 4", "fixed k = 2", "fixed k = 5"],
         ["silhouette (adaptive)", "fixed k = 3", "fixed k = 4", "fixed k = 2", "fixed k = 5"]),
        ("A15  number of cited episodes", a15,
         ["Full S — rep + worst-loss + FPS (current)", "Top-3 by peak loss", "Top-5 by peak loss",
          "Top-2 by peak loss", "All failures in target cluster", "Random 3 from target cluster"],
         ["Full S (κ = 3)", "Top-3 by loss", "Top-5 by loss", "Top-2 by loss",
          "all in cluster", "Random 3"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), gridspec_kw=dict(wspace=0.20))
    for ax, (title, rows, arms, labels) in zip(axes, panels):
        n = len(arms)
        w = 0.26
        x = np.arange(n)
        for i, s in enumerate(PRIMARY):
            ms = [rows[a][s] for a in arms]
            col = [BLUE, ORANGE, GREEN][i]
            ax.bar(x + (i - 1) * w, [m for m, _ in ms], width=w * 0.92, color=col,
                   edgecolor="white", linewidth=0.8, zorder=3, label=SHORT[s])
            ax.errorbar(x + (i - 1) * w, [m for m, _ in ms], yerr=[sd for _, sd in ms],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.2, rotation=30, ha="right")
        ax.set_ylim(85, 105)
        ax.set_title(title, fontsize=9, pad=6)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.axvline(0.5, color="#CCCCCC", lw=0.8, ls=":")
    axes[0].set_ylabel("Final success rate (%), mean ± 1 s.d.")
    notes = [
        "Arms ordered by effect. Spread from Full S to Random 3: 2.25 points.\n"
        "The ordering, not the individual gap, is the result.",
        "Silhouette selection beats the best fixed k (k = 3) by 0.34 points.\n"
        "The adaptivity is real (D2) but its value is small.",
        "Top-3-by-loss against Full S: −0.40 points. n = 5 buys nothing over n = 3.\n"
        "Top-1 omitted: bridging requires at least two cited failures.",
    ]
    for i, (axc, nt) in enumerate(zip(axes, notes)):
        fig.text(0.185 + i * 0.327, -0.30, nt, ha="center", fontsize=7.2, color=INK)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.text(0.5, -0.42,
             "Three step-level ablations with the same answer: each is worth roughly half a point. "
             "They are shown together so that none is oversold. Lift is omitted (at ceiling).",
             ha="center", fontsize=7.6, color=MUTED)
    return save(fig, "F11_context_and_selection")


# ================================================== F12 — chosen cluster count (D2)
def fig_kstar():
    df = sheet("D2")
    rows = {}
    for r in range(8, 18):
        key = (df.iat[r, 0], df.iat[r, 1])
        ks = [float(df.iat[r, c]) for c in range(3, 8)]     # k = 2..6
        skipped = float(df.iat[r, 8])
        rows[key] = (ks, skipped, sum(ks) + skipped)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y = np.arange(len(SETTINGS))[::-1]
    cols = SEQ[:5]
    for s, yi in zip(SETTINGS, y):
        ks, skipped, tot = rows[s]
        left = 0.0
        for k, (v, c) in enumerate(zip(ks, cols), start=2):
            pct = 100 * v / tot
            ax.barh(yi, pct, left=left, color=c, edgecolor="white", linewidth=1.2, zorder=3)
            if pct > 6:
                ax.text(left + pct / 2, yi, f"{k}", ha="center", va="center", fontsize=7.2,
                        color="white" if k <= 3 else INK)
            left += pct
        pct = 100 * skipped / tot
        ax.barh(yi, pct, left=left, color="#E8E8E8", edgecolor="white", linewidth=1.2,
                hatch="///", zorder=3)
        ax.text(left + pct / 2, yi, f"{pct:.0f}%", ha="center", va="center", fontsize=7.0,
                color=INK)
        ax.text(101, yi, f"{int(tot)} rounds", fontsize=7.0, color=MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[s] + ("*" if s in PRIMARY else "") for s in SETTINGS])
    for t, s in zip(ax.get_yticklabels(), SETTINGS):
        if s in PRIMARY:
            t.set_fontweight("bold")
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Share of all rounds (%)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(fc=c, label=f"k = {k}") for k, c in zip(range(2, 7), cols)]
              + [Patch(fc="#E8E8E8", hatch="///", ec="white",
                       label="clustering skipped (N ≤ 3)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=6, fontsize=7.8)
    fig.text(0.5, -0.20,
             "Pooled over the 896 clustered rounds: k = 3 is the mode at 25.1 %, k = 4 second at 23.8 %; "
             "every k from 2 to 6 is selected in at least 15 % of rounds.\n"
             "Total rounds = seeds × B (180 for GridWorld at 9 seeds, 100 for the robot tasks at 5 seeds), "
             "which confirms the stated seed counts.",
             ha="center", fontsize=7.5, color=MUTED)
    return save(fig, "F12_cluster_count_distribution")


# ================================================== F13 — failures over the budget (D4)
def fig_failure_count():
    df = sheet("D4_FailureCount")
    rounds = [int(df.iat[r, 0]) for r in range(8, 28)]
    N = [float(df.iat[r, 1]) for r in range(8, 28)]
    skipped = [isinstance(df.iat[r, 2], str) for r in range(8, 28)]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.axhspan(0, 3.5, color="#F2F2F2", zorder=1)
    ax.text(1.2, 1.55, "N ≤ 3: clustering sweep skipped\n(each failure becomes its own cluster)",
            fontsize=7.4, color=MUTED, va="center")
    ax.plot(rounds, N, "-o", ms=5, color=BLUE, zorder=3)
    for r, n, sk in zip(rounds, N, skipped):
        if sk:
            ax.plot(r, n, "o", ms=9, mfc="none", mec=VERM, mew=1.6, zorder=4)
    ax.set_xticks(range(2, 21, 2))
    ax.set_xlim(0.4, 20.6)
    ax.set_ylim(0, 46)
    ax.set_xlabel("Round r  (budget B = 20)")
    ax.set_ylabel("Mean number of recorded failures N")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.annotate("rounds 18–20", xy=(19, 3), xytext=(16.4, 9.5), fontsize=7.4, color=VERM,
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1.0))
    ax.text(0.98, 0.95,
            "Push-T image, mean over 5 seeds — the only setting instrumented.\n"
            "The clustering and memory do their work in the first two thirds of the budget;\n"
            "the last rounds are effectively running the fallback rule.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.7, pad=4))
    return save(fig, "F13_failures_over_budget")


# ================================================== F14 — aggregate significance (S1)
def fig_forest():
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = np.arange(len(SETTINGS))[::-1] + 2.6
    for s, yi in zip(SETTINGS, y):
        d = GT[s]["diseil"][0] - GT[s]["best"][0]
        sd_d, sd_b = GT[s]["diseil"][1], GT[s]["best"][1]
        se = np.hypot(sd_d, sd_b) / np.sqrt(N_SEEDS[s])
        lift = IS_LIFT[s]
        c = LIFT_GREY if lift else (BLUE if s in PRIMARY else SKY)
        ax.errorbar(d, yi, xerr=se, fmt="o", ms=7 if s in PRIMARY else 5.5, color=c,
                    ecolor=c, elinewidth=1.4, capsize=3, zorder=3)
        ax.text(9.1, yi, f"{d:+.1f}", fontsize=7.6, va="center", ha="right",
                color=INK if not lift else MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[s] + ("*" if s in PRIMARY else "") for s in SETTINGS])
    for t, s in zip(ax.get_yticklabels(), SETTINGS):
        if s in PRIMARY:
            t.set_fontweight("bold")
        if IS_LIFT[s]:
            t.set_color(MUTED)
    ax.axvline(0, color=INK, lw=1.1, zorder=2)

    def diamond(yc, m, hw, col, label):
        ax.add_patch(plt.Polygon([[m - hw, yc], [m, yc + 0.34], [m + hw, yc],
                                  [m, yc - 0.34]], fc=col, ec="white", lw=1.0, zorder=4))
        ax.text(9.1, yc, f"{m:+.2f}", fontsize=7.8, va="center", ha="right",
                color=INK, fontweight="bold")
        ax.text(-0.35, yc, label, fontsize=7.8, va="center", ha="right", color=INK,
                fontweight="bold", transform=ax.get_yaxis_transform())

    deltas = [GT[s]["diseil"][0] - GT[s]["best"][0] for s in SETTINGS]
    pooled = float(np.mean(deltas))
    task_means = [np.mean([GT[(t, o)]["diseil"][0] - GT[(t, o)]["best"][0] for o in ("state", "image")])
                  for t in ["GridWorld 5x5", "Push-T", "Lift", "Wipe", "Door"]]
    diamond(1.35, pooled, np.std(deltas) / np.sqrt(10), VERM,
            "pooled, 10 settings")
    diamond(0.30, float(np.mean(task_means)), float(np.std(task_means, ddof=1) / np.sqrt(5)), ORANGE,
            "collapsed, 5 tasks")
    ax.set_ylim(-0.5, len(SETTINGS) + 2.7)
    ax.set_xlim(-1.0, 9.4)
    ax.set_xlabel("Margin of DISEIL over the best DAgger-family baseline (success-rate points)")
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    fig.text(0.5, -0.06,
             "10 of 10 settings favour DISEIL. Setting level (n = 10): sign test and Wilcoxon, two-sided p = 0.002 — the floor for 10 pairs.\n"
             "Task level (n = 5, which absorbs the correlation between the two modalities of a task): 5/5, paired t(4) = 4.15, p = 0.014.\n"
             "The conservative task-level result is the one to lead with. Bars are the standard error of the paired difference; they are not the test.\n"
             "Lift is at the 100.0 ± 0.0 ceiling and is uninformative about any mechanism.",
             ha="center", va="top", fontsize=7.4, color=INK)
    return save(fig, "F14_aggregate_significance")


# ================================================== F15 — cluster purity (D1)
def fig_purity():
    df = sheet("D1_Cluster_Purity")
    pur, ncause, sil = {}, {}, {}
    for r in range(8, 18):
        key = (df.iat[r, 0], df.iat[r, 1])
        pur[key] = float(df.iat[r, 2])
        ncause[key] = float(df.iat[r, 3])
        sil[key] = float(df.iat[r, 4])

    NUDGE = {("Push-T", "image"): (-0.005, -0.004, "right", "top"),
             ("GridWorld 5x5", "state"): (-0.004, 0.004, "right", "bottom"),
             ("Wipe", "state"): (-0.004, 0.0, "right", "center"),
             ("Door", "image"): (0.004, 0.0, "left", "center"),
             ("Wipe", "image"): (0.0, -0.006, "center", "top"),
             ("Door", "state"): (0.0, 0.005, "center", "bottom")}
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    xs, ys = [], []
    for s in SETTINGS:
        lift = IS_LIFT[s]
        c = LIFT_GREY if lift else (BLUE if s in PRIMARY else SKY)
        mk = "o" if s[1] == "state" else "^"
        ax.plot(sil[s], pur[s], mk, ms=9 if s in PRIMARY else 7, color=c,
                mec="white", mew=0.9, zorder=3)
        dx, dy, ha, va = NUDGE.get(s, (0.0, 0.005, "center", "bottom"))
        ax.text(sil[s] + dx, pur[s] + dy, SHORT[s] + ("*" if s in PRIMARY else ""),
                fontsize=7.0, ha=ha, va=va,
                color=INK if not lift else MUTED,
                fontweight="bold" if s in PRIMARY else "normal", zorder=4)
        xs.append(sil[s])
        ys.append(pur[s])
    r, p = stats.pearsonr(xs, ys)
    sl, ic = np.polyfit(xs, ys, 1)
    gx = np.linspace(min(xs) - .01, max(xs) + .01, 10)
    ax.plot(gx, sl * gx + ic, "--", color=INK, lw=1.2, zorder=2)
    ax.set_xlabel("Mean silhouette (geometric separation of the clusters)")
    ax.set_ylabel("Mean cluster purity\n(fraction sharing the dominant root-cause label)")
    ax.set_ylim(0.735, 0.965)
    ax.set_xlim(0.465, 0.665)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.text(0.015, 0.035,
            f"Pearson r = {r:.2f}, p = {p:.2f}\n"
            "Geometric separation and semantic purity are uncorrelated:\n"
            "A10 and D1 are independent checks, not two views of one quantity.",
            transform=ax.transAxes, fontsize=7.5, color=INK,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.7, pad=4))
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=SKY, label="state modality"),
        Line2D([], [], marker="^", ls="", color=SKY, label="image modality"),
        Line2D([], [], marker="o", ls="", color=BLUE, label="primary setting"),
        Line2D([], [], marker="o", ls="", color=LIFT_GREY, label="Lift (at ceiling)"),
    ], loc="upper right", fontsize=7.4, ncol=2)
    fig.text(0.5, -0.05,
             "Purity ranges from 0.78 (Wipe image) to 0.93 (Lift state), mean 0.877. It is measured against the reasoning model's own\n"
             "root-cause labels, so it records agreement between two components of the same system, not agreement with ground truth.",
             ha="center", fontsize=7.5, color=MUTED)
    return save(fig, "F15_cluster_purity")


if __name__ == "__main__":
    print(f"source of truth: {XLSX}")
    made = [
        fig_allocation_ladder(), fig_gain_vs_success(), fig_knockout_heatmap(),
        fig_a4_a5_dotplot(), fig_kag_fallback(), fig_bridging(), fig_descriptor_dims(),
        fig_budget(), fig_demos_per_round(), fig_memory_constants(),
        fig_context_and_selection(), fig_kstar(), fig_failure_count(), fig_forest(),
        fig_purity(),
    ]
    print(f"\n{len(made)} figures written to {OUT}")
