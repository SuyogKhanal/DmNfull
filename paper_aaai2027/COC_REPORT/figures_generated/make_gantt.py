#!/usr/bin/env python
"""
Candidature Gantt chart for the CoC report.

Covers 13 November 2025 (candidature start) to 30 November 2028 (thesis
submission) at month resolution. Every bar and every milestone in this script
is the same object, with the same date, as the corresponding row of the
milestone table in section "Project plan and Gantt chart" of the report; the
two are kept consistent by hand and any disagreement is a defect.

Style follows figures_generated/make_figures.py: Okabe-Ito colourblind-safe
palette, DejaVu Sans, TrueType embedding.

Outputs: gantt_chart.pdf (vector) and gantt_chart.png (preview).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
SKY, PURPLE = "#56B4E9", "#CC79A7"
INK, MUTED, GREY = "#222222", "#666666", "#BBBBBB"

START = date(2025, 11, 13)
END = date(2028, 11, 30)

# Workstreams. Colour is the workstream; hatch marks a low-intensity continuing bar.
STREAMS = {
    "found": (BLUE, "Foundational work"),
    "aim1": (VERM, "Aim 1 (DISEIL)"),
    "aim2": (GREEN, "Aim 2 (reverse vision-language-action)"),
    "aim3": (PURPLE, "Aim 3 (demonstration demand)"),
    "thesis": (ORANGE, "Thesis and candidature"),
}

# (label, start, end, stream, low_intensity)
BARS = [
    ("Literature review (foundational)",            date(2025, 11, 13), date(2026, 4, 30),  "found", False),
    ("Literature review (continuing)",              date(2026, 5, 1),   date(2028, 9, 30),  "found", True),
    ("Problem formulation and research questions",  date(2025, 12, 1),  date(2026, 3, 31),  "found", False),
    ("HDR coursework and compulsory training",      date(2025, 11, 13), date(2026, 6, 30),  "found", False),

    ("Aim 1: methodology development",              date(2026, 1, 1),   date(2026, 4, 30),  "aim1", False),
    ("Aim 1: implementation (10 settings, 6 baselines)", date(2026, 2, 1), date(2026, 6, 15), "aim1", False),
    ("Aim 1: experimentation and ablations",        date(2026, 4, 1),   date(2026, 7, 5),   "aim1", False),
    ("Aim 1: evaluation and statistical analysis",  date(2026, 6, 1),   date(2026, 7, 15),  "aim1", False),
    ("Aim 1: paper writing (AAAI 2027)",            date(2026, 5, 15),  date(2026, 7, 31),  "aim1", False),
    ("AAAI 2027 review, author response, revisions", date(2026, 8, 1),  date(2027, 2, 28),  "aim1", True),

    ("Confirmation of Candidature report",          date(2026, 6, 1),   date(2026, 8, 13),  "thesis", False),

    ("Aim 2: problem formulation and methodology",  date(2026, 9, 1),   date(2026, 12, 31), "aim2", False),
    ("Aim 2: implementation (captioner, coverage memory)", date(2026, 11, 1), date(2027, 3, 31), "aim2", False),
    ("Aim 2: experimentation (matched-information ablation)", date(2027, 2, 1), date(2027, 5, 10), "aim2", False),
    ("Aim 2: evaluation and analysis",              date(2027, 4, 1),   date(2027, 5, 20),  "aim2", False),
    ("Aim 2: paper writing (CoRL 2027)",            date(2027, 3, 15),  date(2027, 5, 31),  "aim2", False),
    ("CoRL 2027 review, rebuttal, camera-ready",    date(2027, 6, 1),   date(2027, 10, 15), "aim2", True),

    ("Aim 3: problem formulation and methodology",  date(2027, 10, 1),  date(2028, 1, 31),  "aim3", False),
    ("Aim 3: human-research ethics application",    date(2027, 11, 1),  date(2028, 3, 15),  "aim3", False),
    ("Aim 3: implementation (skill inventory, demand model)", date(2027, 12, 1), date(2028, 4, 15), "aim3", False),
    ("Aim 3: experimentation (scripted teachers)",  date(2028, 2, 1),   date(2028, 5, 10),  "aim3", False),
    ("Aim 3: human teaching study",                 date(2028, 3, 20),  date(2028, 8, 31),  "aim3", False),
    ("Aim 3: paper writing (CoRL 2028)",            date(2028, 3, 15),  date(2028, 5, 31),  "aim3", False),
    ("CoRL 2028 review, rebuttal, camera-ready",    date(2028, 6, 1),   date(2028, 10, 15), "aim3", True),

    ("Thesis writing (chapters from CoC and papers)", date(2027, 11, 1), date(2028, 8, 31), "thesis", False),
    ("Thesis integration and full draft",           date(2028, 6, 1),   date(2028, 9, 30),  "thesis", False),
    ("Pre-submission review and revision",          date(2028, 9, 1),   date(2028, 11, 15), "thesis", False),
]

# (label, date, done)  -- done = already achieved at the time of writing (13 Jul 2026)
MILESTONES = [
    ("M1 Candidature start",              date(2025, 11, 13), True),
    ("M2 Aim-1 framework specified",      date(2026, 4, 30),  True),
    ("M3 Experimental matrix complete",   date(2026, 6, 30),  True),
    ("M4 AAAI-27 submission",             date(2026, 7, 31),  True),
    ("M5 Confirmation of Candidature",    date(2026, 8, 13),  False),
    ("M6 Aim 1 complete",                 date(2027, 2, 28),  False),
    ("M7 Mid-candidature review",         date(2027, 5, 14),  False),
    ("M8 CoRL-27 submission",             date(2027, 5, 31),  False),
    ("M9 Aim 2 complete",                 date(2027, 11, 15), False),
    ("M10 Ethics approval (target)",      date(2028, 3, 15),  False),
    ("M11 CoRL-28 submission",            date(2028, 5, 31),  False),
    ("M12 Full thesis draft",             date(2028, 9, 30),  False),
    ("M13 Aim 3 complete",                date(2028, 10, 15), False),
    ("M14 Thesis submission",             date(2028, 11, 30), False),
]

n = len(BARS)
MLANE = -1.9          # y of the milestone lane, beneath the bars
fig, ax = plt.subplots(figsize=(13.6, 0.345 * n + 2.6))
fig.subplots_adjust(left=0.30, right=0.985, top=0.94, bottom=0.14)

# Year shading, alternating, drawn behind everything.
year_spans = [
    (date(2025, 11, 13), date(2026, 11, 12), "Year 1"),
    (date(2026, 11, 13), date(2027, 11, 12), "Year 2"),
    (date(2027, 11, 13), date(2028, 11, 30), "Year 3"),
]
for i, (a, b, name) in enumerate(year_spans):
    ax.axvspan(a, b, color="#F3F5F8" if i % 2 == 0 else "#FFFFFF", zorder=0)
    ax.text(a + (b - a) / 2, n + 0.15, name, ha="center", va="bottom",
            fontsize=9.5, color=MUTED, zorder=5)

# Quarter gridlines.
q = date(2026, 1, 1)
while q <= END:
    ax.axvline(q, color="#E1E5EA", lw=0.6, zorder=1)
    m = q.month + 3
    q = date(q.year + (m - 1) // 12, (m - 1) % 12 + 1, 1)

for i, (label, s, e, stream, low) in enumerate(BARS):
    y = n - 1 - i
    colour = STREAMS[stream][0]
    ax.barh(
        y, (e - s).days, left=s, height=0.62,
        color=colour if not low else "white",
        edgecolor=colour,
        linewidth=0.9,
        hatch="///" if low else None,
        alpha=0.92 if not low else 1.0,
        zorder=3,
    )

# Milestone diamonds on a dedicated lane, keyed by number to the milestone table.
ax.axhline(MLANE, color=GREY, lw=0.6, zorder=2)
for k, (label, d, done) in enumerate(MILESTONES):
    ax.plot(d, MLANE, marker="D", markersize=7.5,
            color=INK if done else "white",
            markeredgecolor=INK, markeredgewidth=1.0, zorder=6)
    ax.plot([d, d], [MLANE, n - 0.6], color=INK, lw=0.5, ls=":", alpha=0.28, zorder=2)
    tag = label.split(" ")[0]
    dy = 0.60 if k % 2 == 0 else -0.95
    ax.text(d, MLANE + dy, tag, ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=7.8, color=INK, zorder=6)

ax.set_ylim(MLANE - 2.9, n + 0.9)
ax.set_xlim(date(2025, 10, 25), date(2028, 12, 20))
ax.set_yticks(range(n))
ax.set_yticklabels([b[0] for b in BARS][::-1], fontsize=8.4, color=INK)
ax.tick_params(axis="y", length=0, pad=4)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.tick_params(axis="x", which="minor", length=2, color=GREY)
ax.tick_params(axis="x", which="major", length=4)
ax.spines["bottom"].set_color(MUTED)

# The Confirmation of Candidature rule separates completed work from planned work.
coc = date(2026, 8, 13)
ax.axvline(coc, color=INK, lw=1.4, zorder=5)
ax.text(coc, MLANE - 2.55, "Confirmation of Candidature, 13 Aug 2026",
        ha="center", va="bottom", fontsize=8.2, color=INK, zorder=7,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=INK, lw=0.7))

handles = [Patch(facecolor=c, edgecolor=c, label=l) for c, l in STREAMS.values()]
handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch="///",
                     label="Low-intensity or venue-scheduled activity"))
handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor=INK,
                      markeredgecolor=INK, markersize=7,
                      label="Milestone achieved (see milestone table)"))
handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
                      markeredgecolor=INK, markersize=7, label="Milestone planned"))
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.075),
          ncol=4, handlelength=1.6, columnspacing=1.4, borderpad=0.2)

fig.savefig(OUT / "gantt_chart.pdf")
fig.savefig(OUT / "gantt_chart.png", dpi=200)
print("wrote", OUT / "gantt_chart.pdf")
