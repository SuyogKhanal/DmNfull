# CoC update spec — new ablation run, standard error, figure refresh

Binding. The method is **DISEIL** (never DISTIL/PACE/P4). Source of truth for all numbers:
`ablations_results/sheets/*.csv` (the new run). The report is generated from `build/v2/*.md`
by `build/assemble.py`, built by `build_pdf.sh`. Edit the chapter files, never `CoC_Report.md`.

## Data rule (the crux — get this exactly right)
- The CSVs store **mean ± sample standard deviation (std)** for EVERY method (the GT_SR header
  says "mean ± 1 std"; magnitudes confirm it).
- The report must report **mean ± standard error (SE)** for every method:
  **SE = std / sqrt(n)**, with **n = 5 for the robot tasks (Push-T, Lift, Wipe, Door)** and
  **n = 9 for GridWorld 5x5** (toy problem, 9 seeds).
- Round each SE to one decimal place, in percentage points.
- Error bars in every figure are SE (small), never std/variance.

## Table 7 — final held-out success rate, mean ± SE (AUTHORITATIVE; agents must reproduce this)
Add two leading columns per row: **Ni** (initial demonstrations) and **Init SR** (round-0 success
rate). **Ni is per-TASK** (identical for both modalities), but **Init SR is per-MODALITY**: the count
is chosen per task, so the two modalities of a task do not land at the same round-0 success rate.
DISEIL is best per row; bold it. "—" = method not applicable (Diff-DAgger on GridWorld; Stagger on robots).

| Task | Obs | Ni | Init SR | Safe | Dropout | Ensemble | Thrifty | Stagger | Diff-DAgger | DISEIL (ours) |
|---|---|---|---|---|---|---|---|---|---|---|
| GridWorld 5x5 | state | 20 | 48.9 | 85.3±0.9 | 84.9±0.8 | 86.2±0.7 | 86.8±0.7 | 85.7±0.5 | — | **92.4±0.4** |
| GridWorld 5x5 | image | 20 | 47.0 | 88.8±0.9 | 88.4±0.7 | 88.8±0.9 | 88.7±0.6 | 89.1±0.8 | — | **91.3±0.6** |
| Push-T | state | 20 | 46.2 | 82.0±3.0 | 84.8±2.7 | 85.9±2.6 | 83.2±3.2 | — | 94.1±2.0 | **96.1±1.6** |
| Push-T | image | 20 | 43.3 | 78.1±3.5 | 82.1±3.1 | 83.2±3.0 | 79.3±3.6 | — | 89.0±2.1 | **92.6±2.2** |
| Lift | state | 8 | 67.2 | 99.2±0.7 | 99.2±0.4 | 99.2±0.4 | 100.0±0.0 | — | 99.2±0.4 | **100.0±0.0** |
| Lift | image | 8 | 66.4 | 99.6±0.4 | 97.2±1.6 | 98.8±0.7 | 99.6±0.4 | — | 99.6±0.4 | **100.0±0.0** |
| Wipe | state | 12 | 47.7 | 88.0±1.1 | 88.6±1.8 | 86.8±1.9 | 89.0±1.1 | — | 90.4±2.7 | **93.1±1.3** |
| Wipe | image | 12 | 45.2 | 69.6±2.4 | 83.2±3.0 | 84.4±3.2 | 69.2±4.0 | — | 88.6±1.4 | **92.3±1.4** |
| Door | state | 4 | 56.8 | 91.8±2.1 | 92.5±1.2 | 88.8±3.1 | 89.6±1.7 | — | 93.2±1.9 | **96.6±1.9** |
| Door | image | 4 | 43.1 | 82.4±1.4 | 81.8±1.5 | 83.0±4.9 | 82.8±1.2 | — | 84.2±1.6 | **88.6±1.5** |

Caption: "... mean ± standard error over 5 seeds (robot tasks) and 9 seeds (GridWorld). Ni is the
number of initial demonstrations; Init SR is the held-out success rate before any DISEIL round.
Best per row in bold." (Lift is a ceiling; do not add a sentence saying it is excluded from
ablations, but the table itself keeps all tasks.)
The initial demonstration counts target a round-0 success rate band of roughly 45-50%: enough
competence to produce meaningful rollout failures, low enough to leave headroom for the budget to
matter. The achieved values span **43.1 to 67.2%** (see Init SR). Do NOT claim every setting
"exceeds 45%" — Push-T (image) at 43.3 and Door (image) at 43.1 fall just below the band, because
Ni is set per task and the image modality starts harder.

## Table 8 — per-demonstration information gain
New values from `GT_InfoGain.csv`. Diff-DAgger and DISEIL columns ONLY (round-2 decision).
`GT_InfoGain.csv` has means only (no std), so present means; the boxplot figure carries the
spread. Relabel DISTIL -> DISEIL.

## Figures (current numbering in the 75-page build)
| Fig | What | Action |
|---|---|---|
| 2 | Architecture (Architectural_Diagram.pdf) | REMOVE the "Cluster Memory" box and its connecting (dashed) arrow to "Cluster Engine". Back up the original, write the cleaned version to the SAME path. Add ONE sentence in the methodology text that the cluster memory is a configurable, task-specific component (active only when a task has recurring failure clusters), so it is not shown in the framework figure. |
| 5 | 5-task learning curves | REPLACE with `updated_figures/selected_tasks_SE.pdf` -> copy to `../figures/selected_tasks_SE.pdf`. Panels: GridWorld(image), Push-T(state), Lift(state), Door(state), Wipe(image); already "DISEIL (Ours)", mean±SE. Update the caption to say mean±SE and name the five panels. |
| 7 | Confidence vs realised improvement | REPLACE with the new plot `updated_figures/confidence_vs_success (1).pdf` -> copy to `../figures/confidence_vs_success_v2.pdf` (space/paren-free). It is r=0.82, n=180. Update caption; do not put n=152 anywhere. |
| 10 | Grounding & feasibility (F5, the A6 KAG-off figure) | It is MISSING the SE error bar on the A6 bars — ADD it. Use SE from `A6_KAG_Off.csv` (e.g. GridWorld state 91±1.3 std -> SE). All bars use SE (minimal). |
| 8,9,11,12,13,14,15,16,17,18 | other ablation figures | Regenerate from the new CSVs, SE error bars (minimal), DISEIL labels. Do NOT resurrect the figures deleted in round 2 (memory-constants, aggregate-significance, cluster-purity, compute-cost). Keep the three-ablation-setting restriction (GridWorld image, Push-T state, Door image) and the no-orange-text / no-prose-in-figure rules. |

Global figure rules (unchanged): no orange text, no prose/verdict inside artwork, SE not std,
three ablation settings only, no Lift in ablation figures.

## Naming sweep
`DISTIL` still appears in ~13 CSVs, the xlsx, `make_figures.py`, and possibly figure labels.
Replace with DISEIL everywhere it is the METHOD name (CSV column headers, plot legends, captions).
Do not touch code identifiers like `p4_top3_rotate` (those are code, not prose).

## Do not touch
Ablation study numbering (A1–A18) and figure/table numbering are already set by round 2. Refresh
DATA and LABELS only; do not renumber anything.
