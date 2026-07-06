# AUTHORITATIVE results & facts for the DISTIL paper (supersedes results_placeholders.md — placeholders are DEAD)

Source: `paper_aaai2027/table_data.xlsx` (3 sheets), transcribed exactly. Author-confirmed protocol facts below.
Any number in the paper MUST match this file. No `\PH{}` macros remain anywhere.

## Identity
- **Title:** DISTIL: Demonstration Distillation for Sample-Efficient Imitation Learning
- **Method name:** DISTIL (Ours). Internal code name was `p4_top3_rotate` / "PACE" — NEVER mention either.

## Protocol (author-confirmed)
- **Demonstration budget: 20 expert demos** for ALL tasks (discrete and continuous).
- **Seeds: 9-seeded runs** for the discrete-action task (GridWorld 5×5); **5-seeded runs** for the continuous-action robot tasks (Push-T, Lift, Wipe, Door).
- Policies: GridWorld image = **pure CNN** policy; GridWorld state = **MLP** policy; robot tasks = diffusion policies (state & image). The METHOD is policy-agnostic: write the policy as a generic function (e.g. f_theta), never "our method is for diffusion policies".
- Methods compared: SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Stagger (GridWorld ONLY), Diff-DAgger (robot tasks ONLY), DISTIL (ours).
- Core narrative: expert demonstrations are costly; there is no need to query the expert on every rollout miss. Under a FIXED demo budget, DISTIL distills the budget into the most informative corrective demos and attains the highest success rate in every task × modality cell.

## Table A — Final held-out success rate at the 20-demo budget (mean ± std over seeds)
Format: SR% ± std(pp). (xlsx "demo vs sr", converted ×100.)

| Task | Mode | SafeDAgger | DropoutDAgger | EnsembleDAgger | ThriftyDAgger | Stagger | Diff-DAgger | DISTIL (Ours) |
|---|---|---|---|---|---|---|---|---|
| GridWorld 5×5 | image | 86.1 ± 2.8 | 85.8 ± 2.6 | 85.7 ± 2.2 | 87.1 ± 1.9 | 86.6 ± 2.3 | — | **89.6 ± 1.8** |
| GridWorld 5×5 | state | 85.3 ± 2.7 | 84.9 ± 2.5 | 86.2 ± 2.1 | 86.8 ± 2.0 | 85.7 ± 1.5 | — | **89.9 ± 1.3** |
| Push-T | state | 82.0 ± 6.8 | 84.8 ± 6.1 | 85.9 ± 5.8 | 83.2 ± 7.2 | — | 90.7 ± 4.5 | **96.1 ± 4.5** |
| Push-T | image | 78.1 ± 7.8 | 82.1 ± 6.9 | 83.2 ± 6.6 | 79.3 ± 8.1 | — | 89.0 ± 4.8 | **93.9 ± 4.9** |
| Lift | state | 99.2 ± 1.6 | 99.2 ± 1.0 | 99.2 ± 1.0 | 98.8 ± 2.4 | — | 99.2 ± 1.0 | **100.0 ± 0.0** |
| Lift | image | 99.6 ± 0.8 | 97.2 ± 3.5 | 98.8 ± 1.6 | 99.6 ± 0.8 | — | 99.6 ± 0.8 | **100.0 ± 0.0** |
| Wipe | state | 88.0 ± 2.5 | 89.6 ± 4.1 | 90.8 ± 4.3 | 90.0 ± 2.5 | — | 90.4 ± 6.0 | **95.5 ± 6.0** |
| Wipe | image | 69.6 ± 5.3 | 83.2 ± 6.8 | 84.4 ± 7.1 | 69.2 ± 9.0 | — | 89.6 ± 3.2 | **95.3 ± 3.2** |
| Door | state | 93.2 ± 5.2 | 92.8 ± 2.7 | 88.8 ± 7.0 | 89.6 ± 3.9 | — | 95.2 ± 4.3 | **98.4 ± 4.2** |
| Door | image | 92.4 ± 3.2 | 88.8 ± 3.3 | 86.0 ± 10.9 | 92.8 ± 2.7 | — | 89.2 ± 3.5 | **99.2 ± 3.4** |

DISTIL is best in ALL 10 cells. Headline deltas worth quoting: Push-T state +5.4pp over the best baseline (Diff-DAgger 90.7); Wipe image +5.7pp (89.6→95.3), and +25.7pp over SafeDAgger; Door image +6.4pp; GridWorld image +2.5pp over ThriftyDAgger.

## Table B — Per-demonstration information gain (xlsx "BoxplotSummary"; mean ± std; ~N=168–184 samples/cell)
Units: match the y-axis label of `info_gain_boxplot.pdf` (the figure-analysis agent must read it off the figure).

| Task | Mode | Safe | Dropout | Ensemble | Thrifty | Stagger | DiffDAgger | DISTIL |
|---|---|---|---|---|---|---|---|---|
| GridWorld | image | 2.46±1.61 | 2.53±1.69 | 1.57±1.55 | 1.37±1.12 | 1.88±1.43 | — | **3.21±2.33** |
| GridWorld | state | 2.55±1.68 | 2.95±2.12 | 1.33±1.18 | 1.34±0.93 | 1.83±1.11 | — | **3.55±2.51** |
| Push-T | state | 1.66±0.99 | 2.36±1.63 | 1.11±0.64 | 1.10±0.65 | — | 1.57±1.10 | **2.81±2.09** |
| Push-T | image | 2.04±1.11 | 2.16±1.36 | 1.06±0.64 | 1.20±0.62 | — | 1.80±1.10 | **2.82±1.72** |
| Lift | state | 2.23±1.36 | 2.10±1.25 | 1.12±0.74 | 1.13±0.55 | — | 1.61±1.12 | **2.64±1.65** |
| Lift | image | 2.18±1.40 | 2.17±1.39 | 1.00±0.55 | 1.21±0.64 | — | 1.36±0.85 | **2.93±1.67** |
| Wipe | state | 2.02±0.96 | 2.38±1.55 | 1.23±1.10 | 1.18±0.81 | — | 1.43±0.80 | **2.91±2.02** |
| Wipe | image | 2.50±1.47 | 2.96±2.00 | 1.43±0.89 | 1.52±0.86 | — | 1.95±1.16 | **3.62±2.19** |
| Door | state | 2.53±1.64 | 3.10±2.21 | 1.43±0.82 | 1.42±0.90 | — | 1.84±1.11 | **3.43±2.12** |
| Door | image | 2.35±1.40 | 2.46±1.44 | 1.24±0.76 | 1.26±0.72 | — | 1.58±0.92 | **3.00±1.98** |

DISTIL has the highest mean per-demo information gain in ALL 10 cells. (Note SafeDAgger/DropoutDAgger have higher raw info-gain than Ensemble/Thrifty yet lower final SR — one line of analysis can note info gain is necessary but its *allocation across failure modes* is what converts to SR; do not over-claim.)

## Confidence ↔ success (xlsx "Conf vs SR": Pearson r, DISTIL only)
GridWorld image 0.86, state 0.88 · Push-T state 0.87, image 0.88 · Lift state 0.88, image 0.89 · Wipe state 0.82, image 0.86 · Door state 0.83, image 0.82.
Range to quote: **r = 0.82–0.89 across all ten task×modality settings**.

## Figures (staged in `draft/figures/`, LaTeX-safe names)
| File | Content | Placement rule |
|---|---|---|
| `teaser.png` | Teaser diagram | Intro, within first paragraphs after abstract (page 1, single-column unless clearly too wide) |
| `architecture.pdf` | Full DISTIL loop (rollout→flag→VisionLLM(start/t*/end)→ReasoningLLM+KAG→ClusterEngine(k modes,+memory)→LLM Prescription→Expert demo→Feasibility check(re-prescribe loop)→add demo→update policy) | `figure*` two-column span; do NOT crop or squash to one row |
| `comparison_baselines.pdf` | Bar/side-by-side comparison of the 7 methods on THREE tasks (read figure to confirm which) | Experiments |
| `confidence_vs_success.pdf` | Confidence vs success rate — GridWorld 5×5 IMAGE setting | Experiments/analysis |
| `info_gain_boxplot.pdf` | Per-demo information gain boxplot — GridWorld 5×5 IMAGE setting | Experiments/analysis |
| `clustering_modes_pusht.pdf` | Discovered failure-mode clusters — Push-T IMAGE setting | Method or analysis |

## Framing corrections (author-mandated)
1. Policy is ANY function f_theta (CNN / MLP / diffusion are instantiations). Never "our method is for diffusion policies".
2. The which-failure arm and the where/bridge-demo arm are BOTH integral defaults — the method is a HYBRID; the word "optional(ly)" is banned. "Where to start" is not offered as a choice.
3. No "planned / upcoming / pending / placeholder" language — results are complete.
4. Tables: consolidate — ONE main table (Table A) + at most one secondary table. No ablation table (no data).
5. Compile target: ≤ 9 PDF pages TOTAL including references (~7 body + ~2 refs), zero overfull boxes (equations must not cross the column, tables must not truncate).
