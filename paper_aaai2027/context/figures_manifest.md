# Figures manifest for DISTIL (AAAI-2027 draft)

All content below was read directly OFF the staged figures in `draft/figures/` (titles, axis labels, legends verified visually). Page sizes measured with `pdfinfo` / PIL. Method label in every plot legend is exactly **"DISTIL (Ours)"** — consistent with the paper name.

---

## 1. `figures/teaser.png` — Teaser (Introduction, page 1)

**What it shows.** Hand-drawn black-and-white cartoon of the DISTIL loop in five annotated vignettes: (a) a robot arm flails at a cube-on-table task, captioned "policy keeps failing ?!" with ghosted failed attempts marked ×; (b) a small LLM character (labeled "LLM", with a brain icon) at a printing press, captioned "LLM prescribes demos"; (c) a human with a clipboard teleoperates the arm, captioned "expert gives the demonstration"; (d) sheets of paper labeled "new demo added"; (e) arrow to "update the policy" (neural-net icon under the table) and the arm now succeeds — "policy improved". Reads left-to-right, top-to-bottom as: fail → LLM prescribes → expert demonstrates → dataset grows → retrain → success.

**Geometry & sizing.** 1728 × 2448 px, portrait, aspect ≈ 0.71:1 (height = 1.42 × width). At full `\columnwidth` (~3.3 in) it would be ~4.7 in tall — too much page-1 real estate. Recommend **single-column**, slightly narrowed:
```latex
\includegraphics[width=0.9\columnwidth]{figures/teaser.png}
```
Place in the Introduction so it lands on page 1 (top of the right column, or after the first 1–2 paragraphs). Do NOT make it `figure*`.

**Caption draft.**
```latex
\caption{\textbf{DISTIL in a nutshell.} When the learner policy repeatedly fails,
DISTIL does not query the expert on every miss: an LLM diagnoses the failures and
\emph{prescribes} which corrective demonstration the expert should provide. The
expert supplies only that demonstration, it is added to the dataset, and the policy
is retrained---distilling a fixed demonstration budget into the most informative
corrections.}
```

**Label.** `\label{fig:teaser}`

---

## 2. `figures/architecture.pdf` — DISTIL system architecture (Method)

**What it shows.** Full block diagram of one DISTIL round, left to right: *Initial set of Expert demos* → *Train Policy* (MLP icon) → **Policy Rollout** (robot icon) → **Flag Uncertainty** → three key frames of the rollout ("Start", "High Loss t\*", "End", shown as robot renders) feed a **Vision LLM** → **Reasoning LLM** ("root-cause analysis"), grounded by **KAG Grounding** ("geometry facts") → **Cluster Engine** ("k failure modes", with a scatter-cluster inset and a **Cluster Memory** store) → **LLM Prescription** ("prescribe the demo") → **Expert Demo** → **Feasibility Check** with an orange dashed back-edge "infeasible → re-prescribe" returning to LLM Prescription → green "Add demo" into the dataset → **Update Policy** ("on the augmented dataset") → teal "next round" edge back to Policy Rollout. (Note: the Update Policy box text contains a typo "on the the augmented dataset" — worth fixing in the source if the diagram is ever regenerated.)

**Geometry & sizing.** 863 × 318 pt, aspect ≈ 2.71:1 — a wide single-row banner. **MANDATED: two-column span, never cropped or squashed.**
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/architecture.pdf}
  ...
\end{figure*}
```
At `\textwidth` (~7 in) the height is ~2.6 in — comfortable. Do not scale below `\textwidth`; box labels are small.

**Caption draft.**
```latex
\caption{\textbf{The DISTIL loop.} Rollouts of the current policy $f_\theta$ are
monitored for uncertainty; for each flagged failure, a vision LLM summarizes the
start, highest-loss ($t^*$), and end frames, and a reasoning LLM performs
root-cause analysis grounded in geometric facts (KAG). Failures are clustered into
$k$ recurring modes (with cross-round cluster memory), and the LLM prescribes the
single corrective demonstration the expert should provide. A feasibility check
triggers re-prescription when needed; the accepted demo augments the dataset and
the policy is updated for the next round.}
```

**Label.** `\label{fig:arch}`

---

## 3. `figures/comparison_baselines.pdf` — Learning curves vs. baselines (Experiments)

**What it shows.** NOT a bar chart — three side-by-side **learning-curve panels** (mean line + shaded std band per method). Shared axes: x = "Number of demonstrations added" (0–20), y = "Success rate" (0–1). **The three tasks are:**
1. **GridWorld 5×5 (image)** — legend: SafeDAgger, EnsembleDAgger, DropoutDAgger, ThriftyDAgger, Stagger, DISTIL (Ours). Curves start ~0.47 and end ~0.86–0.91; DISTIL (orange) finishes on top (~0.90).
2. **Door (state)** — legend: DiffDAgger, SafeDAgger, EnsembleDAgger, DropoutDAgger, ThriftyDAgger, DISTIL (Ours). Start ~0.55–0.60; DISTIL reaches ~0.98, Diff-DAgger ~0.95.
3. **Wipe (image)** — same legend as Door. Start ~0.33–0.58; largest separation: DISTIL ~0.95 vs Diff-DAgger ~0.89, ThriftyDAgger/SafeDAgger stall ~0.70.
Baseline rosters match the protocol (Stagger only on GridWorld; Diff-DAgger only on robot tasks); endpoints agree with Table A.

**Geometry & sizing.** 877 × 287 pt, aspect ≈ 3.06:1. Three panels at `\columnwidth` would be illegibly small. Recommend **two-column span**:
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/comparison_baselines.pdf}
  ...
\end{figure*}
```
Height at `\textwidth` ≈ 2.3 in.

**Caption draft.**
```latex
\caption{\textbf{Success rate vs.\ demonstrations added} on GridWorld $5{\times}5$
(image), Door (state), and Wipe (image); mean $\pm$ std over seeds (9 for
GridWorld, 5 for robot tasks). Under the same 20-demo budget, DISTIL dominates
throughout training and attains the highest final success rate on all three
tasks, with the largest margin on image-based Wipe, where uncertainty-gated
baselines plateau.}
```

**Label.** `\label{fig:comparison}`

---

## 4. `figures/confidence_vs_success.pdf` — LLM confidence vs. improvement (Analysis)

**What it shows.** Scatter plot, orange dots. x = "LLM Confidence Score" (50%–100%), y = "Δ Success Rate" (−0.2 to ~0.45, dotted zero line). Inset box: **r = 0.86, n = 152**. Clear positive trend: demos prescribed at <60% confidence yield ≈0 or slightly negative ΔSR; ≥90% confidence yields ~+0.3–0.45. Setting: **GridWorld 5×5, IMAGE** (r = 0.86 matches the GridWorld-image cell of the Conf-vs-SR sheet).

**Geometry & sizing.** 536 × 401 pt, aspect ≈ 1.34:1 (≈4:3). **Single-column**:
```latex
\includegraphics[width=\columnwidth]{figures/confidence_vs_success.pdf}
```

**Caption draft.**
```latex
\caption{\textbf{Prescription confidence predicts policy improvement}
(GridWorld $5{\times}5$, image). Each point is one prescribed demonstration; the
LLM's self-reported confidence correlates strongly with the resulting change in
success rate ($r=0.86$, $n=152$). Across all ten task$\times$modality settings,
$r$ ranges from $0.82$ to $0.89$.}
```

**Label.** `\label{fig:conf}`

---

## 5. `figures/info_gain_boxplot.pdf` — Per-demo information gain (Analysis)

**What it shows.** Box-and-whisker plot, one box per method on the x-axis: SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Stagger, DISTIL (Ours) — i.e., the **GridWorld roster ⇒ GridWorld 5×5, IMAGE setting**. **Y-axis label (exact): "pre-finetune policy loss on the chosen/prescribed demo"**, range 0–14, orange median lines, black outlier dots. So "information gain" is operationalized as the current policy's loss on the newly selected demo *before* finetuning on it — higher = the demo carries more information the policy lacks. DISTIL has the highest median (~2.7) and the widest upper spread (outliers to ~13.4); Ensemble/Thrifty are lowest (median ~1.0–1.1). Medians are consistent with Table B's GridWorld-image means (DISTIL 3.21 ± 2.33, best of all methods).

**Geometry & sizing.** 679 × 350 pt, aspect ≈ 1.94:1. **Single-column** works (height ≈ 1.7 in at `\columnwidth`; rotated tick labels stay legible):
```latex
\includegraphics[width=\columnwidth]{figures/info_gain_boxplot.pdf}
```
If tick labels look cramped in the compiled PDF, pair it with `fig:conf` in a two-column `figure*` with two `\includegraphics[width=0.49\textwidth]` side by side.

**Caption draft.**
```latex
\caption{\textbf{Per-demonstration information gain} (GridWorld $5{\times}5$,
image), measured as the policy's pre-finetune loss on each newly acquired
demonstration. DISTIL's prescribed demos carry the most new information
(highest median and upper quartile), indicating the budget is spent on
corrections the current policy cannot yet produce.}
```

**Label.** `\label{fig:infogain}`

---

## 6. `figures/clustering_modes_pusht.pdf` — Discovered failure modes (Method or Analysis)

**What it shows.** 3 × 3 grid of rendered Push-T simulation frames (Panda arm, red T block, gray goal-T outline), **Push-T IMAGE setting**. Rows = the k = 3 failure-mode clusters discovered by the Cluster Engine, each with a colored row label and per-frame metrics printed above each image ("θerr … · contact …"):
- **M0 "not-well-aligned"** (blue): θerr 162°/162°/169°, contact 0.06–0.08 m — T near goal but flipped/misaligned.
- **M1 "no-contact"** (orange): θerr 102°–120°, contact 0.08–0.11 m — end-effector fails to engage the T; block far from goal.
- **M2 "badly-rotated"** (green): θerr 37°–84°, contact 0.15–0.18 m — contact lost at large distance, grossly rotated placements.
Within-row frames are visually and metrically coherent; rows are well separated, i.e., the clusters are semantically meaningful.

**Geometry & sizing.** 648 × 648 pt, aspect 1:1 (square). **Single-column**:
```latex
\includegraphics[width=\columnwidth]{figures/clustering_modes_pusht.pdf}
```
(≈3.3 × 3.3 in; each thumbnail ~1.1 in — adequate. Do not span two columns; a 7-in square would consume half a page.)

**Caption draft.**
```latex
\caption{\textbf{Failure modes discovered on image-based Push-T.} DISTIL's
cluster engine groups flagged rollouts into $k{=}3$ recurring modes---%
\emph{not-well-aligned}, \emph{no-contact}, and \emph{badly-rotated}---shown with
three representative frames each (annotated with orientation error and contact
distance). Coherent, well-separated clusters let the LLM prescribe one targeted
corrective demonstration per mode instead of querying the expert on every failure.}
```

**Label.** `\label{fig:clusters}`

---

## Quick reference

| Figure | File | Aspect (w:h) | Placement | Label |
|---|---|---|---|---|
| Teaser cartoon | `teaser.png` | 0.71:1 (portrait) | `figure`, Intro p.1, `width=0.9\columnwidth` | `fig:teaser` |
| Architecture | `architecture.pdf` | 2.71:1 | **`figure*`, `width=\textwidth` (mandated, no crop/squash)** | `fig:arch` |
| Learning curves (GridWorld-img / Door-state / Wipe-img) | `comparison_baselines.pdf` | 3.06:1 | `figure*`, `width=\textwidth` | `fig:comparison` |
| Confidence vs ΔSR scatter (GridWorld img) | `confidence_vs_success.pdf` | 1.34:1 | `figure`, `width=\columnwidth` | `fig:conf` |
| Info-gain boxplot (GridWorld img); y = pre-finetune policy loss on chosen/prescribed demo | `info_gain_boxplot.pdf` | 1.94:1 | `figure`, `width=\columnwidth` | `fig:infogain` |
| Push-T failure-mode clusters (image) | `clustering_modes_pusht.pdf` | 1:1 | `figure`, `width=\columnwidth` | `fig:clusters` |
