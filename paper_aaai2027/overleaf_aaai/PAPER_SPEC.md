# DISEIL AAAI-2027 paper — binding spec

## What this paper is
A submission-ready AAAI paper on **Aim 1 only: DISEIL**. It is NOT a summary of the CoC report.
It is a new paper: strongest contribution, strongest narrative, AAAI standards.
Aims 2 and 3 of the PhD are **out of scope** (at most one forward-looking clause in the conclusion;
no Reverse-VLA section, no candidature/Gantt/training material, no research-programme framing).

**Title:** DISEIL: Demonstration Distillation for Sample-Efficient Imitation Learning
**Acronym derivation** (one letter per title word), bolded once at first mention in the abstract:
**D**emonstration d**I**stillation for **S**ample-**E**fficient **I**mitation **L**earning.

## Sources of truth (strict precedence)
1. **PRIMARY — the CoC report**: `COC_REPORT/build/v2/*.md`, and the built `COC_REPORT/CoC_Report.pdf`.
   - `03_gap_rq.md` -> the gap and research questions
   - `02_background.md` -> related work (the standard material lives here, not in Method)
   - `04_aims.md` (section 4.1 only) -> problem formulation + the DISEIL methodology + Algorithm 1
   - `05_progress.md` -> implementation, experimental setup, results (Tables 7/8), all ablations A1..A18
2. **SECONDARY — the old AAAI draft** `draft/paper.tex`: useful wording only. It is **STALE**: it still
   says DISTIL (38 times) and carries pre-re-run numbers. It must NEVER override the CoC report.
3. **STYLE ONLY — accepted papers** in `past_papers_aaai/`: structure, storytelling, density, caption
   style, how evidence is presented. Never copy content or sentences.

## Numbers (from the CoC; never invent, never round differently)
- Success rate is **mean ± standard error**, SE = std/sqrt(n), **n=5 robot tasks, n=9 GridWorld**.
- CoC **Table 7** (final held-out success rate at B=20, with Ni and Init SR) and **Table 8**
  (per-demonstration information gain, Diff-DAgger vs DISEIL, mean ± SE) are the results.
  Copy them from `COC_REPORT/build/v2/05_progress.md` exactly.
- 5 tasks x 2 observation modalities = 10 settings. DISEIL has the best mean in all 10.
- **NO STATISTICAL TESTING ANYWHERE** (author decision, round 4; supersedes any earlier instruction).
  Do not reintroduce p-values, the sign test, the Wilcoxon signed-rank test, paired t / t(4),
  Friedman, Holm-Bonferroni, the word "significance", or the "collapse to five task means / the
  paired difference is positive on every task" framing, in the paper, the supplementary or the CoC.
  The aggregate claim is the plain one the table already shows: DISEIL holds the best mean in all ten
  settings. Do not replace a removed test with a new hedge or a new statistical claim.
- Honest framing already established in the CoC and NOT to be softened:
  - **Lift is at a ceiling** (100.0 +- 0.0), so a tied final number there is not a tie in sample
    efficiency: DISEIL reaches 100% after the 9th demonstration on Lift (state) where ThriftyDAgger
    needs the 17th, and after the 17th on Lift (image). State that; do not extrapolate the
    demos-to-ceiling figure to any other method or setting.
  - the **cluster memory** is a configurable, task-specific component, not a headline contribution.
  - A4 (LLM vs heuristic) and A5 (VLM off) show **small** gaps. State them honestly.

## Hard constraints (violating any of these fails the submission)
- **7 content pages MAXIMUM**, plus up to 2 pages of references. Nothing else.
- **NO layout manipulation of any kind.** Do not touch margins, column width, spacing, font size,
  `\vspace` hacks, `\setlength`, `\small` on body text, or the aaai2027 style. Papers get rejected for
  this. The only legal way to fit is to write less.
- **pdflatex ONLY.** `aaai2027.sty` hard-refuses XeLaTeX ("pdfTeX is required"). Therefore:
  **no literal unicode Greek** in the source; every symbol in math mode (`$\sigma$`, not `σ`).
- `\usepackage[submission]{aaai2027}` (anonymous). Author block = "Anonymous Submission".
- **Content appendices count toward the page limit** (kit, line 531), so the appendix must be a
  SEPARATE `supplementary.pdf`, never appended to the main paper.
- Every `\cite` key must exist in `references.bib` (from `COC_REPORT/build/references_coc.bib`,
  102 verified entries). **Never fabricate a citation.**

## Writing rules
Follow `COC_REPORT/Non-AI content.md` strictly. Additionally, per the author:
- **No em-dashes** anywhere.
- **No hanging words**: a paragraph must not end with a single short word alone on the last line
  (it wastes a whole line). Tighten the wording instead.
- No marketing language, no exaggerated novelty, no filler, no repetition, no generic AI phrasing.
- **The abstract must not discuss ablations** (the CoC did this; it is wrong for a paper).
- Every paragraph answers: why does this problem matter, why do existing methods fail, why is this
  idea necessary, why does it work, why should a reviewer believe it.
- One connected story: Problem -> Gap -> Insight -> Method -> Experiments -> Evidence -> Limitations
  -> Conclusion. Every section leads into the next. No isolated paragraphs.

## Main paper vs supplementary
MAIN (only the strongest evidence needed to convince a reviewer):
- the architecture figure, the main results table (Table 7), the learning curves, the
  information-gain evidence, and a compact ablation summary that justifies the design.
SUPPLEMENTARY (`supplementary.pdf`, separate):
- complete ablations A1..A18, hyperparameters, implementation details, prompts, KAG examples,
  extended derivations, per-setting tables, additional figures, failure cases, qualitative
  examples, extra discussion, compute cost.
The supplementary strengthens the paper but is never required to understand the method.

## Figure assets (copy into overleaf_aaai/figures/, space-free names, no `../` paths)
| asset | source | use |
|---|---|---|
| architecture | `figures/Architectural_Diagram.pdf` (Cluster Memory already removed) | Method figure |
| teaser | `figures/Teaser_Diagram_clean.png` (flattened; the .pdf variants carry a CamScanner watermark in the text layer) | optional intro figure |
| learning curves | `figures/selected_tasks_SE.pdf` (5 panels, mean +- SE) | Experiments |
| confidence | `figures/confidence_vs_success_v2.pdf` (r=0.82, n=180) | Analysis |
| info gain | `figures/info_gain_boxplot.pdf` | Analysis |
| failure modes | `figures/clustering_modes_pushT.pdf` | Method/analysis or supp |
| ablations | `COC_REPORT/figures_generated/F*.pdf` | mostly supplementary |
Do NOT use `all_5_task_comparison.pdf` (superseded by `selected_tasks_SE.pdf`).

## Deliverables (all in `overleaf_aaai/`, uploadable to Overleaf as-is)
`main_paper.tex`, `supplementary.tex`, `references.bib`, `aaai2027.sty`, `aaai2027.bst`,
`figures/`, `latexmkrc` (forces pdflatex), `README.md`, and the built `main_paper.pdf` +
`supplementary.pdf`. Both must compile with zero errors and no unresolved citations/refs.
