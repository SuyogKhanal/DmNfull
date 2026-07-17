# Round-4 changes (author-directed). Binding.

Applies to the AAAI main paper, the supplementary, AND the CoC report. All three must end consistent.

## 1. Teaser figure on page 1 (main paper)
Insert `figures/teaser.png` (already flattened; the .pdf variants carry a CamScanner watermark in the
text layer, so use the .png) **on page 1, immediately after the FIRST paragraph of the Introduction**.
Small is fine and preferred. Verify in the RENDERED PDF that it actually lands on page 1 and reads
after the first paragraph. Space is paid for by the deletions in item 4, not by any layout change.

## 2. Make the Lift ceiling legible (main paper)
Problem: a reader looking at Table 1 sees 100.0 in several Lift cells and cannot tell that DISEIL is
better. The table hides the result.
Author-provided facts (use exactly; do not extrapolate to methods or settings not listed):
 - **Lift (state):** ThriftyDAgger reaches the 100% ceiling only after the **17th** demonstration.
   DISEIL reaches 100% after the **9th**.
 - **Lift (image):** DISEIL reaches 100% after the **17th** demonstration.
State this in the body where Lift is discussed, and make Table 1 readable on this point (a caption
note is acceptable; a new column is NOT, because demos-to-ceiling is known only for the cells above).
The point to land: on Lift the ceiling hides a real difference in how fast the budget gets there, so a
tied final number is not a tie in sample efficiency.

## 3. Table 2: drop the GridWorld rows
Table 2 compares Diff-DAgger against DISEIL. Diff-DAgger does not run on GridWorld (the policy is not
a diffusion policy), so those rows carry no comparison. Remove the GridWorld rows from Table 2 and
adjust the caption and any prose that counts its rows.

## 4. Remove ALL statistical testing (main paper, supplementary, AND CoC report)
Delete every mention of: p-values, the sign test, the Wilcoxon signed-rank test, paired t-tests,
`t(4)`, Friedman, Holm-Bonferroni, "statistically significant"/"significance", and the
"collapse to five task means / the paired difference is positive on every task" framing.
 - **Abstract:** remove the p-value and the collapsed-task-means sentence entirely.
 - **Main paper:** `sec/01_abstract_intro.tex` (lines ~22-23, ~90) and `sec/04_experiments.tex`
   (lines ~47, ~132-138, ~150).
 - **Supplementary:** `supplementary.tex` (~line 339: Wilcoxon, Friedman).
 - **CoC report:** `COC_REPORT/build/v2/05_progress.md` (11 hits), `06_plan.md`, `00_front.md`.
The surviving claim is the plain one the table already shows: DISEIL holds the best mean in all ten
settings. Do not replace the tests with a hedge or a new statistical claim; simply state the result.
KEEP the honest caveats that are not statistical tests: the Lift ceiling, the A4/A5 small gaps, and
the fact that the two modalities of a task are not independent may be stated in plain words if it
still reads naturally, but without any test, p-value or "significance" language.

## 5. Table 1: "n/a" -> an em-dash-free dash
Replace every `n/a` cell in Table 1 with a plain dash (`--` in LaTeX, rendering as an en-dash, or a
simple `-`). Keep the caption's explanation of what a dash means (method not applicable: Diff-DAgger
on GridWorld, Stagger on the robot tasks).

## Constraints that still bind
- 7 content pages MAX + <= 2 reference pages. NO layout/spacing/margin/font manipulation, ever.
- pdflatex only; no literal unicode Greek; anonymous submission; every \cite key must exist.
- No em-dashes. No hanging words. Method is DISEIL.
- Item 4 frees space; item 1 spends some of it. Net must still be <= 7.0 content pages.
