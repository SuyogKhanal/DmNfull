# Round-4 audit report

Auditor pass against the RENDERED PDFs. Binding: `_changes_r4.md`, `PAPER_SPEC.md`.
Method name: DISEIL. All three documents rebuilt from source during this audit.

Artefacts as audited:

| document | pages | built |
|---|---|---|
| `overleaf_aaai/main_paper.pdf` | 9 (content 6.89 + references) | clean, 0 errors, 0 undefined citations |
| `overleaf_aaai/supplementary.pdf` | 22 (separate PDF) | clean, 0 errors, 0 undefined citations |
| `COC_REPORT/CoC_Report.pdf` | 76 | rebuilt via `assemble.py` + `build_pdf.sh`, all checks passed |

## Check table

| # | Check | Verdict | Evidence (from the rendered PDF) |
|---|---|---|---|
| 1 | Teaser on page 1, after first Introduction paragraph | PASS | `main_paper.pdf` p.1: teaser image bbox `[364.8, 216.0, 512.7, 398.4]`, Figure 1 caption bbox `[319.5, 408.5, 558.0, 473.2]` (right column). Intro paragraph 1 ("Behaviour cloning ... is all that is left to choose") bbox `[54.0, 485.1, 292.5, 652.1]` (left column). Two-column reading order puts the figure after paragraph 1. Source placement `sec/01_abstract_intro.tex:38-47` is immediately after paragraph 1. `figures/teaser.png` used (not the watermarked .pdf). |
| 2 | Lift ceiling legible; exact numbers, no extrapolation | PASS | Table 1 caption, rendered: "The Lift rows are at the ceiling, and the tie they show is not a tie in sample efficiency: DISEIL reaches 100 per cent after the ninth demonstration on Lift (state) and the seventeenth on Lift (image), ThriftyDAgger only after the seventeenth on Lift (state)." Restated in body (`main.txt:411-414`, Results). No new column added; no demos-to-ceiling claim for any other method or setting. |
| 3 | Table 2 has no GridWorld rows; caption matches | PASS | Table 2 rendered rows are exactly 8: Push-T/Lift/Wipe/Door x {state, image}. Caption: "it is lower in all eight settings in which it runs. GridWorld is omitted because Diff-DAgger does not run there." Prose counts agree (`sec/04_experiments.tex:173` "higher in all eight settings in which Diff-DAgger runs"). |
| 4 | Table 1 uses dashes, not "n/a"; caption explains | PASS | `grep -owic "n/a"` = **0** in all three rendered PDFs. Table 1 source uses `--` (en-dash). Caption: "A dash marks a method that does not apply: Diff-DAgger on GridWorld, whose policy is not a diffusion policy, and Stagger where it was not run." |
| 5 | No statistical testing in ANY document | PASS | Rendered-text grep counts, all **0/0/0** (main / supp / CoC) for each of: `p-value`, `p ?[=<] ?0\.[0-9]`, `sign test`, `Wilcoxon`, `paired t`, `t(4)`, `Friedman`, `Holm`, `significan`, `collapsed to five`, `five task means`. Source-level counts also 0 across `main_paper.tex`, `sec/*.tex`, `supplementary.tex`, `COC_REPORT/build/v2/*.md`. |
| 6 | Abstract: no p-value, no ablations | PASS | Abstract (`sec/01_abstract_intro.tex:1-22`) greps 0 for `ablat|knockout|removing|A[0-9]|p ?[=<]|signific|sign test|wilcoxon`. Surviving claim is the plain one: "attains the best or joint-best mean held-out success rate in every setting, by a mean of 2.80 points over the strongest baseline in each." |
| 7 | Content <= 7.0 pages, total <= 9, supp separate, CoC builds | PASS | Content = **6.89 pages**: pages 1-6 full; on p.7 the References heading begins at y=564.1 in the right column (text band y=56.0-705.0, col height 649.0) -> 6 + 0.5 + 0.5*(564.1-56.0)/649.0 = 6.891. Total **9** pages (refs on p.7 right column, p.8, p.9). `supplementary.pdf` is a separate 22-page file, never appended. CoC rebuilt: "checks passed: glyphs intact, no double captions, no banned acronym, no table overflow". |
| 8 | No layout manipulation introduced | PASS | `diff overleaf_aaai/aaai2027.sty /tmp/kit27/AuthorKit27/aaai2027.sty` -> **IDENTICAL**. Grep across `main_paper.tex`, `supplementary.tex`, `sec/*.tex`: **0** hits for `\vspace`, `\setlength`, `\hspace`, `\addtolength`, `\enlargethispage`, `\baselinestretch`, `geometry` package. One `\footnotesize` at `sec/04_experiments.tex:9`, scoped **inside** `\begin{table*}` (Table 1) and not body text: legal and standard. No `\small` on body text. |
| 9 | No em-dashes; DISEIL throughout; no unicode Greek; cites resolve | PASS (after fix) | Em-dashes: main **0**, supp **0**, CoC **0** (was 12, fixed this round; see below). No `---` and no U+2014 in any LaTeX source. Naming: DISEIL x64; DISTIL/PACE/P4-LLM/P4 = **0**. No literal unicode Greek in sources (grep `[\x{0370}-\x{03FF}\x{1F00}-\x{1FFF}]` = 0); every symbol in math mode. Citations: `main_paper.blg` and `supplementary.blg` report **0 warnings**; logs report **0** undefined citations/references. |

## Fix applied this round

`COC_REPORT/build/v2/05_progress.md`: the 12 remaining em-dashes (U+2014) were all
"not applicable" table-cell markers in Tables 7 and 8. Replaced with en-dashes (U+2013),
matching the main paper's `--`. Rebuilt with `assemble.py` + `build_pdf.sh`; the rendered
CoC PDF now greps **0** em-dashes and the build's own checks pass. No prose was touched
and no number changed. Backup of the pre-edit file is in the session scratchpad.

## Outstanding

`PAPER_SPEC.md:30-33` is now **stale and self-contradictory** against `_changes_r4.md` item 4.
It still mandates the framing that round 4 deleted: "the aggregate claim is the **collapsed
task-level test** (5 task means, paired t(4)=4.10, p=0.015 two-sided; sign test p=0.031
one-sided)". The three documents correctly follow `_changes_r4.md` (the later, binding
change list), so no document is wrong. The spec text itself should be updated so a future
round does not reintroduce the tests from it. This is a document-hygiene issue, not a
defect in any built artefact, and it was left alone because editing the binding spec is an
author decision, not an auditor's.
