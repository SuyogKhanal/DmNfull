# CoC Report — Round-3 Update: Build + Audit Report

Method name: **DISEIL** (never DISTIL/PACE/P4 in prose, captions, legends, headers).
Data source of truth: `ablations_results/sheets/*.csv` (new run).
SE rule: SE = std / sqrt(n), n = 5 (robot tasks), n = 9 (GridWorld), rounded to 1 dp (pp).

Build pipeline: `build/assemble.py` merged 9 chapter files (100 refs after dropping [19],[49]),
then `build_pdf.sh` (pandoc -> xelatex, 3 passes). All hard-fail gates passed:
glyphs intact, no double captions, no banned acronym (A2I2), no scanner watermark, no table overflow.

## Check table

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Build (assemble + build_pdf) succeeds, all gates pass | PASS | "checks passed: glyphs intact, no double captions, no banned acronym, no table overflow"; 77 pages, 5,151,794 bytes |
| 2 | Table 7 = mean±SE with Ni + Init SR columns, values match authoritative spec table | PASS | PDF p.41: all 10 rows match spec exactly (e.g. GridWorld state 85.3±0.9 … DISEIL 92.4±0.4; Push-T state DISEIL 96.1±1.6; Door image DISEIL 88.6±1.5). Ni and Init SR leading columns present |
| 3 | Table 7 caption says standard error, 5/9 seeds | PASS | Caption: "mean ± standard error over 5 seeds (robot tasks), 9 seeds (GridWorld); Ni = initial demonstrations; Init SR = round-0 held-out success rate; best per row in bold" |
| 4 | Table 8 = Diff-DAgger vs DISEIL info gain, new values | PASS | PDF p.43: Diff/DISEIL columns match GT_InfoGain.csv (DISTIL->DISEIL): GW state —/3.550, Push-T state 1.570/2.810, Door image 1.580/3.000, etc. |
| 5 | Figure 2: no "Cluster Memory" box or its arrow | PASS | PDF p.21 visual inspection: boxes are Flag Uncertainty, Vision LLM, Reasoning LLM, Cluster Engine, KAG Grounding, Prescription LLM, Expert Demo, Policy Rollout, Update/Train Policy — no Cluster Memory box, no dashed arrow to it. `pdftotext` of arch PDF: no "memory". Original kept as `Architectural_Diagram_withmem.pdf` |
| 6 | Methodology has the one task-specific-memory sentence | PASS | `04_aims.md:170`: "The cluster memory is not drawn at all: it is a configurable, task-specific component that is active only when a task exhibits recurring failure clusters … left out of the framework figure" |
| 7 | Figure 5 = five-panel mean±SE learning curves | PASS | PDF p.42: 5 panels GridWorld(image), Push-T(state), Lift(state), Door(state), Wipe(image); title "Success rate vs demonstrations added (mean ± SE)"; legend "DISEIL (Ours)"; SE shaded bands. `selected_tasks_SE.pdf` md5-identical to `updated_figures/` source |
| 8 | Figure 7 = new confidence plot, r=0.82, n=180 | PASS | PDF p.45: inset "r = 0.82 / n = 180"; caption "GridWorld (image), 180 prescriptions, Pearson r = 0.82". `confidence_vs_success_v2.pdf` md5-identical to `updated_figures/confidence_vs_success (1).pdf` |
| 9 | Figure 10 (grounding/feasibility) has the A6 SE error bar | PASS | PDF p.49: top-row A6 bars (Knowledge graph off) each carry an SE error bar — GW image 89.8, Push-T state 93.4, Door image 85.7 — alongside DISEIL(full) bars; caption "SE = std/sqrt(n), including the A6 bar" |
| 10 | All ablation figures: SE bars, DISEIL labels, no Lift, no orange text | PASS | PDF p.47 (Fig 8 allocation ladder) spot-check: 3 settings (GW image, Push-T state, Door image), SE bars, "DISEIL (full)" labels, grey/green/blue palette (no orange), no Lift. `make_figures.py`: 0 DISTIL, 21 DISEIL, both orange hues withdrawn (D-G1), D-G3 keeps Lift out of ablation figures |
| 11 | "DISTIL" appears nowhere in the PDF | PASS | `grep -c DISTIL build/CoC_Report.txt` = 0; also 0 in CoC_Report.md; PACE = 0, \bP4\b = 0 |
| 12 | "n = 152" appears nowhere | PASS | `grep 152 build/CoC_Report.txt` = NONE; Table 8 text uses "168 to 184 loss records per cell" |
| 13 | Error language says standard error, not std | PASS | 14 "standard error" vs residual uses: 2 "SE = std/sqrt(n)" formula definitions, 1 "standard deviation of 1.73" (margin spread, descriptive), 1 "sample standard deviation over language-model-active rounds" (descriptive) — all legitimate, none mislabel an error bar |
| 14 | No table overflows | PASS | `check_tables.py`: "no table overflows its page width"; 12 tables, 1 overfull box (1.81pt, outside tables, non-fatal) |

## Final document statistics

| Metric | Value |
|---|---|
| Pages | 77 |
| File size | 5,151,794 bytes (~5.15 MB) |
| Figures | 19 |
| Tables | 12 |
| References | 100 (dropped uncited [19] AMPLIFY, [49] CLAM) |
| PDF engine | xelatex (Liberation Serif, Greek glyphs intact) |
| Greek glyphs verified | 15 letters + 2 maths symbols survive |

## Notes

- The chapter files (`build/v2/04_aims.md`, `05_progress.md`) and the figure assets
  (`../figures/selected_tasks_SE.pdf`, `confidence_vs_success_v2.pdf`, cleaned
  `Architectural_Diagram.pdf`, regenerated `figures_generated/F*.pdf`) already carried the
  round-3 edits; this pass assembled, built, and audited them end-to-end.
- SE derivation spot-checked against CSVs: GridWorld std/3 (n=9), robot std/sqrt(5).
  e.g. Push-T state DISEIL 3.5 std -> 1.6 SE; GridWorld state DISEIL 1.3 std -> 0.4 SE. Consistent with Table 7.
- No outstanding failures. Build is reproducible via
  `python3 build/assemble.py && bash build_pdf.sh`.
