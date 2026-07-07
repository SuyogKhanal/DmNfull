# DISTIL AAAI-2027 — FINAL build report (final gate, 2026-07-07)

Paper: **DISTIL: Demonstration Distillation for Sample-Efficient Imitation Learning**
Source: `paper_aaai2027/draft/paper.tex`. Build: pdflatex -> bibtex -> pdflatex -> pdflatex
(TinyTeX, `~/.TinyTeX/bin/x86_64-linux/pdflatex`, nonstopmode). PDF written cleanly.
This report supersedes all earlier REPORT.md versions; review history and closing
outcome are in `REVIEW_LOG.md` (4 review rounds + "## Outcome").

## Verdict: PASS (submission candidate archived at drafts/final/)

| Check | Result |
|---|---|
| Page count | **9 pages** total including references (cap 9) — "Output written on paper.pdf (9 pages, 3350463 bytes)" |
| Overfull boxes | **0** ("Overfull" grep of paper.log: 0 hits; 8 underfull vboxes only, cosmetic) |
| Undefined refs/citations | none (no "undefined" or "multiply" warnings in paper.log; no missing-file warnings) |
| Banned strings | **0 hits** — word-boundary, case-insensitive grep of paper.tex for: PACE, P4 (outside \cite keys), placeholder, optional/optionally, planned, upcoming, pending, StackCube, PlugCharger, PickCube; the literal "152" appears nowhere in the source (so no "n=152"/"n = 152" in prose); "hypothes*" appears nowhere (per-demo info gain is argued as a claim, never called a hypothesis) |
| A*/BFS framing | compliant — GridWorld demonstrations are "provided by a human expert; a shortest-path search is used only as the feasibility check on prescribed layouts, never to produce demonstrations"; the only A*/BFS mention is the path-validity check in the feasibility paragraph |
| Em dashes | **0** occurrences of "---" in paper.tex (target <= 3) |
| Figures | **6/6 present and loaded**: teaser.png, architecture.pdf (figure*), comparison_baselines.pdf (figure*), info_gain_boxplot.pdf, confidence_vs_success.pdf, clustering_modes_pusht.pdf — all files exist in draft/figures/, no graphics warnings in the log |
| Numbers vs context/results_data.md | verified cell-for-cell — all 10 rows of Table 1 (final SR at the 20-demo budget) and all 10 rows of Table 2 (per-demo info gain means); abstract/analysis spot checks: 96.1 vs 90.7 (state Push-T), 95.3 vs 89.6 (image Wipe), +25.7 vs SafeDAgger, +6.4 Door image, 100.0 +/- 0.0 Lift both modes, 3.62 vs 2.96 and 3.55 vs 2.95 (Q2); all ten Pearson r values (0.86/0.88, 0.87/0.88, 0.88/0.89, 0.82/0.86, 0.83/0.82) and the quoted 0.82-0.89 range |
| UR5 KAG bounds vs context/kag_ur5_bounds.md | verified — Lift clamp half-width 0.03 m (x and y); Door 0.0135 m (x) / 0.013 m (y) absolute clamp; Wipe select-only (no scene prescription); near-flat-kernel disclosure quotes the same +/-0.03 / ~+/-0.013 ranges |
| Mandated framings | all present — info-gain three-readings argument (novelty proxy, allocation-across-modes analysis); human GridWorld expert + A*/BFS feasibility-only; 5x5 grid with three obstacles; distill-from-the-pool metaphor (intro + setup); Stagger as the GridWorld matched random control and Diff-DAgger as the robot-task control, with roster justification; online/offline bridge in intro contribution (4) and the dedicated "Bridging online and offline imitation" paragraph (per-round decoupling scoping) |
| AAAI-2027 format | template preamble unmodified (aaai2027 [submission], url, graphicx, natbib, caption + permitted algorithm/algorithmic/amsmath/amssymb/booktabs); anonymous author block; no forbidden packages or spacing commands (no hyperref/geometry/titlesec/setlength/vspace/pagestyle); \pdfinfo TemplateVersion 2027.1 kept |
| Invented statistics | none — no p-values or significance tests anywhere; protocol facts limited to author-confirmed values (20-demo budget, 9 seeds discrete / 5 seeds robot, heldout 100 episodes / 200 layouts, 60 rollout episodes per round) |

Only build-log note: one cosmetic BibTeX warning (zhang2017safedagger carries both
volume and number; BibTeX uses one). Output unaffected.

## Review outcome summary

4 rounds; final scores R1 Method 8.5 (satisfied), R2 Experiments 6.5 (not satisfied),
R3 Presentation 8.5 (satisfied). R2's two carried items are source-limited bookkeeping
(scatter granularity vs pipeline logging; Table 1 pooled record counts), documented in
REVIEW_LOG.md "## Outcome" as unresolvable from the paper sources without fabrication;
the paper text now claims only what the sources support. The figure-regeneration pass
and supplementary-material items are camera-ready obligations, not text-revision defects.
No critical or major items remain against the text.

## Snapshot inventory (`paper_aaai2027/drafts/`)

| Snapshot | Contents | Timestamp |
|---|---|---|
| `round_0_pre_review/` | paper.tex (42,153 B), paper.pdf | Jul 6 20:34 |
| `round_1/` | paper.tex (43,009 B), paper.pdf, CHANGES.md | Jul 6 21:07 |
| `round_2/` | paper.tex (45,045 B), paper.pdf, CHANGES.md | Jul 7 01:12 |
| `round_3/` | paper.tex (46,398 B), paper.pdf, CHANGES.md | Jul 7 01:38 |
| `round_4/` | paper.tex (46,749 B), paper.pdf, CHANGES.md | Jul 7 02:15 |
| `final/` | paper.tex (46,749 B, identical to round_4), paper.pdf (9 pages, 3,350,463 B) | Jul 7 02:18 |

`drafts/final/paper.{tex,pdf}` is the submission candidate.
