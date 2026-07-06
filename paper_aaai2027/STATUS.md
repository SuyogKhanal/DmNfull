# DISTIL AAAI-2027 — status (2026-07-06)

**Current draft: `draft/paper.tex` → `draft/paper.pdf` — 9 pages incl. references, 0 overfull boxes, final gate PASS.**
Full build report: `REPORT.md`. Data source of truth: `context/results_data.md` (transcribed from `table_data.xlsx`).

- Title/method: DISTIL — Demonstration Distillation for Sample-Efficient Imitation Learning.
- Compile locally: `cd draft && ~/.TinyTeX/bin/x86_64-linux/pdflatex -interaction=nonstopmode paper && ~/.TinyTeX/bin/x86_64-linux/bibtex paper && (pdflatex ×2)`.
  (aaai2027.sty REQUIRES pdfTeX — tectonic/XeLaTeX will refuse.)
- Backups: pre-DISTIL draft at `draft/paper_pace_backup.tex`; original figures untouched in `figures/`.
- Author sign-offs pending: (1) patched architecture figure in `draft/figures/architecture.pdf` (feasibility-check box moved before expert demo + typo fix — original preserved); (2) headline claim softened to "highest mean SR in all 10 cells" with seed-overlap caveat; (3) the n=152 exclusion explanation in Q3 needs author confirmation.
- Rerunnable pipelines: `distil_rewrite.workflow.js` (this run), `write_paper.workflow.js`, `compress_paper.workflow.js`.
