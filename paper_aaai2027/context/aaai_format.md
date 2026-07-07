# AAAI-2027 formatting cheat-sheet (distilled from the Author Kit)

Kit location: `paper_aaai2027/context/aaai_author_kit/`
Build the paper ON this template: `AnonymousSubmission2027.tex` (copy its preamble UNCHANGED).
Style: `aaai2027.sty`  |  Bib style: `aaai2027.bst` → `\bibliographystyle{aaai2027}`

## Non-negotiable
- `\usepackage[submission]{aaai2027}` — DO NOT modify the style file, its options, spacing, floats, margins, fonts, or font size.
- Fonts load automatically (newtxtext, helvet, courier). Do NOT add `times`/`helvet`/`courier`.
- Citations via **natbib**: `\cite`, `\citep`, `\citet`. Bibliography: `\bibliography{references}`.
- Two-column; standard AAAI margins/spacing. No page numbers.
- Anonymous submission: sole "author" = "Anonymous Submission"; empty affiliations; anonymize self-citations.
- Title in mixed case.

## Length
- Target **7 pages of technical content** (through Conclusion).
- References do NOT count toward the 7; keep them to ~2 pages (≈9 pages total, per author).
- Optional reproducibility checklist / appendix (if permitted) also does not count.

## Provided / allowed packages
- `algorithm`, `algorithmic`  → USE for the PACE algorithm box
- `booktabs`  → USE for all tables
- `graphicx`, `caption`, `natbib`, `url`, `amsmath`/`amssymb` (math), `newfloat`, `listings`, `inputenc`, `microtype`

## FORBIDDEN packages (submission rejected)
`authblk`, `balance`, `CJK`, `float`, `flushend`, `fullpage`, `geometry`, **`hyperref`**, `navigator`,
`indentfirst`, `layout`, `multicol`, `nameref`, `savetrees`, `setspace`, `stfloats`, `tabu`, `titlesec`,
`tocbibind`, `ulem`, `wrapfig` — and anything embedding links.

## FORBIDDEN commands
`\addtolength`, `\balance`, `\clearpage`, `\columnsep`, `\newpage`, `\pagebreak`, `\pagestyle`, `\tiny`,
and any manual page break.

## This paper specifically
- Method described **formally** (math notation from `equations.tex`) + one `\begin{algorithm}` box for PACE.
- **≥3 booktabs tables**: (1) task-suite summary, (2) main results, (3) ablation.
- Every number is a `\PH{...}` placeholder (results pending). Every `\cite` key ∈ `references.bib`.
- Method name is **PACE** (Perceive → Assess → Choose → Execute).
