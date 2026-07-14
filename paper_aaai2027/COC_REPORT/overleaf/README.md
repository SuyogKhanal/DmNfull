# DISEIL — Confirmation of Candidature (Overleaf project)

Self-contained. Upload this whole folder (or a zip of it) to Overleaf.

**Set the compiler to XeLaTeX** (Menu -> Compiler -> XeLaTeX). `latexmkrc` also requests it,
and `main.tex` carries a `% !TeX program = xelatex` line.

XeLaTeX is required, not preferred: the report uses `fontspec` and `unicode-math`, and pdfLaTeX
**silently drops** the Greek characters (sigma, lambda, gamma) rather than erroring.

## Layout
- `main.tex`    the whole report; the preamble is inlined, nothing is included from outside.
- `figures/`    every image, with space-free names. No path escapes the project root.
- `fonts/`      Liberation Serif/Sans/Mono (metrically identical to Times New Roman, and unlike
                Latin Modern they carry Greek). Loaded by PATH, so Overleaf cannot substitute a
                font that lacks the glyphs.
- `latexmkrc`   forces XeLaTeX.

No bibliography step is needed: the references are already typeset in `main.tex`.

## Editing
This project is GENERATED. The source of truth is `COC_REPORT/build/v2/*.md`, assembled by
`build/assemble.py`. To change the text, edit those and re-run `make_overleaf.sh`, otherwise your
Overleaf edits will be overwritten on the next regeneration.
