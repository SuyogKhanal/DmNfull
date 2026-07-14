#!/usr/bin/env bash
# Rebuild CoC_Report.pdf from CoC_Report.md.
#
# CoC_Report.md is assembled from build/v2/*.md by build/assemble.py; edit the
# chapter files there and re-run that script, not this one, to change the text.
#
# Notes on the non-obvious flags:
#   --pdf-engine xelatex     pdflatex CANNOT render the Greek/maths glyphs used here (sigma,
#                            lambda, gamma). With Latin Modern under pdflatex they are SILENTLY DROPPED.
#   mainfont Liberation Serif  the supervisor asked for Times New Roman, which is not installed.
#                            Liberation Serif is metrically identical to Times New Roman AND carries
#                            the Greek glyphs the report needs. Do not swap it for DejaVu (wrong metrics)
#                            or Latin Modern (no Greek).
#   default --columns        every wide table gets wrapped p-columns, sized in proportion to its
#                            content. Raising it (e.g. --columns=120) gives the wide tables plain
#                            l-columns instead, and Table 2 and Table 12 then run off the page.
#                            The results tables keep every cell on one line because their cells are
#                            written "86.1+-2.8" with no spaces, which gives LaTeX nowhere to break;
#                            writing them "86.1 +- 2.8" is what put the mean, the +- and the standard
#                            deviation on three separate lines in the previous build.
#   build/preamble.tex       sets every table one size smaller so no table exceeds the text width.
#   no --toc                 the TOC is inserted by a raw \tableofcontents after the titlepage, so the
#                            cover is page 1.
#   no --number-sections     the report numbers its own sections; pandoc numbering would double it.
#   images have NO alt text  so pandoc emits no caption of its own; the report writes its own
#                            "**Figure N.**" captions.
#
# The two certificates Certificate_of_Completion_-_Respect_at_Deakin...pdf and
# Research_Integrity_...pdf have broken AcroForm fields and cannot be embedded by
# xdvipdfmx; they are pre-rasterised into certs_png/ and the report points there.
#
# The build is pandoc -> .tex -> xelatex (three passes, for the TOC) rather than a single
# pandoc --pdf-engine run, because the LaTeX log is needed to prove that no table overflows
# its page width. The log is left in build/CoC_Report.log.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

pandoc CoC_Report.md -s -o CoC_Report.tex \
  --include-in-header=build/preamble.tex \
  -V geometry:"margin=2.4cm" \
  -V fontsize=11pt \
  -V mainfont="Liberation Serif" \
  -V sansfont="Liberation Sans" \
  -V monofont="Liberation Mono" \
  -V linkcolor=blue -V urlcolor=blue \
  --resource-path=.:figures_generated:../figures:certs_png

for _ in 1 2 3; do
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build CoC_Report.tex >/dev/null
done
mv build/CoC_Report.pdf CoC_Report.pdf

echo "built: $(pdfinfo CoC_Report.pdf | awk '/^Pages/{p=$2} /^File size/{s=$3} END{print p" pages, "s" bytes"}')"

pdftotext CoC_Report.pdf build/CoC_Report.txt

# 1. every Greek glyph the source asks for must survive into the PDF text layer.
#    The required set is derived from the source, so a glyph added later is checked automatically.
python3 build/check_glyphs.py CoC_Report.md build/CoC_Report.txt

# 2. pandoc must not have added figure captions of its own (that would double-number the figures).
dup=$(grep -cE '^Figure [0-9]+:' build/CoC_Report.txt || true)
[ "$dup" -eq 0 ] || { echo "FAIL: $dup pandoc-generated figure captions (double numbering)"; exit 1; }

# 3. the institute is named in full, never by its acronym.
banned=$(grep -c 'A2I2' build/CoC_Report.txt || true)
[ "$banned" -eq 0 ] || { echo "FAIL: $banned occurrences of the banned acronym"; exit 1; }

# 3b. no scanner-app watermark may reach the PDF. The hand-drawn teaser was scanned with
#     CamScanner, which stamps an INVISIBLE text object: it does not render, so it survives
#     visual inspection, but pdftotext (and any similarity checker) reads it. The asset is
#     therefore flattened to Teaser_Diagram_clean.png, which carries no text layer at all.
wm=$(grep -ci 'camscanner' build/CoC_Report.txt || true)
[ "$wm" -eq 0 ] || { echo "FAIL: $wm occurrences of a scanner watermark in the PDF text layer"; exit 1; }

# 4. no table may overflow its page width.
python3 build/check_tables.py CoC_Report.tex build/CoC_Report.log

echo "checks passed: glyphs intact, no double captions, no banned acronym, no table overflow"
