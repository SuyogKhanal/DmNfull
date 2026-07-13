#!/usr/bin/env bash
# Rebuild CoC_Report.pdf from CoC_Report.md
#
# Notes on the non-obvious flags:
#   --pdf-engine=xelatex   pdflatex CANNOT render the Greek/maths glyphs used here (sigma, lambda, chi, approx).
#   mainfont=DejaVu Serif  Latin Modern has no Greek coverage; with it, those characters are SILENTLY DROPPED.
#   no --toc               the TOC is inserted by a raw \tableofcontents after the titlepage, so the cover is page 1.
#   no --number-sections   the report numbers its own sections; pandoc numbering would double it.
#   images have NO alt text  so pandoc emits no caption of its own; the report writes its own "**Figure N.**" captions.
#
# The two certificates Certificate_of_Completion_-_Respect_at_Deakin...pdf and
# Research_Integrity_...pdf have broken AcroForm fields and cannot be embedded by
# xdvipdfmx; they are pre-rasterised into certs_png/ and the report points there.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

pandoc CoC_Report.md -o CoC_Report.pdf \
  --pdf-engine=xelatex \
  -V geometry:"margin=2.4cm" \
  -V fontsize=11pt \
  -V mainfont="DejaVu Serif" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  -V linkcolor=blue -V urlcolor=blue \
  --resource-path=.:figures_generated:../figures:certs_png

echo "built: $(pdfinfo CoC_Report.pdf | awk '/^Pages/{p=$2} /^File size/{s=$3} END{print p" pages, "s" bytes"}')"

# sanity: glyphs must survive, and pandoc must not have added its own figure captions
for g in σ λ ± ×; do
  n=$(pdftotext CoC_Report.pdf - | grep -o "$g" | wc -l)
  [ "$n" -gt 0 ] || { echo "FAIL: glyph $g missing from PDF"; exit 1; }
done
dup=$(pdftotext CoC_Report.pdf - | grep -cE '^Figure [0-9]+:' || true)
[ "$dup" -eq 0 ] || { echo "FAIL: $dup pandoc-generated figure captions (double numbering)"; exit 1; }
echo "checks passed: glyphs intact, no double captions"
