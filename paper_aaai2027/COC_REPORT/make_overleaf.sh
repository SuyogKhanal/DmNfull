#!/usr/bin/env bash
# Build a SELF-CONTAINED Overleaf project from the CoC sources.
#
# Why this script exists: the local build works only because of things Overleaf will not have.
#   1. Six figures live OUTSIDE the project root ("../figures/..."). Overleaf cannot see a parent
#      directory, so those \includegraphics would fail with "File not found".
#   2. Two assets have SPACES in their filenames. \includegraphics with spaces is fragile.
#   3. The fonts are requested BY NAME (\setmainfont{Liberation Serif}). That works here because
#      the font is installed on this machine. On Overleaf it is a coin flip, and if it resolves to
#      a fallback WITHOUT Greek coverage, sigma/lambda/gamma are dropped SILENTLY (this exact bug
#      already bit us once with Latin Modern). So the .ttf files are shipped inside the project and
#      loaded by PATH, which cannot fall back.
#   4. The preamble arrives via --include-in-header at build time; the shipped .tex must be standalone.
#
# Output: overleaf/  -> zip it (or upload the folder) to Overleaf, set compiler to XeLaTeX, done.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

OUT=overleaf
FONTDIR=/usr/share/fonts/truetype/liberation

rm -rf "$OUT"
mkdir -p "$OUT/figures" "$OUT/fonts"

# ---------------------------------------------------------------- 1. regenerate the standalone .tex
# -s makes it standalone; --include-in-header inlines the preamble INTO the file, so the shipped
# main.tex needs no external header.
pandoc CoC_Report.md -s -o "$OUT/main.tex" \
  --include-in-header=build/preamble.tex \
  -V geometry:"margin=2.4cm" \
  -V fontsize=11pt \
  -V mainfont="Liberation Serif" \
  -V sansfont="Liberation Sans" \
  -V monofont="Liberation Mono" \
  -V linkcolor=blue -V urlcolor=blue \
  --resource-path=.:figures_generated:../figures:certs_png

# ---------------------------------------------------------------- 2. collect every image, flatten
# Every \includegraphics target is copied into overleaf/figures/ under a SPACE-FREE name, and the
# path in main.tex is rewritten to point there. Nothing outside the project is referenced afterwards.
python3 - "$OUT" <<'PY'
import os, re, shutil, sys
out = sys.argv[1]
tex = os.path.join(out, 'main.tex')
src = open(tex, encoding='utf-8').read()

search = ['.', 'figures_generated', '../figures', 'certs_png']
seen, missing = {}, []

def resolve(p):
    if os.path.isabs(p) and os.path.exists(p):
        return p
    for d in search:
        c = os.path.join(d, p)
        if os.path.exists(c):
            return c
    return None

def repl(m):
    pre, path = m.group(1), m.group(2)
    real = resolve(path)
    if real is None:
        missing.append(path)
        return m.group(0)
    if real not in seen:
        # space-free, collision-free destination name
        base = os.path.basename(real).replace(' ', '_')
        dest = os.path.join(out, 'figures', base)
        n = 1
        while os.path.exists(dest) and os.path.abspath(real) != os.path.abspath(seen.get(dest, '')):
            root, ext = os.path.splitext(base)
            base = f"{root}_{n}{ext}"; dest = os.path.join(out, 'figures', base); n += 1
        shutil.copy2(real, dest)
        seen[real] = base
    return f"{pre}{{figures/{seen[real]}}}"

src = re.sub(r'(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}', repl, src)

if missing:
    print("FAIL: unresolved images:", missing); sys.exit(1)

# ------------------------------------------------------------ 3. load fonts BY PATH, not by name
# A name lookup can silently fall back to a font without Greek. A path cannot.
for cmd, fam in (('setmainfont', 'LiberationSerif'),
                 ('setsansfont', 'LiberationSans'),
                 ('setmonofont', 'LiberationMono')):
    src = re.sub(
        r'\\%s\[[^\]]*\]\{[^}]*\}' % cmd,
        r'\\%s{%s}[Path=fonts/, Extension=.ttf, UprightFont=*-Regular, BoldFont=*-Bold, '
        r'ItalicFont=*-Italic, BoldItalicFont=*-BoldItalic]' % (cmd, fam),
        src)

open(tex, 'w', encoding='utf-8').write(src)
print(f"  images bundled: {len(seen)}")
PY

# ---------------------------------------------------------------- 4. ship the fonts
for f in Serif Sans Mono; do
  for v in Regular Bold Italic BoldItalic; do
    cp "$FONTDIR/Liberation$f-$v.ttf" "$OUT/fonts/"
  done
done

# ---------------------------------------------------------------- 5. tell Overleaf to use XeLaTeX
printf '%% !TeX program = xelatex\n' | cat - "$OUT/main.tex" > "$OUT/.t" && mv "$OUT/.t" "$OUT/main.tex"
cat > "$OUT/latexmkrc" <<'EOF'
# Overleaf reads this: the document REQUIRES XeLaTeX (fontspec + unicode-math).
# pdflatex cannot render the Greek glyphs and drops them silently.
$pdf_mode = 5;   # xelatex
EOF

cat > "$OUT/README.md" <<'EOF'
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
EOF

echo "  fonts bundled:  $(ls "$OUT/fonts" | wc -l)"
echo "built: $OUT/  ($(du -sh "$OUT" | cut -f1))"
