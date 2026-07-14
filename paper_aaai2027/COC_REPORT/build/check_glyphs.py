#!/usr/bin/env python3
"""Fail the build if a glyph the source asks for is missing from the built PDF.

The required set is derived from the markdown rather than hard-coded, so a Greek letter
added to the report later is checked without anyone having to remember to add it here.
sigma, lambda, gamma and chi are reported explicitly whether or not the source uses them,
because they are the glyphs the font choice was made for.

Greek set in maths is drawn from the OpenType maths font and lands in the PDF text layer
as a MATHEMATICAL ITALIC codepoint (gamma is U+1D6FE, not U+03B3), so the check accepts
any codepoint whose Unicode name names the letter. Comparing against U+03B3 alone reports
a false failure on a PDF whose glyphs are perfectly intact.
"""
import re
import sys
import unicodedata

GREEK_LOWER = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma",
    "tau", "upsilon", "phi", "chi", "psi", "omega",
]
GREEK_UPPER = ["Delta", "Sigma", "Phi", "Omega", "Lambda", "Theta"]
LITERAL = {
    "gamma": "γ", "sigma": "σ", "lambda": "λ", "chi": "χ", "delta": "δ",
    "theta": "θ", "phi": "φ", "rho": "ρ", "mu": "μ", "eta": "η", "xi": "ξ",
    "tau": "τ", "pi": "π", "epsilon": "ε", "alpha": "α", "kappa": "κ",
}
OTHER = {"±": "plus-minus", "×": "times", "≈": "approximately equal"}
HEADLINE = ["sigma", "lambda", "gamma", "chi"]

src, txt = sys.argv[1], sys.argv[2]
source = open(src, encoding="utf-8").read()
pdf = open(txt, encoding="utf-8", errors="replace").read()

# every distinct character in the PDF text layer, with its Unicode name
names = []
for ch in set(pdf):
    try:
        names.append(unicodedata.name(ch))
    except ValueError:
        pass


# Unicode spells some Greek letters differently from LaTeX: U+03BB is GREEK SMALL LETTER LAMDA.
UNICODE_SPELLING = {"lambda": "lamda", "Lambda": "lamda"}


def renders(letter, capital):
    """Is this Greek letter present in the PDF, in any of its Unicode spellings?"""
    word = re.compile(r"\b%s\b" % UNICODE_SPELLING.get(letter, letter).upper())
    for n in names:
        if not word.search(n) or "GREEK" not in n and "MATHEMATICAL" not in n:
            continue
        if capital:
            if " CAPITAL " in n:
                return True
        elif " SMALL " in n or n.endswith(" SYMBOL"):
            return True
    return False


def wanted(letter):
    """Does the source ask for this letter, as a LaTeX command or as a literal?"""
    if re.search(r"\\%s\b" % letter, source):
        return True
    return LITERAL.get(letter, "\0") in source


required = [(l, False) for l in GREEK_LOWER if wanted(l)]
required += [(l, True) for l in GREEK_UPPER if wanted(l)]

missing = [l for l, cap in required if not renders(l, cap)]
for ch, name in OTHER.items():
    if ch in source and ch not in pdf:
        missing.append(f"{name} ({ch})")

for letter in HEADLINE:
    if not any(l == letter for l, _ in required):
        state = "not used in the source"
    elif renders(letter, False):
        state = "renders"
    else:
        state = "MISSING FROM THE PDF"
    print(f"  {letter}: {state}")

if missing:
    print("FAIL: glyphs dropped by the build: " + ", ".join(missing))
    sys.exit(1)
print(f"  all {len(required)} Greek letters the source uses survive the build, "
      f"as do {sum(ch in source for ch in OTHER)} maths symbols")
