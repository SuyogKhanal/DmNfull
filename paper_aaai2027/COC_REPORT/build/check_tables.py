#!/usr/bin/env python3
"""Fail the build if any table overflows its page width.

LaTeX reports a box that sticks out past the text width as an Overfull \\hbox, with the
line range of the .tex that produced it. Every table in this report is a longtable, so the
check is: find the line span of each longtable in the .tex, and fail if any Overfull \\hbox
in the log falls inside one of them. Overfull boxes elsewhere (verbatim prompt blocks, for
instance) are reported but are not fatal.
"""
import re
import sys

tex, log = sys.argv[1], sys.argv[2]
lines = open(tex, encoding="utf-8").read().split("\n")

spans, start = [], None
for i, line in enumerate(lines, start=1):
    if "\\begin{longtable}" in line:
        start = i
    elif "\\end{longtable}" in line and start is not None:
        spans.append((start, i))
        start = None

overfull = re.findall(
    r"Overfull \\hbox \(([\d.]+)pt too wide\) in \w+ at lines (\d+)--(\d+)",
    open(log, encoding="utf-8", errors="replace").read(),
)

in_table, elsewhere = [], []
for pt, a, b in overfull:
    a, b = int(a), int(b)
    if any(s <= b and a <= e for s, e in spans):
        in_table.append((float(pt), a, b))
    else:
        elsewhere.append((float(pt), a, b))

print(f"  {len(spans)} tables; {len(overfull)} overfull boxes in the document")
if elsewhere:
    worst = max(elsewhere)[0]
    print(f"  {len(elsewhere)} overfull boxes outside tables (worst {worst}pt), not fatal")
if in_table:
    for pt, a, b in in_table:
        print(f"FAIL: table overflows the page width by {pt}pt (CoC_Report.tex lines {a}--{b})")
    sys.exit(1)
print("  no table overflows its page width")
