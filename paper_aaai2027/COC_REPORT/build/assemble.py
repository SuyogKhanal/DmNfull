#!/usr/bin/env python3
"""Assemble CoC_Report.md from build/v2/*.md (filename order).

Two reference entries carried by the working bibliography are cited nowhere in the
body ([19] AMPLIFY and [49] CLAM). The reference list must contain every cited key
and nothing uncited, so both are dropped here and every citation in the body is
renumbered against the surviving list. Numbering is otherwise unchanged.
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
V2 = ROOT / "build" / "v2"
OUT = ROOT / "CoC_Report.md"

DROP = {19, 49}  # uncited entries

parts = [p.read_text() for p in sorted(V2.glob("*.md"))]
doc = "\n\n".join(p.rstrip("\n") for p in parts) + "\n"

# split body / reference list
head, sep, tail = doc.partition("\n# References\n")
assert sep, "no References heading found"

# --- reference list: drop the uncited entries, renumber sequentially
entries = re.findall(r"^\[(\d+)\] (.*)$", tail, flags=re.M)
assert entries, "no reference entries parsed"
old_nums = [int(n) for n, _ in entries]
assert old_nums == sorted(old_nums), "reference list is not in ascending order"

kept = [(n, txt) for n, txt in ((int(n), t) for n, t in entries) if n not in DROP]
remap = {old: new for new, (old, _) in enumerate(kept, start=1)}

new_tail = "\n\n".join(f"[{remap[old]}] {txt}" for old, txt in kept) + "\n"

# --- body: renumber every citation bracket
CITE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def fix(m):
    nums = [int(x) for x in m.group(1).replace(" ", "").split(",")]
    for n in nums:
        if n in DROP:
            raise SystemExit(f"FAIL: body cites dropped reference [{n}]")
        if n not in remap:
            raise SystemExit(f"FAIL: body cites unknown reference [{n}]")
    return "[" + ", ".join(str(remap[n]) for n in nums) + "]"


new_head = CITE.sub(fix, head)

OUT.write_text(new_head + "\n# References\n\n" + new_tail)
print(f"merged {len(parts)} files -> {OUT}")
print(f"references: {len(entries)} -> {len(kept)} (dropped {sorted(DROP)})")
