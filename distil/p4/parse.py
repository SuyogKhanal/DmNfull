"""SELECT / BRIDGE decision parsing — task-agnostic (ported verbatim from the
ManiSkill P4-LLM V3 hybrid, pool_rl_robo/p4_subtask/planner.py::_parse_choice).

The LLM emits a decision label like 'SELECT ep7' or 'BRIDGE ep3,ep9'. This pulls
the choice + the cited episode ids out of free text, tolerant of case, spacing,
punctuation, and an optional 'ep' prefix. Returns (None, []) if neither tag is
found (caller applies a geometric fallback)."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple


def parse_choice(label: str) -> Tuple[Optional[str], List[int]]:
    """('select', [id]) | ('bridge', [ids...up to 3]) | (None, [])."""
    s = str(label or "")
    m = re.search(r"\bSELECT\s*[: ]\s*(?:ep\s*)?(\d+)", s, re.IGNORECASE)
    if m:
        return "select", [int(m.group(1))]
    m = re.search(r"\bBRIDGE\s*[: ]\s*((?:(?:ep\s*)?\d+[,\s]*)+)", s, re.IGNORECASE)
    if m:
        ids = [int(x) for x in re.findall(r"\d+", m.group(1))]
        return ("bridge", ids[:3]) if ids else (None, [])
    return None, []
