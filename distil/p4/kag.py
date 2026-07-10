"""KAG (Knowledge Anchoring Graph) grounding — ported from the PushT p4/kag.py.

Each task has a structured graph JSON (kag/<task>.json) with meta/nodes/edges/
reasoning_implications; format_kag_context renders it to the prompt text that the
reasoning + decision stages consume, so the LLM grounds its analysis on REAL
workspace bounds and failure-mode rules rather than hallucinated numbers.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

_KAG_DIR = Path(__file__).resolve().parent / "kag"


def load_kag_graph(task: str) -> Dict:
    p = _KAG_DIR / f"{task}.json"
    if p.is_file():
        return json.loads(p.read_text())
    print(f"[p4-kag] WARNING: no KAG graph at {p}")
    return {}


def format_kag_context(kag: Dict) -> str:
    if not kag:
        return ""
    meta = kag.get("meta", {})
    nodes = kag.get("nodes", [])
    edges = kag.get("edges", [])
    impl = kag.get("reasoning_implications", {})
    nl = {n["id"]: n for n in nodes}
    lines = ["=== KAG — TASK KNOWLEDGE GRAPH ===",
             f"Domain: {meta.get('domain', '')}",
             f"Description: {meta.get('description', '')}", ""]
    by_type: Dict[str, list] = {}
    for n in nodes:
        by_type.setdefault(n.get("type", "Node"), []).append(n)
    for t, ns in by_type.items():
        lines.append(f"[{t}]")
        for n in ns:
            props = ", ".join(f"{kk}={vv}" for kk, vv in n.get("properties", {}).items())
            lines.append(f"  - {n.get('label', n['id'])} (id={n['id']}): {props}")
        lines.append("")
    if edges:
        lines.append("[RELATIONS]")
        for e in edges:
            src = nl.get(e.get("source"), {}).get("label", e.get("source"))
            tgt = nl.get(e.get("target"), {}).get("label", e.get("target"))
            lines.append(f"  {src} --[{e.get('relation', 'RELATED_TO')}]--> {tgt}")
        lines.append("")
    if impl:
        lines.append("[REASONING IMPLICATIONS]")
        for k, v in impl.items():
            lines.append(f"  * {k}: {v}")
    return "\n".join(lines)


@lru_cache(maxsize=8)
def load_kag_text(task: str) -> str:
    return format_kag_context(load_kag_graph(task))
