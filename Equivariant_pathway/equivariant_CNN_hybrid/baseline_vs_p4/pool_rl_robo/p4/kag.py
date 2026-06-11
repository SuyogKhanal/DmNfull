"""KAG (Knowledge-Augmented Generation) for P4-LLM — per-ManiSkill-task.

Each task has a structured knowledge GRAPH at ``p4/kag/{suite_env}.json`` (the
paper-facing KAG: nodes/edges/reasoning_implications, mirroring the maze
``kag_maze_knowledge.json`` schema) describing the robot, objects, controller,
observation layout, success condition, and failure taxonomy. ``format_kag_context``
renders it to the prompt text the fresh P4 components inject.

The fork's tested P4 engine (used for the first PushT submission) reads a TEXT KAG
doc via ``analyzer.kag_path``. ``kag_text_path`` returns one:
  * PushT-v1 → the fork's complete, tested ``kag_document.txt`` (reused as-is).
  * other tasks → the rendered text of this package's structured JSON graph.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from ..envs import env_setup as E

_KAG_DIR = Path(__file__).resolve().parent / "kag"
_FORK_PUSHT_KAG = (E.FORK_ROOT / "diffdagger" / "main_analysis" / "kag_document.txt")


def kag_json_path(suite_env: str) -> Path:
    return _KAG_DIR / f"{suite_env}.json"


def load_kag_graph(suite_env: str) -> Dict:
    p = kag_json_path(suite_env)
    if p.is_file():
        return json.loads(p.read_text())
    print(f"[p4-kag] WARNING: no KAG graph at {p}")
    return {}


def format_kag_context(kag: Dict) -> str:
    """Render a structured KAG graph into prompt text (nodes by type, relations,
    reasoning implications). Same rendering contract as the maze KAG."""
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
        for kk, vv in impl.items():
            lines.append(f"  * {kk}: {vv}")
        lines.append("")
    return "\n".join(lines)


def load_kag_text(suite_env: str) -> str:
    return format_kag_context(load_kag_graph(suite_env))


def kag_text_path(suite_env: str) -> Optional[str]:
    """Path to a TEXT KAG doc the fork P4 engine reads (analyzer.kag_path). EVERY env
    (incl. PushT-v1) renders ITS OWN structured graph at p4/kag/{suite_env}.json — which
    carries the task's object spawn box + arm/TCP bounds — to a cached .kag.txt. (The
    fork's generic kag_document.txt is NO LONGER used for PushT: it lacks the per-task
    bounds, which is why prescriptions weren't grounded.) Returns None if no graph exists."""
    txt = load_kag_text(suite_env)
    if not txt:
        print(f"[p4-kag] WARNING: no KAG for {suite_env}; LLM gets task_description only")
        return None
    out = _KAG_DIR / f"{suite_env}.kag.txt"
    out.write_text(txt)
    return str(out)
