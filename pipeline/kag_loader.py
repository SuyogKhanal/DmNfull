import json
from pathlib import Path
from typing import Dict, Optional


def load_kag_document(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"KAG document not found at {p}")
    with open(p, "r") as f:
        return json.load(f)


def _node_lookup(kag: Dict) -> Dict[str, Dict]:
    return {n["id"]: n for n in kag.get("nodes", [])}


def format_kag_context(kag: Dict) -> str:
    """Render the KAG graph as a human-readable, LLM-friendly context string."""
    meta = kag.get("meta", {})
    schema = kag.get("schema", {})
    nodes = kag.get("nodes", [])
    edges = kag.get("edges", [])
    reasoning_implications = kag.get("reasoning_implications", {})
    nl = _node_lookup(kag)

    lines = []
    lines.append("=== KAG — ENVIRONMENT KNOWLEDGE GRAPH ===")
    lines.append(f"Domain: {meta.get('domain', 'dynamic_maze_navigation')}")
    lines.append(f"Coordinate system: {meta.get('coordinate_system', '(row, col), origin top-left')}")
    lines.append(f"Grid shape: {meta.get('grid_shape', [5, 5])}")
    lines.append(f"Description: {meta.get('description', '')}")
    lines.append("")

    by_type: Dict[str, list] = {}
    for n in nodes:
        by_type.setdefault(n["type"], []).append(n)

    type_order = [t["type"] for t in schema.get("node_types", [])]
    for t in type_order:
        if t not in by_type:
            continue
        lines.append(f"[{t}]")
        for n in by_type[t]:
            props = n.get("properties", {})
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
            lines.append(f"  - {n['label']} (id={n['id']}): {prop_str}")
        lines.append("")

    lines.append("[RELATIONS]")
    for e in edges:
        src = nl.get(e["source"], {}).get("label", e["source"])
        tgt = nl.get(e["target"], {}).get("label", e["target"])
        rel = e.get("relation", "RELATED_TO")
        props = e.get("properties", {})
        extra = f"  {props}" if props else ""
        lines.append(f"  {src} --[{rel}]--> {tgt}{extra}")
    lines.append("")

    if reasoning_implications:
        lines.append("[REASONING IMPLICATIONS]")
        for k, v in reasoning_implications.items():
            lines.append(f"  * {k}: {v}")
        lines.append("")

    return "\n".join(lines)


def load_and_format(path: str) -> str:
    return format_kag_context(load_kag_document(path))


def save_kag_context(text: str, episode_dir: Path) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)
    out = episode_dir / "kag_context.txt"
    out.write_text(text, encoding="utf-8")
    return out