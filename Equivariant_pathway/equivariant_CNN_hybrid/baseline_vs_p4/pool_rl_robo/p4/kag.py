"""KAG (knowledge-augmented generation) loader for P4-LLM.

Each environment has a structured Markdown knowledge document at
``p4/kag/{env_id}.md`` holding environment facts as key-value pairs (task,
observation/action semantics, reward, success criterion, what a good
demonstration looks like, novice failure modes). The P4 prompt injects the
matching env's KAG so the LLM reasons with concrete, environment-specific
knowledge rather than a vague generic description.

Modular + configurable: the KAG directory defaults to this package's ``kag/``
but can be overridden via ``set_kag_dir`` (wired from ``config.yaml::p4.kag_dir``
by ``p4/pipeline_p4.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

_DEFAULT_KAG_DIR = Path(__file__).resolve().parent / "kag"
_KAG_DIR = _DEFAULT_KAG_DIR
_WARNED: Set[str] = set()


def set_kag_dir(d: Optional[str]) -> None:
    global _KAG_DIR
    _KAG_DIR = Path(d) if d else _DEFAULT_KAG_DIR


def kag_dir() -> Path:
    return _KAG_DIR


def kag_path(env_id: str) -> Path:
    return _KAG_DIR / f"{env_id}.md"


def load_kag(env_id: str, resolved_env_id: Optional[str] = None) -> str:
    """Return the KAG markdown for ``env_id`` (trying the resolved id as a
    fallback). Empty string + a one-time warning if no doc exists."""
    for eid in (env_id, resolved_env_id):
        if not eid:
            continue
        p = _KAG_DIR / f"{eid}.md"
        if p.exists():
            return p.read_text(encoding="utf-8")
    if env_id not in _WARNED:
        print(f"[p4-kag] WARNING: no KAG document for {env_id!r} in {_KAG_DIR}", flush=True)
        _WARNED.add(env_id)
    return ""
