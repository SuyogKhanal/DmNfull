"""Append-only JSON log of LLM prescription reasoning, for manual overlap
inspection on the first ``prescription_overlap_runs`` runs (default 3).

Each entry records:
  {
    "round": int,
    "prescribed_layouts": [
        {"start_pos": [r,c], "goal_pos": [r,c], "fire_positions": [...],
         "steps": "..."},
        ...
    ],
    "llm_reasoning_text": str,   # raw aggregator output
    "compression_ratio": float
  }

We write the file as a single JSON document with a top-level "rounds"
list so it stays valid JSON even mid-run. Each write rewrites the file —
the per-run round count is small (≤ max_rounds ≤ 50) so this is cheap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class PrescriptionOverlapLog:
    def __init__(self, path: Union[str, Path], enabled: bool = True):
        self.path = Path(path)
        self.enabled = bool(enabled)
        self._rounds: List[Dict] = []
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    doc = json.load(f) or {}
                self._rounds = list(doc.get("rounds", []) or [])
            except (json.JSONDecodeError, OSError):
                self._rounds = []
        else:
            self._flush()

    def write_round(
        self,
        round_idx: int,
        prescribed_layouts: List[Dict],
        llm_reasoning_text: str,
        compression_ratio: Optional[float] = None,
        extras: Optional[Dict] = None,
    ) -> None:
        if not self.enabled:
            return
        entry: Dict = {
            "round": int(round_idx),
            "prescribed_layouts": prescribed_layouts,
            "llm_reasoning_text": str(llm_reasoning_text or ""),
        }
        if compression_ratio is not None:
            entry["compression_ratio"] = float(compression_ratio)
        if extras:
            entry["extras"] = extras
        self._rounds.append(entry)
        self._flush()

    def _flush(self) -> None:
        with open(self.path, "w") as f:
            json.dump({"rounds": self._rounds}, f, indent=2, default=str)
