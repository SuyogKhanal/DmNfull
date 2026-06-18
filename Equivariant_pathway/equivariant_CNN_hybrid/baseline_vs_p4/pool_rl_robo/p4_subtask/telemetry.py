"""Exhaustive per-round telemetry — "log every minute detail".

Writes, under ``<work_dir>/telemetry/``:
  * ``round_<n>.jsonl``  — one JSON object per logged event (machine-readable).
  * ``round_<n>.log``    — the same, human-readable.
  * ``subtask_summary.jsonl`` — one compact row per round (quick scan).

Every write is defensive: a telemetry failure must never crash the pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class Telemetry:
    def __init__(self, work_dir: str):
        self.root = Path(work_dir) / "telemetry"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._t0 = {}

    # ---- timing ----------------------------------------------------------
    def tic(self, key: str) -> None:
        self._t0[key] = time.time()

    def toc(self, key: str) -> float:
        t = self._t0.get(key)
        return round(time.time() - t, 3) if t is not None else -1.0

    # ---- writers ---------------------------------------------------------
    def event(self, rnd: int, kind: str, payload: Dict[str, Any]) -> None:
        rec = {"round": int(rnd), "kind": str(kind), **payload}
        try:
            with open(self.root / f"round_{int(rnd):04d}.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            with open(self.root / f"round_{int(rnd):04d}.log", "a") as f:
                f.write(f"[{kind}] " + json.dumps(payload, indent=2, default=str) + "\n")
        except Exception:
            pass

    def summary_row(self, payload: Dict[str, Any]) -> None:
        try:
            with open(self.root / "subtask_summary.jsonl", "a") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            pass
