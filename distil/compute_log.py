"""Additive per-round COMPUTE telemetry for the D5 table (side-file only).

Writes one JSON object per round to ``<work_dir>/telemetry/compute.jsonl``. It never
touches ``result.json`` (whose ``history`` schema stays exactly as it was) and never
alters control flow — a telemetry failure is swallowed, like distil/p4/telemetry.py.

Why a side-file: a round's wall-clock is dominated by the from-scratch policy RETRAIN,
which BOTH DISTIL and the when-to-query baselines pay. So `sec` alone under-states
nothing and over-states nothing about the *method* — it just isn't the number that
characterises DISTIL's overhead. The field that does is `sec_reasoning`:

    sec_reasoning = sec_screen + sec_llm + sec_prescribe

i.e. rollout screening + VLM/analysis/decision calls + prescription & feasibility —
everything DISTIL does that a baseline does NOT. The baseline arm writes the same row
with sec_llm = 0.0 and sec_screen = its own gate cost, so the two are directly
comparable and the "reasoning overhead" is a difference of measured quantities.

Keys (all seconds are wall-clock, rounded to 3dp; absent => not applicable to the arm):
    round, arm, n_demos_at_eval, sec_total, sec_train, sec_eval, sec_screen,
    sec_llm, sec_prescribe, sec_reasoning, tokens
`tokens` is the verbatim usage dict from DistilLLM.decide() (prompt/completion/total/
calls/reasoning/by_stage{vlm,analysis,decision}/kag_calls/kag_chars), or {} when no LLM
call was made in that round.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def log_round(work_dir: str, row: Dict[str, Any]) -> None:
    """Append one compute row. Never raises."""
    try:
        d = Path(work_dir) / "telemetry"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "compute.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        pass


def round_row(rnd: int, arm: str, *, n_demos_at_eval: int, sec_total: float,
              sec_train: float = 0.0, sec_eval: float = 0.0, sec_screen: float = 0.0,
              sec_llm: float = 0.0, sec_prescribe: float = 0.0,
              tokens: Dict[str, Any] = None) -> Dict[str, Any]:
    r3 = lambda v: round(float(v), 3)  # noqa: E731
    return {
        "round": int(rnd), "arm": str(arm), "n_demos_at_eval": int(n_demos_at_eval),
        "sec_total": r3(sec_total), "sec_train": r3(sec_train), "sec_eval": r3(sec_eval),
        "sec_screen": r3(sec_screen), "sec_llm": r3(sec_llm),
        "sec_prescribe": r3(sec_prescribe),
        "sec_reasoning": r3(sec_screen + sec_llm + sec_prescribe),
        "tokens": tokens or {},
    }
