"""P4-LLM demonstration selector (continuous-control analogue of the maze
``p4/pipeline_p4`` prescription step). Returns an index into ``records``; robust
by construction — malformed / out-of-range / failed LLM responses fall back to
the highest-discrepancy state (the maze 'infeasible -> fall back' discipline).
Budget is identical to the baselines: one state (one demonstration) per round.
"""
from __future__ import annotations

import time
from typing import List, Optional

from . import kag, prompts


def select(env_id: str, records: List[dict], llm_client, rng,
           resolved_env_id: Optional[str] = None) -> int:
    if not records:
        return -1
    cand_idx = prompts.candidate_indices(records)
    fallback = cand_idx[0]
    if llm_client is None:
        return fallback
    kag_text = kag.load_kag(env_id, resolved_env_id)
    user = prompts.build_user_prompt(env_id, records, cand_idx, kag_text)
    try:
        obj = llm_client.chat_json(prompts.SYSTEM, user)
        si = int(obj.get("selected_index"))
        if 0 <= si < len(cand_idx):
            return cand_idx[si]
        print(f"[{time.strftime('%H:%M:%S')}] [p4-llm] selected_index {si} out of range; fallback", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{time.strftime('%H:%M:%S')}] [p4-llm] LLM select failed ({e}); fallback to highest-discrepancy", flush=True)
    return fallback
