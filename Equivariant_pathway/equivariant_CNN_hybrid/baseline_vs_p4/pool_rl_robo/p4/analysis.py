"""Component 2a — AnalysisClass (root cause + trajectory phase).

The Reasoning LLM receives the VLM failure description (+ the task KAG) and
classifies the root-cause category and the trajectory phase, grounded in KAG facts.
"""
from __future__ import annotations

from typing import Dict

from . import prompts as P
from .runner import extract_json


class AnalysisClass:
    def __init__(self, client, task_description: str, kag_text: str):
        self.client = client
        self.task = task_description
        self.kag = kag_text

    def run(self, vlm_report: str) -> Dict:
        """Returns {root_cause, phase, rationale}, validated against the enums.
        Falls back to a safe default classification on any failure."""
        default = {"root_cause": "approach_failure", "phase": "pre_grasp",
                   "rationale": "fallback: LLM unavailable or unparseable."}
        if self.client is None:
            return default
        try:
            txt = self.client.reason(P.analysis_prompt(self.task, self.kag, vlm_report),
                                     system_prompt=P.ANALYSIS_SYSTEM)
            obj = extract_json(txt) or {}
        except Exception:
            return default
        rc = obj.get("root_cause")
        ph = obj.get("phase")
        if rc not in P.ROOT_CAUSES:
            rc = "approach_failure"
        if ph not in P.PHASES:
            ph = "pre_grasp"
        return {"root_cause": rc, "phase": ph,
                "rationale": str(obj.get("rationale", ""))[:300]}
