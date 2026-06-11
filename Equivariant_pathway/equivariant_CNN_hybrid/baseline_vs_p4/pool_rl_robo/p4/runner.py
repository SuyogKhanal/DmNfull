"""LLM transport for the fresh P4 components.

Wraps the fork's tested Responses-API clients (main_analysis.llm_clients:
VLMClient / ReasoningClient / PlainClient) — the same transport the proxy serves
(client.responses.create, <think> stripping, input_image.detail injection, the
PROXY_MAX_OUTPUT_TOKENS cap). The VLM model name routes to qwen3-vl-32b; reasoning
/ config (plain) route to qwen3-32b. Reads OPENAI_BASE_URL / model names from env
(set by run_pool_rl_robo.sh), so importing this never requires a live server.
"""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from typing import Optional

from ..envs import env_setup as E

E.bootstrap_fork_path()

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """Strip ``` fences / <think> remnants and return the first balanced JSON
    object, or None. Shared by the P4 components."""
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    i = text.find("{")
    while i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except Exception:
                        break
        i = text.find("{", i + 1)
    return None


def _cfg(model: str, effort: str = "high", max_output_tokens: int = 16384):
    return SimpleNamespace(model=model, effort=effort,
                           max_output_tokens=int(max_output_tokens))


class P4Client:
    """Three-headed client (VLM / reasoning / plain) for the P4 components."""

    def __init__(self, *, llm_model: Optional[str] = None,
                 vlm_model: Optional[str] = None, max_output_tokens: int = 16384):
        from diffdagger.main_analysis.llm_clients import (
            VLMClient, ReasoningClient, PlainClient,
        )
        self.llm_model = llm_model or os.environ.get("LLM_MODEL_NAME", "qwen3-32b")
        self.vlm_model = vlm_model or os.environ.get("VLM_MODEL_NAME", "qwen3-vl-32b")
        self.vlm = VLMClient(_cfg(self.vlm_model, "low", max_output_tokens))
        self.reasoner = ReasoningClient(_cfg(self.llm_model, "high", max_output_tokens))
        self.plain = PlainClient(_cfg(self.llm_model, "low", max_output_tokens))

    # ---- thin call wrappers (each returns the model's text) ----
    def analyze_frame(self, image_path: str, text_prompt: str,
                      system_prompt: Optional[str] = None) -> str:
        return self.vlm.analyze_frame(image_path, text_prompt,
                                      system_prompt=system_prompt).get("text", "")

    def reason(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        return self.reasoner.reason(user_prompt, system_prompt=system_prompt).get("text", "")

    def structure(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        return self.plain.structure(user_prompt, system_prompt=system_prompt).get("text", "")


def make_client(**kw) -> Optional[P4Client]:
    """Build a P4Client iff a server endpoint is configured, else None
    (callers fall back gracefully)."""
    if not os.environ.get("OPENAI_BASE_URL"):
        return None
    return P4Client(**kw)
