"""P4 ANALYSIS pass (VLM). For each top-K failure episode, Qwen3-VL ingests the
START / FIRST-MISTAKE / END frames + a numeric summary and returns a concise
failure analysis. This is the first of the two reasoning passes (analysis ->
prescription)."""
from __future__ import annotations

from typing import List

from ..envs.experts import ENV_PROMPTS

_SYSTEM = (
    "You are a robotics failure analyst. You are shown rendered frames of a "
    "behavior-cloning policy FAILING an episode (START, the FIRST-MISTAKE moment "
    "= highest action-discrepancy vs the expert, and END), plus a numeric "
    "summary. In 2-3 sentences name the failure MODE and what the policy should "
    "have done differently. Be concrete and specific to this environment."
)


def analyze_failure(env_id: str, summary_text: str, frames_b64: List[str], client) -> str:
    user = (
        f"Environment: {env_id}\n{ENV_PROMPTS.get(env_id, '')}\n\n"
        f"{summary_text}\n\n"
        "The attached images are the START, the FIRST-MISTAKE moment, and the END "
        "of the failed episode. Diagnose the failure."
    )
    try:
        return client.chat(_SYSTEM, user, images=frames_b64, max_tokens=512).strip()
    except Exception as e:  # noqa: BLE001
        return f"(VLM analysis unavailable: {type(e).__name__}: {e})"
