"""Measure the KAG block's exact token contribution against the SERVING tokenizer.

The KAG text is inserted verbatim into (a) every analysis prompt (one per analysed
failure) and (b) the single decision prompt. So its per-round cost is

    kag_tokens_per_round = kag_calls * tokens(kag_block)

where kag_calls = (#analysed failures + 1) is logged per round by DistilLLM.decide().

tokens(kag_block) is obtained here by a PAIRED PROMPT-TOKEN DIFF: the same prompt is
rendered with and without the KAG block and each variant is sent to the real endpoint
with max_tokens=1; OpenRouter reports usage.prompt_tokens for both. The difference is
the KAG cost as counted by the model that actually serves the run -- no local
tokenizer, no estimate.

    python -m distil.scripts.measure_kag_tokens --tasks Door Wipe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from distil.p4 import prompts as P            # noqa: E402
from distil.p4.kag import load_kag_text       # noqa: E402
from distil.run import _find_root, _load_dotenv  # noqa: E402

# A representative Stage-A output; the KAG delta is insertion-verbatim and so does not
# depend on this text, but using a realistic body keeps any tokenizer boundary effect
# in the measurement rather than idealising it away.
VLM_REPORT = (
    "In the start frame the gripper is above and behind the handle, roughly 12 cm away "
    "in +x. By the flagged frame the fingers have closed early, in free space, short of "
    "the handle; the handle has not moved. In the end frame the arm has drifted upward "
    "and the door remains fully closed. The failure is an early grasp commitment before "
    "the gripper reached the handle.")

MEMBERS = [
    {"ep_id": 3000001, "ox": 0.112, "oy": -0.043, "t_star": 41, "T": 350,
     "peak_loss": 1.2152,
     "analysis": {"root_cause": "grasp_failure", "phase": "pre_grasp"}},
    {"ep_id": 3000007, "ox": 0.128, "oy": -0.031, "t_star": 38, "T": 350,
     "peak_loss": 1.0913,
     "analysis": {"root_cause": "grasp_failure", "phase": "pre_grasp"}},
    {"ep_id": 3000019, "ox": 0.097, "oy": -0.055, "t_star": 44, "T": 350,
     "peak_loss": 0.9877,
     "analysis": {"root_cause": "grasp_failure", "phase": "pre_grasp"}},
]


def prompt_tokens(client, model, system, user):
    """Exact prompt-token count from the serving endpoint (1 sampled token)."""
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=1,
        extra_body={"reasoning": {"enabled": False}})
    return int(r.usage.prompt_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["Door", "Wipe"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    _load_dotenv(_find_root(), log=lambda m: None)
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"], timeout=120, max_retries=3)
    model = os.environ.get("LLM_MODEL_NAME", "qwen/qwen3-32b")

    out = {"llm_model": model, "method": "paired prompt-token diff (max_tokens=1)",
           "tasks": {}}
    for task in a.tasks:
        desc = P.TASK_DESC.get(task, task)
        kag = load_kag_text(task)
        rc, ph = P.enums_for(task)

        a_with = P.analysis_prompt(desc, kag, VLM_REPORT, root_causes=rc, phases=ph)
        a_without = P.analysis_prompt(desc, "", VLM_REPORT, root_causes=rc, phases=ph)
        d_with = P.decision_prompt(desc, kag, MEMBERS, True)
        d_without = P.decision_prompt(desc, "", MEMBERS, True)

        ta_w = prompt_tokens(client, model, P.ANALYSIS_SYSTEM, a_with)
        ta_o = prompt_tokens(client, model, P.ANALYSIS_SYSTEM, a_without)
        td_w = prompt_tokens(client, model, P.DECISION_SYSTEM, d_with)
        td_o = prompt_tokens(client, model, P.DECISION_SYSTEM, d_without)

        rec = {
            "kag_chars": len(kag),
            "analysis_prompt_tokens_with_kag": ta_w,
            "analysis_prompt_tokens_without_kag": ta_o,
            "analysis_kag_delta": ta_w - ta_o,
            "decision_prompt_tokens_with_kag": td_w,
            "decision_prompt_tokens_without_kag": td_o,
            "decision_kag_delta": td_w - td_o,
        }
        out["tasks"][task] = rec
        print(f"[{task}] kag_chars={rec['kag_chars']}  "
              f"analysis: {ta_o} -> {ta_w} (+{rec['analysis_kag_delta']})  "
              f"decision: {td_o} -> {td_w} (+{rec['decision_kag_delta']})")

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"[write] {a.out}")


if __name__ == "__main__":
    main()
