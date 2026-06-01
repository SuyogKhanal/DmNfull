#!/usr/bin/env python
"""LLM smoke test (run AFTER the expert smoke test).

Two checks:
1. TOKEN BUDGET — build the worst-case P4-LLM prompt for every env (full
   max_candidates pool + the env's injected KAG document) and tokenize it with
   the real Qwen3 tokenizer; confirm prompt_tokens + output cap stays under the
   40960 context. This catches the KAG docs blowing the budget.
2. LIVE (optional) — if OPENAI_BASE_URL points at a running vLLM server, do one
   real P4 selection round-trip and confirm valid JSON is returned.

    python smoke_llm.py                 # token-budget check (+ live if a server is up)
    python smoke_llm.py --live
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.envs.experts import (  # noqa: E402
    EXPERTS,
)
from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.p4 import (  # noqa: E402
    kag, prompts,
)

MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "/weka/s226137394/models/Qwen3-32B")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "40960"))
OUTPUT_CAP = int(os.environ.get("PROXY_MAX_OUTPUT_TOKENS", "8192"))


def synth_records(env_id: str, n: int = 120):
    """A realistic, oversized candidate pool (the prompt keeps the top
    max_candidates by discrepancy — the worst case for prompt length)."""
    rng = np.random.default_rng(0)
    is_fetch = env_id.startswith("Fetch")
    return [{"t": t, "discrepancy": float(rng.uniform(0, 2)),
             "return_so_far": float(-t * 0.5), "reward_to_go": float(-(n - t) * 0.5),
             "goal_dist": (float(rng.uniform(0, 0.5)) if is_fetch else None)}
            for t in range(n)]


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        print(f"[token] real tokenizer unavailable ({type(e).__name__}: {e}); "
              f"falling back to a chars/4 estimate", flush=True)
        return None


def _flatten_ids(ids):
    """Normalize apply_chat_template / tokenizer output to a flat id list.
    transformers 5.x may return a dict/BatchEncoding or a batched [[...]]."""
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return ids


def count_tokens(tok, system: str, user: str):
    if tok is None:
        return (len(system) + len(user)) // 4, True
    try:
        out = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True)
        return len(_flatten_ids(out)), False
    except Exception:  # noqa: BLE001
        return len(_flatten_ids(tok(system + "\n" + user))), False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_candidates", type=int, default=prompts.MAX_CANDIDATES)
    ap.add_argument("--live", action="store_true",
                    help="also do a live LLM round-trip (needs OPENAI_BASE_URL)")
    args = ap.parse_args()

    tok = get_tokenizer()
    input_budget = MAX_MODEL_LEN - OUTPUT_CAP
    print(f"Qwen3 ctx={MAX_MODEL_LEN}  output_cap={OUTPUT_CAP}  -> input budget={input_budget} tokens\n")
    worst = 0
    ok = True
    for env in EXPERTS:
        recs = synth_records(env)
        ci = prompts.candidate_indices(recs)[:args.max_candidates]
        kt = kag.load_kag(env)
        user = prompts.build_user_prompt(env, recs, ci, kt)
        n, approx = count_tokens(tok, prompts.SYSTEM, user)
        worst = max(worst, n)
        over = (n + OUTPUT_CAP) >= MAX_MODEL_LEN
        ok = ok and not over
        tag = " (~chars/4)" if approx else ""
        print(f"{env:24s} kag={len(kt):5d}ch cand={len(ci):2d} "
              f"prompt_tokens={n:6d}{tag}  +cap={n + OUTPUT_CAP:6d}/{MAX_MODEL_LEN}  "
              f"[{'OVER BUDGET' if over else 'OK'}]")
    print(f"\nworst-case prompt = {worst} tokens; worst + output cap = {worst + OUTPUT_CAP} "
          f"vs ctx {MAX_MODEL_LEN}  ->  {'PASS (budget OK)' if ok else 'FAIL (over budget)'}")

    if args.live or os.environ.get("OPENAI_BASE_URL"):
        base = os.environ.get("OPENAI_BASE_URL")
        if not base:
            print("\n[live] OPENAI_BASE_URL unset -> skipping live round-trip. Launch a vLLM "
                  "server (submit_one_qwen.sh on a GPU node) and set OPENAI_BASE_URL to test live.")
        else:
            from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.p4 import (
                selector as sel,
            )
            from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.p4.runner import (
                QwenClient,
            )
            env = "FetchPickAndPlace-v4"
            recs = synth_records(env)
            print(f"\n[live] {base} :: P4 selection round-trip on {env} ...")
            idx = sel.select(env, recs, QwenClient(), random.Random(0))
            print(f"[live] selected record index = {idx} of {len(recs)}  "
                  f"-> {'OK (valid JSON in range)' if 0 <= idx < len(recs) else 'fell back'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
