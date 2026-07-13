"""Build the Push-T / image row of D5_Compute from the measured telemetry.

Consumes the two side files written by
  Equivariant_pathway/.../pool_rl_robo/telemetry_d5.py  (D5_TELEMETRY=1):

  results/PushT-v1/run_<RUN_ID>/p4_subtask/results/telemetry/d5_events.jsonl   (DISEIL)
  results/PushT-v1/run_<RUN_ID>/safe_dagger/results/telemetry/d5_events.jsonl  (baseline)

and distil/results/_compute/kag_tokens_pusht.json (measure_kag_tokens_pusht.py).

Definitions (invents nothing; every number traces to a logged event):

  DISEIL total s/round     round_start(1) -> end of the LAST span of round 1
                           (rollout_and_detect + analyze_and_prescribe +
                            collect_prescribed_demos + train_policy +
                            evaluate_heldout). The from-scratch retrain dominates
                            it and BOTH arms pay it.
  DISEIL reasoning s/round sum(analyze_and_prescribe) + sum(collect_prescribed_demos)
                           = rollout analysis + VLM + reasoning LLM + prescription
                           + feasibility/expert-solve. This is the DISEIL-SPECIFIC
                           cost; it is reported separately from the total.
  Baseline total s/round   first iil_episode AFTER collect_initial_demos ->
                           end of the round's evaluate_heldout. (The bootstrap
                           demo collection is logged as its own span and excluded
                           from both arms.)
  Overhead x               DISEIL total s/round / Baseline total s/round.

  VLM tok/round            sum(prompt+completion) over stage=="VLM" calls.
  LLM tok/round            sum(prompt+completion) over stage in {Reasoning, Plain}.
  Reasoning LLM tok/round  sum over stage=="Reasoning" calls of
                           completion_tokens - tokens(visible completion_text).
                           The Qwen proxy strips <think>...</think> from the text
                           but vLLM still bills those tokens in output_tokens, so
                           the difference IS the hidden thinking-token count.
                           Needs the serving tokenizer -> run with the vllm_embed
                           interpreter; without it the field is UNMEASURED.
  KAG tok/round            (# calls with kag_in_prompt) * kag_tokens.

Usage:
  /home/s226137394/.conda/envs/vllm_embed/bin/python \
      distil/scripts/build_pusht_d5_row.py --run-id 511
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DMN = Path("/weka/s226137394/DmNfull")
SUITE = DMN / "Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo"
KAG_JSON = DMN / "distil/results/_compute/kag_tokens_pusht.json"
OUT = DMN / "distil/results/_compute/PushT_image_d5.json"

UNMEASURED = "UNMEASURED"


def _events(run_id: int, method: str) -> List[Dict[str, Any]]:
    p = (SUITE / "results" / "PushT-v1" / f"run_{run_id}" / method /
         "results" / "telemetry" / "d5_events.jsonl")
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _spans(ev, stage):
    return [e for e in ev if e.get("kind") == "span" and e.get("stage") == stage]


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("/weka/s226137394/models/Qwen3-32B")
    except Exception:
        return None


def diseil(ev, tok) -> Dict[str, Any]:
    if not ev:
        return {"status": "no telemetry"}
    starts = [e for e in ev if e.get("kind") == "round_start"
              and int(e.get("round", 0)) >= 1]
    if not starts:
        return {"status": "no round_start event (arm never reached round 1)"}
    r1 = min(int(e["round"]) for e in starts)
    t0 = min(float(e["ts"]) for e in starts if int(e["round"]) == r1)
    in_r1 = [e for e in ev if int(e.get("round", -1)) == r1]

    spans = [e for e in in_r1 if e.get("kind") == "span"]
    if not spans:
        return {"status": "round 1 logged no spans"}
    t_end = max(float(e["ts_end"]) for e in spans)

    def s(stage):
        return round(sum(float(e["dur_s"]) for e in spans
                         if e.get("stage") == stage), 1)

    calls = [e for e in in_r1 if e.get("kind") == "llm_call"
             and "error" not in e]
    vlm = [c for c in calls if c.get("stage") == "VLM"]
    reason = [c for c in calls if c.get("stage") == "Reasoning"]
    plain = [c for c in calls if c.get("stage") == "Plain"]

    def tot(cs):
        return sum(int(c.get("prompt_tokens", 0)) +
                   int(c.get("completion_tokens", 0)) for c in cs)

    think = UNMEASURED
    if tok is not None:
        n = 0
        for c in reason:
            vis = len(tok(c.get("completion_text", "") or "",
                          add_special_tokens=False)["input_ids"])
            n += max(0, int(c.get("completion_tokens", 0)) - vis)
        think = n

    kag_calls = sum(1 for c in calls if c.get("kag_in_prompt"))
    kag_tok = UNMEASURED
    if KAG_JSON.is_file():
        kag_tok = kag_calls * int(json.loads(KAG_JSON.read_text())["kag_tokens"])

    reasoning_only = round(s("analyze_and_prescribe") +
                           s("collect_prescribed_demos"), 1)
    return {
        "status": "measured",
        "round": r1,
        "total_s_per_round": round(t_end - t0, 1),
        "reasoning_s_per_round": reasoning_only,
        "breakdown_s": {
            "rollout_and_detect": s("rollout_and_detect"),
            "analyze_and_prescribe": s("analyze_and_prescribe"),
            "collect_prescribed_demos": s("collect_prescribed_demos"),
            "train_policy": s("train_policy"),
            "evaluate_heldout": s("evaluate_heldout"),
        },
        "n_calls": {"vlm": len(vlm), "reasoning": len(reason),
                    "plain": len(plain), "kag_carrying": kag_calls},
        "vlm_tokens_per_round": tot(vlm),
        "llm_tokens_per_round": tot(reason) + tot(plain),
        "reasoning_llm_tokens_per_round": think,
        "kag_tokens_per_round": kag_tok,
        "models": sorted({c.get("model", "?") for c in calls}),
    }


def baseline(ev) -> Dict[str, Any]:
    if not ev:
        return {"status": "no telemetry"}
    spans = [e for e in ev if e.get("kind") == "span"]
    boot = _spans(ev, "collect_initial_demos")
    eps = _spans(ev, "iil_episode")
    tr = _spans(ev, "train_policy")
    ho = _spans(ev, "evaluate_heldout")
    if not (eps and tr and ho):
        return {"status": "incomplete telemetry (need iil_episode + "
                          "train_policy + evaluate_heldout)"}
    boot_end = max((float(e["ts_end"]) for e in boot), default=None)
    first_ep = min(float(e["ts_start"]) for e in eps
                   if boot_end is None or float(e["ts_start"]) >= boot_end)
    last_ho = max(float(e["ts_end"]) for e in ho)
    return {
        "status": "measured",
        "total_s_per_round": round(last_ho - first_ep, 1),
        "breakdown_s": {
            "rollout_episodes_until_query":
                round(sum(float(e["dur_s"]) for e in eps), 1),
            "n_rollout_episodes": len(eps),
            "train_policy": round(sum(float(e["dur_s"]) for e in tr), 1),
            "evaluate_heldout": round(sum(float(e["dur_s"]) for e in ho), 1),
        },
        "bootstrap_collect_s_excluded":
            round(sum(float(e["dur_s"]) for e in boot), 1),
        "vlm_tokens_per_round": 0,
        "llm_tokens_per_round": 0,
        "reasoning_llm_tokens_per_round": 0,
        "kag_tokens_per_round": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, default=511)
    a = ap.parse_args()

    tok = _tokenizer()
    d = diseil(_events(a.run_id, "p4_subtask"), tok)
    b = baseline(_events(a.run_id, "safe_dagger"))

    row: Dict[str, Any] = {"task": "PushT", "modality": "image",
                           "run_id": a.run_id, "baseline_arm": "safe_dagger",
                           "diseil": d, "baseline": b}
    if d.get("status") == "measured" and b.get("status") == "measured" \
            and b["total_s_per_round"] > 0:
        row["overhead_x"] = round(d["total_s_per_round"] /
                                  b["total_s_per_round"], 2)
    else:
        row["overhead_x"] = UNMEASURED
    if tok is None:
        row["tokenizer_note"] = (
            "transformers unavailable -> hidden-thinking (reasoning) tokens are "
            "UNMEASURED. Re-run with /home/s226137394/.conda/envs/vllm_embed/bin/python.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
