#!/usr/bin/env python3
"""Collect ONE cell of the OpenRouter DISEIL re-run into cells/<Task>_<modality>.json.

Every number traces to a run that completed on this cluster. Nothing is estimated,
extrapolated or filled in by hand. A seed with no finished run is reported as UNRUN,
never invented.

Two result schemas:

  distil (Door)  -> <out>/result.json
      history[] rows carry `tokens` ({total, by_stage:{vlm,analysis,decision}}),
      `mode` ('select'|'bridge'|None) and `n_screen_failures`.

  pool_rl_robo (Push-T) -> <out>/p4_subtask/results/learning_curve.json
      + <out>/p4_subtask/results/telemetry/d5_events.jsonl (needs D5_TELEMETRY=1),
      whose `llm_call` events carry stage (VLM|Reasoning|Plain), round and tokens.

FALLBACK-ROUND RULE (the whole point of the audit). A round is:
  * llm_active   — the LLM was called: tokens > 0 for that round.
  * budget_free  — no usable failures were found, so there was nothing to prescribe and
                   the LLM was legitimately not called. NOT a fallback.
  * FALLBACK     — a demonstration was acquired WITHOUT the LLM having been called
                   (distil: mode is not None but the round drew 0 tokens). This is the
                   deterministic geometric planner standing in for DISEIL. Any run with
                   one or more such rounds is INVALID under the re-run protocol.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _distil_seed(run_dir: Path) -> Optional[Dict[str, Any]]:
    f = run_dir / "result.json"
    if not f.is_file():
        return None
    r = json.loads(f.read_text())
    hist: List[Dict[str, Any]] = r.get("history") or []

    llm_active = budget_free = fallback = 0
    vlm_tok = llm_tok = tot_tok = 0
    prescriptions = 0
    for h in hist:
        t = h.get("tokens") or {}
        total = int(t.get("total", 0) or 0)
        bs = t.get("by_stage") or {}
        vlm_tok += int((bs.get("vlm") or {}).get("total", 0) or 0)
        llm_tok += int((bs.get("analysis") or {}).get("total", 0) or 0)
        llm_tok += int((bs.get("decision") or {}).get("total", 0) or 0)
        tot_tok += total
        acquired = h.get("mode") is not None          # a demo was collected this round
        if total > 0:
            llm_active += 1
            if h.get("confidence") is not None:
                prescriptions += 1
        elif acquired:
            fallback += 1                              # demo acquired with NO LLM call
        else:
            budget_free += 1                           # nothing to prescribe

    n_init = int(r.get("n_init_demos") or 0)
    n_demos = int(r.get("n_demos") or 0)
    return {
        "final_success_rate": r.get("final_success"),
        "demos_acquired": n_demos - n_init,
        "n_init_demos": n_init,
        "budget": r.get("budget"),
        "rounds_total": len(hist),
        "rounds_llm_active": llm_active,
        "rounds_budget_free": budget_free,
        "rounds_fallback": fallback,
        "prescriptions": prescriptions,
        "vlm_tokens": vlm_tok,
        "llm_tokens": llm_tok,
        "total_tokens": tot_tok,
        "wall_sec": r.get("wall_sec"),
    }


def _pusht_seed(run_dir: Path) -> Optional[Dict[str, Any]]:
    """run_dir = <POOL_RESULTS_ROOT>/PushT-v1/run_<id>/p4_subtask/"""
    lc = run_dir / "results" / "learning_curve.json"
    if not lc.is_file():
        return None
    c = json.loads(lc.read_text())
    rows = c.get("rounds") or c.get("history") or []

    # per-round LLM token/call records (D5_TELEMETRY=1)
    tele = run_dir / "results" / "telemetry" / "d5_events.jsonl"
    per_round: Dict[int, Dict[str, int]] = {}
    vlm_tok = llm_tok = tot_tok = calls = 0
    if tele.is_file():
        for line in tele.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("kind") != "llm_call":
                continue
            rnd = int(e.get("round", -1))
            p = int(e.get("prompt_tokens", 0) or 0)
            comp = int(e.get("completion_tokens", 0) or 0)
            n = p + comp
            stage = str(e.get("stage", ""))
            d = per_round.setdefault(rnd, {"vlm": 0, "llm": 0, "calls": 0})
            if stage.upper() == "VLM":
                d["vlm"] += n; vlm_tok += n
            else:
                d["llm"] += n; llm_tok += n
            d["calls"] += 1
            tot_tok += n; calls += 1

    llm_active = sum(1 for v in per_round.values() if v["vlm"] + v["llm"] > 0)
    demos = c.get("final_performance", {}).get("n_queries")
    if demos is None:
        demos = sum(int(r.get("demos_added", 0) or 0) for r in rows)
    # rounds that acquired a demo but drew no tokens = deterministic fallback
    fallback = 0
    for r in rows:
        rnd = int(r.get("round", -1))
        if int(r.get("demos_added", 0) or 0) > 0:
            v = per_round.get(rnd, {"vlm": 0, "llm": 0})
            if v["vlm"] + v["llm"] == 0:
                fallback += 1

    sr = c.get("final_performance", {}).get("success_rate")
    if sr is None:
        sr = c.get("final_heldout_success_rate")
    return {
        "final_success_rate": sr,
        "demos_acquired": demos,
        "n_init_demos": c.get("initial_demos", 20),
        "budget": c.get("budget"),
        "rounds_total": len(rows),
        "rounds_llm_active": llm_active,
        "rounds_budget_free": max(0, len(rows) - llm_active - fallback),
        "rounds_fallback": fallback,
        "llm_calls": calls,
        "vlm_tokens": vlm_tok,
        "llm_tokens": llm_tok,
        "total_tokens": tot_tok,
        "stop_reason": c.get("stop_reason") or c.get("stopped_reason"),
        "telemetry_present": tele.is_file(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["Door", "PushT"])
    ap.add_argument("--modality", required=True, choices=["state", "image"])
    ap.add_argument("--seed-dir", action="append", required=True,
                    help="one per seed, in seed order: seed=<N>:<path>")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seeds: Dict[str, Any] = {}
    for spec in a.seed_dir:
        tag, path = spec.split(":", 1)
        n = int(tag.split("=")[1])
        d = Path(path)
        rec = _distil_seed(d) if a.task == "Door" else _pusht_seed(d)
        seeds[str(n)] = rec if rec is not None else {"status": "UNRUN",
                                                     "reason": f"no result file under {d}"}

    done = {k: v for k, v in seeds.items() if "status" not in v}
    srs = [v["final_success_rate"] for v in done.values()
           if v.get("final_success_rate") is not None]

    valid = {k: v for k, v in done.items() if v.get("rounds_fallback", 0) == 0
             and v.get("demos_acquired") == a.budget}
    vsrs = [v["final_success_rate"] for v in valid.values()
            if v.get("final_success_rate") is not None]

    cell = {
        "task": a.task,
        "modality": a.modality,
        "method": "DISEIL",
        "arm_in_code": "full" if a.task == "Door" else "p4_subtask",
        "api": "OpenRouter",
        "vlm_model": "qwen/qwen3-vl-30b-a3b-instruct",
        "llm_model": "qwen/qwen3-32b",
        "budget_B": a.budget,
        "demos_per_round_D": 1,
        "heldout_episodes": 100,
        "seeds": seeds,
        "n_seeds_completed": len(done),
        "n_seeds_valid": len(valid),
        "final_success_rate_mean": round(statistics.mean(vsrs), 4) if vsrs else None,
        "final_success_rate_std": (round(statistics.stdev(vsrs), 4)
                                   if len(vsrs) > 1 else (0.0 if vsrs else None)),
        "final_success_rate_mean_all_completed": round(statistics.mean(srs), 4) if srs else None,
        "total_fallback_rounds": sum(v.get("rounds_fallback", 0) for v in done.values()),
        "note": ("mean/std are over VALID seeds only: zero fallback rounds AND the full "
                 "budget acquired. Seeds failing either test are listed but excluded."),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cell, indent=2))
    print(json.dumps(cell, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
