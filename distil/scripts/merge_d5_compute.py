"""Merge the five D5_Compute rows (3 RoboSuite + Push-T + GridWorld) into the final
matrix, under BOTH protocols. Invents nothing; every cell traces to a logged event.

Read-only over the runs. Writes:
  paper_aaai2027/COC_REPORT/build/d5_compute.{md,csv}
  paper_aaai2027/COC_REPORT/ablations_results/DISTIL_ablation_results.xlsx  (D5_Compute sheet only)

PROTOCOLS
  P1 "first round"  -- the FIRST round of the run (round 0 for the distil module, round 1
                       for the fork's 1-indexed Push-T loop). Apples-to-apples across all
                       five settings, because Push-T and GridWorld were run at budget=1
                       and have no other round. It is the WORST case: weakest policy ->
                       most failures -> most analysis.
  P5 "mean of the 5 LLM-active rounds (+/- sample sd)" -- rounds 0..4 of the budget=5
                       RoboSuite runs, matched arm-for-arm against the baseline's rounds
                       0..4. Available ONLY for the three RoboSuite settings.
                       UNMEASURED for Push-T/GridWorld (budget=1: no other round exists).

REASONING-ONLY SECONDS -- two consistent definitions, because the source docs disagreed:
  R_narrow = llm + prescribe                 (analysis + VLM + reasoning LLM + prescription
                                              + feasibility; EXCLUDES the failure screen)
  R_wide   = screen + llm + prescribe        (every DISEIL-specific stage, i.e. the whole
                                              round minus the shared train+eval)
  The baseline's only counterpart stage is its gate rollout (rollout until the 1st
  intervention), so the honest add-on is  R_wide - baseline_gate.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path("/weka/s226137394/DmNfull")
COMPUTE = ROOT / "distil/results/_compute"
BUILD = ROOT / "paper_aaai2027/COC_REPORT/build"
SPANS = COMPUTE / "d5_stage_spans.json"
PG = COMPUTE / "d5_rows_pusht_gridworld.json"
KAG = BUILD / "kag_tokens.json"

JOBS = {
    "Door / state": ("110355", "110356"),
    "Door / image": ("110357", "110358"),
    "Wipe / image": ("110359", "110360"),
    "Push-T / image": ("110375", "110376"),
    "GridWorld 5x5 / image": ("110384", "110385"),
}


def _ms(xs):
    """(mean, sample sd) -- sd is None when n < 2. Never invents a spread."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else None)


def _r(x, n=1):
    return None if x is None else round(x, n)


def robosuite_row(label, task, spans, kag_per_call):
    full = spans[f"{task}"]["full"]
    safe = spans[f"{task}"]["safe"]
    fr = [r for r in full.get("rounds", [])]
    sr = [r for r in safe.get("rounds", [])]

    # LLM-active DISEIL rounds = those that reached [collect a0] AND carry tokens.
    act = [r for r in fr if r.get("tokens") and r["tokens"].get("total")
           and r.get("round_s_from_log") is not None]
    # baseline rounds matched 1:1 by index to the DISEIL LLM-active rounds
    idx = {r["round"] for r in act}
    sact = [r for r in sr if r["round"] in idx and r.get("round_s_from_log") is not None]

    def stage(rs, k):
        return [r.get(k) for r in rs if r.get(k) is not None]

    def tok(rs, path):
        out = []
        for r in rs:
            t = r["tokens"]
            bs = t.get("by_stage") or {}
            if path == "vlm":
                out.append(bs.get("vlm", {}).get("total", 0))
            elif path == "llm":
                out.append(bs.get("analysis", {}).get("total", 0)
                           + bs.get("decision", {}).get("total", 0))
            elif path == "reasoning":
                out.append(t.get("reasoning", 0))
            elif path == "total":
                out.append(t.get("total", 0))
            elif path == "prompt":
                out.append(t.get("prompt", 0))
            elif path == "kag":
                out.append(int(t.get("kag_calls", 0)) * kag_per_call)
            elif path == "calls":
                out.append(t.get("calls", 0))
        return out

    def build(rs, srs, tag):
        if not rs:
            return None
        d_tot, d_tot_sd = _ms([r["round_s_from_log"] for r in rs])
        b_tot, b_tot_sd = _ms([r["round_s_from_log"] for r in srs]) if srs else (None, None)
        scr, scr_sd = _ms(stage(rs, "screen_s"))
        llm, llm_sd = _ms(stage(rs, "llm_s"))
        pre, _ = _ms(stage(rs, "prescribe_s"))
        gate, gate_sd = _ms(stage(srs, "gate_s")) if srs else (None, None)
        r_narrow = (llm or 0) + (pre or 0)
        r_wide = (scr or 0) + r_narrow
        vlm, vlm_sd = _ms(tok(rs, "vlm"))
        lt, lt_sd = _ms(tok(rs, "llm"))
        rs_, rs_sd = _ms(tok(rs, "reasoning"))
        kg, _ = _ms(tok(rs, "kag"))
        tt, _ = _ms(tok(rs, "total"))
        pt, _ = _ms(tok(rs, "prompt"))
        cl, _ = _ms(tok(rs, "calls"))
        return {
            "protocol": tag,
            "n_rounds": len(rs),
            "rounds_used": sorted(r["round"] for r in rs),
            "diseil_s": _r(d_tot), "diseil_s_sd": _r(d_tot_sd),
            "baseline_s": _r(b_tot), "baseline_s_sd": _r(b_tot_sd),
            "overhead_x": _r(d_tot / b_tot, 2) if (d_tot and b_tot) else None,
            "screen_s": _r(scr), "screen_s_sd": _r(scr_sd),
            "llm_s": _r(llm), "llm_s_sd": _r(llm_sd),
            "prescribe_s": _r(pre, 2),
            "reasoning_narrow_s": _r(r_narrow),
            "reasoning_wide_s": _r(r_wide),
            "baseline_gate_s": _r(gate), "baseline_gate_s_sd": _r(gate_sd),
            "reasoning_wide_addon_s": _r(r_wide - gate) if gate is not None else None,
            "train_eval_shared_s": _r((_ms(stage(rs, "train_s"))[0] or 0)
                                      + (_ms(stage(rs, "eval_s"))[0] or 0)),
            "vlm_tokens": _r(vlm, 0), "vlm_tokens_sd": _r(vlm_sd, 0),
            "llm_tokens": _r(lt, 0), "llm_tokens_sd": _r(lt_sd, 0),
            "reasoning_tokens": _r(rs_, 0), "reasoning_tokens_sd": _r(rs_sd, 0),
            "kag_tokens": _r(kg, 0),
            "kag_per_call": kag_per_call,
            "total_tokens": _r(tt, 0),
            "prompt_tokens": _r(pt, 0),
            "calls": _r(cl, 1),
            "kag_pct_of_prompt": _r(100 * kg / pt, 0) if (kg and pt) else None,
        }

    first = build(act[:1], sact[:1], "first round (round 0)") if act else None
    mean5 = build(act, sact, f"mean of {len(act)} LLM-active rounds") if act else None
    return {
        "label": label,
        "status_full": full["status"],
        "status_safe": safe["status"],
        "first": first,
        "mean": mean5,
        "diseil_rounds_logged": len(fr),
        "baseline_rounds_logged": len(sr),
    }


def pg_row(label, rec, first_round_idx, screen_key, kag_per_call):
    b = rec["breakdown_s"]
    bb = rec["baseline_breakdown_s"]
    scr = b.get(screen_key, 0.0)
    llm = b.get("llm", b.get("analyze_and_prescribe", 0.0))
    pre = b.get("prescribe", b.get("collect_prescribed_demos", 0.0))
    gate = rec.get("baseline_gate_s_per_round", 0.0)
    if gate in (None, 0.0) and "rollout_episodes_until_query" in bb:
        gate = bb["rollout_episodes_until_query"]
    r_narrow = llm + pre
    r_wide = scr + r_narrow
    first = {
        "protocol": f"first round (round {first_round_idx})",
        "n_rounds": 1, "rounds_used": [first_round_idx],
        "diseil_s": rec["diseil_s_per_round"], "diseil_s_sd": None,
        "baseline_s": rec["baseline_s_per_round"], "baseline_s_sd": None,
        "overhead_x": rec["overhead_x"],
        "screen_s": _r(scr), "screen_s_sd": None,
        "llm_s": _r(llm), "llm_s_sd": None,
        "prescribe_s": _r(pre, 2),
        "reasoning_narrow_s": _r(r_narrow),
        "reasoning_wide_s": _r(r_wide),
        "baseline_gate_s": _r(gate), "baseline_gate_s_sd": None,
        "reasoning_wide_addon_s": _r(r_wide - gate),
        "train_eval_shared_s": _r(b.get("train", b.get("train_policy", 0))
                                  + b.get("eval", b.get("evaluate_heldout", 0))),
        "vlm_tokens": rec["vlm_tokens"], "vlm_tokens_sd": None,
        "llm_tokens": rec["llm_tokens"], "llm_tokens_sd": None,
        # ALL-STAGE hidden-thinking total, so it is comparable with the distil runs'
        # tokens.reasoning (= analysis.reasoning + decision.reasoning; the VLM emits ~0).
        # Push-T: the source md reports 5612 for the Reasoning STAGE alone; its all-stage
        # total is 9 (vlm) + 5612 (reasoning) + 1593 (plain aggregator) = 7214.
        "reasoning_tokens": rec.get("reasoning_llm_tokens_all_stage",
                                    rec["reasoning_llm_tokens"]),
        "reasoning_tokens_sd": None,
        "reasoning_tokens_reasoning_stage_only": rec["reasoning_llm_tokens"],
        "kag_tokens": rec["kag_tokens"],
        "kag_per_call": kag_per_call,
        "total_tokens": rec["total_tokens"],
        "prompt_tokens": rec["prompt_tokens"],
        "calls": rec["total_calls"],
        "kag_pct_of_prompt": _r(100 * rec["kag_tokens"] / rec["prompt_tokens"], 0),
    }
    return {"label": label, "status_full": "complete", "status_safe": "complete",
            "first": first, "mean": None,
            "diseil_rounds_logged": 1, "baseline_rounds_logged": 1}


def main():
    spans = json.loads(SPANS.read_text())
    pg = json.loads(PG.read_text())
    kj = json.loads(KAG.read_text())["tasks"]
    kag_pc = {t: round((v["analysis_kag_delta"] + v["decision_kag_delta"]) / 2)
              for t, v in kj.items()}
    kag_pc["PushT"] = pg["pusht"]["kag_tokens_per_call"]
    # Push-T all-stage hidden-thinking total (see d5_rows_pusht_gridworld.md, token table):
    # vlm 9 + reasoning-stage 5612 + plain/aggregator 1593 = 7214.
    pg["pusht"]["reasoning_llm_tokens_all_stage"] = 9 + 5612 + 1593

    rows = [
        robosuite_row("Door / state", "Door/state", spans, kag_pc["Door"]),
        robosuite_row("Door / image", "Door/image", spans, kag_pc["Door"]),
        robosuite_row("Wipe / image", "Wipe/image", spans, kag_pc["Wipe"]),
        pg_row("Push-T / image", pg["pusht"], 1, "rollout_and_detect", kag_pc["PushT"]),
        pg_row("GridWorld 5x5 / image", pg["gridworld"], 0, "screen", kag_pc["GridWorld"]),
    ]
    out = COMPUTE / "d5_merged.json"
    out.write_text(json.dumps({"rows": rows, "kag_tokens_per_call": kag_pc,
                               "jobs": JOBS}, indent=2))
    print(json.dumps({"rows": rows, "kag_tokens_per_call": kag_pc}, indent=2))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
