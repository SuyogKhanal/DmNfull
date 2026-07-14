#!/usr/bin/env python3
"""Render fallback_audit_hpcA.md from the collected cells/*.json.

Answers, per run, the only question that decides whether a run is DISEIL at all:
did the LLM actually run every round it was supposed to, or did the loop quietly
fall back to the deterministic geometric planner?

Round taxonomy (see collect_rerun_cell.py for the exact rule):
  llm_active  — the LLM was called (non-zero VLM+LLM tokens).
  budget_free — no usable failures were found, so there was nothing to prescribe.
                The LLM was legitimately not called. NOT a fallback.
  FALLBACK    — a demonstration was acquired WITHOUT any LLM call. The deterministic
                planner stood in for DISEIL. Any run with >=1 such round is INVALID.

A run is VALID iff  fallback_rounds == 0  AND  demos_acquired == B.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CELLS = ["Door_state", "Door_image", "PushT_state", "PushT_image"]


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = []
    out.append("# Fallback audit — DISEIL OpenRouter re-run (HPC-A)\n")
    out.append("Per run: rounds that used the LLM vs. rounds that fell back to the "
               "deterministic geometric planner.\n")
    out.append("**A run is VALID iff `fallback_rounds == 0`** — i.e. the LLM ran on every "
               "round it was called for. Any run that fell back to the deterministic "
               "planner is INVALID: it is not DISEIL, and it is discarded and re-run, "
               "never reported.\n")
    out.append("\n`budget_free` rounds are rounds in which the policy produced no usable "
               "failures, so there was nothing to prescribe and the LLM was correctly not "
               "called. Those are not fallbacks and do not invalidate a run.\n")
    out.append("\n**A budget shortfall is not a fallback and does not invalidate a run.** "
               "The loop stops when the policy yields no usable failures for 4 consecutive "
               "rounds (`distil/p4/loop.py:110-113`): the policy has saturated and the "
               "remaining budget is *unspendable*, because there is no failure left to "
               "prescribe a correction for. This is existing method behaviour, and the "
               "PUBLISHED runs contain it too — published Door (state) seed 4 acquired only "
               "11/20. Such seeds are reported with their shortfall and their reason, and "
               "are included in the mean; excluding them would make the re-run "
               "non-comparable to the numbers it is meant to reproduce. `saturation_patience` "
               "was NOT touched.\n")
    out.append("\nAll runs were launched with `DISEIL_STRICT_LLM=1`, under which an "
               "unavailable or failing LLM **aborts** the run rather than degrading to the "
               "fallback. A run that completed therefore cannot contain a fallback round; "
               "the table below is the post-hoc confirmation of that, not a substitute "
               "for it.\n")

    any_cell = False
    for name in CELLS:
        f = base / "cells" / f"{name}.json"
        out.append(f"\n---\n\n## {name.replace('_', ' (')})\n")
        if not f.is_file():
            out.append("\n**UNRUN** — no result file. Not reported, not invented.\n")
            continue
        any_cell = True
        c = json.loads(f.read_text())
        out.append(f"\nArm `{c['arm_in_code']}` · B={c['budget_B']} · D={c['demos_per_round_D']} "
                   f"· held-out {c['heldout_episodes']} eps · "
                   f"VLM `{c['vlm_model']}` · LLM `{c['llm_model']}`\n")
        out.append("\n| seed | demos acquired | rounds | LLM-active | budget-free | "
                   "**fallback** | VLM tokens | LLM tokens | final SR | valid (LLM ran) |\n")
        out.append("|---|---|---|---|---|---|---|---|---|---|\n")
        for s in sorted(c["seeds"], key=int):
            v = c["seeds"][s]
            if "status" in v:
                out.append(f"| {s} | — | — | — | — | — | — | — | — | **UNRUN** |\n")
                continue
            fb = v.get("rounds_fallback", 0)
            short = "" if v.get("budget_spent_in_full", True) else " ⚠️"
            out.append(
                f"| {s} | {v.get('demos_acquired')}/{c['budget_B']}{short} "
                f"| {v.get('rounds_total')} "
                f"| {v.get('rounds_llm_active')} | {v.get('rounds_budget_free')} "
                f"| **{fb}** | {v.get('vlm_tokens', 0):,} | {v.get('llm_tokens', 0):,} "
                f"| {v.get('final_success_rate')} | {'YES' if fb == 0 else '**NO**'} |\n")
        tot = c.get("total_fallback_rounds", 0)
        mean, sd = c.get("final_success_rate_mean"), c.get("final_success_rate_std")
        out.append(f"\n**Cell total fallback rounds: {tot}.** "
                   f"{c['n_seeds_valid']}/{c['n_seeds_completed']} completed seeds are valid "
                   f"(the LLM ran on every round it was called for).\n")
        if mean is not None:
            out.append(f"\n**Final held-out success rate: {mean * 100:.1f} ± "
                       f"{(sd or 0) * 100:.1f}** over {c['n_seeds_valid']} seeds.\n")
        sh = c.get("seeds_with_budget_shortfall") or []
        if sh:
            out.append(f"\n⚠️ Budget shortfall (policy saturated, remaining budget "
                       f"unspendable) on seed(s) **{', '.join(sh)}** — marked ⚠️ above. Not a "
                       f"fallback; the LLM ran throughout. Reported, not corrected.\n")
        if tot:
            out.append("\n🔴 **This cell contains FALLBACK rounds. The affected seeds are "
                       "INVALID (not DISEIL) and are excluded from the mean above.**\n")

    if not any_cell:
        out.append("\n_No cell has completed yet._\n")

    p = base / "fallback_audit_hpcA.md"
    p.write_text("".join(out))
    print(f"[write] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
