"""Prompt addendum builders for P4 sequential and batch modes.

Both modes reuse upstream's full LLM pipeline (analysis + KAG + RAG +
prescription + aggregator). We only inject two strings on top:

  1. A *self-awareness* block (always present) — gives the LLM round
     number, demos added so far this round, demos remaining in the
     budget, and a one-line failure summary derived from the current
     correction-pool rollout.
  2. A *mode directive* — sequential vs batch, "prescribe ONE most-
     informative layout" vs "prescribe AS MANY layouts as needed to
     cover all failure modes within remaining budget".

The two strings are concatenated with the existing budget-addendum from
upstream (`p4_budget._budget_addendum`) so the LLM gets all three pieces
of context in a single coherent block.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import (  # noqa: E402
    p4_budget as _upstream,
)


def self_awareness_block(
    round_idx: int,
    demos_added_this_round: int,
    demos_remaining_in_budget: int,
    n_failures_observed: int,
    failure_mode_counts: Optional[Dict[str, int]] = None,
) -> str:
    """A human-readable status block prepended to every LLM addendum."""
    counts = failure_mode_counts or {}
    if counts:
        mode_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    else:
        mode_str = "n/a (no per-failure cluster signal available this round)"
    return (
        "EXPERIMENT STATE — always-on context for this LLM call:\n"
        f"  * round number: {int(round_idx)}\n"
        f"  * demos added so far this round: {int(demos_added_this_round)}\n"
        f"  * demos remaining in the global budget: {int(demos_remaining_in_budget)}\n"
        f"  * failures observed in current correction-pool rollout: "
        f"{int(n_failures_observed)}\n"
        f"  * failure modes (count per mode): {mode_str}\n"
    )


def mode_directive(mode: str) -> str:
    """Mode-specific text injected after the self-awareness block."""
    if mode == "sequential":
        return (
            "SEQUENTIAL MODE — prescribe EXACTLY ONE layout this round.\n"
            "Choose the SINGLE most informative layout: the one whose demo, "
            "once added and learned, will close the most failure modes in "
            "the correction pool. The orchestrator will hard-cap your output "
            "to ONE layout, so do not waste tokens recommending more.\n"
            "If you have multiple candidate clusters, pick the one whose "
            "failure pattern is the most repeated and whose prescribed "
            "pathway is also the most TRANSFERABLE to nearby failures."
        )
    if mode == "batch":
        return (
            "BATCH MODE — prescribe AS MANY layouts as needed to cover all "
            "current failure modes, subject to the remaining budget.\n"
            "Cluster failures into the smallest set of distinct modes, then "
            "recommend the smallest set of layouts per cluster that fixes "
            "the missing behaviour. Total layouts must not exceed the "
            "remaining budget — the orchestrator will hard-cap any excess."
        )
    if mode == "top3":
        return (
            "TOP-3 MODE — you are shown the 3 HIGHEST-LOSS failures from this "
            "round's correction-pool rollout. Prescribe EXACTLY ONE layout / "
            "demonstration that best covers these 3 failures. The orchestrator "
            "hard-caps your output to ONE layout, so do not recommend more.\n"
            "Pick the single demonstration whose learned behaviour is the most "
            "TRANSFERABLE across the three failure patterns shown.\n"
            "\n"
            "OUTPUT REQUIREMENT (strict, non-negotiable):\n"
            "  * The single recommended_layout MUST be CONCRETE and FULLY "
            "specified: provide start_pos: [r, c], goal_pos: [r, c], "
            "fire_positions: [[r, c], ...] (all integers in 0..4), AND a "
            "non-empty `steps` corridor string of the form "
            "\"(r,c)->(r,c)->...\" from start to goal.\n"
            "  * DO NOT emit an empty recommended_layouts list, a null cluster, "
            "or null n_demos_needed. An incomplete prescription is treated as "
            "ZERO prescriptions and this round will collect ZERO demos.\n"
            "  * If unsure which layout to prescribe, pick ONE of the 3 shown "
            "failure layouts verbatim (copy its start/goal/fires) and design a "
            "safe corridor that avoids every fire cell."
        )
    if mode == "all":
        return (
            "ALL-FAILURES MODE — you are shown ALL failures from this round's "
            "correction-pool rollout. Prescribe EXACTLY ONE layout / "
            "demonstration that best covers the full set of failures shown. "
            "The orchestrator hard-caps your output to ONE layout, so do not "
            "recommend more.\n"
            "Choose the SINGLE most informative demonstration: the one whose "
            "learned behaviour closes the largest number of the observed "
            "failure modes. If you cluster failures, prescribe the layout "
            "that fixes the LARGEST cluster.\n"
            "\n"
            "OUTPUT REQUIREMENT (strict, non-negotiable):\n"
            "  * The single recommended_layout MUST be CONCRETE and FULLY "
            "specified: provide start_pos: [r, c], goal_pos: [r, c], "
            "fire_positions: [[r, c], ...] (all integers in 0..4), AND a "
            "non-empty `steps` corridor string of the form "
            "\"(r,c)->(r,c)->...\" from start to goal.\n"
            "  * DO NOT emit an empty recommended_layouts list, a null "
            "cluster, or null n_demos_needed. An incomplete prescription is "
            "treated as ZERO prescriptions and this round will collect ZERO "
            "demos (the round's budget is then wasted).\n"
            "  * When in doubt with many failures shown, pick ONE failure "
            "episode from the LARGEST cluster verbatim (copy its start/goal/"
            "fires) and design a safe corridor that avoids every fire cell. "
            "Better to prescribe a known-failing layout than to emit nothing."
        )
    raise ValueError(f"unknown P4 mode: {mode!r}")


def infeasible_feedback_block(records, max_items: int = 12) -> str:
    """Feedback listing prior rounds' rejected (infeasible) prescriptions.

    The A*/BFS feasibility checker rejects corridors that step on fire,
    leave the grid, skip cells, or don't connect start->goal. We surface
    those rejections back to the LLM so it stops repeating them. Returns
    "" when there is nothing to report (so it's omitted from the prompt).
    """
    recs = [r for r in (records or []) if r]
    if not recs:
        return ""
    recs = recs[-max_items:]
    lines = []
    for r in recs:
        lines.append(
            f"  * round {r.get('round', '?')}: start={r.get('start_pos')} "
            f"goal={r.get('goal_pos')} fires={r.get('fire_positions')} "
            f"steps={r.get('steps') or '(none)'}  -> REJECTED: "
            f"{r.get('reason', 'infeasible')}"
        )
    return (
        "INFEASIBLE-PRESCRIPTION FEEDBACK — learn from earlier rounds.\n"
        "In previous rounds you prescribed corridor pathways that the "
        "A*/BFS feasibility checker REJECTED, so they produced ZERO usable "
        "demonstrations and wasted the round. Do NOT repeat these. Every "
        "pathway you prescribe this round MUST: (a) start exactly at the "
        "episode start cell, (b) end exactly at the goal cell, (c) move to "
        "a 4-adjacent cell each step (one of up/down/left/right, no "
        "diagonals or jumps), (d) stay inside the grid, and (e) never step "
        "on a FIRE cell. Re-derive the path carefully against the layout's "
        "fires before emitting it.\n"
        "Rejected prescriptions so far:\n" + "\n".join(lines)
    )


def reasoning_addendum(
    mode: str, round_idx: int, demos_added_this_round: int,
    demos_remaining_in_budget: int, n_failures_observed: int,
    failure_mode_counts: Optional[Dict[str, int]] = None,
    budget_total: int = 0, already_used: int = 0,
    infeasible_feedback: str = "",
) -> str:
    """The string passed as ``prompt_addendum_reasoning`` (Phase B)."""
    parts = [
        _upstream.REASONING_ADDENDUM_BASE,
        self_awareness_block(
            round_idx, demos_added_this_round, demos_remaining_in_budget,
            n_failures_observed, failure_mode_counts,
        ),
        mode_directive(mode),
        _upstream._budget_addendum(budget_total, already_used),
    ]
    if infeasible_feedback:
        parts.append(infeasible_feedback)
    return "\n\n".join(parts)


def aggregator_addendum(
    mode: str, round_idx: int, demos_added_this_round: int,
    demos_remaining_in_budget: int, n_failures_observed: int,
    failure_mode_counts: Optional[Dict[str, int]] = None,
    budget_total: int = 0, already_used: int = 0,
    infeasible_feedback: str = "",
) -> str:
    """The string passed as ``prompt_addendum_aggregator`` (Phase C)."""
    parts = [
        _upstream.AGGREGATOR_ADDENDUM_BASE,
        self_awareness_block(
            round_idx, demos_added_this_round, demos_remaining_in_budget,
            n_failures_observed, failure_mode_counts,
        ),
        mode_directive(mode),
        _upstream._budget_addendum(budget_total, already_used),
    ]
    if infeasible_feedback:
        parts.append(infeasible_feedback)
    return "\n\n".join(parts)


def summarize_failure_modes(failure_records: Iterable[Dict]) -> Dict[str, int]:
    """Quick clustering of the round's failures into rough mode labels.

    The labels are intentionally coarse — start-quadrant + goal-quadrant.
    They give the LLM a *count* of how many failures fall into each
    spatial bucket without pretending we've done the real Phase-C
    clustering. (Phase C still runs and can override.)
    """
    def quadrant(pos) -> str:
        try:
            r, c = int(pos[0]), int(pos[1])
        except (TypeError, ValueError, IndexError):
            return "?"
        rb = "top" if r < 2 else ("mid" if r == 2 else "bot")
        cb = "left" if c < 2 else ("mid" if c == 2 else "right")
        return f"{rb}-{cb}"

    counter: Counter = Counter()
    for rec in failure_records:
        layout = rec.get("layout", {}) if isinstance(rec, dict) else {}
        sq = quadrant(layout.get("start_pos", [0, 0]))
        gq = quadrant(layout.get("goal_pos", [0, 0]))
        counter[f"{sq}->{gq}"] += 1
    return dict(counter)
