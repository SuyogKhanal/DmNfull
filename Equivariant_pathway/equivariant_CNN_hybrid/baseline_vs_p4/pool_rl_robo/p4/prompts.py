"""P4-LLM prompt construction (continuous-control analogue of pool_x_selector's
``p4/prompts.py``). Builds an environment-specific text prompt summarising the
novice's visited states; the LLM returns the index of the single state to
request an expert demonstration for. Strict JSON output requirement (the maze
OUTPUT-REQUIREMENT discipline)."""
from __future__ import annotations

from typing import List

from ..envs.experts import ENV_PROMPTS

MAX_CANDIDATES = 30

SYSTEM = (
    "You select the single most informative state for which to request an expert "
    "demonstration, to improve a behavior-cloning policy via DAgger. "
    "Reply with ONLY a JSON object of the form "
    '{"selected_index": <int>, "rationale": "<short reason>"} '
    "and nothing else — no markdown, no prose."
)


def candidate_indices(records: List[dict]) -> List[int]:
    """Top-MAX_CANDIDATES original indices, ranked by discrepancy (descending)."""
    order = sorted(range(len(records)), key=lambda i: -float(records[i].get("discrepancy", 0.0)))
    return order[:MAX_CANDIDATES]


def build_user_prompt(env_id: str, records: List[dict], cand_idx: List[int],
                      kag_text: str = "") -> str:
    rows = []
    for j, oi in enumerate(cand_idx):
        r = records[oi]
        parts = [
            f"[{j}]",
            f"t={int(r.get('t', 0))}",
            f"discrepancy={float(r.get('discrepancy', 0.0)):.3f}",
            f"return_so_far={float(r.get('return_so_far', 0.0)):.1f}",
            f"reward_to_go={float(r.get('reward_to_go', 0.0)):.1f}",
        ]
        gd = r.get("goal_dist")
        if gd is not None:
            parts.append(f"gripper_to_goal_dist={float(gd):.3f}")
        rows.append(" ".join(parts))
    task = ENV_PROMPTS.get(env_id, "")
    kag_block = f"ENVIRONMENT KNOWLEDGE (KAG) for {env_id}:\n{kag_text.strip()}\n\n" if kag_text.strip() else ""
    return (
        f"You are improving a behavior-cloning (DAgger) policy on the {env_id} "
        "environment — a continuous-control task with the specifics below.\n\n"
        f"{kag_block}"
        f"TASK SUMMARY: {task}\n\n"
        "A novice policy was rolled out in THIS environment. Below are candidate states it "
        "visited. 'discrepancy' is the L2 distance between the novice's action and the "
        "expert's action at that state (higher = the novice disagrees more with the expert, "
        "so an expert demonstration there is likely more corrective). 'return_so_far' and "
        "'reward_to_go' are cumulative rewards before/after the state.\n\n"
        "Use the environment knowledge above — its success criterion, key state features, "
        "failure modes, and the 'Selecting an informative state' guidance — to choose the "
        "single state whose expert demonstration would most improve the policy ON THIS "
        "SPECIFIC ENVIRONMENT.\n\n"
        "CANDIDATE STATES:\n" + "\n".join(rows) + "\n\n"
        f"Pick the ONE state (by its [index], 0..{len(cand_idx) - 1}). Return JSON "
        '{"selected_index": i, "rationale": "<why, referencing the environment knowledge>"}.'
    )
