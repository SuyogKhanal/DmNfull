"""P4-LLM V3 hybrid prompts (ported from the PushT pipeline's p4/prompts.py).

Faithful to the multi-stage contract:
  VLM (effort low)      : read the failure frames, say what went wrong.
  ANALYSIS (effort high): classify root cause + phase from the VLM report + KAG.
  DECISION (effort high): given the dominant cluster's members + their analyses +
                          KAG, choose SELECT ep# (correct one real failure) or
                          BRIDGE ep#,ep# (prescribe one middle-ground scene).
Every reasoning stage is KAG-grounded (real workspace bounds + failure-mode rules).
"""
from __future__ import annotations

from typing import Dict, List

# Per-task natural-language description (fed to every stage).
TASK_DESC = {
    "Lift": ("A UR5e robot with a parallel-jaw gripper must GRASP a small cube resting "
             "on a table and LIFT it above a height threshold. Non-trivial: the grasp "
             "must be centered and secure before lifting."),
    "Door": ("A UR5e robot must reach a door handle, engage it, and PULL the door OPEN "
             "past a hinge angle of 0.3 rad. The door's position and yaw are randomized "
             "each episode; the arm must adapt its approach and pull direction."),
    "Wipe": ("A UR5e robot with a wiping pad (no fingers) must WIPE a trail of dirt "
             "markers off the table by pressing down and sweeping along the dirt path "
             "until all markers are cleared (coverage)."),
    "GridWorld": ("A point agent on a 5x5 grid must reach a GOAL cell using four discrete "
                  "moves (UP/DOWN/LEFT/RIGHT). Three FIRE cells terminate the episode in "
                  "failure if stepped on; a fire-free path to the goal always exists. "
                  "Coordinates are (row, col), origin top-left."),
}

# Generic manipulation root-cause / phase enums (shared with the PushT pipeline).
ROOT_CAUSES = ["grasp_failure", "approach_failure", "placement_error",
               "contact_instability", "pose_mismatch", "timeout"]
PHASES = ["pre_grasp", "grasp", "transport", "placement", "insertion"]

# GridWorld-appropriate enums (06_..md note): maze failure taxonomy + phases.
GW_ROOT_CAUSES = ["wrong_direction", "hit_fire", "wall_thrashing", "timeout"]
GW_PHASES = ["approach", "corridor", "junction"]


def enums_for(task):
    """(root_causes, phases) for a task — GridWorld gets the maze taxonomy."""
    if task == "GridWorld":
        return GW_ROOT_CAUSES, GW_PHASES
    return ROOT_CAUSES, PHASES

# ── Stage A: VLM (effort low) ────────────────────────────────────────────────
VLM_SYSTEM = (
    "You are analysing a robot manipulation failure from rendered frames. Be "
    "concrete and spatial; describe what you actually see, not generic advice.")
VLM_SYSTEM_NAV = (
    "You are analysing a grid-navigation failure from rendered top-down frames. Be "
    "concrete and spatial (use (row, col) cells); describe what you actually see, "
    "not generic advice.")


def vlm_system(task=None) -> str:
    return VLM_SYSTEM_NAV if task == "GridWorld" else VLM_SYSTEM


def vlm_prompt(task_description: str, roles: List[str], t_star: int, task=None) -> str:
    """One VLM call showing the failure's start / peak-loss(t*) / end frames."""
    which = ", ".join(roles)
    if task == "GridWorld":
        return (
            "You are analysing a grid-navigation failure. The attached top-down frames "
            f"are, in order: {which} (the high-loss frame is the policy's most-uncertain "
            f"step, t*={t_star}). The blue cell is the agent, green is the GOAL, red-orange "
            "are FIRE hazards, grey is free space.\n"
            f"Task: {task_description}\n"
            "Describe what went wrong. Focus on: where on the grid the agent is at the "
            "high-loss step, which direction it is heading, and whether it is walking into "
            "a fire, a boundary, or away from the goal. ~110 words, concrete, use (row,col).")
    return (
        "You are analysing a robot manipulation failure. The attached frames are, in "
        f"order: {which} (the peak-loss frame is the policy's most-uncertain step, "
        f"t*={t_star}).\n"
        f"Task: {task_description}\n"
        "Describe what went wrong. Focus on: where in the trajectory the failure "
        "occurs, the robot/gripper configuration at peak loss, and what object or "
        "contact state caused it. ~120 words, concrete and spatial.")


# ── Stage B: ANALYSIS (effort high) ──────────────────────────────────────────
ANALYSIS_SYSTEM = (
    "You are a robot-manipulation failure analyst. Classify the root cause and "
    "trajectory phase using ONLY the provided categories and the KAG facts. "
    "Output strict JSON, no prose, no code fences.")


def analysis_prompt(task_description: str, kag_text: str, vlm_report: str,
                    root_causes=None, phases=None) -> str:
    root_causes = root_causes or ROOT_CAUSES
    phases = phases or PHASES
    return (
        f"TASK: {task_description}\n\n"
        f"{kag_text}\n\n"
        f"VLM FAILURE DESCRIPTION (the only visual evidence):\n{vlm_report}\n\n"
        "Identify the root cause category and the trajectory phase where the "
        f"failure occurred.\nroot_cause ∈ {root_causes}\nphase ∈ {phases}\n\n"
        'Output ONLY this JSON:\n'
        '{"root_cause": "<one of the categories>", '
        '"phase": "<one of the phases>", '
        '"rationale": "<one sentence grounded in the VLM description and a KAG fact>"}')


# ── Stage C: DECISION — SELECT vs BRIDGE + CONFIDENCE (effort high) ──────────
DECISION_SYSTEM = (
    "You are a demonstration coach for an interactive imitation-learning loop. Each "
    "round you spend ONE expert demonstration to fix the dominant failure mode. You "
    "decide HOW to spend it, grounded in the KAG facts and the per-failure analyses. "
    "Reason briefly, then end with EXACTLY two lines: (1) a decision line in the exact "
    "required format, and (2) a confidence line "
    "'CONFIDENCE: <integer 0-100> - <one-line rationale>' reporting how confident you "
    "are that this demonstration will improve the policy.")


def _members_block(members: List[Dict]) -> str:
    lines = []
    for m in members:
        a = m.get("analysis", {})
        lines.append(
            f"  - ep{m['ep_id']}: object_xy=({m['ox']:.3f},{m['oy']:.3f}) "
            f"progress={m['t_star']}/{m['T']} peak_loss={m['peak_loss']:.4f} "
            f"root_cause={a.get('root_cause','?')} phase={a.get('phase','?')}")
    return "\n".join(lines)


def decision_prompt(task_description: str, kag_text: str, members: List[Dict],
                    bridge_supported: bool) -> str:
    select_opt = (
        "  (A) SELECT ep<ID> — one recorded failure represents the whole mode. That "
        "exact scene is re-run and the expert corrects it on-policy from the divergence "
        "point t*. Use when the cluster is TIGHT or one failure clearly dominates.\n")
    bridge_opt = (
        "  (B) BRIDGE ep<ID>,ep<ID> — no single failure covers the mode. Prescribe ONE "
        "new object placement in the MIDDLE GROUND between 2-3 cited failures (e.g. "
        "failures at (1,1) and (5,5) → a demo near (3,3)); the expert demonstrates from "
        "there. Use when the members are geometrically SPREAD but share a root cause.\n")
    decision_line = ("'SELECT ep<ID>'"
                     + (" or 'BRIDGE ep<ID>,ep<ID>'" if bridge_supported
                        else " (this task supports SELECT only — no single object pose to bridge)"))
    out = (
        "  End your reply with EXACTLY these two lines:\n"
        f"    {decision_line}\n"
        "    CONFIDENCE: <integer 0-100> - <one-line rationale>\n")
    return (
        f"TASK: {task_description}\n\n"
        f"{kag_text}\n\n"
        "DOMINANT FAILURE CLUSTER (members with their VLM+analysis findings):\n"
        f"{_members_block(members)}\n\n"
        "Decide how to spend the one demonstration this round:\n"
        + select_opt + (bridge_opt if bridge_supported else "")
        + "\nWeigh the spread of the object positions and whether the root causes agree.\n"
        + out)
