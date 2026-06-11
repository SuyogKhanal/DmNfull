"""Prompt templates for the fresh P4-LLM three-component pipeline.

Generic across all four ManiSkill tasks — failure modes are NOT hard-coded; the
VLM infers them from the frames and the Reasoning LLM classifies them against the
task's KAG. The load-bearing piece is OUTPUT_REQUIREMENT, adapted from the maze
§6c HARD FLOOR (and the fork's REASONING/AGGREGATOR addenda): a failure is present,
so the prescription MUST be a concrete, non-empty config — an empty/refused
prescription raises PrescriptionEmptyError and is retried.
"""
from __future__ import annotations

from typing import List

ROOT_CAUSES = ["grasp_failure", "approach_failure", "placement_error",
               "contact_instability", "pose_mismatch", "timeout"]
PHASES = ["pre_grasp", "grasp", "transport", "placement", "insertion"]

# ---- Component 1: VLM (failure localisation) ---------------------------
VLM_SYSTEM = (
    "You are analysing a robot manipulation failure from rendered frames. Be "
    "concrete and spatial; describe what you actually see, not generic advice.")


def vlm_prompt(task_description: str, role: str, t_star: int) -> str:
    return (
        f"You are analysing a robot manipulation failure.\n"
        f"This frame is the '{role}' frame"
        + (f" (peak diffusion loss at timestep {t_star})" if role == "high_loss_frame"
           else "") + ".\n"
        f"Task: {task_description}\n"
        "Describe what went wrong. Focus on: where the failure occurs in the "
        "trajectory, what the robot's configuration was at peak loss, and what "
        "object or gripper state caused the failure. ~120 words.")


# ---- Component 2a: AnalysisClass (root cause + phase) -------------------
ANALYSIS_SYSTEM = (
    "You are a robot-manipulation failure analyst. Classify the root cause and "
    "trajectory phase using ONLY the provided categories and the KAG facts. "
    "Output strict JSON, no prose, no code fences.")


def analysis_prompt(task_description: str, kag_text: str, vlm_report: str) -> str:
    return (
        f"TASK: {task_description}\n\n"
        f"{kag_text}\n\n"
        f"VLM FAILURE DESCRIPTION (the only evidence):\n{vlm_report}\n\n"
        "Identify the root cause category and the trajectory phase where the "
        f"failure occurred.\nroot_cause ∈ {ROOT_CAUSES}\nphase ∈ {PHASES}\n\n"
        'Output ONLY this JSON:\n'
        '{"root_cause": "<one of the categories>", '
        '"phase": "<one of the phases>", '
        '"rationale": "<one sentence grounded in the VLM description and a KAG fact>"}')


# ---- Component 2b: PrescriptionClass (correction config) ---------------
OUTPUT_REQUIREMENT = (
    "OUTPUT REQUIREMENT (strict, non-negotiable):\n"
    "- A failure IS present, so you MUST prescribe a CONCRETE, FULLY-SPECIFIED "
    "correction episode. Provide gripper_pose_at_start ([x,y,z,qx,qy,qz,qw]), "
    "object_pose ([x,y,z,qx,qy,qz,qw]), and environment_variant (a short string; "
    "use \"no change\" if none).\n"
    "- DO NOT emit an empty object, null fields, or refuse. An incomplete or empty "
    "prescription is treated as ZERO prescriptions and this round collects ZERO "
    "demos — the worst possible outcome.\n"
    "- Keep all poses inside the task's reliable workspace (see the KAG). When "
    "unsure, copy a real failure episode's start/object pose and adjust minimally "
    "toward the goal. Better to prescribe a known-hard config than to emit nothing.")

PRESCRIPTION_SYSTEM = (
    "You are a demonstration coach. You prescribe the configuration of the NEXT "
    "corrective episode so the expert can demonstrate the missing behaviour. "
    "Apply every OUTPUT REQUIREMENT. Output strict JSON, no prose, no code fences.")


def prescription_prompt(task_description: str, kag_text: str,
                        analysis_json: str, failures_summary: str) -> str:
    return (
        f"TASK: {task_description}\n\n"
        f"{kag_text}\n\n"
        f"ROOT-CAUSE ANALYSIS (authoritative):\n{analysis_json}\n\n"
        f"TOP FAILURE EPISODES (compress these into ONE correction config):\n"
        f"{failures_summary}\n\n"
        f"{OUTPUT_REQUIREMENT}\n\n"
        'Output ONLY this JSON:\n'
        '{"gripper_pose_at_start": [x,y,z,qx,qy,qz,qw], '
        '"object_pose": [x,y,z,qx,qy,qz,qw], '
        '"environment_variant": "<string>", '
        '"rationale": "<one sentence tying this config to the failure>"}')


# ---- Cube-prescription variant (StackCube/PickCube — prescribe cube poses) ----
CUBE_OUTPUT_REQUIREMENT = (
    "OUTPUT REQUIREMENT (strict, non-negotiable):\n"
    "- A failure IS present, so you MUST prescribe a CONCRETE corrective scene: the "
    "world XY positions of cube A (the cube to pick up) and cube B (the base cube to "
    "stack onto). Provide cubeA_xyz ([x,y,z]) and cubeB_xyz ([x,y,z]); z is on the "
    "table (~0.02) and will be pinned automatically. Optionally cubeA_zrot / "
    "cubeB_zrot (z-rotation in radians; omit/null = random).\n"
    "- DO NOT emit an empty object, null cube positions, or refuse. An empty/invalid "
    "prescription is treated as ZERO prescriptions and collects ZERO demos this round "
    "— the worst possible outcome.\n"
    "- Keep cube XY inside the reachable table workspace: x∈[-0.15,0.15], y∈[-0.25,"
    "0.25] (out-of-range or overlapping cubes are auto-clamped to a graspable layout).\n"
    "- CHOOSE the layout to TEACH the missing behaviour the analysis identified: e.g. "
    "for grasp failures, place cube A in the open so the expert demonstrates a clean "
    "grasp; for placement/alignment errors, place A and B at a separation that "
    "rehearses precise stacking; for approach failures, vary A's position/orientation. "
    "When unsure, copy a real failure episode's cube layout and adjust minimally.")

CUBE_PRESCRIPTION_SYSTEM = (
    "You are a demonstration coach for a cube-stacking robot. You prescribe the cube "
    "layout of the NEXT corrective episode so the motion-planner expert can demonstrate "
    "the missing behaviour. Apply every OUTPUT REQUIREMENT. Output strict JSON, no "
    "prose, no code fences.")


def cube_prescription_prompt(task_description: str, kag_text: str,
                             analysis_json: str, failures_summary: str) -> str:
    return (
        f"TASK: {task_description}\n\n"
        f"{kag_text}\n\n"
        f"ROOT-CAUSE ANALYSIS (authoritative):\n{analysis_json}\n\n"
        f"TOP FAILURE EPISODES (compress these into ONE corrective cube layout):\n"
        f"{failures_summary}\n\n"
        f"{CUBE_OUTPUT_REQUIREMENT}\n\n"
        'Output ONLY this JSON:\n'
        '{"cubeA_xyz": [x,y,z], "cubeB_xyz": [x,y,z], '
        '"cubeA_zrot": <float|null>, "cubeB_zrot": <float|null>, '
        '"n_demos": 1, '
        '"rationale": "<one sentence tying this cube layout to the failure>"}')


# ---- Component 3: Configuration LLM (env.reset config) -----------------
CONFIG_SYSTEM = (
    "You translate a manipulation-correction prescription into a concrete "
    "simulator reset configuration. Output strict JSON, no prose, no code fences.")


def config_prompt(task_description: str, prescription_json: str,
                  reset_keys_desc: str) -> str:
    return (
        f"TASK: {task_description}\n\n"
        f"PRESCRIPTION (authoritative):\n{prescription_json}\n\n"
        f"Translate it into the simulator reset configuration for this task. "
        f"{reset_keys_desc}\n\n"
        'Output ONLY this JSON (omit keys that do not apply to this task):\n'
        '{"robot_init_qpos": [<7 floats> | null], '
        '"obj_init_pose": {"p": [x,y,z], "q": [qx,qy,qz,qw]}, '
        '"goal_pose": {"p": [x,y,z], "q": [qx,qy,qz,qw]}}')


def failures_summary(failures: List[dict]) -> str:
    """One line per top-K failure (objective numbers the LLM can ground on)."""
    if not failures:
        return "(no failures this round)"
    rows = []
    for i, f in enumerate(failures):
        rows.append(f"  [{i}] t*={f.get('t_star')} peak_loss={f.get('peak_loss')} "
                    f"final_dist={f.get('final_dist')} steps={f.get('steps')}")
    return "\n".join(rows)
