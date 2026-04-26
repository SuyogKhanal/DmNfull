import base64
import json
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _oai_client():
    from openai import OpenAI
    return OpenAI()


def _chat_vlm(client, model: str, image_b64: str, text_prompt: str, system_prompt: str, max_tokens: int) -> str:
    content = [
        {"type": "input_text", "text": text_prompt},
        {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
    ]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    r = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
        reasoning={"effort": "low"},
    )
    return r.output_text or ""


def _format_goal_fire_description(dyn_cfg: Dict) -> str:
    start = dyn_cfg.get("start_pos", [0, 0])
    goal  = dyn_cfg.get("goal_pos",  [4, 4])
    fires = dyn_cfg.get("fire_positions", [])
    fires_str = ", ".join([f"({r},{c})" for r, c in fires]) if fires else "(none)"
    return (
        f"Episode start position: (row={start[0]}, col={start[1]}). "
        f"Goal position: (row={goal[0]}, col={goal[1]}). "
        f"Fire hazards at: {fires_str}."
    )


def analyse_failure(
    episode: Dict,
    llm_cfg: Dict,
    cache=None,
) -> Tuple[str, List[Dict]]:
    """Run chained VLM analysis over the 3 key frames. Returns (combined_report, per_frame_list)."""
    if cache is not None:
        scope = cache.episode_scope(episode["episode_id"])
        cached = cache.load(scope, "vlm_output")
        if cached is not None:
            return cached.get("report", ""), cached.get("per_frame", [])

    model = str(llm_cfg.get("vlm_model", llm_cfg.get("model", "gpt-5-nano-2025-08-07")))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))
    client = _oai_client()

    dyn_cfg = episode.get("dynamic_config", {})
    ascii_grid = episode.get("ascii_grid", "")
    goal_fire_desc = _format_goal_fire_description(dyn_cfg)

    system_prompt = (
        "You are a demonstration coach for maze navigation imitation learning. "
        "An agent is learning to navigate a 5x5 grid maze via a vision-conditioned diffusion policy "
        "trained on human demonstrations. Every episode, the start position, goal position, and "
        "fire hazard positions are RANDOMISED. The maze contains FIRE hazards (red cells) that "
        "terminate the episode on contact. When the policy fails, a human operator must record a "
        "corrective demonstration. Your job is to observe what the agent did wrong and describe what "
        "the CORRECT demonstration should look like. Focus on: movement direction, proximity to fire "
        "hazards, distance to goal, and optimal path choice. Be spatially precise — use grid "
        "coordinates (row, col). Roughly 150 words per frame."
    )

    first_frame_template = (
        "Observe this {role} from a maze navigation episode.\n\n"
        "MAZE LAYOUT (·=free, █=wall, F=fire, G=goal, A=agent):\n{ascii_grid}\n\n"
        "EPISODE CONFIGURATION:\n{goal_fire_desc}\n\n"
        "CURRENT STATE DESCRIPTION:\n{state_desc}\n\n"
        "ADJACENT CELLS: {neighbourhood}\n\n"
        "You are coaching a human operator who will record a corrective demonstration.\n"
        "Describe:\n"
        "1. Where is the agent on the grid? What is at each adjacent cell (up/down/left/right)?\n"
        "   State the exact (row, col) position.\n"
        "2. Where is the GOAL relative to the agent? What is the Manhattan distance?\n"
        "   How many cells right/left and how many cells down/up?\n"
        "3. Where are the FIRE hazards relative to the agent?\n"
        "   Is the agent in danger of stepping into fire on any adjacent move?\n"
        "4. What direction should a human demonstrator move FIRST from this position?\n"
        "   Justify by referencing which adjacent cells are safe vs dangerous.\n\n"
        "Reference the state description data. Be spatially specific — use coordinates."
    )

    rolling_frame_template = (
        "Observe this {role} from the maze episode (step {step_idx}).\n\n"
        "PREVIOUS FRAME ANALYSIS:\n{previous_summary}\n\n"
        "EPISODE CONFIGURATION:\n{goal_fire_desc}\n\n"
        "CURRENT STATE DESCRIPTION:\n{state_desc}\n\n"
        "ADJACENT CELLS: {neighbourhood}\n\n"
        "Compare to the previous frame and assess what the agent is doing wrong:\n"
        "1. How has the agent's position changed since the last frame? State old and new\n"
        "   (row, col). Did it move closer to the goal or further away? By how many cells?\n"
        "2. Did the agent move toward FIRE? Is it now adjacent to one of the fire cells?\n"
        "   If it stepped INTO fire, state which fire cell and from which direction.\n"
        "3. Was the agent's movement efficient? Did it revisit a cell it already visited?\n"
        "   Did it move in a direction that doesn't reduce Manhattan distance to goal?\n"
        "4. What should the human demonstrator do differently from THIS position?\n"
        "   Give the exact action (UP/DOWN/LEFT/RIGHT) and explain why it avoids fire\n"
        "   and progresses toward the goal.\n\n"
        "Reference specific coordinates. Roughly 150 words."
    )

    per_frame: List[Dict] = []
    previous_summary = ""
    frame_paths = episode.get("frame_paths", {}) or {}

    try:
        for i, kf in enumerate(episode["key_frames"]):
            role = kf["role"]
            idx  = kf["step_idx"]
            step = episode["steps"][idx]
            state_desc   = step.get("info", {}).get("llm_state_description", "")
            neighbourhood= step.get("info", {}).get("neighbourhood", {})
            fpath = frame_paths.get(role)
            if not fpath or not os.path.exists(fpath):
                per_frame.append({"role": role, "step_idx": idx, "summary": "[SKIPPED: frame path missing]"})
                continue

            img_b64 = _b64(fpath)
            if i == 0:
                prompt = first_frame_template.format(
                    role=role.replace("_", " "),
                    ascii_grid=ascii_grid,
                    goal_fire_desc=goal_fire_desc,
                    state_desc=state_desc,
                    neighbourhood=json.dumps(neighbourhood, indent=2),
                )
            else:
                prompt = rolling_frame_template.format(
                    role=role.replace("_", " "),
                    step_idx=idx,
                    previous_summary=previous_summary,
                    goal_fire_desc=goal_fire_desc,
                    state_desc=state_desc,
                    neighbourhood=json.dumps(neighbourhood, indent=2),
                )

            summary = _chat_vlm(client, model, img_b64, prompt, system_prompt, max_tokens)
            per_frame.append({"role": role, "step_idx": idx, "summary": summary})
            previous_summary = summary
    except Exception as e:
        traceback.print_exc()
        per_frame.append({"role": "error", "step_idx": -1, "summary": f"[VLM ERROR: {e}]"})

    header = (
        f"VISION REPORT  Maze={episode['maze_name']}  "
        f"Steps={episode['total_steps']}  "
        f"Reward={episode['total_reward']:.2f}  "
        f"Success={episode['success']}"
    )
    lines = [header]
    for f_ in per_frame:
        lines.append(f"--- {f_['role']} (step {f_['step_idx']}) ---")
        lines.append(f_["summary"])
        lines.append("")

    report = "\n".join(lines)

    if cache is not None:
        scope = cache.episode_scope(episode["episode_id"])
        cache.save(scope, "vlm_output", {"report": report, "per_frame": per_frame})

    return report, per_frame


def save_vlm_report(text: str, episode_dir: Path) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)
    out = episode_dir / "vlm_report.txt"
    out.write_text(text, encoding="utf-8")
    return out