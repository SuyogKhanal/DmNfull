"""Reasoning module — analysis and prescription are separate passes.

Cache policy (enforced here):
  - run_analysis and run_prescription are NEVER cached.  Each call must
    produce a fresh output so that component ablations (RAG on/off, KAG
    on/off, TKF block present/absent) always produce distinct results.
  - Only VLM outputs are cached — that happens in vlm_analyser.py.

Flow (enforced by pipeline_runner.py):
  Pass 1 → run_analysis(...)        → analysis_text
  TKF    → run_knowledge_check(analysis_text, ...)  → tkf_block
  Pass 2 → run_prescription(analysis_text, tkf_block, ...)  → prescription_text

Note on 'steps' field:
  Episodes loaded from phase_a.all_rollouts (ablation suite rollout dirs)
  only contain summary-level fields written by _save_rollout_only — there
  is NO 'steps' list.  All helpers below must treat 'steps' as optional.
"""
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _oai_client():
    from openai import OpenAI
    return OpenAI()


def _chat_reasoning(client, model: str, messages: List[Dict], max_tokens: int, effort: str = "high") -> str:
    r = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
        reasoning={"effort": effort},
    )
    return r.output_text or ""


def _chat_plain(client, model: str, messages: List[Dict], max_tokens: int) -> str:
    r = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
        reasoning={"effort": "low"},
    )
    return r.output_text or ""


def _format_trajectory(episode: Dict, max_entries: int = 60) -> str:
    """Format step-by-step trajectory.  Returns empty string if 'steps' is absent."""
    steps = episode.get("steps") or []
    if not steps:
        return "[step log not available — loaded from rollout summary]"
    entries = []
    for s in steps:
        if s.get("action") is None:
            continue
        pos = s.get("info", {}).get("agent_pos", "?")
        entries.append(
            f"  step {s['step_idx']}: at {pos} -> {s.get('action_name','?')} -> reward={s.get('reward',0.0):+.2f}"
        )
    return "\n".join(entries[:max_entries])


def _format_key_states(episode: Dict) -> str:
    """Format key-frame state descriptions.  Safe when 'steps' is absent."""
    steps = episode.get("steps") or []
    lines = []
    for kf in episode.get("key_frames", []):
        idx = kf["step_idx"]
        desc = ""
        if steps and idx < len(steps):
            desc = steps[idx].get("info", {}).get("llm_state_description", "")
        lines.append(f"[{kf['role']}, step {idx}] {desc}")
    return "\n\n".join(lines)


def _format_dynamic_config(episode: Dict) -> str:
    dyn = episode.get("dynamic_config", {})
    return (
        f"Start: {dyn.get('start_pos','?')} | "
        f"Goal: {dyn.get('goal_pos','?')} | "
        f"Fires: {dyn.get('fire_positions','?')}"
    )


def build_analysis_prompt(
    episode: Dict,
    vision_report: str,
    kag_context: str,
    rag_context: str,
) -> str:
    traj_str   = _format_trajectory(episode)
    key_states = _format_key_states(episode)
    dyn_str    = _format_dynamic_config(episode)

    # Final position: prefer last step entry; fall back to graceful placeholder.
    steps = episode.get("steps") or []
    final_pos = (
        steps[-1].get("info", {}).get("agent_pos", "?")
        if steps else "[not available]"
    )

    sections = []
    sections.append(
        "You are a navigation engineer analysing a failed maze episode.\n\n"
        "TASK: A vision-conditioned diffusion policy is learning to navigate a 5x5 grid maze "
        "from a RANDOMISED start to a RANDOMISED goal while avoiding RANDOMISED fire hazards. "
        "Training is done via human demonstrations. When the policy fails, a human must record "
        "a CORRECTIVE demonstration to improve it. Your analysis will be used to tell the human "
        "what to demonstrate.\n\n"
        "Grid cells: 0=free, 1=wall, 2=fire (terminates episode), 3=goal (success).\n"
        "Actions: 0=UP (row-1), 1=DOWN (row+1), 2=LEFT (col-1), 3=RIGHT (col+1)."
    )

    if kag_context:
        sections.append(f"ENVIRONMENT KNOWLEDGE GRAPH (KAG):\n{kag_context}")

    sections.append(
        f"MAZE LAYOUT (this episode):\n{episode.get('ascii_grid','')}\n\n"
        f"EPISODE DYNAMIC CONFIG:\n  {dyn_str}\n\n"
        f"EPISODE OUTCOME:\n"
        f"  Success      : {episode.get('success', False)}\n"
        f"  Total steps  : {episode.get('total_steps', 0)}\n"
        f"  Total reward : {episode.get('total_reward', 0.0):.2f}\n"
        f"  Final pos    : {final_pos}\n"
    )

    sections.append(f"STEP-BY-STEP TRAJECTORY:\n{traj_str}")
    sections.append(f"KEY STATE DESCRIPTIONS:\n{key_states}")

    if vision_report:
        sections.append(f"VISION ANALYSIS FROM VLM:\n{vision_report}")

    if rag_context:
        sections.append(rag_context)

    sections.append(
        "ROOT CAUSE ANALYSIS — show your working:\n"
        "1. TRAJECTORY RECONSTRUCTION\n"
        "2. FIRST WRONG DECISION\n"
        "3. WHY THE POLICY FAILED (reference the dynamic config and the KAG if provided)\n"
        "4. WHAT THE POLICY GOT WRONG\n"
        "5. REGIONS / CONFIGURATIONS NEEDING DATA"
    )

    return "\n\n".join(sections)


def build_prescription_prompt(
    episode: Dict,
    analysis_text: str,
    tkf_block: str,
) -> str:
    dyn_str = _format_dynamic_config(episode)
    return (
        "You are a demonstration coach for a DAgger-style imitation learning system.\n\n"
        "A diffusion policy failed at a maze navigation task. A human expert now needs to record "
        "CORRECTIVE DEMONSTRATIONS to improve the policy.\n\n"
        "IMPORTANT RULES FOR YOUR RESPONSE:\n"
        "  - Write in plain, friendly English that anyone can understand.\n"
        "  - Do NOT use technical terms like 'Manhattan distance', 'DAgger', 'diffusion policy', "
        "'imitation learning', or action codes.\n"
        "  - Do NOT give step-by-step action sequences or grid coordinates.\n"
        "  - Instead, describe the TYPE OF SITUATION and WHERE IN THE MAZE a demonstration is needed.\n"
        "  - Remember: the maze is RANDOMISED every episode, so prescriptions should generalise.\n\n"
        f"ROOT CAUSE ANALYSIS:\n{analysis_text}\n\n"
        f"{tkf_block}\n"
        f"EPISODE CONTEXT:\n  {dyn_str}\n\n"
        "In plain English, answer these 4 questions:\n"
        "1. WHERE in the maze does the AI get confused?\n"
        "2. WHAT should a good walkthrough look like in that area?\n"
        "3. WHAT will the AI learn from seeing that walkthrough?\n"
        "4. HOW MANY walkthroughs are needed and how should they vary?"
    )


def run_analysis(
    episode: Dict,
    vision_report: str,
    kag_context: str,
    rag_context: str,
    llm_cfg: Dict,
) -> str:
    """Pass 1: Root-cause analysis only.  No cache — always fresh.

    Returns analysis_text (str).  This output is then fed into TKF so the
    knowledge fetcher reasons about the ACTUAL failure, not a placeholder.
    """
    model = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))
    effort = str(llm_cfg.get("reasoning_effort", "high"))

    client = _oai_client()
    analysis_prompt = build_analysis_prompt(episode, vision_report, kag_context, rag_context)

    try:
        return _chat_reasoning(
            client, model,
            [
                {"role": "system", "content":
                    "You are a navigation failure analyst for DAgger imitation learning. "
                    "Walk through the agent's trajectory step by step. Identify what went wrong "
                    "and why. Think step by step."},
                {"role": "user", "content": analysis_prompt},
            ],
            max_tokens, effort,
        )
    except Exception as e:
        traceback.print_exc()
        return f"[ANALYSIS ERROR: {e}]"


def run_prescription(
    episode: Dict,
    analysis_text: str,
    tkf_block: str,
    llm_cfg: Dict,
) -> Tuple[str, str]:
    """Pass 2: Prescription — uses the real analysis_text AND the real tkf_block.

    Called AFTER TKF so the tkf_block is populated with actual retrieval
    results (including sim scores and fallback demos when below threshold).

    Returns (prescription_text, combined_text).
    No cache — always fresh.
    """
    model = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))
    effort = str(llm_cfg.get("reasoning_effort", "high"))

    client = _oai_client()
    prescription_prompt = build_prescription_prompt(episode, analysis_text, tkf_block)
    prescription_text = ""

    try:
        prescription_text = _chat_reasoning(
            client, model,
            [
                {"role": "system", "content":
                    "You are a friendly demonstration coach. Write in plain, everyday English. "
                    "No coordinates, no technical jargon, no action sequences. "
                    "Describe what the human should show the AI in simple terms."},
                {"role": "user", "content": prescription_prompt},
            ],
            max_tokens, effort,
        )
    except Exception as e:
        traceback.print_exc()
        prescription_text = f"[PRESCRIPTION ERROR: {e}]"

    combined = (
        "=== ROOT CAUSE ANALYSIS ===\n"
        f"{analysis_text}\n\n"
        f"{tkf_block}\n"
        "=== DEMONSTRATION PRESCRIPTION ===\n"
        f"{prescription_text}\n"
    )
    return prescription_text, combined


def run_reasoning(
    episode: Dict,
    vision_report: str,
    kag_context: str,
    rag_context: str,
    tkf_block: str,
    llm_cfg: Dict,
    cache=None,           # ignored — kept for backwards-compat signature only
    use_vlm: bool = True, # unused here, kept for compat
    use_kag: bool = True,
    use_rag: bool = True,
) -> Tuple[str, str, str]:
    """Backwards-compat wrapper: analysis + prescription in one call.

    Prefer calling run_analysis() then run_prescription() directly (as
    pipeline_runner does) so TKF can run in between the two passes.

    Returns (analysis_text, prescription_text, combined_text).
    """
    analysis_text = run_analysis(
        episode=episode,
        vision_report=vision_report if use_vlm else "",
        kag_context=kag_context if use_kag else "",
        rag_context=rag_context if use_rag else "",
        llm_cfg=llm_cfg,
    )
    prescription_text, combined_text = run_prescription(
        episode=episode,
        analysis_text=analysis_text,
        tkf_block=tkf_block,
        llm_cfg=llm_cfg,
    )
    return analysis_text, prescription_text, combined_text


def summarise_episode(
    reasoning_text: str,
    episode_id: int,
    llm_cfg: Dict,
    cache=None,           # ignored — kept for backwards-compat signature only
    use_vlm: bool = True,
    use_kag: bool = True,
    use_rag: bool = True,
) -> str:
    """Condense full reasoning into a 3-5 sentence summary for Phase C input.

    Not cached — must reflect whatever combined_text was produced this run.
    """
    model = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))

    client = _oai_client()
    try:
        text = _chat_plain(
            client, model,
            [
                {"role": "system", "content": "Extract a concise 3-5 sentence summary. No preamble."},
                {"role": "user", "content":
                    f"Below is a detailed analysis of a failed maze navigation episode (id={episode_id}).\n"
                    f"Extract a concise summary (3-5 sentences) covering:\n"
                    f"- What went wrong (root cause)\n"
                    f"- Key configuration (start, goal, fire placement)\n"
                    f"- What the correct path should have been\n\n"
                    f"ANALYSIS:\n{reasoning_text}\n\n"
                    f"Output ONLY the summary, nothing else."},
            ],
            max_tokens,
        )
    except Exception as e:
        traceback.print_exc()
        text = reasoning_text

    return text


def save_reasoning(text: str, episode_dir: Path) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)
    out = episode_dir / "reasoning.txt"
    out.write_text(text, encoding="utf-8")
    return out
