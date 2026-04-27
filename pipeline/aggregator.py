import json
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _oai_client():
    from openai import OpenAI
    return OpenAI()


def _chat_reasoning(client, model: str, messages: List[Dict], max_tokens: int, effort: str = "high") -> str:
    r = client.responses.create(model=model, input=messages, max_output_tokens=max_tokens, reasoning={"effort": effort})
    return r.output_text or ""


def _chat_plain(client, model: str, messages: List[Dict], max_tokens: int) -> str:
    r = client.responses.create(model=model, input=messages, max_output_tokens=max_tokens, reasoning={"effort": "low"})
    return r.output_text or ""


def _build_summary_block(failure_summaries: List[Dict], failure_ids: List[int]) -> str:
    out = []
    for es in failure_summaries:
        eid = es.get("episode_id")
        if eid not in failure_ids:
            continue
        dyn = es.get("dynamic_config", {})
        out.append(
            f"\n--- Episode {eid} (seed={es.get('seed','?')}, steps={es.get('total_steps','?')}, "
            f"reward={es.get('total_reward', 0.0):.2f}) ---\n"
            f"  dynamic_config: start={dyn.get('start_pos','?')} goal={dyn.get('goal_pos','?')} fires={dyn.get('fire_positions','?')}\n"
            f"{es.get('summary', '')}\n"
        )
    return "".join(out)


def cross_episode_reasoning(
    failure_summaries: List[Dict],
    failure_ids: List[int],
    maze_ascii: str,
    llm_cfg: Dict,
    cache=None,
) -> str:
    if cache is not None:
        cached = cache.load(cache.run_scope(), "aggregator_cross")
        if cached is not None:
            return cached.get("text", "")

    n = len(failure_summaries)
    client = _oai_client()
    model = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))
    effort = str(llm_cfg.get("reasoning_effort", "high"))
    summary_block = _build_summary_block(failure_summaries, failure_ids)
    valid_ids_str = ", ".join(str(i) for i in sorted(failure_ids))

    try:
        text = _chat_reasoning(
            client, model,
            [
                {"role": "system", "content":
                    "You are a cross-episode failure analyst for imitation learning in a "
                    "RANDOMISED 5x5 maze (start, goal, and fire placements all vary per episode). "
                    "Identify patterns across failures and prescribe demonstrations. "
                    "IMPORTANT: Only use the episode IDs listed in CONFIRMED FAILURE EPISODES. "
                    "Do NOT invent or include any other episode IDs."},
                {"role": "user", "content":
                    f"Below are summaries from {n} FAILED episodes in a 5x5 dynamic maze.\n\n"
                    f"REPRESENTATIVE MAZE ASCII (one episode):\n{maze_ascii}\n\n"
                    f"CONFIRMED FAILURE EPISODE IDs (ONLY these are failures): [{valid_ids_str}]\n"
                    f"Do NOT include any episode ID outside this list in failure_clusters.\n\n"
                    f"PER-FAILURE SUMMARIES:\n{summary_block}\n\n"
                    f"Answer:\n"
                    f"1. FAILURE CLUSTERING: Group failures by failure mode. Each cluster's "
                    f"episodes_in_cluster must only contain IDs from the list above.\n"
                    f"2. WHERE DOES THE POLICY STRUGGLE? Refer to regions / configurations, not fixed cells.\n"
                    f"3. WHAT DEMONSTRATIONS ARE NEEDED? Describe in plain English relative to the random config.\n"
                    f"4. HOW MANY AND HOW DIVERSE? 3-10 total demonstrations total."},
            ],
            max_tokens, effort,
        )
    except Exception as e:
        traceback.print_exc()
        text = f"[CROSS-EPISODE ERROR: {e}]"

    if cache is not None:
        cache.save(cache.run_scope(), "aggregator_cross", {"text": text})
    return text


def final_structured_prescription(
    cross_reasoning: str,
    failure_summaries: List[Dict],
    failure_ids: List[int],
    llm_cfg: Dict,
    cache=None,
) -> Tuple[str, Dict]:
    if cache is not None:
        cached = cache.load(cache.run_scope(), "aggregator_structured")
        if cached is not None:
            return cached.get("raw", ""), cached.get("parsed", {})

    n = len(failure_summaries)
    client = _oai_client()
    model = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))
    valid_ids_str = ", ".join(str(i) for i in sorted(failure_ids))
    failure_ids_set = set(failure_ids)

    try:
        raw = _chat_plain(
            client, model,
            [
                {"role": "system", "content":
                    "You are a JSON formatting assistant. Output ONLY valid JSON. "
                    "IMPORTANT: episodes_in_cluster must ONLY contain IDs from the confirmed failure list. "
                    "Guidance text must be written in plain, friendly English — no coordinates, no action codes, no technical jargon."},
                {"role": "user", "content":
                    f"A diffusion policy failed across {n} episodes in a RANDOMISED 5x5 maze. "
                    f"Cross-episode analysis:\n\n{cross_reasoning}\n\n"
                    f"CONFIRMED FAILURE EPISODE IDs: [{valid_ids_str}]\n"
                    f"Each cluster's episodes_in_cluster must ONLY contain IDs from that list.\n\n"
                    "Output ONLY this JSON (no fences, no preamble):\n"
                    "{\n"
                    f'  "n_failure_episodes_analysed": {n},\n'
                    '  "failure_modes_found": <int>,\n'
                    '  "failure_clusters": [\n'
                    '    {\n'
                    '      "cluster_label": "<short human-readable label>",\n'
                    f'      "episodes_in_cluster": [<only IDs from: {valid_ids_str}>],\n'
                    '      "root_cause": "<fire_collision | looping | wrong_direction | timeout | wall_thrashing>",\n'
                    '      "where_it_fails": "<plain English description of where in the maze>",\n'
                    '      "what_it_does_wrong": "<plain English>"\n'
                    '    }\n'
                    '  ],\n'
                    '  "demonstration_prescriptions": [\n'
                    '    {\n'
                    '      "demo_id": 1,\n'
                    '      "guidance": "<plain English, describe the SITUATION to show>",\n'
                    '      "target_region": "<plain English description of the maze area>",\n'
                    '      "what_it_teaches": "<plain English>",\n'
                    '      "n_repetitions": <1-3>\n'
                    '    }\n'
                    '  ],\n'
                    '  "total_demonstrations_needed": <int 3-10>,\n'
                    '  "overall_summary": "<2-3 plain English sentences>",\n'
                    '  "confidence": <float 0.0-1.0>\n'
                    "}"
                },
            ],
            max_tokens,
        )
    except Exception as e:
        traceback.print_exc()
        if cache is not None:
            cache.save(cache.run_scope(), "aggregator_structured", {"raw": f"[STRUCTURED ERROR: {e}]", "parsed": {}})
        return f"[STRUCTURED ERROR: {e}]", {}

    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {"raw_output": raw}

    clusters = parsed.get("failure_clusters", []) if isinstance(parsed, dict) else []
    if isinstance(clusters, list):
        for cluster in clusters:
            ep_ids = cluster.get("episodes_in_cluster", [])
            cluster["episodes_in_cluster"] = [eid for eid in ep_ids if eid in failure_ids_set]

    if cache is not None:
        cache.save(cache.run_scope(), "aggregator_structured", {"raw": raw, "parsed": parsed})

    return raw, parsed


def run_aggregator(
    failure_summaries: List[Dict],
    failure_ids: List[int],
    maze_ascii: str,
    llm_cfg: Dict,
    pipeline_flags: Optional[Dict] = None,
    cache=None,
) -> Tuple[str, str, Dict]:
    """
    Phase C wrapper.  Returns (cross_text, raw_structured, parsed_structured).

    If pipeline_flags["use_cross_episode_reasoning"] is False, the expensive
    cross-episode reasoning LLM call is skipped and the aggregator goes
    straight to final_structured_prescription with an empty cross_text.
    """
    flags = pipeline_flags or {}
    use_cer = bool(flags.get("use_cross_episode_reasoning", True))

    if use_cer:
        cross_text = cross_episode_reasoning(
            failure_summaries, failure_ids, maze_ascii, llm_cfg, cache=cache
        )
    else:
        cross_text = "[CROSS-EPISODE REASONING DISABLED FOR THIS PROFILE]"

    raw, parsed = final_structured_prescription(
        cross_text, failure_summaries, failure_ids, llm_cfg, cache=cache
    )
    return cross_text, raw, parsed


def save_final_prescription(cross_text: str, structured: Dict, run_dir: Path, episode_prescriptions: List[Dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    prescription_path = run_dir / "ablation_summary.json"
    payload = {
        "cross_episode_reasoning":  cross_text,
        "parsed_prescription":      structured,
        "per_episode_prescriptions":episode_prescriptions,
    }
    with open(prescription_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return prescription_path


def save_episode_final_prescription(text: str, episode_dir: Path) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)
    out = episode_dir / "final_prescription.txt"
    out.write_text(text, encoding="utf-8")
    return out
