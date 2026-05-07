import json
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

_FINAL_REC_RE = re.compile(
    r"<<<FINAL_REC>>>(.*?)<<<END_FINAL_REC>>>",
    re.DOTALL,
)

_VALID_CORRIDORS = {
    "left_edge", "top_edge", "right_edge", "bottom_edge", "central_mixed",
}


def _parse_final_rec(text: str) -> Dict:
    """Extract corridor / steps / n_demos / demo_variations / rationale from a
    FINAL_REC block. Returns {} if missing or malformed so callers can detect it.

    Looks for the LAST occurrence in the text — the prescription stage echoes the
    block back at the top of its own output, so when reasoning_combined contains
    both the analysis and the prescription, the prescription's copy is the most
    recent view of the recommendation.
    """
    if not text:
        return {}
    matches = list(_FINAL_REC_RE.finditer(text))
    if not matches:
        return {}
    body = matches[-1].group(1).strip()

    out: Dict = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip().lower()] = v.strip()

    if "n_demos" in out:
        try:
            out["n_demos"] = int(re.sub(r"[^\d-]", "", out["n_demos"]) or "0")
        except ValueError:
            pass
    if "corridor" in out:
        out["corridor"] = out["corridor"].strip().lower()
    return out


def _aggregate_n_demos(parsed_recs: List[Dict]) -> int:
    """Median (rounded up) of per-episode n_demos. Conservative floor of 1."""
    vals = [r.get("n_demos") for r in parsed_recs if isinstance(r.get("n_demos"), int) and r["n_demos"] > 0]
    if not vals:
        return 1
    vals = sorted(vals)
    mid = len(vals) // 2
    if len(vals) % 2 == 0:
        med = (vals[mid - 1] + vals[mid]) / 2
    else:
        med = vals[mid]
    import math
    return max(1, int(math.ceil(med)))


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
    """Build per-episode context for the aggregator with FINAL_REC parsed out.

    The aggregator should use the parsed FINAL_REC fields — corridor, steps,
    n_demos — as the authoritative recommendation. The full reasoning is still
    included for context but explicitly marked 'do NOT contradict FINAL_REC'.
    """
    out = []
    for es in failure_summaries:
        eid = es.get("episode_id")
        if eid not in failure_ids:
            continue
        dyn = es.get("dynamic_config", {})

        full_reasoning = es.get("reasoning_combined", "").strip()
        rec = _parse_final_rec(full_reasoning)

        header = (
            f"\n--- Episode {eid} (seed={es.get('seed','?')}, "
            f"steps={es.get('total_steps','?')}, "
            f"reward={es.get('total_reward', 0.0):.2f}) ---\n"
            f"  dynamic_config: start={dyn.get('start_pos','?')} "
            f"goal={dyn.get('goal_pos','?')} "
            f"fires={dyn.get('fire_positions','?')}\n"
        )

        if rec:
            rec_block = (
                "  FINAL_REC (parsed — authoritative for this episode):\n"
                f"    corridor:        {rec.get('corridor','?')}\n"
                f"    steps:           {rec.get('steps','?')}\n"
                f"    n_demos:         {rec.get('n_demos','?')}\n"
                f"    demo_variations: {rec.get('demo_variations','?')}\n"
                f"    rationale:       {rec.get('rationale','?')}\n"
            )
        else:
            rec_block = "  FINAL_REC: <missing or malformed — flag this in the cluster output>\n"

        short_summary = es.get("summary", "").strip()
        body = (
            f"  SUMMARY: {short_summary}\n\n"
            f"{rec_block}\n"
            "  FULL REASONING (for context only — must not contradict FINAL_REC):\n"
            f"{full_reasoning}\n"
        )
        out.append(header + body)
    return "".join(out)


def cross_episode_reasoning(
    failure_summaries: List[Dict],
    failure_ids: List[int],
    maze_ascii: str,
    llm_cfg: Dict,
    cache=None,
    kag_context: str = "",
) -> str:
    # Update 3: cross-episode reasoning is NEVER cached so it always reflects
    # the freshly generated (uncached) per-episode reasoning_combined texts.
    n = len(failure_summaries)
    client = _oai_client()
    model = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))
    effort = str(llm_cfg.get("reasoning_effort", "high"))
    summary_block = _build_summary_block(failure_summaries, failure_ids)
    valid_ids_str = ", ".join(str(i) for i in sorted(failure_ids))

    kag_block = ""
    if kag_context and kag_context.strip():
        kag_block = (
            "ENVIRONMENT KNOWLEDGE GRAPH (KAG) — HIGH PRIORITY. The corridor names\n"
            "and failure-mode taxonomy below are the ONLY allowed vocabulary for\n"
            "your clusters. When you describe a cluster's region or failure mode,\n"
            "you must ground it in the KAG facts.\n"
            f"{kag_context}\n\n"
        )

    user_content = (
    f"Below are summaries from {n} FAILED episodes in a 5x5 dynamic maze.\n\n"
    f"{kag_block}"
    f"REPRESENTATIVE MAZE ASCII (one episode):\n{maze_ascii}\n\n"
    f"CONFIRMED FAILURE EPISODE IDs (ONLY these are failures): [{valid_ids_str}]\n"
    f"Do NOT include any episode ID outside this list in failure_clusters.\n\n"
    f"PER-FAILURE SUMMARIES (each contains a parsed FINAL_REC block — those values "
    f"are authoritative for that episode and you must not contradict them):\n"
    f"{summary_block}\n\n"
    "Answer:\n"
    "1. FAILURE CLUSTERING: Group failures by corridor (from FINAL_REC; corridor names\n"
    "   must come from the KAG when KAG is provided) and by failure mode.\n"
    "2. WHERE DOES THE POLICY STRUGGLE? Refer to regions / corridors using KAG names.\n"
    "3. WHAT DEMONSTRATIONS ARE NEEDED? For each cluster, name the corridor (from FINAL_REC)\n"
    "   and aggregate the n_demos values (median rounded up) for that cluster.\n"
    "4. WHAT LAYOUTS COVER EACH CLUSTER? For each cluster, propose 1-5 concrete maze\n"
    "   layouts (start_pos, goal_pos, fire_positions) within the 5x5 grid that exercise\n"
    "   the failure mode. Pick the SMALLEST set that genuinely covers the cluster — if\n"
    "   one well-chosen layout teaches the corridor, do not pad with three. The layouts\n"
    "   must lie inside the cluster's corridor where possible, place fires that block the\n"
    "   failed direct path, and respect:\n"
    "     - all positions in [0..4] x [0..4]\n"
    "     - start_pos != goal_pos and Manhattan(start, goal) >= 4\n"
    "     - fire_positions disjoint from start_pos and goal_pos\n"
    "     - exactly 3 fire cells per layout\n"
    "5. HOW MANY AND HOW DIVERSE? Total n_demos should reflect the sum across clusters.\n"
    "   Each recommended layout should be demonstrated 1-5 times to cover variation;\n"
    "   pick the count that matches how multi-modal the failure actually is, not a fixed\n"
    "   number. The orchestrator stops when held-out SR >= 90%, so neither over- nor\n"
    "   under-prescribing helps — be honest about the gap each layout closes."
    )

    # Optional gated addendum: callers (e.g. Equivariant_pathway/p4_only) may
    # supply llm_cfg["prompt_addendum_aggregator"] to add a run-specific
    # directive (e.g. holistic sample-efficiency emphasis). Empty / missing
    # leaves the legacy prompt untouched.
    addendum = str(llm_cfg.get("prompt_addendum_aggregator", "") or "").strip()
    if addendum:
        user_content = f"{user_content}\n\n{addendum}"

    try:
        text = _chat_reasoning(
            client, model,
            [
                {"role": "system", "content":
                    "You are a cross-episode failure analyst for imitation learning in a "
                    "RANDOMISED 5x5 maze (start, goal, and fire placements all vary per episode). "
                    "Identify patterns across failures and prescribe demonstrations. The KAG "
                    "corridor / failure-mode vocabulary, when provided, is HIGH PRIORITY and "
                    "your clusters MUST use those exact corridor names. "
                    "IMPORTANT: Only use the episode IDs listed in CONFIRMED FAILURE EPISODES. "
                    "Do NOT invent or include any other episode IDs."},
                {"role": "user", "content": user_content},
            ],
            max_tokens, effort,
        )
    except Exception as e:
        traceback.print_exc()
        text = f"[CROSS-EPISODE ERROR: {e}]"

    return text


def _validate_layout(layout: Dict, grid_size: int = 5, n_fires: int = 3) -> Tuple[bool, str]:
    """Return (ok, reason). A valid layout has 5x5-bounded start/goal/fires,
    Manhattan(start, goal) >= 4, 3 fire cells, no overlap between start/goal/fires."""
    def _coerce_pair(p):
        try:
            r, c = int(p[0]), int(p[1])
            return (r, c)
        except Exception:
            return None
    sp = _coerce_pair(layout.get("start_pos") or [])
    gp = _coerce_pair(layout.get("goal_pos") or [])
    fp_raw = layout.get("fire_positions") or []
    if sp is None or gp is None:
        return False, "missing start_pos or goal_pos"
    for r, c in (sp, gp):
        if not (0 <= r < grid_size and 0 <= c < grid_size):
            return False, f"position out of grid: {(r,c)}"
    if sp == gp:
        return False, "start equals goal"
    if (abs(sp[0] - gp[0]) + abs(sp[1] - gp[1])) < 4:
        return False, "Manhattan(start, goal) < 4"
    fires = []
    for f in fp_raw:
        cf = _coerce_pair(f)
        if cf is None:
            return False, f"malformed fire entry: {f}"
        if not (0 <= cf[0] < grid_size and 0 <= cf[1] < grid_size):
            return False, f"fire out of grid: {cf}"
        fires.append(cf)
    if len(set(fires)) != len(fires):
        return False, "duplicate fire positions"
    if len(fires) != n_fires:
        return False, f"expected {n_fires} fires, got {len(fires)}"
    if sp in fires or gp in fires:
        return False, "fire overlaps start or goal"
    return True, "ok"


def final_structured_prescription(
    cross_reasoning: str,
    failure_summaries: List[Dict],
    failure_ids: List[int],
    llm_cfg: Dict,
    cache=None,
    kag_context: str = "",
) -> Tuple[str, Dict]:
    """Convert cross-episode reasoning + per-episode FINAL_RECs into structured JSON.

    Never cached — must reflect the freshly produced cross_reasoning each run.
    """
    n = len(failure_summaries)
    client = _oai_client()
    model = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))
    valid_ids_str = ", ".join(str(i) for i in sorted(failure_ids))
    failure_ids_set = set(failure_ids)

    # Build per-episode FINAL_REC list so the aggregator and post-validator can
    # see exactly what each episode recommended.
    per_ep_recs: List[Dict] = []
    for es in failure_summaries:
        eid = es.get("episode_id")
        if eid not in failure_ids_set:
            continue
        rec = _parse_final_rec(es.get("reasoning_combined", "") or "")
        rec["episode_id"] = eid
        per_ep_recs.append(rec)

    rec_table = "\n".join(
        f"  ep={r.get('episode_id')}  corridor={r.get('corridor','?')}  "
        f"n_demos={r.get('n_demos','?')}  steps={r.get('steps','?')}"
        for r in per_ep_recs
    )
    aggregated_n = _aggregate_n_demos(per_ep_recs)

    kag_block = ""
    if kag_context and kag_context.strip():
        kag_block = (
            "ENVIRONMENT KNOWLEDGE GRAPH (KAG) — HIGH PRIORITY. Use these corridor\n"
            "names verbatim and respect the failure-mode taxonomy. Layouts you\n"
            "recommend should fall inside the corridor regions defined here:\n"
            f"{kag_context}\n\n"
        )

    user_lead = (
        f"A diffusion policy failed across {n} episodes in a RANDOMISED 5x5 maze.\n\n"
        f"{kag_block}"
        f"CROSS-EPISODE REASONING (primary source):\n{cross_reasoning}\n\n"
        f"PER-EPISODE FINAL_REC TABLE (authoritative, do not contradict):\n{rec_table}\n\n"
        f"AGGREGATED n_demos (median rounded up across the FINAL_RECs above): {aggregated_n}\n\n"
        f"CONFIRMED FAILURE EPISODE IDs: [{valid_ids_str}]\n"
        f"Each cluster's episodes_in_cluster MUST only contain IDs from that list.\n\n"
    )
    # Optional gated addendum: callers (e.g. Equivariant_pathway/p4_only) may
    # supply llm_cfg["prompt_addendum_aggregator"] to inject a run-specific
    # directive (holistic / sample-efficiency / minimum-layouts framing).
    # Inserted between the context block and the legacy RULES so it informs
    # how the LLM reads the rules. Empty / missing = identical legacy prompt.
    addendum = str(llm_cfg.get("prompt_addendum_aggregator", "") or "").strip()
    if addendum:
        user_lead = f"{user_lead}{addendum}\n\n"
    try:
        raw = _chat_plain(
            client, model,
            [
                {"role": "system", "content":
                    "You are a JSON formatting assistant. "
                    "Output ONLY valid JSON — no fences, no preamble. "
                    "Every field must be derived from the cross-episode reasoning AND the "
                    "per-episode FINAL_REC table. Do not invent a different corridor, "
                    "different n_demos, or different episode IDs. "
                    "When a KAG block is provided, corridor names MUST come from it."},
                {"role": "user", "content":
                    user_lead +
                    "RULES:\n"
                    "  - failure_clusters[].corridor MUST be one of left_edge, top_edge, "
                    "right_edge, bottom_edge, central_mixed, or 'mixed' if the cluster spans corridors.\n"
                    "  - failure_clusters[].source_final_recs MUST be the list of corridor strings "
                    "from the FINAL_REC table for the episodes in that cluster, in the same order.\n"
                    "  - demonstration_prescriptions[].corridor MUST be a single corridor enum value.\n"
                    "  - demonstration_prescriptions[].n_repetitions MUST be derived from FINAL_REC "
                    "n_demos values for the episodes in the matching cluster (use the cluster median, "
                    "rounded up; never less than 1).\n"
                    "  - demonstration_prescriptions[].recommended_layouts is REQUIRED and is a list\n"
                    "    of 1-5 concrete layouts the human should record. Pick the SMALLEST list\n"
                    "    that addresses the cluster — padding hurts, because the orchestrator runs\n"
                    "    until held-out SR >= 90% and every extra demo is more BFS-collection time.\n"
                    "    Each layout MUST satisfy:\n"
                    "      * start_pos and goal_pos are integer [r, c] with 0<=r,c<=4\n"
                    "      * Manhattan(start_pos, goal_pos) >= 4\n"
                    "      * fire_positions is exactly 3 distinct [r,c] cells, none equal to start or goal\n"
                    "      * the layout exercises the cluster's failure mode in the named corridor\n"
                    "    Each layout has its own n_repetitions (1-5) describing how many demos to\n"
                    "    record on that exact layout. The sum of layout n_repetitions across a\n"
                    "    prescription's recommended_layouts MUST equal that prescription's n_repetitions.\n"
                    "  - total_demonstrations_needed MUST equal the sum of all "
                    "demonstration_prescriptions[].n_repetitions and SHOULD be close to "
                    f"the aggregated_n value above ({aggregated_n}); deviate only if you split "
                    "across clusters that need different corridors.\n\n"
                    "Output ONLY this JSON (no fences, no preamble):\n"
                    "{\n"
                    f'  "n_failure_episodes_analysed": {n},\n'
                    '  "failure_modes_found": <int>,\n'
                    '  "failure_clusters": [\n'
                    '    {\n'
                    '      "cluster_label": "<short, run-specific>",\n'
                    f'      "episodes_in_cluster": [<only IDs from: {valid_ids_str}>],\n'
                    '      "root_cause": "<fire_collision | looping | wrong_direction | timeout | wall_thrashing>",\n'
                    '      "corridor": "<left_edge | top_edge | right_edge | bottom_edge | central_mixed | mixed>",\n'
                    '      "where_it_fails": "<plain English, specific to this run>",\n'
                    '      "what_it_does_wrong": "<plain English, specific to this run>",\n'
                    '      "source_final_recs": [<corridor strings from FINAL_REC for the episodes in this cluster>]\n'
                    '    }\n'
                    '  ],\n'
                    '  "demonstration_prescriptions": [\n'
                    '    {\n'
                    '      "demo_id": 1,\n'
                    '      "corridor": "<one corridor enum value>",\n'
                    '      "guidance": "<plain English>",\n'
                    '      "target_region": "<plain English>",\n'
                    '      "what_it_teaches": "<plain English>",\n'
                    '      "n_repetitions": <integer derived from FINAL_REC n_demos>,\n'
                    '      "recommended_layouts": [\n'
                    '        {\n'
                    '          "start_pos": [<int 0-4>, <int 0-4>],\n'
                    '          "goal_pos":  [<int 0-4>, <int 0-4>],\n'
                    '          "fire_positions": [[<int>,<int>], [<int>,<int>], [<int>,<int>]],\n'
                    '          "n_repetitions": <int 1-5>,\n'
                    '          "rationale": "<one sentence tying this layout to the cluster failure>"\n'
                    '        }\n'
                    '      ]\n'
                    '    }\n'
                    '  ],\n'
                    '  "total_demonstrations_needed": <int>,\n'
                    '  "overall_summary": "<2-3 sentences specific to these failures>",\n'
                    '  "confidence": <float 0.0-1.0>\n'
                    "}"
                },
            ],
            max_tokens,
        )
    except Exception as e:
        traceback.print_exc()
        return f"[STRUCTURED ERROR: {e}]", {}

    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {"raw_output": raw}

    # Filter cluster ep IDs to confirmed failures only.
    if isinstance(parsed, dict):
        for cluster in parsed.get("failure_clusters", []) or []:
            ep_ids = cluster.get("episodes_in_cluster", []) or []
            cluster["episodes_in_cluster"] = [eid for eid in ep_ids if eid in failure_ids_set]

        # Post-parse validation: warn (don't fail) if cluster.corridor disagrees
        # with its source_final_recs. This is the audit trail that tells you
        # whether the aggregator is summarising or genuinely cross-reasoning.
        for cluster in parsed.get("failure_clusters", []) or []:
            srcs = cluster.get("source_final_recs", []) or []
            decl = (cluster.get("corridor") or "").strip().lower()
            unique_srcs = {(s or "").strip().lower() for s in srcs if s}
            unique_srcs.discard("")
            if decl and unique_srcs and decl not in unique_srcs and decl != "mixed":
                print(
                    f"[Aggregator] WARNING cluster '{cluster.get('cluster_label')}': "
                    f"declared corridor='{decl}' but source FINAL_RECs were "
                    f"{sorted(unique_srcs)}"
                )

        # Sanity: total_demonstrations_needed should equal the sum.
        total_decl = parsed.get("total_demonstrations_needed")
        prescriptions = parsed.get("demonstration_prescriptions", []) or []
        total_calc = sum(int(d.get("n_repetitions", 0) or 0) for d in prescriptions)
        if isinstance(total_decl, int) and total_calc and total_decl != total_calc:
            print(
                f"[Aggregator] WARNING total_demonstrations_needed={total_decl} "
                f"but sum(n_repetitions)={total_calc} — using the sum."
            )
            parsed["total_demonstrations_needed"] = total_calc

        # Validate recommended_layouts on each prescription. Drop invalid ones,
        # warn so the run log shows which were filtered. The active loop only
        # launches play_maze for layouts that pass validation, so an invalid
        # layout cannot crash the demo-collection step.
        for pres in prescriptions:
            kept_layouts: List[Dict] = []
            raw_layouts = pres.get("recommended_layouts", []) or []
            for li, layout in enumerate(raw_layouts):
                ok, reason = _validate_layout(layout)
                if not ok:
                    print(
                        f"[Aggregator] WARNING demo_id={pres.get('demo_id','?')} "
                        f"layout #{li}: dropped ({reason}) — {layout}"
                    )
                    continue
                # Coerce types so downstream consumers (play_maze CLI) get clean ints.
                layout["start_pos"]      = [int(layout["start_pos"][0]), int(layout["start_pos"][1])]
                layout["goal_pos"]       = [int(layout["goal_pos"][0]),  int(layout["goal_pos"][1])]
                layout["fire_positions"] = [[int(r), int(c)] for r, c in layout["fire_positions"]]
                try:
                    layout["n_repetitions"] = max(1, int(layout.get("n_repetitions", 1) or 1))
                except Exception:
                    layout["n_repetitions"] = 1
                kept_layouts.append(layout)
            pres["recommended_layouts"] = kept_layouts
            if not kept_layouts:
                print(
                    f"[Aggregator] WARNING demo_id={pres.get('demo_id','?')} has no "
                    f"valid recommended_layouts after filtering."
                )

    return raw, parsed


def run_aggregator(
    failure_summaries: List[Dict],
    failure_ids: List[int],
    maze_ascii: str,
    llm_cfg: Dict,
    pipeline_flags: Optional[Dict] = None,
    cache=None,
    kag_context: str = "",
) -> Tuple[str, str, Dict]:
    """
    Phase C wrapper.  Returns (cross_text, raw_structured, parsed_structured).

    kag_context, when non-empty, is injected into both the cross-episode
    reasoning prompt and the structured-prescription prompt with HIGH
    priority — corridor names must come from the KAG and recommended
    layouts must lie inside the KAG corridor regions.
    """
    flags = pipeline_flags or {}
    use_cer = bool(flags.get("use_cross_episode_reasoning", True))
    use_kag = bool(flags.get("use_kag", True))
    kag_for_aggregator = kag_context if use_kag else ""

    if use_cer:
        cross_text = cross_episode_reasoning(
            failure_summaries, failure_ids, maze_ascii, llm_cfg,
            cache=None, kag_context=kag_for_aggregator,
        )
    else:
        cross_text = "[CROSS-EPISODE REASONING DISABLED FOR THIS PROFILE]"

    raw, parsed = final_structured_prescription(
        cross_text, failure_summaries, failure_ids, llm_cfg,
        cache=None, kag_context=kag_for_aggregator,
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
