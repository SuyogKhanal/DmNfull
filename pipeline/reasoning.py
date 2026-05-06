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
  is NO 'steps' list.  All helpers below must treat 'steps' as optional and
  build a usable fallback summary from the rollout-summary fields that ARE
  present (total_steps, total_reward, success, dynamic_config, key_frames).
  Without this fallback, the cross-episode LLM sees no trajectory evidence
  and defaults to a generic wrong label like "uniform looping".
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
    """Format step-by-step trajectory.

    When the per-step log is absent (rollout-summary episodes loaded from
    phase_a.all_rollouts in an ablation-suite full_output.json), build a
    fallback summary from the fields that ARE present so the cross-episode
    LLM has enough evidence to correctly identify fire-collision vs timeout
    vs wall-thrashing — instead of defaulting to a generic "looping" label.
    """
    steps = episode.get("steps") or []
    if not steps:
        dyn = episode.get("dynamic_config", {}) or {}
        total_steps  = episode.get("total_steps", "?")
        total_reward = episode.get("total_reward", 0.0)
        success      = episode.get("success", False)
        start_pos    = dyn.get("start_pos", "?")
        goal_pos     = dyn.get("goal_pos", "?")
        fires        = dyn.get("fire_positions", "?")
        kf_parts = []
        for kf in episode.get("key_frames", []) or []:
            role = kf.get("role", "?")
            idx  = kf.get("step_idx", "?")
            kf_parts.append(f"{role} at step {idx}")
        kf_str = ", ".join(kf_parts) if kf_parts else "(no key frames)"
        try:
            reward_str = f"{float(total_reward):.2f}"
        except (TypeError, ValueError):
            reward_str = str(total_reward)
        return (
            f"[Step log not available. Summary: {total_steps} steps, "
            f"reward={reward_str}, success={success}.\n"
            f"Start={start_pos}, Goal={goal_pos}, Fires={fires}.\n"
            f"Key frames: {kf_str}]"
        )
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
        "a CORRECTIVE demonstration to improve it. Your analysis will be CONSUMED downstream by "
        "a prescription LLM that will derive the structured FINAL_REC (corridor, path, n_demos, "
        "demo_variations, rationale) FROM YOUR ANALYSIS — so be specific and grounded.\n\n"
        "Grid cells: 0=free, 1=wall, 2=fire (terminates episode), 3=goal (success).\n"
        "Actions: 0=UP (row-1), 1=DOWN (row+1), 2=LEFT (col-1), 3=RIGHT (col+1)."
    )

    if kag_context:
        sections.append(
            "ENVIRONMENT KNOWLEDGE GRAPH (KAG) — the corridor names and failure-mode "
            "definitions in this block are the ONLY allowed corridor vocabulary. "
            "When you discuss the chosen corridor in section 5 you MUST name it using "
            "a corridor key from this KAG and quote the supporting KAG fact verbatim "
            f"(in single quotes).\n{kag_context}"
        )

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
        sections.append(
            "RETRIEVED SIMILAR PAST FAILURES (RAG) — these are real prior episodes "
            "with similar visual / config signatures. In section 4 below you MUST "
            "consider each retrieved case explicitly: name its rank, state whether "
            "its corridor / path applies to THIS episode, and either reuse it or "
            "justify why a different choice is needed. Do not silently ignore the "
            f"retrievals.\n{rag_context}"
        )

    grounding_rules = []
    grounding_rules.append(
        "- Section 3 must name the failure mode using KAG terminology if a KAG block "
        "was provided above; otherwise describe it in plain language."
        if kag_context
        else "- Section 3 describes the failure mode in plain language (no KAG provided)."
    )
    grounding_rules.append(
        "- Section 4 must enumerate every retrieved RAG case (by rank) and give a "
        "one-sentence verdict for each: APPLIES / PARTIAL / DOES NOT APPLY, with reason."
        if rag_context
        else "- Section 4: no RAG retrievals available — write 'No retrieved cases.'"
    )
    grounding_rules.append(
        "- Section 5 must pick the corridor key from the KAG corridor list and quote "
        "the KAG corridor fact (in single quotes) that justifies the pick."
        if kag_context
        else "- Section 5: no KAG provided — describe the corridor in plain geometric terms "
             "(e.g. 'top edge row', 'left edge column')."
    )

    sections.append(
        "ROOT CAUSE ANALYSIS — produce ALL FIVE sections below. Be specific, ground "
        "each claim in the evidence above, and follow these grounding rules:\n"
        + "\n".join(grounding_rules) + "\n\n"
        "1. TRAJECTORY RECONSTRUCTION — where the agent went, step by step.\n"
        "2. FIRST WRONG DECISION — the earliest step where the agent diverged from a safe path,\n"
        "   with the (row,col) and the action taken vs the action it should have taken.\n"
        "3. WHY THE POLICY FAILED — root cause referencing the dynamic config and KAG if provided.\n"
        "4. RAG CROSS-EPISODE EVIDENCE — per-rank verdicts as described in the rules above.\n"
        "5. CORRIDOR & PATH PROPOSAL — name the corridor (from KAG if provided), give a concrete\n"
        "   coordinate path (start -> ... -> goal) avoiding all fires, and a one-sentence\n"
        "   justification grounded in section 3 and (if applicable) section 4.\n\n"
        "DO NOT emit a <<<FINAL_REC>>> block. The prescription stage owns FINAL_REC and will\n"
        "derive it from your sections 1-5. Producing FINAL_REC here would let you bypass the\n"
        "grounding rules above and is explicitly forbidden."
    )

    return "\n\n".join(sections)


def build_prescription_prompt(
    episode: Dict,
    analysis_text: str,
    tkf_block: str,
    kag_context: str = "",
    rag_context: str = "",
) -> str:
    dyn_str = _format_dynamic_config(episode)

    have_kag = bool(kag_context and kag_context.strip())
    have_rag = bool(rag_context and rag_context.strip() and "no matches above threshold" not in rag_context.lower())
    have_tkf = bool(tkf_block and tkf_block.strip() and "DISABLED" not in tkf_block and "ERROR" not in tkf_block)

    grounding_rules = ["STRICT GROUNDING RULES — these decide whether your FINAL_REC is valid:\n"]

    if have_kag:
        grounding_rules.append(
            "  K1. The `corridor` field in FINAL_REC MUST be the corridor key the analysis selected\n"
            "      in section 5, and that key MUST appear in the KAG corridor block below. The\n"
            "      `rationale` field MUST quote (in single quotes) the KAG corridor fact that\n"
            "      justifies it. If section 5 of the analysis did not pick a KAG corridor, you\n"
            "      MUST pick the closest matching KAG corridor and cite its fact — do not invent.\n"
        )
    else:
        grounding_rules.append(
            "  K1. No KAG was supplied. Pick `corridor` from the enum directly using the analysis\n"
            "      geometry described in section 5.\n"
        )

    if have_rag:
        grounding_rules.append(
            "  R1. The `demo_variations` field MUST reference the RAG retrievals by rank when\n"
            "      they apply (e.g. 'covers rag rank=1 left_edge variant and rank=2 short-loop'),\n"
            "      OR the `rationale` MUST explicitly state which retrieved cases were rejected\n"
            "      and why. Do NOT produce a FINAL_REC that ignores the retrieved cases.\n"
            "  R2. If two or more retrieved cases share the same corridor and that corridor\n"
            "      matches your chosen one, n_demos MUST be at least 2 to cover variation.\n"
        )
    else:
        grounding_rules.append(
            "  R1. No RAG retrievals available. demo_variations should describe natural\n"
            "      variation (start offset, fire-side preference) without citing retrieved cases.\n"
        )

    if have_tkf:
        grounding_rules.append(
            "  T1. A TKF block is present. demo_variations MUST acknowledge whether the existing\n"
            "      demos cover the corridor (FOUND), partially cover it (PARTIAL), or are missing\n"
            "      (NOT_FOUND), and the n_demos MUST reflect that gap (NOT_FOUND -> at least 3,\n"
            "      PARTIAL -> at least 2, FOUND -> 1 is acceptable).\n"
        )

    grounding_rules.append(
        "  S1. The `steps` field MUST be a coordinate path that:\n"
        "      (a) starts at the episode start_pos and ends at the goal_pos,\n"
        "      (b) never enters a fire cell, and\n"
        "      (c) lies inside the chosen corridor for the bulk of its length.\n"
        "      Reuse the path the analysis proposed in section 5 unless it violates (a)-(c).\n"
        "  S2. If you cannot produce a FINAL_REC that satisfies the rules above, output a single\n"
        "      line '[FINAL_REC UNGROUNDED: <one-sentence reason>]' instead of inventing one.\n"
    )

    rules_block = "\n".join(grounding_rules)

    enum_line = (
        "corridor: <left_edge | top_edge | right_edge | bottom_edge | central_mixed>"
    )

    parts = [
        "You are a demonstration coach for a DAgger-style imitation learning system.",
        "",
        "Your job is to produce the FINAL_REC structured block by combining the upstream",
        "components (analysis, KAG, RAG, TKF) and then explain it in plain English.",
        "",
        "BUDGET GUIDANCE (read carefully — the orchestrator runs until heldout SR >= 90%,",
        "so over-prescribing wastes BFS-collection time and under-prescribing prolongs",
        "the loop). Pick n_demos and the layout count that BEST address the failure",
        "without padding:",
        "  - n_demos: any integer in [1, 20]. Use 1-2 when the failure is local (one",
        "    corridor, one direction); use 3-5 for genuinely multi-modal failures (two",
        "    valid paths, both must be taught); use 6-10 only when KAG/RAG/TKF jointly",
        "    show several distinct gaps; >10 is reserved for cases where TKF reports",
        "    NOT_FOUND on multiple corridors at once.",
        "  - recommended_layouts: 1-5 concrete layouts per cluster. Prefer fewer, more",
        "    targeted layouts over many redundant ones. Each layout must teach something",
        "    the others don't (different start, different fire pattern, etc.).",
        "Be honest with these numbers — a too-conservative n_demos that misses the",
        "failure mode is just as expensive as an inflated one, because the loop will",
        "have to run another round.",
        "",
        "INPUTS YOU MUST USE (each one materially changes FINAL_REC):",
        "  1. ROOT-CAUSE ANALYSIS  — sections 1-5 from the reasoning LLM (always present).",
        f"  2. KAG CORRIDOR / FAILURE GRAPH  — {'PRESENT' if have_kag else 'absent'}.",
        f"  3. RAG RETRIEVED PRIOR FAILURES — {'PRESENT' if have_rag else 'absent'}.",
        f"  4. TKF DEMO COVERAGE BLOCK     — {'PRESENT' if have_tkf else 'absent'}.",
        "",
        rules_block,
        "",
        "=== ANALYSIS (sections 1-5) ===",
        analysis_text,
        "",
    ]
    if have_kag:
        parts += ["=== KAG (corridor + failure-mode vocabulary) ===", kag_context, ""]
    if have_rag:
        parts += ["=== RAG (retrieved similar past failures) ===", rag_context, ""]
    if have_tkf:
        parts += ["=== TKF (demo coverage check) ===", tkf_block, ""]
    parts += [
        f"EPISODE CONTEXT:\n  {dyn_str}",
        "",
        "OUTPUT FORMAT — produce EXACTLY this structure (no preamble before the markers):",
        "",
        "<<<FINAL_REC>>>",
        enum_line,
        "steps: <(r,c)->(r,c)->...->(r,c)>",
        "n_demos: <integer 1-20 — pick the smallest count that genuinely addresses the failure>",
        "demo_variations: <one-line description (must satisfy R1/T1 if applicable)>",
        "rationale: <one sentence (must satisfy K1 if KAG present)>",
        "<<<END_FINAL_REC>>>",
        "",
        "PLAIN-ENGLISH EXPLANATION FOR THE HUMAN DEMONSTRATOR:",
        "1. WHERE in the maze does the AI get confused? (reference the corridor named in FINAL_REC)",
        "2. WHAT should a good walkthrough look like in that area? (describe the same path FINAL_REC specifies)",
        "3. WHAT will the AI learn from seeing that walkthrough?"
        + (" (tie this to the RAG-retrieved cases by rank)" if have_rag else ""),
        "4. HOW MANY walkthroughs are needed and how should they vary? (state n_demos from FINAL_REC and reuse demo_variations)",
    ]

    return "\n".join(parts)


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
    kag_context: str = "",
    rag_context: str = "",
) -> Tuple[str, str]:
    """Pass 2: Prescription — DERIVES FINAL_REC from analysis + KAG + RAG + TKF.

    Previously this stage merely forwarded a FINAL_REC produced by the analysis
    pass. That made FINAL_REC byte-identical across ablation profiles even when
    KAG / RAG / TKF were toggled, because the analysis LLM tended to anchor
    FINAL_REC on the VLM report and ignore the richer signals it had just
    written into prose. The fix:

      - Analysis no longer emits FINAL_REC.
      - Prescription receives kag_context, rag_context, and tkf_block as
        first-class inputs and DERIVES FINAL_REC fresh, with strict grounding
        rules (see build_prescription_prompt). This means flipping use_kag,
        use_rag, or use_tkf actually changes the FINAL_REC fields, not just
        the surrounding prose.

    Returns (prescription_text, combined_text). No cache — always fresh.
    """
    model = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))
    effort = str(llm_cfg.get("reasoning_effort", "high"))

    client = _oai_client()
    prescription_prompt = build_prescription_prompt(
        episode=episode,
        analysis_text=analysis_text,
        tkf_block=tkf_block,
        kag_context=kag_context,
        rag_context=rag_context,
    )
    prescription_text = ""

    try:
        prescription_text = _chat_reasoning(
            client, model,
            [
                {"role": "system", "content":
    "You are a demonstration coach. You DERIVE the FINAL_REC structured block "
    "(corridor, steps, n_demos, demo_variations, rationale) by combining the "
    "root-cause analysis with the KAG corridor vocabulary, RAG retrieved cases, "
    "and TKF demo-coverage block when those are provided. You apply every "
    "STRICT GROUNDING RULE in the user prompt — a FINAL_REC that ignores "
    "available KAG / RAG / TKF inputs is invalid. Corridor names "
    "(left_edge, top_edge, right_edge, bottom_edge, central_mixed) are the "
    "shared vocabulary."},
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


def run_plain_prescription(
    episode: Dict,
    vision_report: str,
    llm_cfg: Dict,
) -> Tuple[str, str]:
    """P1-style prescription: a plain LLM reads ONLY the VLM report and emits a
    FINAL_REC + plain-English explanation. No reasoning chain, no KAG, no RAG,
    no TKF. This makes profile P1 (`use_reasoning=false`, `use_plain_llm=true`)
    actually produce a FINAL_REC the aggregator can cluster, instead of an
    empty `[REASONING DISABLED]` placeholder.

    Returns (prescription_text, combined_text).
    """
    model = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))
    max_tokens = int(llm_cfg.get("max_output_tokens", 16384))

    dyn_str = _format_dynamic_config(episode)
    ascii_grid = episode.get("ascii_grid", "")
    success     = episode.get("success", False)
    total_steps = episode.get("total_steps", 0)
    total_reward = episode.get("total_reward", 0.0)

    user_prompt = (
        "You are a demonstration coach for a DAgger-style imitation learning system.\n"
        "You have ONLY the VLM (vision) report of a failed maze episode — no reasoning,\n"
        "no knowledge graph, no retrieved cases, no demo-coverage check. Produce the\n"
        "FINAL_REC structured block from the VLM report alone, then a short plain-English\n"
        "explanation. This is the P1 baseline.\n\n"
        f"MAZE LAYOUT:\n{ascii_grid}\n\n"
        f"EPISODE DYNAMIC CONFIG:\n  {dyn_str}\n\n"
        f"EPISODE OUTCOME:\n"
        f"  Success      : {success}\n"
        f"  Total steps  : {total_steps}\n"
        f"  Total reward : {float(total_reward):.2f}\n\n"
        f"VISION ANALYSIS FROM VLM:\n{vision_report or '(no VLM report)'}\n\n"
        "RULES:\n"
        "  - corridor MUST be one of: left_edge | top_edge | right_edge | bottom_edge | central_mixed\n"
        "  - steps MUST be a coordinate path using arrow notation, e.g. (0,1)->(0,0)->(1,0)\n"
        "  - n_demos MUST be an integer between 1 and 20 (pick the SMALLEST that addresses the failure)\n"
        "  - demo_variations MUST be a single line\n"
        "  - rationale MUST be a single sentence and reference what the VLM observed\n\n"
        "OUTPUT FORMAT — produce EXACTLY this structure (no preamble before the markers):\n\n"
        "<<<FINAL_REC>>>\n"
        "corridor: <left_edge | top_edge | right_edge | bottom_edge | central_mixed>\n"
        "steps: <(r,c)->(r,c)->...->(r,c)>\n"
        "n_demos: <integer 1-10>\n"
        "demo_variations: <one-line description>\n"
        "rationale: <one sentence grounded in the VLM report>\n"
        "<<<END_FINAL_REC>>>\n\n"
        "PLAIN-ENGLISH EXPLANATION FOR THE HUMAN DEMONSTRATOR:\n"
        "1. WHERE in the maze does the AI get confused? (use the corridor named in FINAL_REC)\n"
        "2. WHAT should a good walkthrough look like? (describe the path FINAL_REC specifies)\n"
        "3. WHAT will the AI learn from seeing that walkthrough?\n"
        "4. HOW MANY walkthroughs are needed and how should they vary? (state n_demos and demo_variations)"
    )

    client = _oai_client()
    prescription_text = ""
    try:
        prescription_text = _chat_plain(
            client, model,
            [
                {"role": "system", "content":
                    "You are a demonstration coach producing a FINAL_REC block from a "
                    "VLM report only. Be specific (corridor, coordinate path, n_demos), "
                    "ground every field in something the VLM actually said, and never "
                    "invent KAG / RAG / TKF references — those signals are not available "
                    "in this profile."},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens,
        )
    except Exception as e:
        traceback.print_exc()
        prescription_text = f"[PRESCRIPTION ERROR: {e}]"

    combined = (
        "=== VLM-ONLY PLAIN LLM PRESCRIPTION (P1 baseline) ===\n"
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
        kag_context=kag_context if use_kag else "",
        rag_context=rag_context if use_rag else "",
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