import json
import os
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Dict, List, Optional


ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


_clip_model = None
_clip_processor = None


def _load_clip(model_id: str):
    global _clip_model, _clip_processor
    if _clip_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = _clip_model.to(device).eval()
    return _clip_model, _clip_processor


def _embed_text(text: str, model_id: str):
    import torch
    import numpy as np
    model, processor = _load_clip(model_id)
    device = next(model.parameters()).device
    words = text.split()
    chunk_size = 55
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, max(len(words), 1), chunk_size)]
    vecs = []
    for chunk in chunks:
        enc = processor(text=[chunk], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            tv = model.get_text_features(**enc)
            tv = torch.nn.functional.normalize(tv, dim=-1)[0].cpu().float().numpy()
        vecs.append(tv)
    import numpy as np
    vec = np.mean(np.stack(vecs), axis=0).astype("float32")
    norm = np.linalg.norm(vec)
    return (vec / (norm + 1e-9)).astype("float32")


def _classify_corridor(trajectory: List) -> str:
    positions = [tuple(p) for p in trajectory]
    n = max(len(positions), 1)
    left_edge  = sum(1 for r, c in positions if c == 0) / n
    top_edge   = sum(1 for r, c in positions if r == 0) / n
    right_edge = sum(1 for r, c in positions if c == 4) / n
    bottom_edge= sum(1 for r, c in positions if r == 4) / n
    candidates = {"left_edge": left_edge, "top_edge": top_edge, "right_edge": right_edge, "bottom_edge": bottom_edge}
    best_name, best_val = max(candidates.items(), key=lambda kv: kv[1])
    return best_name if best_val > 0.4 else "central_mixed"


def _describe_trajectory(trajectory: List, actions: List, start_pos: List, goal_pos: List, fire_positions: List) -> str:
    positions = [tuple(p) for p in trajectory]
    n = len(positions)
    start = tuple(start_pos) if start_pos else (positions[0] if positions else (0, 0))
    end   = positions[-1] if positions else start
    goal  = tuple(goal_pos) if goal_pos else end
    success = (end == goal)

    fire_set = {tuple(p) for p in (fire_positions or [])}
    if fire_set and positions:
        min_fire_dist = min(
            min(abs(r-fr)+abs(c-fc) for fr, fc in fire_set)
            for r, c in positions
        )
    else:
        min_fire_dist = 999

    if min_fire_dist <= 1:
        fire_proximity = "very close to fire (1 step away)"
    elif min_fire_dist <= 2:
        fire_proximity = "moderately close to fire (2 steps away)"
    else:
        fire_proximity = "safely away from fire hazards"

    corridor = _classify_corridor(positions)
    corridor_desc = {
        "left_edge":    "hugging the left edge of the maze",
        "top_edge":     "staying near the top row before descending",
        "right_edge":   "using the right column as the main route",
        "bottom_edge":  "running along the bottom row",
        "central_mixed":"navigating through the central area",
    }.get(corridor, "using a mixed path")

    action_counts = {}
    for a in actions:
        name = ACTION_NAMES.get(int(a), "?")
        action_counts[name] = action_counts.get(name, 0) + 1
    action_summary = ", ".join(f"{v}x {k}" for k, v in sorted(action_counts.items(), key=lambda x: -x[1])) if action_counts else "no actions"

    outcome = "reached the goal successfully" if success else f"ended at position {end} (did not reach goal)"

    return (
        f"A demonstration starting at {start} with goal at {goal} and fires at {sorted(fire_set) if fire_set else 'none'}. "
        f"The run was {corridor_desc}, {fire_proximity}. "
        f"The demo took {max(n-1, 0)} steps with actions: {action_summary}. "
        f"The agent {outcome}. "
        f"Path type: {corridor}."
    )


def _parse_demo_json(filepath: Path, demo_id: int) -> Dict:
    with open(filepath, "r") as f:
        data = json.load(f)
    trajectory     = data.get("trajectory", [])
    actions        = data.get("actions", [])
    start_pos      = data.get("start_pos", trajectory[0] if trajectory else [0, 0])
    goal_pos       = data.get("goal_pos", trajectory[-1] if trajectory else [4, 4])
    fire_positions = data.get("fire_positions", [])
    maze_name      = data.get("maze_name", "unknown")
    timestamp      = data.get("timestamp", 0)
    description    = _describe_trajectory(trajectory, actions, start_pos, goal_pos, fire_positions)
    success        = bool(data.get("success", tuple(trajectory[-1]) == tuple(goal_pos) if trajectory else False))
    return {
        "id":             demo_id,
        "filename":       filepath.name,
        "maze_name":      maze_name,
        "timestamp":      timestamp,
        "n_steps":        max(len(trajectory) - 1, 0),
        "start":          list(start_pos),
        "goal":           list(goal_pos),
        "fire_positions": [list(p) for p in fire_positions],
        "success":        success,
        "corridor":       _classify_corridor(trajectory),
        "outcome":        "success" if success else "failure",
        "description":    description,
    }


def _ingest_demos(demos_dir: Path) -> List[Dict]:
    json_files = sorted(demos_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No demo JSON files in {demos_dir}")
    metadata = []
    for i, fp in enumerate(json_files, 1):
        try:
            metadata.append(_parse_demo_json(fp, i))
        except Exception as e:
            print(f"[TKF] Skipping {fp.name}: {e}")
    if not metadata:
        raise RuntimeError(f"Could not parse any demo files in {demos_dir}")
    return metadata


def _build_index(metadata: List[Dict], index_dir: Path, clip_model: str):
    import faiss
    import numpy as np
    texts = [
        f"{d['description']} | corridor: {d['corridor']} | outcome: {d['outcome']} | n_steps: {d['n_steps']}"
        for d in metadata
    ]
    vecs = [_embed_text(t, clip_model) for t in texts]
    matrix = np.stack(vecs)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "demo_faiss.index"))
    with open(index_dir / "demo_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return index, metadata


def _load_index(index_dir: Path):
    import faiss
    idx_file = index_dir / "demo_faiss.index"
    meta_file = index_dir / "demo_metadata.json"
    if not idx_file.exists() or not meta_file.exists():
        return None, []
    index = faiss.read_index(str(idx_file))
    with open(meta_file, "r") as f:
        meta = json.load(f)
    return index, meta


def _query_index(index, meta: List[Dict], query_text: str, clip_model: str, top_k: int) -> List[Dict]:
    import numpy as np
    qv = _embed_text(query_text, clip_model).reshape(1, -1)
    scores, idxs = index.search(qv, min(top_k, index.ntotal))
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
        if idx < 0:
            continue
        results.append({"rank": rank, "score": float(score), "metadata": meta[int(idx)]})
    return results


def _verdict(results: List[Dict], found_thresh: float, partial_thresh: float) -> str:
    if not results:
        return "NOT_FOUND"
    top = results[0]["score"]
    if top >= found_thresh:
        return "FOUND"
    if top >= partial_thresh:
        return "PARTIAL"
    return "NOT_FOUND"


def _oai_plain(messages: List[Dict], model: str, max_tokens: int = 16384) -> str:
    from openai import OpenAI
    client = OpenAI()
    r = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
        reasoning={"effort": "low"},
    )
    return r.output_text or ""


def _llm_identify_needs(reasoning_text: str, model: str) -> str:
    prompt = textwrap.dedent(f"""
        You are reviewing the failure analysis of a maze navigation agent.
        The agent learned from human demonstrations (imitation learning) and failed.
        Below is the COMPLETE failure reasoning text from the pipeline.

        Your job is to extract, in plain everyday English (no jargon, no coordinates),
        what kind of human demonstration is MISSING from the training data.

        Write 3-5 sentences covering:
        - What behaviour is the agent NOT showing that it should?
        - What kind of situation or path does it need to see demonstrated?
        - Why would seeing that demonstration help it avoid failing next time?

        COMPLETE FAILURE REASONING:
        {reasoning_text}

        OUTPUT ONLY the plain-English needs description. Nothing else.
    """)
    return _oai_plain(
        [
            {"role": "system", "content": "You extract missing demonstration needs in plain English. No technical jargon."},
            {"role": "user",   "content": prompt},
        ],
        model,
    ).strip()


def _build_fallback_retrieval_block(results: List[Dict]) -> str:
    """Format the top-2 best matches for the reasoning LLM when below threshold.

    Each entry includes the similarity score so the LLM can reason about how
    relevant the existing demos actually are.
    """
    if not results:
        return "No demos exist in the training bank at all."
    lines = ["Best available demos (below the confidence threshold — included for context):"]
    for r in results[:2]:
        m = r["metadata"]
        lines.append(
            f"  Demo (similarity={r['score']:.3f} / 100% would be a perfect match): "
            f"{m.get('description', '?')} "
            f"[corridor={m.get('corridor','?')}, outcome={m.get('outcome','?')}, "
            f"n_steps={m.get('n_steps','?')}]"
        )
    return "\n".join(lines)


def _llm_adjust_prescription(
    needs: str,
    retrieval_summary: str,
    verdict: str,
    reasoning: str,
    model: str,
    fallback_block: str = "",
) -> str:
    """Produce final adjusted guidance.

    When verdict is NOT_FOUND but we have fallback demos (below threshold),
    the LLM is shown those demos WITH their sim scores and asked to reason
    about what specifically went off the margin — and whether variety or a
    genuinely new demo type is needed.
    """
    if verdict == "NOT_FOUND" and fallback_block:
        extra_section = textwrap.dedent(f"""

        CLOSEST EXISTING DEMOS (did NOT meet the confidence threshold):
        {fallback_block}

        Given the similarity scores above, reason about:
        1. Are these existing demos roughly covering the right situation, just
           not quite matching (high sim ~0.4-0.6) — suggesting more VARIETY of
           the same type is needed?
        2. Or are they completely unrelated (low sim <0.3) — meaning a brand
           new type of demo must be recorded from scratch?
        3. What specific aspect of the failure situation is NOT covered by the
           closest existing demos?
        """)
    else:
        extra_section = ""

    prompt = textwrap.dedent(f"""
        You are finalising guidance for a human who will record corrective demonstrations.

        NEEDS DESCRIPTION (what the policy is missing):
        {needs}

        TRAINING-DATA RETRIEVAL VERDICT: {verdict}
        RETRIEVAL SUMMARY:
        {retrieval_summary}
        {extra_section}
        ORIGINAL REASONING:
        {reasoning}

        Write 4-6 sentences of plain-English guidance.
        - If verdict is FOUND: explain why the policy likely still failed despite having this kind of demo, and recommend more variety.
        - If verdict is PARTIAL: describe what the existing demos cover and what new variation to add.
        - If verdict is NOT_FOUND with close demos (sim >= 0.3): explain what is MISSING from those close demos and whether the human needs more variety or a genuinely new scenario.
        - If verdict is NOT_FOUND with no close demos (sim < 0.3): confirm this demonstration is entirely missing and describe what the human should record from scratch.
        No coordinates, no action codes, no technical jargon.
    """)
    return _oai_plain(
        [
            {"role": "system", "content": "You write practical demonstration guidance in plain English."},
            {"role": "user",   "content": prompt},
        ],
        model,
    ).strip()


def _run_direct(
    reasoning_text: str,
    index,
    meta,
    clip_model: str,
    found_thresh: float,
    partial_thresh: float,
    top_k: int,
    llm_model: str,
    cache=None,
    episode_id: Optional[int] = None,
) -> Dict:
    needs = _llm_identify_needs(reasoning_text, llm_model)
    results = _query_index(index, meta, needs, clip_model, top_k)
    verdict = _verdict(results, found_thresh, partial_thresh)

    if results:
        lines = [f"Verdict: {verdict}", ""]
        for r in results:
            m = r["metadata"]
            lines.append(
                f"  Match {r['rank']} (similarity={r['score']:.3f}): "
                f"{m.get('description','?')} [corridor={m.get('corridor','?')}, outcome={m.get('outcome','?')}]"
            )
        summary = "\n".join(lines)
    else:
        summary = f"Verdict: {verdict}\nNo matching demonstrations found."

    # --- Update 2: TKF fallback — pass top-2 with sim scores when below threshold ---
    fallback_block = ""
    if verdict == "NOT_FOUND":
        fallback_block = _build_fallback_retrieval_block(results)

    adjusted = _llm_adjust_prescription(
        needs, summary, verdict, reasoning_text, llm_model, fallback_block=fallback_block
    )
    return {
        "needs_description":     needs,
        "verdict":               verdict,
        "top_matches":           results,
        "retrieval_summary":     summary,
        "adjusted_prescription": adjusted,
        "fallback_used":         verdict == "NOT_FOUND" and bool(results),
    }


def _run_crewai(
    reasoning_text: str,
    index,
    meta,
    clip_model: str,
    found_thresh: float,
    partial_thresh: float,
    top_k: int,
    llm_model: str,
    cache=None,
    episode_id: Optional[int] = None,
) -> Dict:
    try:
        from crewai import Agent, Task, Crew, Process
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError as e:
        print(f"[TKF] CrewAI not installed ({e}) — falling back to direct pipeline.")
        return _run_direct(reasoning_text, index, meta, clip_model, found_thresh, partial_thresh, top_k, llm_model, cache, episode_id)

    needs_holder: Dict = {}
    retrieval_holder: Dict = {}

    class NeedsQueryInput(BaseModel):
        reasoning_text: str = Field(description="Full failure reasoning text from the pipeline")

    class FAISSQueryInput(BaseModel):
        needs_description: str = Field(description="Plain-English description of what demo is needed")

    class NeedsIdentifierTool(BaseTool):
        name: str = "identify_demonstration_needs"
        description: str = "Reads failure reasoning text and returns a plain-English description of what demo is missing."
        args_schema: type = NeedsQueryInput
        def _run(self, reasoning_text: str) -> str:
            result = _llm_identify_needs(reasoning_text, llm_model)
            needs_holder["needs"] = result
            return result

    class FAISSRetrieverTool(BaseTool):
        name: str = "query_training_demo_database"
        description: str = "Queries a FAISS demo database. Returns a FOUND/PARTIAL/NOT_FOUND verdict plus top matches."
        args_schema: type = FAISSQueryInput
        def _run(self, needs_description: str) -> str:
            results = _query_index(index, meta, needs_description, clip_model, top_k)
            v = _verdict(results, found_thresh, partial_thresh)
            retrieval_holder["results"] = results
            retrieval_holder["verdict"] = v
            if not results:
                summary = f"Verdict: {v}\nNo matching demonstrations found."
            else:
                lines = [f"Verdict: {v}", ""]
                for r in results:
                    m = r["metadata"]
                    lines.append(
                        f"  Match {r['rank']} (similarity={r['score']:.3f}): "
                        f"{m.get('description','?')} [corridor={m.get('corridor','?')}, outcome={m.get('outcome','?')}]"
                    )
                summary = "\n".join(lines)
            retrieval_holder["summary"] = summary
            return summary

    try:
        needs_tool = NeedsIdentifierTool()
        faiss_tool = FAISSRetrieverTool()

        agent1 = Agent(role="Needs Identifier", goal="Identify what demo is missing.", backstory="An imitation-learning QA analyst.", tools=[needs_tool], verbose=False, allow_delegation=False)
        agent2 = Agent(role="Training DB Retriever", goal="Check if the demo already exists.", backstory="A FAISS-backed retrieval specialist.", tools=[faiss_tool], verbose=False, allow_delegation=False)
        agent3 = Agent(role="Prescription Adjuster", goal="Produce final guidance from verdict + needs.", backstory="A demonstration coach.", verbose=False, allow_delegation=False)

        task1 = Task(description=f"Read this reasoning text and identify the missing demo:\n\n{reasoning_text}\n\nUse the identify_demonstration_needs tool.", agent=agent1, expected_output="A plain-English needs description.")
        task2 = Task(description="Given the needs description from the previous task, query the training database with the query_training_demo_database tool. Return the verdict and the top matches.", agent=agent2, context=[task1], expected_output="A verdict (FOUND/PARTIAL/NOT_FOUND) and top matches.")
        task3 = Task(description="Using the needs description and the retrieval result, write 4-6 sentences of plain-English guidance tailored to the verdict. No coordinates or jargon.", agent=agent3, context=[task1, task2], expected_output="Final plain-English prescription.")

        crew = Crew(agents=[agent1, agent2, agent3], tasks=[task1, task2, task3], process=Process.sequential, verbose=False)
        crew_result = crew.kickoff()

        needs   = needs_holder.get("needs") or _llm_identify_needs(reasoning_text, llm_model)
        verdict = retrieval_holder.get("verdict", "NOT_FOUND")
        matches = retrieval_holder.get("results", [])
        summary = retrieval_holder.get("summary", f"Verdict: {verdict}")
        adjusted = str(crew_result).strip() if crew_result else ""

        # --- Update 2: apply fallback block even for CrewAI path ---
        fallback_block = ""
        if verdict == "NOT_FOUND":
            fallback_block = _build_fallback_retrieval_block(matches)
        if not adjusted:
            adjusted = _llm_adjust_prescription(needs, summary, verdict, reasoning_text, llm_model, fallback_block=fallback_block)

        return {
            "needs_description":     needs,
            "verdict":               verdict,
            "top_matches":           matches,
            "retrieval_summary":     summary,
            "adjusted_prescription": adjusted,
            "fallback_used":         verdict == "NOT_FOUND" and bool(matches),
        }
    except Exception as e:
        traceback.print_exc()
        print(f"[TKF] CrewAI error ({e}) — falling back to direct pipeline.")
        return _run_direct(reasoning_text, index, meta, clip_model, found_thresh, partial_thresh, top_k, llm_model, cache, episode_id)


def run_knowledge_check(
    reasoning_text: str,
    tkf_cfg: Dict,
    llm_cfg: Dict,
    cache=None,
    episode_id: Optional[int] = None,
) -> Dict:
    demo_dir   = Path(tkf_cfg.get("demo_dir", "demos"))
    index_dir  = Path(tkf_cfg.get("index_path", "results/demo_knowledge_base"))
    clip_model = str(tkf_cfg.get("clip_model", "openai/clip-vit-large-patch14"))
    found_th   = float(tkf_cfg.get("sim_threshold_found", 0.45))
    partial_th = float(tkf_cfg.get("sim_threshold_partial", 0.30))
    top_k      = int(tkf_cfg.get("top_k", 5))
    use_crewai = bool(tkf_cfg.get("use_crewai", True))
    llm_model  = str(llm_cfg.get("model", "gpt-5-nano-2025-08-07"))

    index, meta = _load_index(index_dir)
    if index is None:
        try:
            metadata = _ingest_demos(demo_dir)
            index, meta = _build_index(metadata, index_dir, clip_model)
        except Exception as e:
            traceback.print_exc()
            return {
                "needs_description":     "",
                "verdict":               "NOT_FOUND",
                "top_matches":           [],
                "retrieval_summary":     f"TKF skipped — could not build demo index: {e}",
                "adjusted_prescription": "",
                "fallback_used":         False,
            }

    if use_crewai:
        return _run_crewai(reasoning_text, index, meta, clip_model, found_th, partial_th, top_k, llm_model, cache, episode_id)
    return _run_direct(reasoning_text, index, meta, clip_model, found_th, partial_th, top_k, llm_model, cache, episode_id)


def format_tkf_block(tkf_result: Optional[Dict]) -> str:
    if not tkf_result:
        return ""
    fallback_note = ""
    if tkf_result.get("fallback_used"):
        fallback_note = (
            "\n[NOTE: No demo met the confidence threshold. "
            "The 2 closest matches were used as context — see sim scores above.]\n"
        )
    return (
        "=== TRAINING KNOWLEDGE CHECK ===\n"
        f"Verdict: {tkf_result.get('verdict','NOT_FOUND')}\n"
        f"{fallback_note}"
        f"What was found in existing training demos:\n{tkf_result.get('retrieval_summary','(empty)')}\n"
        f"Adjusted guidance: {tkf_result.get('adjusted_prescription','(empty)')}\n"
    )


def save_tkf_result(tkf_result: Dict, episode_dir: Path) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)
    out = episode_dir / "tkf_result.json"
    with open(out, "w") as f:
        json.dump(tkf_result, f, indent=2, default=str)
    return out
