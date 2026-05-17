# P4 Architecture — Complete Replication Specification

> Audience: an engineer or LLM with **no access to this repo** who must reimplement
> "P4" from scratch. Everything needed is embedded verbatim below (prompts, JSON
> schemas, configs, code). `path:line` citations are for traceability only — you do
> not need the repo to follow this document.

---

## 1. TL;DR

**P4 is a closed-loop, budget-constrained, DAgger-style active-learning cycle** that
improves an imitation-learning navigation policy by having an LLM *read the policy's
failures and prescribe new training demonstrations*.

The **"P4" LLM profile** is one point on an ablation ladder. It enables exactly:

| Component | P4 |
|---|---|
| VLM (vision analysis of failure frames) | **on** |
| Per-episode Reasoning (root-cause + prescription) | **on** |
| KAG (domain Knowledge-graph injected into prompts) | **on** |
| Cross-episode Reasoning (Phase C aggregator) | **on** |
| Plain-LLM aggregator (structured JSON prescription) | **on** |
| RAG (retrieval of past similar failures) | **off** |
| TKF (Trusted-Knowledge-Filter vs existing demos) | **off** |

One sentence: *roll out the policy → take the failed episodes → for each, run
VLM→KAG→analysis→prescription → aggregate all failures into a structured set of
"go record these maze layouts" demonstrations → BFS-collect those demos → retrain →
re-evaluate → repeat until the held-out success rate hits target or the demo budget
is spent.*

---

## 2. The P4 cycle (big picture)

```
                 ┌─────────────────────────────────────────────────────────┐
                 │  initial policy (trained on N seed demos)                │
                 └───────────────────────────┬─────────────────────────────┘
                                             │
            ┌────────────────────────────────▼────────────────────────────────┐
   ROUND r  │ 1. ROLLOUT the policy on the CORRECTION layout pool (rollout_test)│
            │    -> full_output.json  (per-episode success/frames/config)      │
            └────────────────────────────────┬────────────────────────────────┘
                                             │  failures = episodes with success==False
            ┌────────────────────────────────▼────────────────────────────────┐
            │ 2. P4 ANALYSIS  (pipeline.rerun_pipeline_only on the rollout)     │
            │    Phase A: load saved rollout + KAG context                     │
            │    Phase B (per failed episode, fan-out):                        │
            │        VLM 3 frames → KAG → (RAG off) → ANALYSIS →               │
            │        (TKF off) → PRESCRIPTION(FINAL_REC) → SUMMARY             │
            │    Phase C: cross-episode reasoning → STRUCTURED JSON            │
            │             (failure_clusters + demonstration_prescriptions      │
            │              + recommended_layouts + total_demonstrations_needed)│
            └────────────────────────────────┬────────────────────────────────┘
                                             │ prescription_report.json
            ┌────────────────────────────────▼────────────────────────────────┐
            │ 3. FLATTEN -> recommended_layouts.json                           │
            │ 4. HARD-CAP layouts to remaining demo budget                     │
            │ 5. collect_demos --layouts_from  (A* / BFS expert)               │
            │    -> new demo JSON files                                        │
            │ 6. RETRAIN policy on ALL accumulated demos                       │
            │ 7. ROLLOUT on the static HELDOUT set -> heldout_sr               │
            │ 8. append to learning_curve.json                                 │
            └────────────────────────────────┬────────────────────────────────┘
                                             │ stop if: heldout_sr>=target_sr,
                                             │ budget exhausted, no new demos,
                                             │ or max_rounds reached
                                             ▼
                              next round r+1 (loop back to 1)
```

Why this design: the policy is stochastic and trained from human-style
demonstrations. When it fails, the *correct* intervention is "show it a good
demonstration of the missing behaviour." P4 replaces the human who would otherwise
have to (a) watch every failure, (b) decide what demonstration is missing, and (c)
decide how many demos and on which layouts — with an LLM pipeline that does exactly
that, grounded in a domain knowledge graph (KAG) and a strict structured-output
contract.

---

## 3. Repository map

| File | Role |
|---|---|
| `configs/ablation_profiles/p4_vlm_reasoning_kag_cross_plain_llm.yaml` | The P4 feature-toggle profile (12 lines). |
| `configs/experiment_config.yaml` | Master config (models, maze, rollout, kag/rag/tkf, tracking). Profile is deep-merged on top. |
| `Equivariant_pathway/_analysis_common.py` | `run_profile_analysis()` — loads profile+master, runs the pipeline, writes the prescription report. |
| `Equivariant_pathway/analyze_p4.py` | Canonical CLI entrypoint for one-shot P4 analysis of a rollout. |
| `pipeline/pipeline_runner.py` | The engine: `rerun_pipeline_only()` (Phase A/B/C). |
| `pipeline/vlm_analyser.py` | VLM frame analysis (cached). |
| `pipeline/reasoning.py` | Per-episode analysis + prescription (FINAL_REC) + summary prompts. |
| `pipeline/aggregator.py` | Phase C: cross-episode reasoning + structured JSON prescription. |
| `pipeline/kag_loader.py` | Loads + renders the KAG JSON into the prompt string. |
| `pipeline/knowledge_fetcher.py` | TKF (off in P4). |
| `pipeline/rag_bank.py` | RAG store/retrieve (off in P4). |
| `pipeline/_oai_retry.py` | Shared OpenAI client + token bucket + retry. |
| `knowledge/kag_maze_knowledge.json` | The domain knowledge graph injected by KAG. |
| `Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/p4_budget.py` | The **budget cycle wrapper** (`run_p4_budget`). |
| `Equivariant_pathway/collect_demos.py` | BFS/A* expert that turns prescribed layouts into demo files. |
| `Equivariant_pathway/equivariant_CNN_hybrid/rollout_test.py` | Produces the rollout `full_output.json` input contract. |
| `Equivariant_pathway/equivariant_CNN_hybrid/train.py` | Retrains the policy on accumulated demos. |

---

## 4. Configuration

### 4.1 The P4 profile (verbatim)

`configs/ablation_profiles/p4_vlm_reasoning_kag_cross_plain_llm.yaml`:

```yaml
# Profile 4: VLM + Reasoning + KAG + Plain LLM + Cross-Episode Reasoning + Plain LLM Aggregator
# Adds KAG to P3 — domain knowledge is now injected into per-episode reasoning.
pipeline:
  use_vlm: true
  use_reasoning: true
  use_cross_episode_reasoning: true
  use_kag: true
  use_rag: false
  use_tkf: false
  use_plain_llm: true
  use_aggregator: true
```

### 4.2 Master config (verbatim)

`configs/experiment_config.yaml`:

```yaml
maze:
  name: multimodal
  randomize_start: true
  randomize_goal: true
  randomize_fire: true
  num_fire_tiles: 3

rollout:
  n_episodes: 100
  seed: 42
  checkpoint_path: checkpoints/best_model_ema.pth
  fallback_checkpoint_path: checkpoints/best_model.pth
  render: false

pipeline:
  use_vlm: true
  use_kag: true
  use_rag: true
  use_reasoning: true
  use_cross_episode_reasoning: true
  use_tkf: true
  use_aggregator: true
  use_plain_llm: true

kag:
  document_path: knowledge/kag_maze_knowledge.json

rag:
  bank_path: results/rag_bank
  top_k: 3
  sim_threshold: 0.3
  clip_model: openai/clip-vit-large-patch14

tkf:
  demo_dir: demos
  index_path: results/demo_knowledge_base
  clip_model: openai/clip-vit-large-patch14
  sim_threshold_found:   0.80
  sim_threshold_partial: 0.70
  use_crewai: true

llm:
  model: gpt-5-nano-2025-08-07
  vlm_model: gpt-5-nano-2025-08-07
  reasoning_effort: high
  max_output_tokens: 16384

tracking:
  save_frames: true
  save_per_episode_json: true
  save_prescriptions: true
  save_config_snapshot: true
  output_dir: results/runs
```

### 4.3 Merge semantics

The effective config = `deep_merge(master_config, profile)` then
`deep_merge(result, extra_overrides)`. The recursive merge (override wins on leaves,
recurse on dict-vs-dict) — `Equivariant_pathway/_analysis_common.py:84`:

```python
def _deep_merge(base: Dict, override: Dict) -> Dict:
    import copy
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
```

So for P4 the net `pipeline` block becomes the master toggles **overridden** by the
P4 profile → `use_rag:false, use_tkf:false` (the rest stay true). `extra_overrides`
(set by the budget wrapper, §10) injects per-round prompt addenda, `tkf.demo_dir`,
and optionally `pipeline.phase_b_max_workers`.

---

## 5. Input contract — the rollout `full_output.json`

P4 analysis never runs the environment itself; it **consumes a saved rollout**. The
rollout producer (`rollout_test.py`) writes a directory containing:

- `full_output.json`
- `config_used.yaml` (minimal: `maze`, `rollout`, `tracking.output_dir`)
- `episodes/episode_<id>/frames/{start,high_loss,end}.png` and `trajectory.gif`
- `episodes/episode_<id>/episode_data.json` (per-step log; optional for rerun)

`full_output.json` schema (real example, fire_positions are stored as **strings**):

```json
{
  "metadata": {
    "run_id": "correction_rollout",
    "timestamp": "2026-05-16T15:55:18.603552",
    "n_episodes": 40,
    "n_unique_layouts": 40,
    "seed_base": 1000,
    "n_successes": 26,
    "n_failures": 14,
    "phase_a_only": true,
    "model_type": "equivariant_cnn_hybrid",
    "checkpoint": ".../best_hybrid_policy.pth",
    "layouts_yaml": ".../shared/correction_layouts.yaml"
  },
  "config": { "maze": {...}, "rollout": {...} },
  "phase_a": {
    "all_rollouts": [
      {
        "episode_id": 0,
        "maze_name": "corr_r00_001",
        "seed": 1000,
        "total_steps": 5,
        "total_reward": 9.5,
        "success": true,
        "ascii_grid": "· F F · ·\nA F · · ·\n· · · · G\n· · · · ·\n· · · · ·",
        "dynamic_config": {
          "start_pos": [1, 0],
          "goal_pos": [2, 4],
          "fire_positions": [["0","1"], ["0","2"], ["1","1"]]
        },
        "key_frames": [
          {"role": "start_frame",        "step_idx": 0},
          {"role": "highest_loss_frame", "step_idx": 1},
          {"role": "end_frame",          "step_idx": 5}
        ],
        "frame_paths": {
          "start_frame": "/abs/.../episodes/episode_0/frames/start.png",
          "highest_loss_frame": "/abs/.../high_loss.png",
          "end_frame": "/abs/.../end.png",
          "trajectory_gif": "/abs/.../trajectory.gif"
        }
      }
    ],
    "success_episode_ids": [0, ...],
    "failure_episode_ids": [3, 6, 7, ...]
  }
}
```

**Definition of a failure:** any `phase_a.all_rollouts[]` entry with
`success == false` (equivalently, `episode_id ∈ phase_a.failure_episode_ids`). Only
failures enter Phase B.

The 3 `key_frames` (`start_frame`, `highest_loss_frame`, `end_frame`) + their PNG
paths in `frame_paths` are what the VLM reads.

---

## 6. The pipeline engine — `rerun_pipeline_only`

`pipeline/pipeline_runner.py` re-runs Phase B + Phase C over a saved rollout with
the merged config. Verbatim body (`pipeline_runner.py:603-714`):

```python
    _snapshot_config(config, out_dir)

    full_out_src = saved / "full_output.json"
    if not full_out_src.exists():
        raise FileNotFoundError(full_out_src)
    with open(full_out_src, "r") as f:
        saved_full = json.load(f)

    saved_failure_ids = saved_full.get("phase_a", {}).get("failure_episode_ids", [])

    # Use the rollout dir name as rollout_id so VLM cache is shared correctly
    # across all profiles in the same ablation suite.
    rollout_id = saved.name

    failures = _load_episodes_from_saved_run(saved, saved_failure_ids)
    if not failures:
        print(f"[Rerun] WARNING: no failure episodes could be loaded from {saved}.")

    kag_context = ""
    if config.get("pipeline", {}).get("use_kag", True):
        try:
            kag_context = load_and_format(
                config.get("kag", {}).get("document_path", "knowledge/kag_maze_knowledge.json")
            )
        except Exception as e:
            print(f"[Rerun] KAG load failed: {e}")

    rag_bank = None
    if config.get("pipeline", {}).get("use_rag", True):
        try:
            rag_cfg = config.get("rag", {})
            rag_bank = RAGBank(
                bank_path=rag_cfg.get("bank_path", "results/rag_bank"),
                top_k=int(rag_cfg.get("top_k", 3)),
                sim_threshold=float(rag_cfg.get("sim_threshold", 0.3)),
                clip_model=rag_cfg.get("clip_model", "openai/clip-vit-large-patch14"),
                owner_run_id=out_dir.name,
            )
        except Exception as e:
            print(f"[Rerun] RAG init failed: {e}")

    per_episode = _run_phase_b_parallel(
        failures=failures,
        run_dir=out_dir,
        config=config,
        kag_context=kag_context,
        rag_bank=rag_bank,
        run_id=out_dir.name,
        cache=cache,
        rollout_id=rollout_id,
        log_prefix="Rerun",
    )

    cross_text = ""
    structured: Dict = {}
    if config.get("pipeline", {}).get("use_aggregator", True) and failures:
        maze_ascii = failures[0].get("ascii_grid", "")
        cross_text, _raw, structured = run_aggregator(
            failure_summaries=per_episode,
            failure_ids=[f["episode_id"] for f in failures],
            maze_ascii=maze_ascii,
            llm_cfg=config.get("llm", {}),
            pipeline_flags=config.get("pipeline", {}),
            cache=cache,
            kag_context=kag_context,
        )

    full_output = {
        "metadata": { "run_id": out_dir.name, "timestamp": datetime.now().isoformat(),
            "rerun_of": str(saved), "pipeline_flags": config.get("pipeline", {}),
            "n_episodes": saved_full.get("metadata", {}).get("n_episodes", 0),
            "n_successes": saved_full.get("metadata", {}).get("n_successes", 0),
            "n_failures": saved_full.get("metadata", {}).get("n_failures", len(saved_failure_ids)),
            "seed_base": saved_full.get("metadata", {}).get("seed_base") },
        "config":  config,
        "phase_a": saved_full.get("phase_a", {}),
        "phase_b": {"per_episode": per_episode},
        "phase_c": {"cross_episode_reasoning": cross_text, "parsed_prescription": structured},
    }
    with open(out_dir / "full_output.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    save_final_prescription(cross_text, structured, out_dir, per_episode)
    return full_output
```

Key points for replication:
- **Phase A is reused, not recomputed**: `phase_a` is copied straight from the saved
  rollout. P4 only re-runs B + C.
- `rollout_id = saved.name` scopes the VLM cache (same rollout ⇒ shared VLM outputs).
- For **P4** specifically: `use_kag:true` ⇒ `kag_context` is the rendered KAG string;
  `use_rag:false` ⇒ `rag_bank` stays `None`.

### 6.1 Phase B fan-out

`pipeline_runner.py:14-32` decides parallelism (default = one worker per failure;
set `pipeline.phase_b_max_workers: 1` for fully sequential):

```python
def _phase_b_max_workers(config: Dict, n_failures: int) -> int:
    pf = config.get("pipeline", {}) or {}
    requested = pf.get("phase_b_max_workers", None)
    if requested is None:
        return max(1, n_failures)
    try:
        n = int(requested)
    except (TypeError, ValueError):
        return max(1, n_failures)
    if n <= 0:
        return max(1, n_failures)
    return max(1, min(n, n_failures))
```

`_run_phase_b_parallel` runs `_build_phaseB_for_episode` per failure; with
`max_workers == 1` it is a plain sequential `for` loop, else a
`ThreadPoolExecutor`. Cross-episode (Phase C) always runs **after** all Phase B
completes.

### 6.2 The per-episode chain `_build_phaseB_for_episode` (strict order)

Docstring (verbatim, `pipeline_runner.py:167`):

```
Phase B per-episode flow — strict 3-pass order:
  1. VLM  (cached per (rollout_id, episode_id) — VLM ONLY)
  2. KAG  (no cache)
  3. RAG  (no cache)
  4. run_analysis(VLM + KAG + RAG)  -> analysis_text   [no cache]
  5. TKF(analysis_text)             -> tkf_block        [no cache]
  6. run_prescription(analysis_text, tkf_block) -> prescription_text [no cache]
```

Each pass is gated by the merged flags. For **P4** the realized chain per failed
episode is:

1. **VLM** (`use_vlm:true`) — analyse the 3 key frames → `vlm_report`.
2. **KAG** (`use_kag:true`) — `kag_ctx_for_ep = kag_context` (the rendered graph).
3. **RAG** (`use_rag:false`) — skipped, `rag_ctx = ""`; saves `rag_retrieved.txt = "DISABLED"`.
4. **ANALYSIS** (`use_reasoning:true`) — `run_analysis(episode, vlm_report, kag_ctx, "")` → `analysis_text` (5-section root cause).
5. **TKF** (`use_tkf:false`) — skipped, `tkf_block = ""`; saves `tkf_result.json = {"verdict":"DISABLED"}`.
6. **PRESCRIPTION** (`use_reasoning:true`) — `run_prescription(episode, analysis_text, tkf_block="", kag_ctx, rag_ctx="")` → `prescription_text` containing the `<<<FINAL_REC>>>` block.
7. **SUMMARY** (`use_reasoning && use_plain_llm`) — `summarise_episode(combined_text, ...)` → 3-5 sentence `summary` consumed by Phase C.

Per-episode it returns a dict (consumed by Phase C) with keys:
`episode_id, seed, total_steps, total_reward, success, dynamic_config, summary,
vlm_report, kag_context, rag_context, analysis_text, prescription,
reasoning_combined, tkf_result, adjusted_prescription, frame_paths`.

When `tracking.save_prescriptions` (true by default) each episode dir gets:
`vlm_report.txt, kag_context.txt, rag_retrieved.txt, reasoning.txt,
tkf_result.json, final_prescription.txt`.

---

## 7. The LLM calls — verbatim prompts & schemas (the heart of P4)

All calls go through the **OpenAI Responses API**: `client.responses.create(...)`
with `reasoning={"effort": effort}` and `r.output_text`. Models default to
`gpt-5-nano-2025-08-07`; `reasoning_effort: high`; `max_output_tokens: 16384`.

### 7.1 Shared call wrappers (`pipeline/reasoning.py:33`, `aggregator.py:71`)

```python
def _chat_reasoning(client, model, messages, max_tokens, effort="high"):
    from pipeline._oai_retry import call_with_retry
    r = call_with_retry(client.responses.create, label="reasoning",
        model=model, input=messages, max_output_tokens=max_tokens,
        reasoning={"effort": effort})
    return r.output_text or ""

def _chat_plain(client, model, messages, max_tokens):
    from pipeline._oai_retry import call_with_retry
    r = call_with_retry(client.responses.create, label="reasoning-plain",
        model=model, input=messages, max_output_tokens=max_tokens,
        reasoning={"effort": "low"})
    return r.output_text or ""
```

`reasoning`/`prescription`/`cross-episode` use `_chat_reasoning` (effort `high`);
`summary` and the final JSON aggregator use `_chat_plain` (effort `low`).

### 7.2 VLM frame call (`pipeline/vlm_analyser.py`)

Per key frame (start / highest_loss / end), one image+text call. Prompt builder
(`vlm_analyser.py:33`):

```python
def _frame_prompt(role: str, step_idx: int, episode: Dict) -> str:
    dyn = episode.get("dynamic_config", {})
    return (
        f"You are analysing frame '{role}' (step {step_idx}) of a failed maze navigation episode.\n"
        f"Episode config: start={dyn.get('start_pos','?')}, goal={dyn.get('goal_pos','?')}, "
        f"fires={dyn.get('fire_positions','?')}.\n\n"
        "Provide a structured analysis (~150 words):\n"
        "1. Agent location and adjacent cells (note any walls/fires/goal nearby).\n"
        "2. Goal location and Manhattan distance.\n"
        "3. Fire hazards relative to the agent.\n"
        "4. First corrective demonstration move (exact action + brief justification)."
    )
```

The call (`vlm_analyser.py:62`) — image is base64 PNG inline:

```python
r = call_with_retry(
    client.responses.create,
    label="vlm",
    model=model,                       # llm.vlm_model
    input=[{"role": "user", "content": [
        {"type": "input_text",  "text": prompt},
        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
    ]}],
    max_output_tokens=max_tokens,
)
```

The 3 frame texts are concatenated into a `VISION REPORT` string. VLM output is the
**only cached** stage, keyed `rollout_<rollout_id>_episode_<episode_id>`.

### 7.3 Analysis pass (`pipeline/reasoning.py`)

System message (`reasoning.py:394`):

```
You are a navigation failure analyst for DAgger imitation learning. Walk through
the agent's trajectory step by step. Identify what went wrong and why. Think step
by step.
```

User prompt = `build_analysis_prompt(episode, vision_report, kag_context,
rag_context)` (verbatim, `reasoning.py:126-228`). The fixed lead section:

```
You are a navigation engineer analysing a failed maze episode.

TASK: A vision-conditioned diffusion policy is learning to navigate a 5x5 grid maze
from a RANDOMISED start to a RANDOMISED goal while avoiding RANDOMISED fire hazards.
Training is done via human demonstrations. When the policy fails, a human must record
a CORRECTIVE demonstration to improve it. Your analysis will be CONSUMED downstream by
a prescription LLM that will derive the structured FINAL_REC (corridor, path, n_demos,
demo_variations, rationale) FROM YOUR ANALYSIS — so be specific and grounded.

Grid cells: 0=free, 1=wall, 2=fire (terminates episode), 3=goal (success).
Actions: 0=UP (row-1), 1=DOWN (row+1), 2=LEFT (col-1), 3=RIGHT (col+1).
```

Because P4 has KAG **on**, this block is appended right after the lead:

```
ENVIRONMENT KNOWLEDGE GRAPH (KAG) — the corridor names and failure-mode definitions
in this block are the ONLY allowed corridor vocabulary. When you discuss the chosen
corridor in section 5 you MUST name it using a corridor key from this KAG and quote
the supporting KAG fact verbatim (in single quotes).
<KAG STRING — see §8>
```

Then maze layout / dynamic config / outcome / trajectory / key-state / vision report
sections are appended (RAG section omitted since `use_rag:false`). Finally the
grounding rules + the 5-section instruction (verbatim):

```
ROOT CAUSE ANALYSIS — produce ALL FIVE sections below. Be specific, ground each
claim in the evidence above, and follow these grounding rules:
- Section 3 must name the failure mode using KAG terminology if a KAG block was provided above; otherwise describe it in plain language.
- Section 4: no RAG retrievals available — write 'No retrieved cases.'
- Section 5 must pick the corridor key from the KAG corridor list and quote the KAG corridor fact (in single quotes) that justifies the pick.

1. TRAJECTORY RECONSTRUCTION — where the agent went, step by step.
2. FIRST WRONG DECISION — the earliest step where the agent diverged from a safe path,
   with the (row,col) and the action taken vs the action it should have taken.
3. WHY THE POLICY FAILED — root cause referencing the dynamic config and KAG if provided.
4. RAG CROSS-EPISODE EVIDENCE — per-rank verdicts as described in the rules above.
5. CORRIDOR & PATH PROPOSAL — name the corridor (from KAG if provided), give a concrete
   coordinate path (start -> ... -> goal) avoiding all fires, and a one-sentence
   justification grounded in section 3 and (if applicable) section 4.

DO NOT emit a <<<FINAL_REC>>> block. The prescription stage owns FINAL_REC and will
derive it from your sections 1-5. Producing FINAL_REC here would let you bypass the
grounding rules above and is explicitly forbidden.
```

The grounding-rule lines are chosen by these conditionals (verbatim
`reasoning.py:190-209`) — note KAG-present vs RAG-absent branches:

```python
grounding_rules.append(
    "- Section 3 must name the failure mode using KAG terminology if a KAG block "
    "was provided above; otherwise describe it in plain language."
    if kag_context
    else "- Section 3 describes the failure mode in plain language (no KAG provided).")
grounding_rules.append(
    "- Section 4 must enumerate every retrieved RAG case (by rank) ..."
    if rag_context
    else "- Section 4: no RAG retrievals available — write 'No retrieved cases.'")
grounding_rules.append(
    "- Section 5 must pick the corridor key from the KAG corridor list and quote "
    "the KAG corridor fact (in single quotes) that justifies the pick."
    if kag_context
    else "- Section 5: no KAG provided — describe the corridor in plain geometric terms ...")
```

`run_analysis` then optionally appends a caller addendum (the budget directive, §10)
— `reasoning.py:386`:

```python
addendum = str(llm_cfg.get("prompt_addendum_reasoning", "") or "").strip()
if addendum:
    analysis_prompt = f"{analysis_prompt}\n\n{addendum}"
```

### 7.4 Prescription pass — produces `FINAL_REC` (`pipeline/reasoning.py:231-360`)

System message (`reasoning.py:450`):

```
You are a demonstration coach. You DERIVE the FINAL_REC structured block (corridor,
steps, n_demos, demo_variations, rationale) by combining the root-cause analysis
with the KAG corridor vocabulary, RAG retrieved cases, and TKF demo-coverage block
when those are provided. You apply every STRICT GROUNDING RULE in the user prompt —
a FINAL_REC that ignores available KAG / RAG / TKF inputs is invalid. Corridor names
(left_edge, top_edge, right_edge, bottom_edge, central_mixed) are the shared
vocabulary.
```

`build_prescription_prompt` computes input presence flags:

```python
have_kag = bool(kag_context and kag_context.strip())
have_rag = bool(rag_context and rag_context.strip() and "no matches above threshold" not in rag_context.lower())
have_tkf = bool(tkf_block and tkf_block.strip() and "DISABLED" not in tkf_block and "ERROR" not in tkf_block)
```

For **P4**: `have_kag=True`, `have_rag=False`, `have_tkf=False`. The active
grounding rules become (verbatim — the K1/R1/S1/S2 lines that fire for P4):

```
STRICT GROUNDING RULES — these decide whether your FINAL_REC is valid:

  K1. The `corridor` field in FINAL_REC MUST be the corridor key the analysis selected
      in section 5, and that key MUST appear in the KAG corridor block below. The
      `rationale` field MUST quote (in single quotes) the KAG corridor fact that
      justifies it. If section 5 of the analysis did not pick a KAG corridor, you
      MUST pick the closest matching KAG corridor and cite its fact — do not invent.
  R1. No RAG retrievals available. demo_variations should describe natural
      variation (start offset, fire-side preference) without citing retrieved cases.
  S1. The `steps` field MUST be a coordinate path that:
      (a) starts at the episode start_pos and ends at the goal_pos,
      (b) never enters a fire cell, and
      (c) lies inside the chosen corridor for the bulk of its length.
      Reuse the path the analysis proposed in section 5 unless it violates (a)-(c).
  S2. If you cannot produce a FINAL_REC that satisfies the rules above, output a single
      line '[FINAL_REC UNGROUNDED: <one-sentence reason>]' instead of inventing one.
```

The prompt also contains BUDGET GUIDANCE (n_demos selection) and ends with the
**exact required output format** (`reasoning.py:342-358`):

```
OUTPUT FORMAT — produce EXACTLY this structure (no preamble before the markers):

<<<FINAL_REC>>>
corridor: <left_edge | top_edge | right_edge | bottom_edge | central_mixed>
steps: <(r,c)->(r,c)->...->(r,c)>
n_demos: <integer 1-20 — pick the smallest count that genuinely addresses the failure>
demo_variations: <one-line description (must satisfy R1/T1 if applicable)>
rationale: <one sentence (must satisfy K1 if KAG present)>
<<<END_FINAL_REC>>>

PLAIN-ENGLISH EXPLANATION FOR THE HUMAN DEMONSTRATOR:
1. WHERE in the maze does the AI get confused? (reference the corridor named in FINAL_REC)
2. WHAT should a good walkthrough look like in that area? (describe the same path FINAL_REC specifies)
3. WHAT will the AI learn from seeing that walkthrough?
4. HOW MANY walkthroughs are needed and how should they vary? (state n_demos from FINAL_REC and reuse demo_variations)
```

`run_prescription` returns `(prescription_text, combined_text)` where
`combined_text` = `"=== ROOT CAUSE ANALYSIS ===\n{analysis}\n\n{tkf_block}\n===
DEMONSTRATION PRESCRIPTION ===\n{prescription}\n"`. The aggregator parses the
**last** `<<<FINAL_REC>>>…<<<END_FINAL_REC>>>` block out of this combined text.

### 7.5 Episode summary (`pipeline/reasoning.py:597`)

```python
_chat_plain(client, model, [
  {"role":"system","content":"Extract a concise 3-5 sentence summary. No preamble."},
  {"role":"user","content":
     f"Below is a detailed analysis of a failed maze navigation episode (id={episode_id}).\n"
     f"Extract a concise summary (3-5 sentences) covering:\n"
     f"- What went wrong (root cause)\n- Key configuration (start, goal, fire placement)\n"
     f"- What the correct path should have been\n\nANALYSIS:\n{reasoning_text}\n\n"
     f"Output ONLY the summary, nothing else."},
], max_tokens)
```

### 7.6 Phase C — cross-episode reasoning (`pipeline/aggregator.py:142`)

System message:

```
You are a cross-episode failure analyst for imitation learning in a RANDOMISED 5x5
maze (start, goal, and fire placements all vary per episode). Identify patterns
across failures and prescribe demonstrations. The KAG corridor / failure-mode
vocabulary, when provided, is HIGH PRIORITY and your clusters MUST use those exact
corridor names. IMPORTANT: Only use the episode IDs listed in CONFIRMED FAILURE
EPISODES. Do NOT invent or include any other episode IDs.
```

User prompt (verbatim, `aggregator.py:170-200`); for P4 the KAG block **is**
present (`kag_block` prefix):

```
ENVIRONMENT KNOWLEDGE GRAPH (KAG) — HIGH PRIORITY. The corridor names and
failure-mode taxonomy below are the ONLY allowed vocabulary for your clusters.
When you describe a cluster's region or failure mode, you must ground it in the
KAG facts.
<KAG STRING>

Below are summaries from {n} FAILED episodes in a 5x5 dynamic maze.

REPRESENTATIVE MAZE ASCII (one episode):
{maze_ascii}

CONFIRMED FAILURE EPISODE IDs (ONLY these are failures): [{valid_ids}]
Do NOT include any episode ID outside this list in failure_clusters.

PER-FAILURE SUMMARIES (each contains a parsed FINAL_REC block — those values are
authoritative for that episode and you must not contradict them):
{summary_block}

Answer:
1. FAILURE CLUSTERING: Group failures by corridor (from FINAL_REC; corridor names
   must come from the KAG when KAG is provided) and by failure mode.
2. WHERE DOES THE POLICY STRUGGLE? Refer to regions / corridors using KAG names.
3. WHAT DEMONSTRATIONS ARE NEEDED? For each cluster, name the corridor (from FINAL_REC)
   and aggregate the n_demos values (median rounded up) for that cluster.
4. WHAT LAYOUTS COVER EACH CLUSTER? For each cluster, propose 1-5 concrete maze
   layouts (start_pos, goal_pos, fire_positions) within the 5x5 grid that exercise
   the failure mode. Pick the SMALLEST set that genuinely covers the cluster — if
   one well-chosen layout teaches the corridor, do not pad with three. The layouts
   must lie inside the cluster's corridor where possible, place fires that block the
   failed direct path, and respect:
     - all positions in [0..4] x [0..4]
     - start_pos != goal_pos and Manhattan(start, goal) >= 4
     - fire_positions disjoint from start_pos and goal_pos
     - exactly 3 fire cells per layout
5. HOW MANY AND HOW DIVERSE? Total n_demos should reflect the sum across clusters.
   Each recommended layout should be demonstrated 1-5 times to cover variation;
   pick the count that matches how multi-modal the failure actually is, not a fixed
   number. The orchestrator stops when held-out SR >= 90%, so neither over- nor
   under-prescribing helps — be honest about the gap each layout closes.
```

The per-failure `summary_block` gives, per episode: header (seed/steps/reward),
`dynamic_config`, the **parsed FINAL_REC** (corridor/steps/n_demos/demo_variations/
rationale, marked *authoritative*), the short summary, and the full reasoning
"for context only — must not contradict FINAL_REC".

Optional addendum precedence (`aggregator.py:209`): prefers
`llm_cfg["prompt_addendum_cross_episode"]`, else falls back to
`llm_cfg["prompt_addendum_aggregator"]` (the budget wrapper sets the latter).

### 7.7 Phase C — structured JSON prescription (`pipeline/aggregator.py:276-517`)

A second call converts the cross-episode prose + the per-episode FINAL_REC table
into strict JSON. System message:

```
You are a JSON formatting assistant. Output ONLY valid JSON — no fences, no
preamble. Every field must be derived from the cross-episode reasoning AND the
per-episode FINAL_REC table. Do not invent a different corridor, different n_demos,
or different episode IDs. When a KAG block is provided, corridor names MUST come
from it.
```

User content = context block (KAG + cross-reasoning + per-episode FINAL_REC table +
verbatim per-failure layout coordinates + `aggregated_n` = median-rounded-up of the
FINAL_REC `n_demos`) + optional budget addendum + RULES + **the exact output JSON**
(`aggregator.py:400-437`, verbatim):

```
Output ONLY this JSON (no fences, no preamble):
{
  "n_failure_episodes_analysed": <n>,
  "failure_modes_found": <int>,
  "failure_clusters": [
    {
      "cluster_label": "<short, run-specific>",
      "episodes_in_cluster": [<only IDs from the confirmed failure list>],
      "root_cause": "<fire_collision | looping | wrong_direction | timeout | wall_thrashing>",
      "corridor": "<left_edge | top_edge | right_edge | bottom_edge | central_mixed | mixed>",
      "where_it_fails": "<plain English, specific to this run>",
      "what_it_does_wrong": "<plain English, specific to this run>",
      "source_final_recs": [<corridor strings from FINAL_REC for the episodes in this cluster>]
    }
  ],
  "demonstration_prescriptions": [
    {
      "demo_id": 1,
      "corridor": "<one corridor enum value>",
      "guidance": "<plain English>",
      "target_region": "<plain English>",
      "what_it_teaches": "<plain English>",
      "n_repetitions": <integer derived from FINAL_REC n_demos>,
      "recommended_layouts": [
        {
          "start_pos": [<int 0-4>, <int 0-4>],
          "goal_pos":  [<int 0-4>, <int 0-4>],
          "fire_positions": [[<int>,<int>], [<int>,<int>], [<int>,<int>]],
          "n_repetitions": <int 1-5>,
          "rationale": "<one sentence tying this layout to the cluster failure>"
        }
      ]
    }
  ],
  "total_demonstrations_needed": <int>,
  "overall_summary": "<2-3 sentences specific to these failures>",
  "confidence": <float 0.0-1.0>
}
```

The accompanying RULES (verbatim, `aggregator.py:375-399`) constrain corridor enums,
`source_final_recs`, layout validity (Manhattan ≥ 4, 3 distinct fires not on
start/goal), and `total_demonstrations_needed == sum(n_repetitions)`.

**Post-parse hardening** (do reimplement this — `aggregator.py:446-517`): strip
``` fences; `json.loads`; filter `episodes_in_cluster` to confirmed failure IDs;
warn if a cluster's `corridor` disagrees with its `source_final_recs`; force
`total_demonstrations_needed = sum(n_repetitions)` if they disagree; **validate &
drop** invalid `recommended_layouts` via `_validate_layout`:

```python
def _validate_layout(layout, grid_size=5, n_fires=3):
    # ok if: start/goal/fires in [0,5)^2 ; start != goal ;
    #        Manhattan(start,goal) >= 4 ; exactly 3 distinct fires ;
    #        no fire on start or goal.  Returns (ok, reason).
```

`_aggregate_n_demos` = median (ceil) of per-episode FINAL_REC `n_demos`, floor 1.
`_parse_final_rec` extracts the **last** `<<<FINAL_REC>>>` block, parses
`key: value` lines, coerces `n_demos` to int and lowercases `corridor`.

### 7.8 OpenAI client / pacing (`pipeline/_oai_retry.py`)

Process-shared synchronous client (no async anywhere):

```python
def make_client():
    from openai import OpenAI
    return OpenAI(max_retries=_SDK_MAX_RETRIES, timeout=_SDK_TIMEOUT)
```

Env-tunable pacing (`_oai_retry.py:86`):

```python
_TPM_BUDGET      = env_int  ("OAI_TPM_BUDGET",      200_000)
_TPM_SHARE       = env_float("OAI_TPM_SHARE",       1.0)
_MAX_IN_FLIGHT   = env_int  ("OAI_MAX_IN_FLIGHT",   4)
_SDK_MAX_RETRIES = env_int  ("OAI_SDK_MAX_RETRIES", 8)
_SDK_TIMEOUT     = env_float("OAI_SDK_TIMEOUT",     120.0)
_TARGET_TPM      = max(1_000, int(_TPM_BUDGET * _TPM_SHARE))
```

`call_with_retry(fn, *args, label, **kwargs)`: (1) token-bucket `acquire` (TPM
pacing, refills `capacity/60` per sec), (2) in-flight `Semaphore(_MAX_IN_FLIGHT)`,
(3) call `fn`, (4) on `RateLimitError` parse Retry-After and back off (≤ max_delay,
jittered), also retry on transient `APIConnectionError/APITimeoutError/
APIStatusError`. A minimal reimplementation can be just the SDK call + exponential
backoff; the bucket only matters under heavy parallelism.

---

## 8. KAG — the defining P4 feature

P4 = P3 **+ KAG**. KAG injects a verified domain knowledge graph so the LLM uses a
fixed corridor/failure vocabulary and grounded environment facts instead of
free-form guessing.

### 8.1 The knowledge graph (verbatim, `knowledge/kag_maze_knowledge.json`)

```json
{
  "meta": {
    "schema_version": "1.0",
    "document_type": "knowledge_augmented_generation",
    "domain": "dynamic_maze_navigation",
    "description": "Structured knowledge graph describing the dynamic 5x5 maze navigation environment used in the diffusion-policy imitation learning research project. Consumed by the Reasoning LLM to ground failure analysis in verified environment facts (geometry, dynamics, rewards, failure taxonomy, policy constraints). All coordinates use (row, col) with top-left = (0,0) and bottom-right = (4,4).",
    "coordinate_system": "(row, col), origin top-left, row increases downward, col increases rightward",
    "grid_shape": [5, 5]
  },
  "schema": {
    "node_types": [
      {"type": "GridWorld", "description": "The 5x5 discrete maze world itself."},
      {"type": "Tile", "description": "A single cell in the grid (FREE, WALL, FIRE, or GOAL)."},
      {"type": "Agent", "description": "The navigating entity controlled by the diffusion policy."},
      {"type": "Goal", "description": "The target cell the agent must reach for success."},
      {"type": "FireHazard", "description": "A dangerous cell that terminates the episode on contact."},
      {"type": "Wall", "description": "An impassable cell; stepping into one costs a penalty but the agent does not move."},
      {"type": "Action", "description": "One of four discrete moves available to the agent."},
      {"type": "Corridor", "description": "A named safe navigation route that avoids fire in the multimodal layout."},
      {"type": "FailureMode", "description": "A category of episode termination without success."},
      {"type": "RewardSignal", "description": "A scalar reward associated with an event."},
      {"type": "ObservationComponent", "description": "A sub-element of the agent's observation dict."},
      {"type": "DynamicConfig", "description": "A randomisation knob controlling per-episode start/goal/fire placement."},
      {"type": "PolicyConstraint", "description": "A structural constraint imposed by the MazeDiffusionPolicy architecture."}
    ],
    "relation_types": [
      {"relation": "HAS_TILE", "description": "GridWorld contains a Tile."},
      {"relation": "CAN_ENTER", "description": "Agent can legally move onto a Tile type."},
      {"relation": "TERMINATES_ON", "description": "Episode ends when the agent contacts this Tile."},
      {"relation": "SUCCEEDS_ON", "description": "Episode succeeds when the agent contacts this Tile."},
      {"relation": "PENALISES", "description": "An event emits a negative RewardSignal."},
      {"relation": "NAVIGATES_VIA", "description": "Agent uses a Corridor to reach the Goal."},
      {"relation": "OBSERVES", "description": "Agent observes an ObservationComponent each step."},
      {"relation": "RANDOMISES", "description": "DynamicConfig randomises a Tile or Agent placement per episode."},
      {"relation": "CAUSES_FAILURE", "description": "An event triggers a FailureMode."},
      {"relation": "AVOIDS", "description": "A Corridor avoids a FireHazard region."},
      {"relation": "CONNECTS_TO", "description": "A Corridor connects one grid region to another."},
      {"relation": "ENFORCES", "description": "A PolicyConstraint enforced by the architecture."}
    ]
  },
  "nodes": [
    {"id": "gw_main", "type": "GridWorld", "label": "Dynamic 5x5 Maze",
     "properties": {"grid_size": 5, "num_cells": 25, "top_left": [0,0], "bottom_right": [4,4], "max_steps_per_episode": 200, "layout_name": "multimodal"}},
    {"id": "tile_free", "type": "Tile", "label": "FREE",
     "properties": {"value": 0, "enterable": true, "reward_on_enter": -0.1, "description": "Normal passable cell. Entering costs the step penalty only."}},
    {"id": "tile_wall", "type": "Tile", "label": "WALL",
     "properties": {"value": 1, "enterable": false, "reward_on_hit": -0.5, "description": "Impassable cell. Attempting to move into a wall keeps the agent in place and applies the wall-hit penalty."}},
    {"id": "tile_fire", "type": "Tile", "label": "FIRE",
     "properties": {"value": 2, "enterable": true, "terminates_episode": true, "reward_on_enter": -10.0, "description": "Hazard cell. Entering terminates the episode immediately with large negative reward."}},
    {"id": "tile_goal", "type": "Tile", "label": "GOAL",
     "properties": {"value": 3, "enterable": true, "terminates_episode": true, "reward_on_enter": 10.0, "description": "Target cell. Entering terminates the episode with large positive reward (success)."}},
    {"id": "agent_main", "type": "Agent", "label": "Navigating Agent",
     "properties": {"controller": "MazeDiffusionPolicy", "action_selection": "argmax over one-hot action_dim=4 channels of the first predicted action in the denoised sequence", "render_colour_rgb": [30,100,220]}},
    {"id": "goal_main", "type": "Goal", "label": "Episode Goal",
     "properties": {"randomised_per_episode": true, "render_colour_rgb": [50,200,80]}},
    {"id": "fire_hazard_main", "type": "FireHazard", "label": "Fire Hazards",
     "properties": {"default_count": 3, "configurable": true, "randomised_per_episode": true, "render_colour_rgb": [220,60,20], "danger": "Instant termination on contact. Agent must never step onto a FIRE tile."}},
    {"id": "wall_main", "type": "Wall", "label": "Maze Walls",
     "properties": {"source": "configs/maze_layouts.py layout['grid'] values == 1"}},
    {"id": "act_up",    "type": "Action", "label": "UP",    "properties": {"index": 0, "delta": [-1, 0]}},
    {"id": "act_down",  "type": "Action", "label": "DOWN",  "properties": {"index": 1, "delta": [ 1, 0]}},
    {"id": "act_left",  "type": "Action", "label": "LEFT",  "properties": {"index": 2, "delta": [ 0,-1]}},
    {"id": "act_right", "type": "Action", "label": "RIGHT", "properties": {"index": 3, "delta": [ 0, 1]}},
    {"id": "corridor_left_edge",  "type": "Corridor", "label": "Left-edge corridor",
     "properties": {"path_description": "Stay in column 0 while descending to row 4, then travel along row 4 toward the goal column.", "ideal_for": "Start positions near the top-left quadrant when fire occupies the central rows."}},
    {"id": "corridor_top_edge",   "type": "Corridor", "label": "Top-edge corridor",
     "properties": {"path_description": "Stay in row 0 while traversing to the goal column, then descend along that column.", "ideal_for": "Start positions near the top rows when fire occupies the middle rows."}},
    {"id": "corridor_right_col",  "type": "Corridor", "label": "Right-column corridor",
     "properties": {"path_description": "Reach column 4 along the top, then descend column 4 to the goal row.", "ideal_for": "Start positions near the top-right when the central and bottom-left regions are blocked by fire."}},
    {"id": "fm_fire_collision", "type": "FailureMode", "label": "fire_collision",
     "properties": {"description": "Agent entered a FIRE tile. Most common failure. Indicates the policy failed to generalise fire-avoidance to this (start, goal, fire-placement) configuration."}},
    {"id": "fm_timeout",        "type": "FailureMode", "label": "timeout",
     "properties": {"description": "Episode reached MAX_STEPS (200) without reaching the goal. Indicates looping, hesitation, or oscillation."}},
    {"id": "fm_wall_thrashing", "type": "FailureMode", "label": "wall_thrashing",
     "properties": {"description": "Agent repeatedly hit walls while failing to progress toward the goal. Usually co-occurs with timeout."}},
    {"id": "rew_goal",    "type": "RewardSignal", "label": "GOAL_REWARD",    "properties": {"value":  10.0}},
    {"id": "rew_fire",    "type": "RewardSignal", "label": "FIRE_PENALTY",   "properties": {"value": -10.0}},
    {"id": "rew_wall",    "type": "RewardSignal", "label": "WALL_PENALTY",   "properties": {"value":  -0.5}},
    {"id": "rew_step",    "type": "RewardSignal", "label": "STEP_PENALTY",   "properties": {"value":  -0.1}},
    {"id": "rew_revisit", "type": "RewardSignal", "label": "REVISIT_PENALTY","properties": {"value": -0.05}},
    {"id": "obs_state", "type": "ObservationComponent", "label": "state_vector",
     "properties": {"shape": [14], "dtype": "float32", "layout": "[agent_row/4, agent_col/4, goal_row/4, goal_col/4, 3x3 local neighbourhood values / 3.0 (9 entries), steps_remaining / 200]", "notes": "The goal position is always present in state[2:4], so the policy can condition on any goal placement."}},
    {"id": "obs_image", "type": "ObservationComponent", "label": "image",
     "properties": {"shape": [64,64,3], "dtype": "uint8 (rendered); normalised to float32 [0,1] before encoder", "description": "Bird's-eye RGB render. Blue=agent, Green=goal, Red=fire, Dark=wall, Light=free."}},
    {"id": "cfg_rand_start", "type": "DynamicConfig", "label": "randomize_start", "properties": {"flag": "randomize_start", "default": true, "target": "agent_pos"}},
    {"id": "cfg_rand_goal",  "type": "DynamicConfig", "label": "randomize_goal",  "properties": {"flag": "randomize_goal",  "default": true, "target": "goal_pos"}},
    {"id": "cfg_rand_fire",  "type": "DynamicConfig", "label": "randomize_fire",  "properties": {"flag": "randomize_fire",  "default": true, "target": "fire_positions", "num_fire_tiles_default": 3}},
    {"id": "cfg_min_sg_dist","type": "DynamicConfig", "label": "min_start_goal_manhattan", "properties": {"value": 4, "description": "Enforced minimum Manhattan distance between start and goal to prevent trivially short episodes."}},
    {"id": "pc_stochastic_ddpm", "type": "PolicyConstraint", "label": "Stochastic DDPM Inference", "properties": {"num_diffusion_steps": 200, "description": "Inference draws a fresh Gaussian noise tensor and denoises over 200 steps. Identical observations can yield different action sequences run-to-run — this is intended probabilism, not a bug."}},
    {"id": "pc_cell_encoder", "type": "PolicyConstraint", "label": "CellAlignedEncoder", "properties": {"architecture": "1x1 Conv2d stack followed by AvgPool2d(kernel_size=cell_px, stride=cell_px)", "feature_dim": 128, "grid_size": 5, "cell_px": 16, "description": "Custom grid-cell-aware vision encoder with zero cross-cell bleed. Each grid cell's pixels are pooled independently into one feature before projection. Not a ResNet."}},
    {"id": "pc_horizons", "type": "PolicyConstraint", "label": "Observation/Prediction Horizons", "properties": {"obs_horizon": 4, "pred_horizon": 3, "action_dim": 4, "description": "Policy observes the last 4 (state, image) pairs and predicts the next 3 one-hot actions; only the first action of the sequence is executed each step."}},
    {"id": "pc_unet_shape", "type": "PolicyConstraint", "label": "UNet1D backbone", "properties": {"dim": 64, "dim_mults": [1, 2, 4]}}
  ],
  "edges": [
    {"source": "gw_main", "target": "tile_free", "relation": "HAS_TILE"},
    {"source": "gw_main", "target": "tile_wall", "relation": "HAS_TILE"},
    {"source": "gw_main", "target": "tile_fire", "relation": "HAS_TILE"},
    {"source": "gw_main", "target": "tile_goal", "relation": "HAS_TILE"},
    {"source": "agent_main", "target": "tile_free", "relation": "CAN_ENTER"},
    {"source": "agent_main", "target": "tile_fire", "relation": "CAN_ENTER", "properties": {"warning": "Entering terminates the episode with -10 reward."}},
    {"source": "agent_main", "target": "tile_goal", "relation": "CAN_ENTER"},
    {"source": "agent_main", "target": "tile_wall", "relation": "CAN_ENTER", "properties": {"blocked": true, "effect": "Agent remains in place; wall penalty -0.5 applied."}},
    {"source": "agent_main", "target": "tile_fire", "relation": "TERMINATES_ON", "properties": {"failure_mode": "fire_collision"}},
    {"source": "agent_main", "target": "tile_goal", "relation": "SUCCEEDS_ON"},
    {"source": "tile_goal", "target": "rew_goal", "relation": "PENALISES", "properties": {"sign": "positive"}},
    {"source": "tile_fire", "target": "rew_fire", "relation": "PENALISES"},
    {"source": "tile_wall", "target": "rew_wall", "relation": "PENALISES"},
    {"source": "agent_main","target": "rew_step", "relation": "PENALISES", "properties": {"applied_every_step": true}},
    {"source": "agent_main","target": "rew_revisit", "relation": "PENALISES", "properties": {"applied_when": "agent re-enters a previously visited free cell"}},
    {"source": "agent_main", "target": "corridor_left_edge",  "relation": "NAVIGATES_VIA"},
    {"source": "agent_main", "target": "corridor_top_edge",   "relation": "NAVIGATES_VIA"},
    {"source": "agent_main", "target": "corridor_right_col",  "relation": "NAVIGATES_VIA"},
    {"source": "corridor_left_edge", "target": "fire_hazard_main", "relation": "AVOIDS"},
    {"source": "corridor_top_edge",  "target": "fire_hazard_main", "relation": "AVOIDS"},
    {"source": "corridor_right_col", "target": "fire_hazard_main", "relation": "AVOIDS"},
    {"source": "corridor_left_edge", "target": "goal_main", "relation": "CONNECTS_TO"},
    {"source": "corridor_top_edge",  "target": "goal_main", "relation": "CONNECTS_TO"},
    {"source": "corridor_right_col", "target": "goal_main", "relation": "CONNECTS_TO"},
    {"source": "agent_main", "target": "obs_state", "relation": "OBSERVES"},
    {"source": "agent_main", "target": "obs_image", "relation": "OBSERVES"},
    {"source": "cfg_rand_start", "target": "agent_main",       "relation": "RANDOMISES"},
    {"source": "cfg_rand_goal",  "target": "goal_main",        "relation": "RANDOMISES"},
    {"source": "cfg_rand_fire",  "target": "fire_hazard_main", "relation": "RANDOMISES"},
    {"source": "cfg_min_sg_dist","target": "gw_main",          "relation": "RANDOMISES", "properties": {"constraint": "min_manhattan(start, goal) >= 4"}},
    {"source": "tile_fire", "target": "fm_fire_collision", "relation": "CAUSES_FAILURE"},
    {"source": "gw_main",   "target": "fm_timeout",        "relation": "CAUSES_FAILURE", "properties": {"trigger": "step_count >= 200 and agent has not reached goal"}},
    {"source": "tile_wall", "target": "fm_wall_thrashing", "relation": "CAUSES_FAILURE", "properties": {"trigger": "repeated wall-hit penalties without progress"}},
    {"source": "agent_main", "target": "pc_stochastic_ddpm", "relation": "ENFORCES"},
    {"source": "agent_main", "target": "pc_cell_encoder",    "relation": "ENFORCES"},
    {"source": "agent_main", "target": "pc_horizons",        "relation": "ENFORCES"},
    {"source": "agent_main", "target": "pc_unet_shape",      "relation": "ENFORCES"}
  ],
  "reasoning_implications": {
    "fire_collision_per_episode": "If the agent stepped onto FIRE, the policy has not generalised fire-avoidance to this particular (start, goal, fire-placement) configuration. Prescribe a corrective demonstration that starts from a similar region and shows the safe corridor (left-edge, top-edge, or right-column) appropriate to this fire placement.",
    "timeout_per_episode": "If the agent timed out, it is likely oscillating or hesitating. Prescribe demonstrations that commit to one corridor without backtracking.",
    "wall_thrashing_per_episode": "If the agent hit many walls, it is not using the state vector's local-neighbourhood signal effectively. Prescribe demonstrations that include approach-and-turn behaviour near walls.",
    "dynamic_randomisation_caveat": "Because start, goal, and fire are all randomised per episode, the policy must generalise across configurations. A demo prescription should describe the SITUATION (fire placement pattern relative to start/goal) rather than a fixed coordinate path."
  }
}
```

### 8.2 KAG → prompt string (`pipeline/kag_loader.py:18`)

```python
def format_kag_context(kag: Dict) -> str:
    meta, schema = kag.get("meta", {}), kag.get("schema", {})
    nodes, edges = kag.get("nodes", []), kag.get("edges", [])
    reasoning_implications = kag.get("reasoning_implications", {})
    nl = {n["id"]: n for n in nodes}
    lines = ["=== KAG — ENVIRONMENT KNOWLEDGE GRAPH ===",
             f"Domain: {meta.get('domain','dynamic_maze_navigation')}",
             f"Coordinate system: {meta.get('coordinate_system','(row, col), origin top-left')}",
             f"Grid shape: {meta.get('grid_shape',[5,5])}",
             f"Description: {meta.get('description','')}", ""]
    by_type = {}
    for n in nodes: by_type.setdefault(n["type"], []).append(n)
    for t in [x["type"] for x in schema.get("node_types", [])]:
        if t not in by_type: continue
        lines.append(f"[{t}]")
        for n in by_type[t]:
            prop_str = ", ".join(f"{k}={v}" for k, v in n.get("properties", {}).items())
            lines.append(f"  - {n['label']} (id={n['id']}): {prop_str}")
        lines.append("")
    lines.append("[RELATIONS]")
    for e in edges:
        src = nl.get(e["source"], {}).get("label", e["source"])
        tgt = nl.get(e["target"], {}).get("label", e["target"])
        extra = f"  {e.get('properties',{})}" if e.get("properties") else ""
        lines.append(f"  {src} --[{e.get('relation','RELATED_TO')}]--> {tgt}{extra}")
    lines.append("")
    if reasoning_implications:
        lines.append("[REASONING IMPLICATIONS]")
        for k, v in reasoning_implications.items():
            lines.append(f"  * {k}: {v}")
        lines.append("")
    return "\n".join(lines)
# load_and_format(path) = format_kag_context(json.load(path))
```

This single string is injected into (a) every per-episode **analysis** prompt, (b)
every per-episode **prescription** prompt, (c) the Phase-C **cross-episode** prompt,
and (d) the Phase-C **structured JSON** prompt — always as "HIGH PRIORITY, the ONLY
allowed corridor vocabulary."

---

## 9. Output contract

### 9.1 `<rollout>/p4_analysis/full_output.json`
The engine's return value (§6) serialized: `metadata`, `config`, `phase_a` (copied),
`phase_b.per_episode[]`, `phase_c.{cross_episode_reasoning, parsed_prescription}`.

### 9.2 `<rollout>/p4_analysis/<label>_prescription_report.json`
Written by `run_profile_analysis` (`_analysis_common.py:139`). Real (truncated):

```json
{
  "profile": "p4_vlm_reasoning_kag_cross_plain_llm.yaml",
  "label": "p4_only_budget_hybrid",
  "pathway": "equivariant",
  "rollout_dir": ".../round_001/correction_rollout",
  "run_dir": ".../round_001/p4_analysis",
  "n_failures": 14,
  "test_layout_failures": [
    {"episode_id": 3, "maze_name": "corr_r00_004",
     "start_pos": [3,3], "goal_pos": [0,4],
     "fire_positions": [[0,1],[2,2],[2,3]], "success": false, "total_steps": 60}
  ],
  "prescription": {
    "total_demonstrations_needed": 3,
    "demonstration_prescriptions": [
      {"cluster": null, "n_demos_needed": 1,
       "recommended_layouts": [
         {"start_pos": [3,1], "goal_pos": [0,0],
          "fire_positions": [[0,2],[2,3],[4,4]], "n_repetitions": 1,
          "rationale": "..."}]}
    ],
    "failure_clusters": [ ... ]
  }
}
```

`demonstration_prescriptions` here is `_summarise_prescription(structured)`
(`_analysis_common.py:40`) — pulls `cluster/cluster_label`, `n_demos_needed`,
`recommended_layouts`, `rationale` from the Phase-C JSON.

### 9.3 `recommended_layouts.json` (flattened, budget wrapper writes this)
```json
{
  "layouts": [
    {"parent_demo_id": null, "layout_index": 0, "repetition": 1, "n_repetitions": 1,
     "start_pos": [3,3], "goal_pos": [0,0],
     "fire_positions": [[0,2],[1,2],[2,2]],
     "rationale": "This layout uses the left-edge corridor ..."}
  ],
  "n_layouts": 1,
  "n_layouts_proposed": 1,
  "n_layouts_capped_by_budget": 0
}
```

### 9.4 Per-episode Phase B artifacts
`<rollout>/p4_analysis/episodes/episode_<id>/`:
`vlm_report.txt`, `kag_context.txt`, `rag_retrieved.txt` (`"DISABLED"` for P4),
`reasoning.txt` (analysis+prescription combined), `tkf_result.json`
(`{"verdict":"DISABLED"}` for P4), `final_prescription.txt` (the FINAL_REC block +
plain-English explanation).

---

## 10. The budget cycle wrapper (`p4_budget.py`)

This is what turns "one-shot P4 analysis" into the closed loop. Verbatim constants
and prompt addenda (`p4_budget.py:35-77`):

```python
PROFILE_YAML = "p4_vlm_reasoning_kag_cross_plain_llm.yaml"
LABEL = "p4_only_budget_hybrid"
OUT_SUBDIR = "p4_analysis"

REASONING_ADDENDUM_BASE = (
    "SAMPLE-EFFICIENCY DIRECTIVE — pick the smallest n_demos that closes the failure mode,\n"
    "but never zero when a failure is present.\n"
    "HARD FLOOR: this episode IS a failure. n_demos for this episode must be >= 1."
)
AGGREGATOR_ADDENDUM_BASE = (
    "HOLISTIC SAMPLE-EFFICIENCY DIRECTIVE — minimise total layouts but NEVER zero, and\n"
    "scale recommendations with the diversity of the failure pool.\n"
    "1. Cluster the failures into the smallest set of distinct modes.\n"
    "2. Recommend the SMALLEST set of layouts per cluster that fixes the missing behaviour.\n"
    "3. HARD FLOOR: if n_failure_episodes >= 1, the response MUST contain at least one\n"
    "   cluster, one demonstration_prescription, one recommended_layout, and\n"
    "   total_demonstrations_needed >= 1.\n"
)

def _budget_addendum(budget_total: int, already_used: int) -> str:
    remaining = max(0, budget_total - already_used)
    return (
        "BUDGET CONSTRAINT (hard) — this comparison runs on a finite demo\n"
        f"budget of {budget_total} extra demonstrations on top of the initial 20.\n"
        f"  * demonstrations already prescribed in earlier rounds: {already_used}\n"
        f"  * demonstrations REMAINING for this round AND every future round: {remaining}\n"
        "Recommend AT MOST that many layouts here. The orchestrator will hard-\n"
        "cap your output to that count, so any layouts beyond it are wasted.\n"
        "If you can defer some learnings to a later round (after the model is\n"
        "retrained on what you prescribe now), prescribe FEWER now and revisit.\n"
        "If you choose to spend your entire remaining budget here, do so on the\n"
        "layouts that will close the most failure modes per demo."
    )
```

`_run_analysis` builds `extra_overrides` (this is how the budget directive reaches
the prompts of §7.3/§7.6) and calls `run_profile_analysis`:

```python
extra = {
  "llm": {
    "prompt_addendum_reasoning":  REASONING_ADDENDUM_BASE  + "\n\n" + _budget_addendum(budget_total, already_used),
    "prompt_addendum_aggregator": AGGREGATOR_ADDENDUM_BASE + "\n\n" + _budget_addendum(budget_total, already_used),
  },
  "tkf": {"demo_dir": str(demo_dir)},
}
# optional: extra["pipeline"] = {"phase_b_max_workers": int(os.environ["P4_PHASE_B_MAX_WORKERS"])}  (sequential mode)
return run_profile_analysis(profile_yaml_name=PROFILE_YAML, rollout_dir=..., out_subdir_name=OUT_SUBDIR,
                            out_dir_override=..., master_config_path=None, label=LABEL, extra_overrides=extra)
```

`_flatten_recommended_layouts` turns the report's
`prescription.demonstration_prescriptions[].recommended_layouts[]` into the flat
`recommended_layouts.json`. `_cap_recommendations(rec_path, remaining)` keeps only
`layouts[:remaining]` and records `n_layouts_proposed` / `n_layouts_capped_by_budget`.
`_collect_prescribed` runs `python -m Equivariant_pathway.collect_demos
--layouts_from <rec_path> --demo_dir <round_dir> --seed <s>`.

The loop `run_p4_budget(...)` (verbatim core, `p4_budget.py:286-373`):

```python
extras_saved = 0
for rnd in range(1, max_rounds + 1):
    remaining = budget - extras_saved
    if remaining <= 0: break                                   # budget gate

    corr_dir = round_dir / "correction_rollout"
    _rollout(ckpt_dir, correction_yaml, corr_dir, seed=seed+rnd*1000, max_steps=...)
    corr_metrics = _read_sr(corr_dir)                          # success rate on correction pool

    _run_analysis(corr_dir, round_dir/"p4_analysis", demo_dir,
                  budget_total=budget, already_used=extras_saved)   # Phase A/B/C

    rec_path = _flatten_recommended_layouts(round_dir/"p4_analysis")
    cap_audit, n_new_demos = {"proposed":0,"kept":0,"capped":0}, 0
    if rec_path is not None:
        cap_audit = _cap_recommendations(rec_path, remaining=remaining)
        if cap_audit["kept"] > 0:
            n_new_demos = _collect_prescribed(rec_path, demo_dir/f"round_{rnd:03d}", seed=seed+1000+rnd)
            if n_new_demos > (budget - extras_saved):          # clip n_repetitions overspill
                ... unlink extras ...
    extras_saved += n_new_demos

    if n_new_demos > 0:
        _retrain(demo_dir, ckpt_dir, seed=seed, epochs=round_epochs, train_from_scratch=...)

    post = _read_sr_after(_rollout(ckpt_dir, heldout_yaml, round_dir/"heldout_eval", seed=seed+rnd, ...))
    history.append({... "heldout_sr": post["success_rate"], "correction_sr": corr_metrics["success_rate"],
                     "extra_demos": extras_saved, "n_prescribed_layouts": cap_audit["proposed"],
                     "n_capped_by_budget": cap_audit["capped"], "n_new_demos": n_new_demos ...})
    _persist_curve(...)                                         # learning_curve.json

    if post["success_rate"] >= target_sr: return {"stopped_reason": "target_hit", ...}
    if n_new_demos == 0:                  return {"stopped_reason": "no_new_demos", ...}
    if extras_saved >= budget:            return {"stopped_reason": "budget_exhausted", ...}
return {"stopped_reason": "max_rounds", ...}
```

Helpers (verbatim behaviour): `_rollout` = `python -m
Equivariant_pathway.equivariant_CNN_hybrid.rollout_test --checkpoint
<ckpt>/best_hybrid_policy.pth --layouts <yaml> --out_dir <dir> --seed s
--max_steps m`; `_read_sr` reads `full_output.json.metadata` →
`n_successes/n_episodes`; `_retrain` = `python -m
Equivariant_pathway.equivariant_CNN_hybrid.train --demo_dir <all demos>
--checkpoint_dir <ckpt> --epochs e --seed s [--resume]`.

Round 0 (before the loop): copy seed demos+checkpoint, roll out the heldout set
once, record baseline; early-return `initial>=target` if already at target.

`learning_curve.json` schema (`_persist_curve`, real example):

```json
{
  "method": "p4_only_budget_hybrid", "run_index": 0, "budget": 10, "target_sr": 0.9,
  "demo_dir": "...", "checkpoint_dir": "...",
  "correction_yaml": "...", "heldout_yaml": "...",
  "history": [
    {"round": 0, "cum_demos": 20, "extra_demos": 0, "budget_remaining": 10,
     "heldout_sr": 0.51, "heldout_n_successes": 102, "heldout_n_episodes": 200,
     "correction_sr": null, "n_prescribed_layouts": 0, "n_capped_by_budget": 0, "n_new_demos": 0},
    {"round": 1, "cum_demos": 24, "extra_demos": 3, "budget_remaining": 7,
     "heldout_sr": 0.735, "heldout_n_successes": 147, "heldout_n_episodes": 200,
     "correction_sr": 0.65, "n_prescribed_layouts": 3, "n_capped_by_budget": 0, "n_new_demos": 3}
  ]
}
```

Stop reasons: `initial>=target`, `target_hit`, `no_new_demos`,
`budget_exhausted`, `max_rounds`.

---

## 11. `collect_demos` — turning prescribed layouts into demos

`Equivariant_pathway/collect_demos.py` consumes `recommended_layouts.json` via
`--layouts_from` and uses an **A\* expert** to record an optimal trajectory per
layout. `collect_from_recommended` parses `spec["layouts"]` (the flat list);
`_record_one` writes one demo JSON per layout. Demo schema (verbatim payload,
`collect_demos.py:109`):

```python
payload = {
  "maze_name": "multimodal",
  "timestamp": int(time.time()*1000) + int(rng.integers(0,1_000_000)),
  "layout_id": layout_id,
  "source": "equivariant_pathway_bfs",
  "start_pos": [int,int],
  "goal_pos":  [int,int],
  "fire_positions": [[int,int], ...],
  "trajectory": [[r,c], ...],                  # expert path
  "observations": [ [14 floats], ... ],        # per step state vector
  "images": [ [[[r,g,b],...]], ... ],          # per step rendered image
  "actions": [int, ...],                       # 0..3
  "rewards": [float, ...],
  "total_reward": float,
  "success": bool                              # rewards[-1] > 5.0
}
```

CLI: `--layouts` (YAML mode, n_repetitions per layout, `--num_demos` cap) **or**
`--layouts_from` (active-loop JSON mode used by P4) — exactly one required.

---

## 12. Minimal end-to-end reproduction

**One-shot P4 analysis of an existing rollout** (canonical entrypoint
`Equivariant_pathway/analyze_p4.py`):

```bash
export OPENAI_API_KEY=sk-...
python Equivariant_pathway/analyze_p4.py \
  --rollout_dir /path/to/<rollout_with_full_output.json> \
  [--out_dir /path/to/out]            # default: <rollout_dir>/p4_analysis
  [--master_config configs/experiment_config.yaml]
# -> writes <out_dir>/p4_prescription_report.json + full_output.json + per-episode artifacts
```

In-process equivalent:

```python
from Equivariant_pathway._analysis_common import run_profile_analysis
run_profile_analysis(
    profile_yaml_name="p4_vlm_reasoning_kag_cross_plain_llm.yaml",
    rollout_dir="/path/to/rollout",
    out_subdir_name="p4_analysis",
    label="p4",
    extra_overrides=None,   # budget wrapper passes prompt addenda + tkf.demo_dir here
)
```

**Full budget cycle:** `run_p4_budget(run_index, p4_root, shared_demo_dir,
shared_ckpt_dir, correction_yaml, heldout_yaml, budget, target_sr, round_epochs,
max_rounds, max_steps, seed, train_from_scratch)`.

Relevant env vars: `OPENAI_API_KEY` (required); optional pacing
`OAI_TPM_BUDGET / OAI_TPM_SHARE / OAI_MAX_IN_FLIGHT / OAI_SDK_MAX_RETRIES /
OAI_SDK_TIMEOUT`; `P4_PHASE_B_MAX_WORKERS=1` forces fully sequential OpenAI calls.

---

## 13. P4 vs the rest (ablation ladder)

| profile | vlm | reasoning | cross_episode | kag | rag | tkf | aggregator |
|---|---|---|---|---|---|---|---|
| baseline_naive | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| p1_vlm_plain_llm | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| p2_vlm_reasoning_plain_llm | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| p3_vlm_reasoning_cross_plain_llm | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| **p4_vlm_reasoning_kag_cross_plain_llm** | ✓ | ✓ | ✓ | **✓** | ✗ | ✗ | ✓ |
| p5_vlm_reasoning_kag_rag_cross_plain_llm | ✓ | ✓ | ✓ | ✓ | **✓** | ✗ | ✓ |
| p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** | ✓ |

What each toggle flips, mechanically:
- **use_vlm**: pass 1 runs the 3-frame VLM and feeds `VISION ANALYSIS FROM VLM` into
  analysis; off ⇒ `[VLM DISABLED]`. If reasoning is off but vlm+plain_llm on (P1),
  `run_plain_prescription` produces FINAL_REC from the VLM report alone.
- **use_reasoning**: enables the analysis + prescription passes (the FINAL_REC
  derivation). Off ⇒ no per-episode FINAL_REC chain.
- **use_kag (P4's addition over P3)**: injects the KAG string and switches the
  analysis grounding rule to "name the corridor using a KAG key + quote the KAG
  fact" and prescription rule **K1**; also KAG-prefixes both Phase-C prompts.
- **use_rag**: builds a `RAGBank`, retrieves similar past failures, injects a RAG
  section + activates analysis section-4 enumeration and prescription rules R1/R2.
  Off in P4 ⇒ `rag_bank=None`, RAG section omitted, R1 = "no retrievals".
- **use_tkf**: runs the demo-coverage check between analysis and prescription,
  injecting a TKF block + prescription rule T1 (NOT_FOUND→≥3, PARTIAL→≥2 demos).
  Off in P4 ⇒ `tkf_block=""`, T1 absent, `tkf_result.json={"verdict":"DISABLED"}`.
- **use_cross_episode_reasoning / use_aggregator**: Phase C. Off ⇒ no clustering /
  no structured prescription (the cycle would have nothing to collect).

---

## 14. "Replicate P4 from scratch" checklist

1. **Environment**: a 5×5 grid world with the reward/termination contract exactly as
   the KAG states — FREE −0.1, WALL −0.5 (no move), FIRE −10 (terminate), GOAL +10
   (terminate), revisit −0.05; randomized start/goal/fire (3 fires), enforce
   `Manhattan(start,goal) ≥ 4`; obs = 14-d state vector + 64×64×3 image.
2. **Policy + rollout producer**: any imitation policy; a `rollout_test` that, given
   a layouts YAML + checkpoint, writes the §5 `full_output.json` (with per-episode
   `success`, `dynamic_config`, `ascii_grid`, 3 `key_frames` + PNG `frame_paths`)
   and `config_used.yaml`.
3. **KAG file**: author the §8.1 JSON for your domain; implement `format_kag_context`
   (§8.2) to render it.
4. **Config system**: master YAML (§4.2) + the 12-line P4 profile (§4.1) + recursive
   `_deep_merge` (§4.3) + `extra_overrides` layering.
5. **Engine** (§6): load saved rollout, take failures only, build `kag_context`
   (KAG on), `rag_bank=None` (RAG off); Phase B per failure = VLM→KAG→(skip RAG)→
   analysis→(skip TKF)→prescription→summary using the **verbatim prompts of §7**;
   then Phase C = cross-episode reasoning + the **exact structured JSON of §7.7**
   with the post-parse validation/repair.
6. **Structured-output parser**: strip ``` fences, `json.loads`, filter cluster
   episode IDs to confirmed failures, force `total_demonstrations_needed =
   Σ n_repetitions`, drop layouts failing `_validate_layout`
   (in-grid, start≠goal, Manhattan ≥ 4, exactly 3 distinct fires off start/goal).
7. **Budget loop** (§10): flatten prescription → cap to remaining budget → A\*-
   collect demos for the kept layouts → retrain on all accumulated demos → roll out
   the static heldout set → append `learning_curve.json` → stop on
   target/budget/no-new-demos/max-rounds. Refresh `_budget_addendum` every round.
8. **Gotchas**:
   - Only `success==false` episodes enter Phase B; Phase C runs once after all
     Phase B; the **last** `<<<FINAL_REC>>>` block in combined text is authoritative.
   - The Phase-C JSON call uses low reasoning effort and must return raw JSON
     (no fences) — keep the validator/repair or it will break collection.
   - VLM is the only cached stage (`rollout_<id>_episode_<id>`); analysis/
     prescription/summary/aggregator/TKF are never cached (so toggles change output).
   - Heldout set is fixed across all rounds; only the correction pool varies.
   - Sequential vs parallel is purely `pipeline.phase_b_max_workers`
     (`1` = sequential; unset = one worker per failure).
   - Reasoning calls use the **Responses API** with `reasoning={"effort": ...}`;
     model defaults `gpt-5-nano-2025-08-07`, `max_output_tokens: 16384`.
   - KAG is what makes P4 ≠ P3: corridor names must come from the KAG and be quoted
     in the prescription rationale (rule K1).

---

*Source of truth: this document was generated by reading the repository at
`/vast/s226137394/DmN/DmNfull`. Every prompt, schema, config and code block above is
quoted verbatim from the cited files.*
