# 06 — LLM prompts (verbatim base + the required confidence addition)

Three LLM roles over **OpenRouter** (`02_...md` #3). OpenRouter natively supports BOTH the stable
**Chat Completions** API (`client.chat.completions.create`) and the BETA **Responses** API
(`client.responses.create`) at `base_url=https://openrouter.ai/api/v1` — **no proxy** (the old
vLLM-proxy requirement is gone). **Recommended: Chat Completions** (stable; the `reasoning` +
`image_url` params are best-documented there); the Responses API also works if you keep the
existing client, but it is BETA (may break, stateless). **Roles / models / effort:** VLM
(perception) = `qwen/qwen3-vl-30b-a3b-instruct` **low**; ANALYSIS + DECISION = `qwen/qwen3-32b`
**high**; aggregator/structuring = `qwen/qwen3-32b` **low** (so the text model runs at both low and
high, per the user's ask). Save the exact prompts **and the resolved model/effort** into **each
run's output folder** (`prompts/`, golden rule 3).

The cleanest base = the robosuite port `diff-dagger-ur5/diffdagger_rs/p4/prompts.py` — reproduced
below. The richer fork PushT variants (rolling 5-frame VLM, 5-section analysis) live in
`diff-dagger/diffdagger/main_pipeline/stage{1,2,3}*.py` and `pool_rl_robo/p4/prompts.py`; port
whichever depth you want but keep the 3-stage contract + effort levels.

## (a) VLM — perception (effort LOW). Shows start / t\* / end frames.
```python
VLM_SYSTEM = ("You are analysing a robot manipulation failure from rendered frames. Be "
              "concrete and spatial; describe what you actually see, not generic advice.")

def vlm_prompt(task_description, roles, t_star):   # roles e.g. ["start","high_loss","end"]
    which = ", ".join(roles)
    return ("You are analysing a robot manipulation failure. The attached frames are, in "
            f"order: {which} (the peak-loss frame is the policy's most-uncertain step, "
            f"t*={t_star}).\nTask: {task_description}\n"
            "Describe what went wrong. Focus on: where in the trajectory the failure occurs, "
            "the robot/gripper configuration at peak loss, and what object or contact state "
            "caused it. ~120 words, concrete and spatial.")
```
> **Note for this run:** `t*` in the prompt should be the `t_flag` (first threshold crossing),
> not the argmax peak (`02_...md` #1). VLM-frame-count is an ablation knob {1,3,5} (`05_...md`).

## (b) ANALYSIS — root cause + phase, KAG-grounded (effort HIGH)
```python
ROOT_CAUSES = ["grasp_failure","approach_failure","placement_error",
               "contact_instability","pose_mismatch","timeout"]
PHASES = ["pre_grasp","grasp","transport","placement","insertion"]

ANALYSIS_SYSTEM = ("You are a robot-manipulation failure analyst. Classify the root cause and "
    "trajectory phase using ONLY the provided categories and the KAG facts. "
    "Output strict JSON, no prose, no code fences.")

def analysis_prompt(task_description, kag_text, vlm_report):
    return (f"TASK: {task_description}\n\n{kag_text}\n\n"
        f"VLM FAILURE DESCRIPTION (the only visual evidence):\n{vlm_report}\n\n"
        "Identify the root cause category and the trajectory phase where the failure occurred.\n"
        f"root_cause ∈ {ROOT_CAUSES}\nphase ∈ {PHASES}\n\n"
        'Output ONLY this JSON:\n{"root_cause":"<one>","phase":"<one>",'
        '"rationale":"<one sentence grounded in the VLM description and a KAG fact>"}')
```
(Add GridWorld-appropriate root causes/phases for the maze — e.g. wrong-direction, hit-wall,
hit-fire, timeout / approach, corridor, junction.)

## (c) DECISION — SELECT vs BRIDGE **+ CONFIDENCE** (effort HIGH)  ← the key change
Base (robosuite `decision_prompt`), **extended to require a confidence score** (`02_...md` #6):
```python
DECISION_SYSTEM = ("You are a demonstration coach for an interactive imitation-learning loop. "
    "Each round you spend ONE expert demonstration to fix the dominant failure mode. You decide "
    "HOW to spend it, grounded in the KAG facts and the per-failure analyses. Reason briefly, "
    "then end with EXACTLY: a decision line AND a confidence line.")

# user prompt = TASK + KAG + the dominant cluster's members (each: ep id, object_xy, progress,
#   peak_loss, root_cause, phase) + the two options:
#   (A) SELECT ep<ID>  — correct one recorded failure on-policy from its divergence point.
#   (B) BRIDGE ep<ID>,ep<ID> — prescribe ONE new middle-ground object placement between 2-3 cited
#       failures; expert demonstrates from there.  (omit (B) for Wipe = SELECT-only)
# OUTPUT contract (REQUIRED — parse both lines):
#   'SELECT ep<ID>'  or  'BRIDGE ep<ID>,ep<ID>'
#   'CONFIDENCE: <integer 0-100>  — <one-line rationale>'
```
Parse the label with the SELECT/BRIDGE regex (`diffdagger_rs/p4/parse.py`) **and** the
`CONFIDENCE:` integer; log both per round (Q3 diagnostic). On unparsable output → geometric
fallback decision, confidence=null, logged.

For the **full BRIDGE pose synthesis** (Eq 10) and the fork's richer prescription JSON
(gripper/object pose, cube layout), see `pool_rl_robo/p4/prompts.py` (`prescription_prompt`,
`cube_prescription_prompt`, `OUTPUT_REQUIREMENT`) and `stage2_prescriptive.py` — but for
robosuite the planner computes the BRIDGE pose from the cited failures, so the LLM only needs to
emit SELECT/BRIDGE + cited ids + confidence.

## Effort / model config (verified against OpenRouter docs)
- **Pinned slugs** (env `VLM_MODEL_NAME` / `LLM_MODEL_NAME`): VLM `qwen/qwen3-vl-30b-a3b-instruct`;
  text `qwen/qwen3-32b`. Client base: `diff-dagger-ur5/diffdagger_rs/p4/llm.py` (OpenAI-compatible;
  change only base_url→OpenRouter, key→`OPENROUTER_API_KEY`, and the two slugs).
- **`qwen3-32b` is binary thinking-on/off — no native effort tiers.** OpenRouter *accepts*
  `reasoning={"effort":"high"|"low"}` but converts it to a token budget
  (`budget=max(min(max_tokens×ratio,128000),1024)`; high≈0.80, medium≈0.50, low≈0.20). Some
  providers collapse low/high to identical "thinking on". **For a reliable low↔high gap, set an
  explicit `max_tokens` and prefer `reasoning={"max_tokens":N}`** (DECISION/analysis high → N≈8192;
  aggregator low → N≈1024 or `reasoning={"enabled":false}` for no-think). Overall cap
  `max_tokens=16384`. Chat Completions call shape:
  ```python
  # analysis / decision (qwen3-32b, HIGH):
  client.chat.completions.create(model="qwen/qwen3-32b",
      messages=[{"role":"system","content":SYS},{"role":"user","content":USER}],
      max_tokens=16384, extra_body={"reasoning": {"max_tokens": 8192}})
  # aggregator / structuring (qwen3-32b, LOW): extra_body={"reasoning": {"enabled": False}}
  # VLM (qwen3-vl-30b-a3b-instruct, no reasoning): user content is a list of parts —
  #   [{"type":"text","text":vlm_prompt(...)},
  #    {"type":"image_url","image_url":{"url":"data:image/png;base64,<...>"}}, ...one per frame]
  ```
  (If you keep the Responses API instead: `client.responses.create(model=..., input=...,
  reasoning={"effort":...}, max_output_tokens=16384)` — same base_url, BETA.)
- LLM-effort + token-cap are also ablation knobs (`05_...md` Tier 3 — tie to the FM-necessity
  story). Log the resolved `{model, reasoning}` per call for the compute table.

## ⚠ Qwen `<think>` parse gotcha (bit us before — cost silent zero-demo rounds)
Qwen3-32B / Qwen3-VL emit `<think>...</think>` inline in the message content. A strict
`json.loads` on the analysis/decision output then fails → empty prescription → **0 demos collected
that round while every API call still returns 200** (heldout SR stays flat, no error). Fixes on
OpenRouter: (1) pass the unified `reasoning` param — `reasoning={"exclude":true}` (or
`include_reasoning:false`) returns reasoning in a **separate field**, not inline in `content`, so
`content` is clean JSON; and/or (2) strip `<think>…</think>` from `content` before `json.loads`
(regex `re.sub(r"<think>.*?</think>", "", txt, flags=re.S)`). Grep symptom if it recurs:
`prescribed=0 kept=0 collected=0` with `content` starting `<think>`.

## Ablation hooks that touch prompts
- `vlm=off` (`05_...md`): skip stage (a); pass only the geometric descriptor + root-cause to
  ANALYSIS/DECISION. `kag=off`: drop `{kag_text}` from (b)+(c). `decision=heuristic`/`fallback_only`:
  bypass (c) entirely (no LLM). Each must be a clean switch.
