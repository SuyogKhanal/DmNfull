# 02 — Design changes for THIS clean run (explicit instructions)

These are deliberate changes from the previous (scattered) implementation. Each has a reason,
several grounded in a code-level study done last session (see `HPC_ANSWERS` findings summarized
inline). Implement all of them.

## 1. Cluster + take over at the FIRST threshold-crossing, NOT the peak (argmax)
- **Old:** the failure's takeover step was `t* = argmax(per-step uncertainty)` — the *peak*,
  which is *late* in a failing episode.
- **Why change:** for SELECT / on-policy correction the expert takes over at `t*` and the env
  terminates at its own horizon, so the expert's real budget is `horizon − t*`. With a late
  peak (measured: Door median t\*/T ≈ 0.91 → ~27 steps left; Lift ≈192/200 → ~8 steps) the
  expert *cannot* finish, and the demo is (wrongly) marked infeasible. Wipe fails *early*
  (t\*≈0.02 → almost full budget) and is almost never infeasible — the infeasible rate tracks
  `(1 − t*/T)`. So the peak is the wrong takeover point.
- **New:** define `t_flag` = the **first step where the per-step uncertainty crosses the OOD
  threshold** (with a small K-patience, exactly like Diff-DAgger's query trigger). Build the
  descriptor / do clustering at `t_flag`, and hand the expert over at `t_flag`. Earlier, less-
  corrupted state + far more budget → far fewer spurious infeasibles.

## 2. Give Lift / Wipe / Door (state AND image) enough steps
- **Why:** same finding — the expert was starved of steps after takeover.
- **New:** (a) raise the env horizons for Lift/Wipe/Door; (b) **decouple the expert-takeover
  budget from `horizon − t_flag`** — give the scripted expert a fixed, generous budget after
  takeover (e.g. the full task horizon), independent of where the flag fell (this is what PushT
  already does: a separate 120-step expert budget). Confirm current horizons in
  `03_TASKS_AND_ENVS.md` and increase them.

## 3. LLM backend = OpenRouter API (no local vLLM, no H100/H200)
- Key: `OPENROUTER_API_KEY` in `.env`. OpenAI-compatible endpoint
  `base_url=https://openrouter.ai/api/v1`. **Pinned models (verified live on OpenRouter):**
  - **VLM (perception, stage a):** `qwen/qwen3-vl-30b-a3b-instruct` — accepts `image_url` content
    parts, 262K ctx, non-thinking *Instruct* variant. Effort = low (no thinking).
  - **Text (stage b analysis + stage c decision):** `qwen/qwen3-32b` at **effort HIGH**.
  - **Text (aggregator / structured-JSON parsing):** `qwen/qwen3-32b` at **effort LOW**.
    → the one text model runs at **both** low and high (the user's "qwen3-32b with reasoning
    effort low and high"), mirroring the old ReasoningClient(high)/PlainClient(low) split.
- **API + reasoning gotcha (verified against OpenRouter docs):** OpenRouter natively supports BOTH
  `client.chat.completions.create` (stable) and `client.responses.create` (Responses API, BETA) —
  **no proxy needed** (the old local-vLLM-proxy requirement is dead). **BUT** `qwen3-32b` is a
  *binary* thinking-on/off model with **no native low/med/high tiers**; OpenRouter accepts
  `reasoning={"effort":"high"|"low"}` and silently converts it to a reasoning-token budget
  (`budget = max(min(max_tokens×ratio, 128000), 1024)`; high≈0.80, low≈0.20). On some providers
  low and high collapse to the same "thinking on". **To make low-vs-high actually differ, pass an
  explicit `max_tokens` AND prefer `reasoning={"max_tokens":N}` directly** (decision→large N,
  aggregator→small N or `{"enabled":false}` for no-think). Exact call shapes in `06_PROMPTS.md`.
- **Consequence:** no 3-GPU VLM/LLM servers, no `--constraint=gpu-h100|gpu-h200`, and **no local
  model weights at all** — the LLM is a pure API call. Each job needs ~1 GPU only (diffusion-policy
  train + rollout); schedules on any partition → faster. **This is also what makes the 2nd HPC
  trivially portable** (it needs no model weights) — see `08_...md` + `SETUP_HPC2.md`.

## 4. Geometric-state clustering for IMAGE tasks too — NO R3M
- The **policy** is image-based (visual obs), but the **descriptor / clustering / LLM config
  recommendations use the privileged GEOMETRIC state** (object pose, gripper, progress, …),
  identical to the state-based runs. Do **not** use R3M visual embeddings for clustering.
- Rationale: geometry gives clean, well-separated clusters and lets the LLM prescribe concrete
  configs; R3M added complexity for no clustering benefit.
- **Heads-up (supervisor Tier 6, privileged-info flip):** this means image runs consume object
  poses for clustering + memory centroids. That is a deliberate, defensible choice — document
  it, and keep the "image descriptors without privileged poses" ablation available.

## 5. Infeasibility check loop (part of the method; matters most for PushT)
- If the scripted expert fails to solve a prescribed demo, that attempt costs **no budget**
  (budget = *successful* demos only), and the LLM **re-prescribes** (a fresh SELECT/BRIDGE),
  bounded by an attempts-before-fallback cap, then a deterministic fallback (nearest untried
  failure). This is stage 7 of the architecture (the infeasibility feedback loop).
- **PushT specifically:** the automated expert is a clockwise-only PPO with a capability gap
  (some prescribed T-rotations need CCW). The infeasibility loop simply **re-prescribes** those
  until a feasible demo is obtained — so there is no separate "anti-clockwise export" step. Keep
  the loop; budget accounting is unchanged (only successful demos count).

## 6. Confidence score every round (NEW prompt requirement)
- The DECISION prompt must **explicitly ask the LLM for a confidence score** (e.g. an integer
  0–100 or a float 0–1) for its SELECT/BRIDGE choice, plus a one-line rationale. Log it per
  round (enables a confidence-vs-ΔSR diagnostic and a calibration check).

## 7. SELECT ⟷ BRIDGE flexibility (unchanged, but keep it central)
- Every round the LLM freely chooses **SELECT** (correct one recorded failure on-policy from
  `t_flag`) **or BRIDGE** (prescribe one new middle-ground scene between 2–3 cited failures).
  Never force one. Log which was chosen (for the targeted-vs-bridge usage diagnostic the
  supervisor wants — Tier 4).

## 8. Budget
- **Main runs: 20 successful demonstrations.** Budget **sweep** for the ablation: run
  **{10, 20, 40}** (supervisor Tier 3; the user also mentioned {5,10,20} — include 5 if cheap)
  on one task per family. Report whether the DISTIL margin grows or shrinks with budget (the
  "framework, not a 20-demo instance" claim depends on it).
