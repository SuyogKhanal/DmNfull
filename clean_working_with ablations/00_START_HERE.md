# 00 — START HERE  (context pack for the DISTIL clean-up + ablation run)

You are the coding agent taking over the **DISTIL** project (formerly "P4-LLM V3 hybrid").
This folder is your **single source of context**. Read the `.md` files in order (00→09), read
the artifacts listed below, then execute. **You do the work — build, submit, aggregate.** The
previous session only authored this context pack.

## The artifacts already in this folder (read them)
- `paper.pdf` — the (older) DISTIL paper. Method + claims + current results. The *figure* in it
  is old; the **claim is the NEW architecture** below.
- `Architectural Diagram.pdf` and `Architectural Diagram.drawio (1).html` — **the updated
  architecture = the claim**. This is what you implement. Open the HTML to see how components
  connect.
- `supervisor_ablation_ask.txt` — the supervisor's full ablation demand (6 tiers). This defines
  the experiment matrix. `05_ABLATIONS.md` distills it into config flags.
- `DISTIL_ablation_preview.xlsx` — the target results structure. Aggregate into this shape.

## The mission (one sentence)
Consolidate the code that is scattered across **four repos** into **one clean, reproducible
DISTIL module**, and re-run **every task (GridWorld 5×5, Lift, Wipe, Door, PushT — each in
state AND image modality) × 5 seeds × the ablation matrix**, at a **20-demonstration budget**
(plus a budget sweep), using the **OpenRouter API** for the LLM, orchestrated across **two
HPCs**, with results aggregated into a clean, inspectable structure.

## Golden rules (non-negotiable — the supervisor rejected the last version for these)
1. **One place.** All code in a single module in this folder (or a single subfolder). No more
   hopping between `diff-dagger-ur5`, `diff-dagger`, `pool_rl_robo`, `DmNfull`. See
   `04_CODEBASE_MAP_AND_CONSOLIDATION.md`.
2. **Reproducible.** Fixed seeds, **byte-identical shared bootstrap** per (task, seed) shared by
   every arm, one config per run, deterministic where possible. Anyone must reproduce any cell
   with one command.
3. **Prompts + KAG travel with every run.** Save the exact LLM prompts and the KAG document
   **into each task/ablation output folder** (`06_PROMPTS.md`, `07_KAG.md` are the masters).
4. **Confidence score every round.** The decision prompt must explicitly ask the LLM to emit a
   per-round **confidence score**; log it.
5. **One ablation branch = one job.** Every ablation/config is a single self-contained job;
   results land in a predictable path and are auto-aggregated (`09_REPRODUCIBILITY...md`).
6. **Two HPCs, prioritized split (not naive half/half).** See `08_ORCHESTRATION_2HPC.md`.
7. **OpenRouter API, no local vLLM, no H100/H200 constraint.** The LLM is an API call; jobs need
   only ~1 GPU for the diffusion policy. Faster + schedules anywhere.

## What is DISTIL (the concept you implement) — full detail in `01_METHOD_DISTIL.md`
Per round: roll the current policy → detect failures by the policy's own uncertainty → build a
geometric descriptor per failure → **cluster** failures into modes → **cluster memory** rotates
which mode to target → assemble a context set → the **reasoning LLM decides SELECT (correct one
real failure on-policy) vs BRIDGE (prescribe one middle-ground scene covering several)** and
emits a **confidence score** → collect **one successful demo** → retrain from scratch → eval on
a frozen held-out set. The LLM has **full flexibility to choose SELECT or BRIDGE every round.**

## The design decisions that are DIFFERENT this run — read `02_DESIGN_CHANGES_THIS_RUN.md`
Short list (details there): (a) cluster on the **first threshold-crossing** uncertainty signal,
**not** the peak/argmax; (b) **increase max_steps / expert budget** for Lift/Wipe/Door (state +
image) — they were under-allocated; (c) **OpenRouter API** LLM backend; (d) **geometric-state
clustering for image tasks too — NO R3M** (the LLM makes config recommendations from geometry);
(e) an **infeasibility re-prescribe loop** (esp. for PushT); (f) **confidence score per round**.

## The crown jewel (where to spend compute)
The supervisor is blunt: the paper is a 9-component pipeline with **no internal ablations**, and
≥5 components risk being shown inert. **The allocation thesis (clustering + memory → coverage →
success) is the one clean causal story — prove THAT first.** Tier-1 knockouts (memory-off,
random-allocation, clustering-off, LLM-vs-heuristic, VLM-off) + the sign test are submission-
critical. Everything else is drawer/rebuttal material. `05_ABLATIONS.md` has the triage.

## Read order
- `01_METHOD_DISTIL.md` — the method + equations (Eq 7 descriptor, Eq 8 k-selection, Eq 9 memory).
- `02_DESIGN_CHANGES_THIS_RUN.md` — the explicit changes for this clean run.
- `03_TASKS_AND_ENVS.md` — every task/modality, env ids, **new** horizons, success, experts.
- `04_CODEBASE_MAP_AND_CONSOLIDATION.md` — the 4 scattered repos → one module; what to port.
- `05_ABLATIONS.md` — the supervisor's matrix → config flags + triage + the budget sweep.
- `06_PROMPTS.md` — verbatim prompts (VLM / analysis / SELECT-BRIDGE decision / confidence).
- `07_KAG.md` — the per-task KAG documents + renderer.
- `08_ORCHESTRATION_2HPC.md` — two-HPC prioritized split + handoff-`.md` mechanism + workflows.
- `09_REPRODUCIBILITY_AND_AGGREGATION.md` — seeds/bootstrap, results tree, stats (sign test),
  the Tier-4 diagnostics to auto-report, the compute table.

## Do NOT
- Do not reintroduce R3M for image clustering. Do not hardcode H100/H200. Do not scatter code.
- Do not silently drop the infeasibility handling. Do not report a metric you did not verify.
