# 08 — Orchestration across TWO HPCs (prioritized split + handoff)

The researcher has **two HPC allocations**. You (the coding agent) run in one session at a time,
tied to whichever HPC that session is on. Your job is to submit the ablation matrix smartly, and
when a session on HPC-A has done what it usefully can, **write a handoff `.md`** so a fresh
session on HPC-B continues seamlessly.

## ⚠ The two HPCs share NO filesystem — git is the transport
HPC-A is the weka cluster (`/weka/s226137394/...`); **HPC-B is a different cluster that cannot see
`/weka` at all.** Non-negotiables the consolidated module must respect:
- **Code reaches HPC-B via `git clone`/`pull`, not `/weka`.** This is the payoff of consolidating
  into one `distil/` package + pushing to GitHub. The custom envs (PushT-v2, robosuite wrappers)
  must be **vendored inside `distil/`** so a clone brings them (`04_...md`). **No absolute `/weka`
  path or import anywhere** — repo-relative root / `$DISTIL_ROOT` (`00` golden rule 8). A single
  hardcoded path breaks HPC-B silently.
- **No model weights travel** — the LLM is an OpenRouter API call, so HPC-B needs only the code, a
  GPU for the diffusion policy, and its own `.env` with `OPENROUTER_API_KEY`. (Biggest portability
  win of the OpenRouter switch.) Bring-up checklist: **`SETUP_HPC2.md`**.
- **Byte-identical bootstrap stays LOCAL by partitioning cleanly:** assign each **whole
  `(task, modality)` cell** — DISTIL + every baseline/ablation + all 5 seeds — to a *single*
  cluster. Its bootstrap is generated once there and every arm loads it (byte-identical *within the
  cell*, which is all fairness needs). **Never split one cell's arms across the two clusters** — or
  you'd compare arms trained on different bootstraps. This is exactly "prioritize + split by
  task-block, not naive half/half". (If you ever must share a bootstrap across clusters it's a
  small `.pkl` — commit it or `rsync`/`scp` it.)
- **Results flow back via git:** HPC-B can't write to `/weka` either. Each cluster writes its own
  light `result.json`/`config.yaml` leaves and pushes ONLY those (never checkpoints/telemetry) to
  a results branch; the aggregator (run on HPC-A) `git pull`s both and builds the master table.
  Commit `RUN_STATE.md` too so both clusters share one ledger.

## Job model (thanks to OpenRouter)
- Every run is **one self-contained sbatch job** = one (task, modality, ablation, seed) OR one
  (task, modality, ablation) that loops seeds internally — your call, but **one ablation branch
  is one job** (golden rule 5).
- Because the LLM is an **OpenRouter API call**, a job needs only **~1 GPU** (diffusion-policy
  train + rollout) and **no** `gpu-h100|gpu-h200` constraint. Submit to whatever partition is
  fastest to start. Put `OPENROUTER_API_KEY` in the job env (source `.env`).
- Deterministic output path per job (see `09_...md`):
  `results/<task>/<modality>/<ablation>/seed<s>/` with `result.json`, `run.log`, `telemetry/`,
  and the **prompts + KAG copied in** (golden rule 3).

## Prioritized split — NOT naive half/half
Rank the queue, submit high-value first, and rebalance dynamically:

1. **Priority order** (from `05_ABLATIONS.md` triage):
   - P0: **full DISTIL**, all tasks × both modalities × 5 seeds, budget 20 (the headline table).
   - P1: **Tier-1 knockouts** (memory-off, random-alloc, clustering-off, LLM-vs-heuristic,
     VLM-off) — the allocation thesis lives here.
   - P2: budget sweep {10,20,40}; Tier-2 drawer ablations.
   - P3: Tier-3 sensitivity sweeps; Tier-4 diagnostics (mostly log-parsing, cheap).
2. **Split by cost, not by count — but always by WHOLE cell.** Wipe and PushT are the long poles
   (500-step episodes / 250-step reroll + more rounds). Spread the *expensive cells* across the two
   HPCs — e.g. Wipe-state on one, Wipe-image + PushT on the other — but keep every arm/seed of a
   single `(task,modality)` cell on ONE cluster (bootstrap fairness, above). Estimate per-cell
   wall-clock (rounds × (train + eval + screen + LLM)) and load-balance on that, not on job count.
3. **Watch start latency + progress.** Monitor the queue. If a job is **stuck pending** (slow to
   start on this HPC) or is **low priority** (P3) while P0/P1 still need slots, **cancel it** and
   move it to the handoff list rather than let it block the important work.

## The handoff mechanism (this is the key ask)
When the current HPC session has submitted what it should and identified work better done
elsewhere:
1. `scancel` the low-priority / slow-to-start jobs you're deferring.
2. **Write `HANDOFF_HPC2.md`** in this folder containing: the exact remaining job list (task,
   modality, ablation, seed, priority, est. cost), the precise `sbatch`/launch command for each,
   which shared-bootstrap files they need, and any partial results already on disk. Make it
   copy-paste runnable.
3. The researcher opens a **new session on the other HPC**, points it at this folder, and says
   "continue from `HANDOFF_HPC2.md`." That session submits the remaining jobs there.
4. Each session **updates a shared `RUN_STATE.md`** (or `run_state.json`) — a live ledger of
   every job: `{task, modality, ablation, seed, hpc, jobid, status, result_path}` — so either
   session (and the aggregator) always knows what's done, running, or pending. Treat this ledger
   as the source of truth; reconcile against `squeue` at session start.

## Use multi-agent workflows
This is a large matrix — orchestrate with the Workflow tool:
- **Build/verify** phase: parallel agents port + unit-test each component (env adapters,
  descriptor, cluster, memory, collect, LLM client, KAG) against the consolidated module.
- **Submit** phase: fan out job submission by (task family) once the smoke passes.
- **Aggregate** phase: parallel agents parse each `result.json`/telemetry into the master table
  + the Tier-4 diagnostics (`09_...md`).
- Smoke EVERY new component on 1 GPU + a cheap OpenRouter call before launching the full matrix
  (1-round, tiny budget) — never launch the matrix on unproven code.

## Guardrails
- **Never** commit `results/` blobs, checkpoints (`*.pt`), or frames to git (they bloat the repo
  — a past incident hit 548 GB). Keep only `result.json` + `run.log` + configs light.
- Cap OpenRouter spend: log tokens/round; the compute table (Tier 4) needs it anyway.
- If OpenRouter rate-limits, back off + retry (the client must handle 429/5xx); a failed LLM call
  falls back to the deterministic decision, logged.
