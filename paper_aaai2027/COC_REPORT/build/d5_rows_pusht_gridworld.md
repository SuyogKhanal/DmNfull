# D5_Compute — the two completed rows: Push-T / image and GridWorld 5x5 / image

Status: **MEASURED.** Every number below traces to a logged event in a run that
completed on this cluster. Nothing is estimated, extrapolated or filled in by hand.
Where a quantity could not be measured it is written `UNMEASURED` with the reason.

Baseline arm = **SafeDAgger**, matching the three RoboSuite D5 jobs (`ablation=safe`).
Budget = **1 round** (the author's explicit instruction: one round is enough to time a
round).

---

## 1. The two rows

| | **Push-T / image** | **GridWorld 5x5 / image** |
|---|---|---|
| DISEIL **total** s/round | **1891.0** | **118.0** |
| Baseline (SafeDAgger) **total** s/round | **688.0** | **54.6** |
| **Overhead ×** (total ÷ total) | **2.75×** | **2.16×** |
| DISEIL **reasoning-only** s/round | **491.7** | **65.5** |
| Baseline gate/screen s/round | 0.0 (no analysis stage) | 2.9 |
| **Reasoning-only overhead (s)** | **+491.7 s** | **+62.6 s** |
| VLM tokens/round | **17 504** (9 calls) | **1 690** (3 calls) |
| LLM tokens/round | **64 612** (11 calls) | **8 045** (4 calls) |
| Total tokens/round | 82 116 (20 calls) | 9 735 (7 calls) |
| Reasoning-LLM tokens/round (hidden thinking) | **5 612** | **3 055** |
| **KAG token contribution** | **11 464** /round (8 calls × 1433) | **3 460** /round (4 calls × 865) |
| KAG as share of the prompt budget | 17 % of 65 700 prompt tok | **54 %** of 6 389 prompt tok |
| Prompt / completion tokens | 65 700 / 16 416 | 6 389 / 3 346 |
| Measured round | round 1 | round 0 |
| LLM backend | local vLLM: `qwen3-32b` (text) + `qwen3-vl-32b` (vision) | OpenRouter `qwen/qwen3-32b` + VLM |
| Job IDs (DISEIL / baseline) | **110375 / 110376** | **110384 / 110385** |
| SLURM state | COMPLETED (34:26 / 12:27) | COMPLETED (02:31 / 01:27) |

### Per-stage wall-clock (seconds, measured spans)

**Push-T / image — DISEIL (round 1, total 1891.0 s)**

| stage | s | DISEIL-specific? |
|---|---|---|
| `rollout_and_detect` (60 screening episodes) | 746.8 | yes, in this configuration — see caveat C3 |
| `analyze_and_prescribe` (clustering + VLM + reasoning LLM + prescription) | 490.0 | **yes** |
| `collect_prescribed_demos` (feasibility / expert solve) | 1.6 | **yes** |
| `train_policy` (from-scratch retrain) | 437.1 | no — both arms pay it |
| `evaluate_heldout` | 215.4 | no — both arms pay it |
| **sum** | **1890.9** ( = 1891.0 total, ✓) | |

**Push-T / image — SafeDAgger (round total 688.0 s)**

| stage | s |
|---|---|
| 1 × `iil_episode` (rollout until the first intervention) | 6.3 |
| `train_policy` | 475.2 |
| `evaluate_heldout` | 206.6 |
| bootstrap `collect_initial_demos` — **excluded from both arms** | 37.4 |

**GridWorld — DISEIL (round 0, total 118.0 s)**: train 43.3 | eval 7.8 | screen 5.1 |
LLM 60.4 | prescribe 0.007 (+1.5 s frame rendering / misc).
**GridWorld — SafeDAgger\* (54.6 s)**: train 41.7 | eval 9.5 | screen (gate) 2.9.

### Push-T token detail (per LLM stage, from `d5_events.jsonl`)

| stage | calls | prompt | completion | total | hidden thinking | KAG-carrying |
|---|---|---|---|---|---|---|
| VLM (`qwen3-vl-32b`) | 9 | 13 823 | 3 681 | 17 504 | 9 | 0 |
| Reasoning (`qwen3-32b`, effort=high) | 7 | 43 508 | 10 212 | 53 720 | **5 612** | 7 |
| Plain / aggregator (`qwen3-32b`) | 4 | 8 369 | 2 523 | 10 892 | 1 593 | 1 |
| **total** | **20** | **65 700** | **16 416** | **82 116** | **7 214** | **8** |

The headline "Reasoning-LLM tokens/round" (5 612) is the hidden-thinking count of the
**Reasoning stage**. Across *all* calls the hidden-thinking total is **7 214**
(the Plain aggregator also emits a `<think>` block). GridWorld's 3 055 is its
all-stage total (analysis 2 018 + decision 1 037; the VLM emits none), so the strictly
comparable Push-T figure is 7 214.

---

## 2. The headline interpretation

**A round's wall-clock is dominated by the from-scratch policy RETRAIN, which BOTH
arms pay.** On Push-T that shared cost is 437–475 s of training plus 207–215 s of
held-out evaluation — i.e. **652.5 s of the DISEIL round and 681.8 s of the baseline
round are the same work**. The overhead ratio is therefore *not* a measure of the
reasoning cost. The number that actually characterises DISEIL is the **reasoning-only**
time:

* Push-T: **491.7 s/round** of reasoning (clustering + 9 VLM calls + 7 reasoning-LLM
  calls + 4 aggregator calls + prescription + feasibility), against a baseline that
  does **none** of it.
* GridWorld: **65.5 s/round** vs the baseline's 2.9 s gate → **+62.6 s**.

---

## 3. Exact commands (reproduce)

```bash
cd /weka/s226137394/DmNfull

# --- Push-T / image (run_511, seed 1, budget=1, max_rounds=1) ---
# DISEIL (p4_subtask): 3 GPUs, H100-only (GPU0 VLM, GPU1 LLM, GPU2 orchestrator)   -> job 110375
sbatch --job-name=d5_PushT_image_full \
  --export=ALL,METHODS=p4_subtask,SEED=1,RUN_ID=511 \
  distil/scripts/run_pusht_d5.sbatch

# SafeDAgger baseline: 1 GPU, no LLM                                                -> job 110376
sbatch --job-name=d5_PushT_image_safe --gpus-per-node=1 \
  --constraint="gpu-h100|gpu-h200" \
  --export=ALL,METHODS=safe_dagger,SEED=1,RUN_ID=511 \
  distil/scripts/run_pusht_d5.sbatch

# --- GridWorld 5x5 / image (seed 1, budget=1) ---
# NOTE: CONDA_ENV=diffdagger is REQUIRED — there is no `distil` conda env on this
# cluster, and without the override `python` resolves to an interpreter with no torch
# (this is what killed the first attempt, jobs 110377/110378).
sbatch --job-name=d5_GridWorld_image_full \
  --export=ALL,MODALITY=image,ABLATION=full,SEED=1,BUDGET=1,CONDA_ENV=diffdagger \
  distil/scripts/run_gridworld_d5.sbatch          # -> job 110384

sbatch --job-name=d5_GridWorld_image_safe \
  --export=ALL,MODALITY=image,ABLATION=safe,SEED=1,BUDGET=1,CONDA_ENV=diffdagger \
  distil/scripts/run_gridworld_d5.sbatch          # -> job 110385
```

Telemetry side-files consumed:

* Push-T: `Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/results/PushT-v1/run_511/{p4_subtask,safe_dagger}/results/telemetry/d5_events.jsonl`
* GridWorld: `distil/results/_compute/GridWorld/image/{full,safe}/seed1/telemetry/compute.jsonl`
* Machine-readable row dump: `distil/results/_compute/d5_rows_pusht_gridworld.json`

---

## 4. CAVEATS — read before quoting any number

**C1 — SINGLE ROUND, SINGLE SEED. No variance is available.** Budget = 1 by
instruction, so each cell is *one* observation of *one* round at seed 1. No standard
deviation, no confidence interval. Do not present these as means.

**C2 — THESE ARE AN UPPER BOUND ON STEADY-STATE PER-ROUND COST.** The measured round is
the **first** round, when the policy is at its weakest and therefore produces the
**most** failures. Failure count drives everything DISEIL-specific: the number of
episodes that must be analysed, hence the number of VLM calls, reasoning-LLM calls and
KAG-carrying prompts, hence the tokens and the reasoning seconds. A later round, with a
stronger policy and fewer failures, costs **less**. Every DISEIL figure in this table is
therefore an **upper bound**, not a steady-state average.

**C3 — Push-T: the rollout is asymmetric BY CONSTRUCTION at budget = 1, and it, not the
retrain, is what inflates the 2.75×.** DISEIL screens a fixed `rollout_episodes: 60`
(746.8 s). SafeDAgger's loop is `while interventions < budget`, so with budget = 1 it
stops at its **first** intervened episode — 1 episode, 6.3 s. The baseline's per-round
rollout is thus atypically small here; at a realistic budget it would roll out many
episodes per retrain and the ratio would **fall**. Decomposing the 1203 s gap:
screening rollout +740.5 s, analysis +490.0 s, prescription +1.6 s, retrain −38.1 s,
eval +8.8 s. **The retrain is not the source of the overhead; the 60-episode screen and
the 490 s analysis are.**

**C4 — GridWorld: the measured round is round 0, whose retrain is ~2× the steady-state
retrain.** GridWorld uses `initial_train_steps=2000` in round 0 but
`round_train_steps=1000` thereafter. Round 0's train term is 43.3 s; the very next
round (logged, round 1) trains in 16.3 s. Since the *shared* retrain sits in the
denominator of Overhead ×, the 2.16× is computed against an inflated shared cost — with
a steady-state (cheaper) retrain the **ratio would be larger**, even though the absolute
reasoning seconds would fall with the failure count (C2). The reasoning-only figure
(+62.6 s) is the robust one. `build_d5_compute.py` deliberately drops round 0 for this
reason; at budget = 1 round 0 is the *only* LLM-bearing round, so it was read directly
from the telemetry side-file instead.

**C5 — Push-T LLM calls run CONCURRENTLY.** The 20 calls sum to 1040.9 s of *serving*
time, but their wall-clock union is 487.9 s — which is why the `analyze_and_prescribe`
span is 490.0 s and not ~1041 s. All reported seconds are **wall-clock**, never summed
call latencies. (All 20 calls fall inside that span; none occur during the rollout.)

**C6 — The two rows use different LLM backends and different token-accounting
mechanisms.** Push-T runs on **local vLLM** (the fork's qwen proxy); its hidden-thinking
tokens are **recovered** as `completion_tokens − tokens(visible text)` with the serving
tokenizer (`/weka/s226137394/models/Qwen3-32B`), because the proxy strips
`<think>…</think>` from the returned text while vLLM still bills those tokens in
`output_tokens`. GridWorld runs on **OpenRouter**; its hidden-thinking tokens are read
directly from `usage.completion_tokens_details.reasoning_tokens`. Both are measured, but
they are not the same instrument, and the two rows' token counts are **not** directly
comparable to each other (different models, different prompt sizes, different task).

**C7 — KAG token contribution is measured per call, then multiplied by the run's own
KAG-carrying call count** (`kag_in_prompt == true` in the telemetry; VLM prompts carry
no KAG):
* Push-T: **1433 tok/call**, by exact tokenization with the serving tokenizer of the
  verbatim block the pipeline splices in; cross-checked by paired prompt-token diff
  (analysis 1431, prescription 1471). × 8 calls = **11 464 tok/round**.
* GridWorld: **865 tok/call**, by paired prompt-token diff at `max_tokens=1`
  (analysis 304→1169, decision 585→1450). × 4 calls = **3 460 tok/round**.
  On GridWorld the KAG is **54 % of the entire prompt budget** — the single largest
  prompt-side term.

**C8 — Both Push-T arms reuse the identical bootstrap checkpoint** (`P4_REUSE_INIT_CKPT`
→ `run_111/shared_baselines/init_ckpt.pth`), so no bootstrap *training* is re-paid and
the two arms start from the same policy. The bootstrap demo *collection* (40.4 s DISEIL /
37.4 s baseline) is logged as its own span and is **excluded from both** per-round
totals.

**C9 — Nothing here is UNMEASURED.** Every field in §1 was read from a telemetry event.
The one field that would have been reported as `UNMEASURED` (Push-T hidden-thinking
tokens, had the serving tokenizer been unavailable) was successfully measured.

---

## 5. Instrumentation note (for the shared repo)

The telemetry is **additive and env-gated**: `pool_rl_robo/telemetry_d5.py` is completely
inert unless `D5_TELEMETRY=1` (verified: with the flag off it patches nothing and creates
no directory), and it only *wraps* functions call-through and appends to a new side-file.
No existing behaviour, default, threshold, control-flow branch or output schema was
changed. `distil/compute_log.py` likewise only appends `telemetry/compute.jsonl` and does
not touch `result.json`'s schema.

One hardening fix was made during the smoke phase: `telemetry_d5.install()` now calls the
suite's idempotent `bootstrap_fork_path()` before patching the fork. Previously it
depended on an import-order side effect (`orchestrator._common` → `envs.env_setup`) to put
`diffdagger` on `sys.path`; if `install()` were ever called first, the LLM/round/span
patches would have silently no-op'd and the entire token measurement would have been lost
with only an `install_warn` line to show for it. The production path always satisfied that
ordering, so this changes no behaviour — it removes a silent-failure mode.

`paper_aaai2027/COC_REPORT/build/kag_tokens.json` gained a `GridWorld` entry. Door and
Wipe were left **byte-identical** (a re-measure drifts Door 798→795, and the existing
Door/Wipe D5 rows depend on the stored values), so the file was merged, not regenerated.
