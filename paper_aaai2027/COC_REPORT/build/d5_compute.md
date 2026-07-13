# D5_Compute — wall-clock and token cost per round (complete five-setting matrix)

Baseline arm = **SafeDAgger** (`ablation=safe`) in all five settings, matching the three RoboSuite D5 jobs. Method = **DISEIL** (`p4_subtask` / `p4_top3_rotate` are code identifiers only). Seed 1 throughout.

Every number below traces to a logged event in a run that executed on this cluster. Nothing is extrapolated or hand-filled. Quantities that could not be measured are written `UNMEASURED` with the reason.

---

## 0. The one thing to take away

**A round's wall-clock is dominated by the from-scratch policy RETRAIN plus the 100-episode held-out EVAL — and BOTH arms pay all of it.** On the three RoboSuite settings that shared cost is 548–1 491 s of the round. The consequence:

* the raw **Overhead ×** on RoboSuite is **1.13–1.53×**, i.e. close to 1, and it **UNDERSTATES** DISEIL's cost, because the large shared denominator dilutes it;
* the quantity that actually characterises DISEIL is the **reasoning-only add-on** — the seconds and the tokens the baseline does **not** spend.

Report both. Neither alone is honest.

---

## 1. Protocols — do not mix these

| | **P1 — first round** | **P5 — mean of the LLM-active rounds** |
|---|---|---|
| Which rounds | the run's **first** round: round **0** for the distil module (Door/Wipe/GridWorld, 0-indexed), round **1** for the fork's 1-indexed Push-T loop | rounds **0–4** (the five LLM-active rounds) of the BUDGET=5 runs, matched arm-for-arm against the baseline's same-indexed rounds |
| Available for | **all five settings** | **the three RoboSuite settings only** |
| Why | Push-T and GridWorld were run at **BUDGET=1** (the author's explicit instruction), so the first round is the *only* round they have. P1 is therefore the only apples-to-apples comparison across all five. | Push-T/GridWorld: **UNMEASURED** — at BUDGET=1 exactly one round exists, so there is no multi-round mean and no spread. |
| What it means | **Worst case / UPPER BOUND.** The first round has the weakest policy → most failures → biggest cluster set → most VLM+LLM calls → highest reasoning cost. | **Best estimate of steady-state cost.** ± is the sample SD of the round-to-round spread *within* one run (**not** a seed-to-seed spread). |

Round 5 of the BUDGET=5 runs is a terminal, eval-only round (budget exhausted, no LLM); it is excluded from **both** arms under P5.

---

## 2. THE MATRIX — Protocol P1 (first round; all five settings)

| Task / Obs | Baseline s/round | DISEIL s/round | Overhead × | VLM tok/round | LLM tok/round | KAG tok contribution | Reasoning-LLM tok/round |
|---|---|---|---|---|---|---|---|
| **Door / state** | 737.0 | 1,054.0 | **1.43×** | 3,258 | 8,253 | 3,200 | 3,486 |
| **Push-T / image** | 688.0 | 1,891.0 | **2.75×** | 17,504 | 64,612 | 11,464 | 7,214 |
| **Wipe / image** | 1,468.0 | 2,195.0 | **1.5×** | 3,228 | 6,332 | 2,392 | 2,303 |
| **Door / image** | 1,247.0 | 1,474.0 | **1.18×** | 3,293 | 8,086 | 3,200 | 3,004 |
| **GridWorld 5x5 / image** | 54.6 | 118.0 | **2.16×** | 1,690 | 8,045 | 3,460 | 3,055 |

### 2b. The same rounds, decomposed into shared vs DISEIL-specific seconds

| Task / Obs | Shared train+eval (both arms) | DISEIL screen | DISEIL analysis+prescription | Baseline gate rollout | **Reasoning-only add-on** |
|---|---|---|---|---|---|
| **Door / state** | 783.0 s | 205.0 s | 66.0 s | 1.0 s | **+270.0 s** |
| **Push-T / image** | 652.5 s | 746.8 s | 491.6 s | 6.3 s | **+1,232.1 s** |
| **Wipe / image** | 1,491.0 s | 431.0 s | 273.0 s | 4.0 s | **+700.0 s** |
| **Door / image** | 1,180.0 s | 201.0 s | 93.0 s | 1.0 s | **+293.0 s** |
| **GridWorld 5x5 / image** | 51.1 s | 5.1 s | 60.4 s | 2.9 s | **+62.6 s** |

* **reasoning-only NARROW** = clustering + VLM + reasoning LLM + prescription + feasibility (the *analysis+prescription* column). It **excludes** the failure-screening rollout. The baseline does **none** of this work.
* **reasoning-only WIDE** = NARROW + the failure-screening rollout = every DISEIL-specific stage (the whole round minus the shared train+eval).
* **Reasoning-only add-on** = WIDE − the baseline's gate rollout. This is the honest "what DISEIL costs you" number.

⚠️ **The two source documents defined `reasoning-only` differently and this merge fixes that.** The Push-T/GridWorld source doc reported Push-T's 491.7 s as NARROW (screen excluded) but GridWorld's 65.5 s as WIDE (screen included). Both definitions are computed consistently for all five settings here:

| Task / Obs | reasoning-only **narrow** s | reasoning-only **wide** s |
|---|---|---|
| Door / state | 66.0 | 271.0 |
| Push-T / image | 491.6 | 1,238.4 |
| Wipe / image | 273.0 | 704.0 |
| Door / image | 93.0 | 294.0 |
| GridWorld 5x5 / image | 60.4 | 65.5 |

---

## 3. THE MATRIX — Protocol P5 (mean ± SD over the LLM-active rounds)

The **n rounds** column states exactly how many rounds each mean is taken over. It is 5 for a completed BUDGET=5 run; a smaller number means the run was still in flight when this matrix was generated and the mean is over the rounds that had actually completed. No round is ever extrapolated.

| Task / Obs | Baseline s/round | DISEIL s/round | Overhead × | VLM tok/round | LLM tok/round | KAG tok contribution | Reasoning-LLM tok/round | n rounds |
|---|---|---|---|---|---|---|---|---|
| **Door / state** | 532.6 ± 145.5 | 782.6 ± 183.5 | **1.47×** | 3,285 ± 22 | 7,644 ± 573 | 3,200 | 2,712 ± 630 | 5 |
| **Push-T / image** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |
| **Wipe / image** | 1,303.3 ± 144.1 | 1,991.0 ± 179.2 | **1.53×** | 3,239 ± 12 | 6,879 ± 903 | 2,392 | 2,909 ± 978 | 3 |
| **Door / image** | 1,034.2 ± 150.2 | 1,168.8 ± 194.8 | **1.13×** | 3,292 ± 9 | 7,495 ± 838 | 3,200 | 2,561 ± 917 | 5 |
| **GridWorld 5x5 / image** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |

Push-T and GridWorld are **UNMEASURED** under P5 for one reason and one reason only: they were run at **BUDGET=1** on the author's explicit instruction, so exactly one round exists. A mean and a spread over 5 rounds cannot be computed from 1 round, and will not be invented here.

### 3b. P5, decomposed

| Task / Obs | Shared train+eval | DISEIL screen | DISEIL analysis+prescription | Baseline gate | **Reasoning-only add-on** |
|---|---|---|---|---|---|
| **Door / state** | 547.8 s | 158.6 s | 76.2 s | 2.0 s | **+232.8 s** |
| Push-T / image | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| **Wipe / image** | 1,332.7 s | 426.7 s | 231.7 s | 2.7 s | **+655.7 s** |
| **Door / image** | 901.6 s | 181.0 s | 86.2 s | 1.0 s | **+266.2 s** |
| GridWorld 5x5 / image | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |

---

## 4. Why Push-T and GridWorld show a much larger Overhead × than Door and Wipe

It is **not** that their reasoning is more expensive relative to the work done. It is that their **shared denominator is small** and, on Push-T, that the baseline's rollout is atypically tiny at BUDGET=1:

* **SafeDAgger's loop is `while interventions < budget`.** At BUDGET=1 it stops at its **first** intervened episode — 1 episode, 6.3 s on Push-T. DISEIL meanwhile screens a fixed 60 episodes (746.8 s). That asymmetry, not the retrain, is what produces the 2.75×. At a realistic budget the baseline would roll out many episodes per retrain and the ratio would **fall**.
* On Door/Wipe the shared train+eval is 548–1 491 s per round — large enough to swamp a 234–704 s reasoning add-on, which is why those ratios sit at 1.13–1.53×.

Both effects push the ratio around without changing the underlying reasoning cost. **This is precisely why the ratio is the wrong headline and the add-on is the right one.**

---

## 5. Provenance — job IDs and exact commands

| Setting | DISEIL job | SafeDAgger job | Budget | Backend |
|---|---|---|---|---|
| Door / state | **110355** | **110356** | 5 | OpenRouter `qwen/qwen3-32b` + VLM |
| Push-T / image | **110375** | **110376** | 1 | local vLLM `qwen3-32b` + `qwen3-vl-32b` |
| Wipe / image | **110359** | **110360** | 5 | OpenRouter `qwen/qwen3-32b` + VLM |
| Door / image | **110357** | **110358** | 5 | OpenRouter `qwen/qwen3-32b` + VLM |
| GridWorld 5x5 / image | **110384** | **110385** | 1 | OpenRouter `qwen/qwen3-32b` + VLM |

```bash
cd /weka/s226137394/DmNfull

# ---- the three RoboSuite settings (BUDGET=5) : jobs 110355-110360 ----
# (already launched by the main build; DO NOT relaunch)
sbatch --job-name=d5_Door_state_full \
  --export=ALL,TASK=Door,MODALITY=state,ABLATION=full,SEED=1,BUDGET=5 \
  distil/scripts/run_d5.sbatch                       # -> 110355
sbatch --job-name=d5_Door_state_safe \
  --export=ALL,TASK=Door,MODALITY=state,ABLATION=safe,SEED=1,BUDGET=5 \
  distil/scripts/run_d5.sbatch                       # -> 110356
#   ... likewise Door/image -> 110357/110358, Wipe/image -> 110359/110360

# ---- Push-T / image (BUDGET=1) : jobs 110375 / 110376 ----
sbatch --job-name=d5_PushT_image_full \
  --export=ALL,METHODS=p4_subtask,SEED=1,RUN_ID=511 \
  distil/scripts/run_pusht_d5.sbatch                 # -> 110375
sbatch --job-name=d5_PushT_image_safe --gpus-per-node=1 \
  --constraint="gpu-h100|gpu-h200" \
  --export=ALL,METHODS=safe_dagger,SEED=1,RUN_ID=511 \
  distil/scripts/run_pusht_d5.sbatch                 # -> 110376

# ---- GridWorld 5x5 / image (BUDGET=1) : jobs 110384 / 110385 ----
# CONDA_ENV=diffdagger is REQUIRED (no `distil` env on this cluster; without the
# override `python` resolves to an interpreter with no torch -- this killed 110377/8)
sbatch --job-name=d5_GridWorld_image_full \
  --export=ALL,MODALITY=image,ABLATION=full,SEED=1,BUDGET=1,CONDA_ENV=diffdagger \
  distil/scripts/run_gridworld_d5.sbatch             # -> 110384
sbatch --job-name=d5_GridWorld_image_safe \
  --export=ALL,MODALITY=image,ABLATION=safe,SEED=1,BUDGET=1,CONDA_ENV=diffdagger \
  distil/scripts/run_gridworld_d5.sbatch             # -> 110385

# ---- rebuild this matrix from the runs (read-only over the runs) ----
python3 distil/scripts/parse_d5_stages.py    # run.log -> d5_stage_spans.json
python3 distil/scripts/merge_d5_compute.py   # + pusht/gridworld -> d5_merged.json
python3 distil/scripts/write_d5_outputs.py   # -> d5_compute.{md,csv} + the workbook
```

### How the seconds were measured

The RoboSuite runs log no per-stage spans, so stage boundaries were reconstructed from the `[HH:MM:SS]` timestamps that `distil/run.py` already prints (`[train]` → `[calibrate]` → `[eval]` → `[screen]` → `[distil-llm] … analysed` → `[collect a0]`). The reconstruction is **checked** against the independently recorded `result.json` `history[].sec`: it agrees within **1 s on every one of the completed rounds**. Push-T and GridWorld carry explicit span events in their own telemetry side-files and needed no reconstruction.

Sources consumed:

* RoboSuite: `distil/results/_compute/{Door,Wipe}/{state,image}/{full,safe}/seed1/{run.log,result.json,telemetry/round_*.jsonl}`
* Push-T: `…/pool_rl_robo/results/PushT-v1/run_511/{p4_subtask,safe_dagger}/results/telemetry/d5_events.jsonl`
* GridWorld: `distil/results/_compute/GridWorld/image/{full,safe}/seed1/telemetry/compute.jsonl`
* KAG tokens/call: `paper_aaai2027/COC_REPORT/build/kag_tokens.json`
* Merged intermediates: `distil/results/_compute/{d5_stage_spans,d5_merged}.json`

---

## 6. CAVEATS — read before quoting any number

**C1 — Single seed (seed 1) everywhere.** The ± in §3 is the round-to-round spread *within one run*. It is **not** a seed-to-seed confidence interval. No cell in this matrix has cross-seed variance.

**C2 — P1 is an UPPER BOUND, not an average.** The first round has the weakest policy and therefore the most failures. Failure count drives everything DISEIL-specific: episodes analysed → VLM calls → reasoning-LLM calls → KAG-carrying prompts → tokens and reasoning seconds. Later rounds cost less (Door/state screen failures fall 19 → 12 → 15 → 9 → 6 across rounds 0–4, and the DISEIL round falls 1 054 s → 587 s). Compare §2 with §3 to see the size of the effect.

**C3 — Round 0 carries a double-length initial train, for BOTH arms.** The distil module uses `initial_train_steps=8000` in round 0 and `round_train_steps=4000` thereafter (Door/state DISEIL train: 250 s in round 0, then ~131–137 s). Since that inflated train sits in the **denominator** of Overhead ×, the P1 ratio is computed against an inflated *shared* cost and is therefore **conservative** (too low). The reasoning-only add-on is immune to this and is the robust figure.

**C4 — The Overhead × is not a measure of the reasoning cost.** See §0 and §4. On RoboSuite it is diluted by the large shared train+eval; on Push-T it is inflated by SafeDAgger's 1-episode rollout at BUDGET=1. It moves for reasons that have nothing to do with how expensive DISEIL's reasoning is.

**C5 — Token counts are NOT comparable across rows.** Door/Wipe/GridWorld run on **OpenRouter** (`qwen/qwen3-32b` + VLM) and read hidden-thinking tokens directly from `usage.completion_tokens_details.reasoning_tokens`. Push-T runs on **local vLLM** (`qwen3-32b` + `qwen3-vl-32b`) and *recovers* hidden-thinking tokens as `completion_tokens − tokens(visible text)` with the serving tokenizer, because the proxy strips `<think>…</think>` from the text while vLLM still bills those tokens. Both are measured, but they are different instruments over different models, prompts and tasks. Push-T's much larger counts (82 k vs ~10 k) reflect 20 LLM calls against 7, not a more expensive method.

**C6 — 'Reasoning-LLM tokens/round' is the ALL-STAGE hidden-thinking total.** For the RoboSuite/GridWorld rows that is `tokens.reasoning` = analysis + decision (the VLM emits ~0). For Push-T the comparable all-stage total is **7 214** (vlm 9 + reasoning-stage 5 612 + plain aggregator 1 593). The Push-T source doc's headline **5 612** is the *Reasoning stage alone* and must not be placed in the same column as the other four rows — this merge uses 7 214.

**C7 — KAG token contribution is measured per call, then multiplied by that run's own KAG-carrying call count** (`kag_calls` in the telemetry; VLM prompts carry no KAG). Tokens per KAG block, by paired prompt-token diff at `max_tokens=1` against the serving tokenizer: **Door 800**, **Wipe 598**, **GridWorld 865**, **Push-T 1 433**. KAG is a large share of the prompt budget — 41–42 % on Door, 35 % on Wipe, **54 % on GridWorld**, 17 % on Push-T.

**C8 — Push-T's LLM calls run CONCURRENTLY.** Its 20 calls sum to 1 040.9 s of *serving* time but their wall-clock union is 487.9 s. All seconds in this document are **wall-clock**, never summed call latencies.

**C9 — Bootstrap demo collection is excluded from every per-round total, in both arms.** Both Push-T arms additionally reuse an identical bootstrap checkpoint (`P4_REUSE_INIT_CKPT`), so no bootstrap training is re-paid and the arms start from the same policy.

**C10 — Nothing in this matrix is `UNMEASURED` except the P5 cells for Push-T and GridWorld**, and those are unmeasurable for the stated structural reason (BUDGET=1 → one round → no mean, no spread). No timing and no token count anywhere in this document was estimated, extrapolated or invented.

---

## 7. Instrumentation note (shared repo)

All D5 instrumentation is **additive and non-breaking**. `pool_rl_robo/telemetry_d5.py` is inert unless `D5_TELEMETRY=1` and only wraps functions call-through, appending to a new side-file. `distil/compute_log.py` only appends `telemetry/compute.jsonl` and does not touch `result.json`'s schema. The three scripts added for this merge (`parse_d5_stages.py`, `merge_d5_compute.py`, `write_d5_outputs.py`) are **read-only** over the runs — they parse `run.log` / `result.json` / telemetry and write new side-files. No existing behaviour, default, threshold, control-flow branch or output schema was changed, and no running job was touched.

_Generated 2026-07-13 19:01 from `distil/results/_compute/d5_merged.json`._
