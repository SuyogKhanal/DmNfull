# D5_Compute — Push-T / IMAGE: how to measure one round of DISEIL and of SafeDAgger

Status: **recon complete, instrumentation in place, NOTHING SUBMITTED.**
Baseline arm = SafeDAgger (`safe_dagger`), matching the three RoboSuite D5 jobs
(`ablation=safe`). Budget = 1 round, per the author's instruction.

---

## 0. Where Push-T actually lives (and why it is not the `distil/` module)

`distil/scripts/build_d5_compute.py` currently marks Push-T **blocked** because
`distil.config.get_config` has no Push-T task. That remains true. The Push-T image
study is implemented only in the fork-backed suite:

    /weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo

Consequences that drive everything below:

* The LLM is **not** OpenRouter. `run_pool_rl_robo.sh` starts **two local vLLM
  servers** (`Qwen3-32B` text on GPU1, `Qwen3-VL-32B` vision on GPU0) plus the
  fork's `qwen/proxy.py`, and points `OPENAI_BASE_URL` at the proxy. The
  orchestrator runs on GPU2. **DISEIL therefore needs 3 GPUs; SafeDAgger needs 1.**
* DISEIL is `p4_subtask` = the fork's `LLMGuidedDAggerPipeline` (profile **P4**:
  VLM + per-episode reasoning + cross-episode reasoning + KAG + plain-LLM
  aggregator; RAG/TKF off) with the suite's `SubtaskPlanner` injected
  (visual R3M clustering → diversity → sub-task anchor).
* DISEIL image is **H100-only** (on h200l the extra rgb reposition env hangs during
  Vulkan render-context setup — confirmed job 102948). SafeDAgger builds no
  reposition env, so h100|h200 is fine for it.

---

## 1. The exact one-round configuration

New file (nothing existing was retuned):
`pool_rl_robo/config_pusht_image_d5.yaml` — a verbatim copy of the study config
`config_pusht_image.yaml` with exactly two knobs moved:

| knob | study | D5 |
|---|---|---|
| `budget` | 100 | **1** |
| `max_rounds` | 100 | **1** |

Everything that *costs time in a round* is unchanged, so the measured second is the
same second the real study pays: `nd_retrain: 1` (from-scratch retrain after the
demo), `round_epochs: 200`, `max_train_steps: 30000`, `rollout_episodes: 60`,
`heldout_n: 100`, `eval_num_envs: 10`, `initial_demos: 20`, `target_sr: 1.0` (no
early stop), and the full V3-hybrid `p4.subtask` block (`cluster_features: visual`,
R3M, PCA-16). It also states `baselines.safe_dagger.tau: 0.1` explicitly — that is
the value the code already defaults to and the value in `config.yaml`, so it is a
documentation change, not a behaviour change.

Why `budget: 1` gives **exactly one round**:
* DISEIL: round 1 collects 1 successful demo → `budget_used(1) >= budget_total(1)`
  → `_retrain_and_log` → `stop_reason="budget_exhausted"`. (If round 1's
  prescription is infeasible/empty the engine may enter round 2 — the telemetry is
  per-round, so round 1 is still the measured unit.)
* SafeDAgger: `while interventions < budget` — the first successful intervened
  episode ends the loop, then retrain + held-out eval, then
  `stop_reason="budget_exhausted"`.

**Both arms reuse the finished image bootstrap byte-identically** via
`P4_REUSE_INIT_CKPT` →
`results/PushT-v1/run_111/shared_baselines/init_ckpt.pth` (`init_sr = 0.27`,
20 bootstrap demos, seeds 0–19). So no bootstrap *training* is re-paid, the two
arms start from the identical policy, and `initial_demos` **must stay 20** (it is
the dataset that checkpoint was trained on).

Env: conda env **`diffdagger`** (`CONDA_ENV=diffdagger`, interpreter
`/home/s226137394/.conda/envs/diffdagger/bin/python`); `PYTHONPATH` is not needed —
`run_pool_rl_robo.sh` `cd`s to `DmNfull` and runs the dotted module. vLLM servers use
`/home/s226137394/.conda/envs/vllm_embed/bin/python`, the proxy uses
`/home/s226137394/.conda/envs/maze/bin/python`. Models come from the launcher's env
(`LLM_MODEL_NAME=qwen3-32b`, `VLM_MODEL_NAME=qwen3-vl-32b`); the repo-root `.env`
carries `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `OPENAI_BASE_URL` but
`python-dotenv` runs with `override=False`, so the launcher's local-proxy
`OPENAI_BASE_URL` wins. **No secret is needed or read for this run.**

---

## 2. THE TWO COMMANDS (do not submit yet)

From `/weka/s226137394/DmNfull`:

```bash
# (a) DISEIL — Push-T, IMAGE, one round.  3 GPUs, H100-only (VLM+LLM+orchestrator).
sbatch --job-name=d5_PushT_image_full \
  --export=ALL,METHODS=p4_subtask,SEED=1,RUN_ID=511 \
  distil/scripts/run_pusht_d5.sbatch

# (b) SafeDAgger baseline — Push-T, IMAGE, one round.  1 GPU, no LLM.
sbatch --job-name=d5_PushT_image_safe --gpus-per-node=1 \
  --constraint="gpu-h100|gpu-h200" \
  --export=ALL,METHODS=safe_dagger,SEED=1,RUN_ID=511 \
  distil/scripts/run_pusht_d5.sbatch
```

They may run concurrently: same `run_511` directory but disjoint sub-trees
(`p4_subtask/` + `shared_p4_subtask/` + `run_summary_p4_subtask.json` vs
`safe_dagger/` + `shared_baselines/` + `run_summary_baselines.json`), and neither
writes into `run_111`. Logs land in `/weka/s226137394/DmNfull/distil/slurm_logs/`.

**sbatch wrapper (new):** `/weka/s226137394/DmNfull/distil/scripts/run_pusht_d5.sbatch`
— modelled on `run_distil.sbatch` (same qos/partition/log pattern, env-var
parameterised: `METHODS`, `SEED`, `RUN_ID`, `REUSE_CKPT`, `CONFIG`,
`D5_TELEMETRY`). It does **not** re-implement the arms: it exports the D5 env and
`exec`s the suite's proven `run_pool_rl_robo.sh` (which starts the vLLM servers +
proxy). Defaults: `gpu-large`, `qos=batch-long`, `gpu-h100`, 3 GPUs,
`cpus-per-gpu=8`, `mem=24G`, `time=12:00:00` — the same resource pattern the
existing `submit_pusht_img_hybrid.sh` uses.

Post-run (both arms finished):

```bash
/home/s226137394/.conda/envs/vllm_embed/bin/python distil/scripts/build_pusht_d5_row.py --run-id 511
```

---

## 3. Was anything already recorded? (findings)

| quantity | already recorded? |
|---|---|
| per-round wall-clock, DISEIL | **Partially.** `IterationRecord.save()` writes `elapsed_seconds` to `p4_engine/iteration_NNN/iteration_record.json`, but `save()` is called *before* the from-scratch retrain, so it **excludes retrain + held-out eval** — it is not a round total. (Measured example from the live study, run_111 round 2: `elapsed_seconds = 2127.8`.) |
| per-round wall-clock, SafeDAgger | **No.** `selection/iil_baselines.py::run_iil_arm` returns only a whole-arm `elapsed_seconds`, which also contains the bootstrap demo collection. |
| VLM / LLM token usage | **Returned but discarded.** The fork's `main_analysis/llm_clients.py::_responses_call` already extracts the Responses-API `usage` object (`_usage()` → `prompt_tokens`, `completion_tokens`) and every stage function bubbles it up in a `token_usage` dict — but `pipeline._analyze_and_prescribe` never reads it and **nothing is persisted anywhere on disk**. |
| reasoning ("thinking") tokens | **No**, and the endpoint does not expose them as a field: the qwen proxy strips `<think>…</think>` from the response *text*, while vLLM still bills those tokens inside `output_tokens`. They are therefore recoverable exactly as `completion_tokens − tokens(visible text)`. |
| KAG token contribution | **No.** |

**⇒ I instrumented it.** Tokens are *not* natively logged.

---

## 4. The instrumentation (additive, env-gated, non-breaking)

### 4.1 New file — `pool_rl_robo/telemetry_d5.py`
Completely **inert unless `D5_TELEMETRY=1`**: `install()` returns immediately, so
every existing run is byte-identical to before this file existed. When armed it
only *wraps* functions (call-through, return value untouched) and appends to a new
side file `results/PushT-v1/run_<ID>/<method>/results/telemetry/d5_events.jsonl`.
No prompt, threshold, default, control-flow branch or existing output schema is
modified. Every write is inside `try/except`.

Wrapped (verified live — all report `_d5_wrapped=True` after install):

* `main_analysis.llm_clients._responses_call` → one `llm_call` event per LLM/VLM
  call: `stage` (`VLM`/`Reasoning`/`Plain`), `model`, `effort`, `prompt_tokens`,
  `completion_tokens`, `total_tokens`, `kag_in_prompt`, `dur_s`, and the visible
  `completion_text` (so thinking tokens are recoverable exactly).
* `main_pipeline.pipeline.IterationRecord.__init__` → `round_start` event
  (the true round boundary) + round tagging for every later event.
* `LLMGuidedDAggerPipeline._init_llm_clients` → captures the rendered KAG text for
  the substring probe.
* `LLMGuidedDAggerPipeline._analyze_and_prescribe` → span (rollout analysis +
  clustering + VLM + reasoning LLM + prescription).
* `LLMGuidedDAggerPipeline._collect_prescribed_demos` → span (feasibility /
  expert-solve).
* `main_pipeline.sim_bridge.{train_policy, rollout_and_detect, evaluate_heldout,
  collect_initial_demos_shared}` → spans.
* `selection.iil_baselines.{train_policy, evaluate_heldout,
  collect_initial_demos_shared, _run_one_episode_iil}` → spans (the SafeDAgger arm
  binds these at module import, hence the separate patch).

### 4.2 The only edit to existing code — 11 added lines
`pool_rl_robo/orchestrator/_common.py`, at the top of `run_method()`:

```python
    # D5 compute telemetry (paper D5_Compute table). INERT unless D5_TELEMETRY=1 ...
    try:
        from .. import telemetry_d5 as _D5
        _D5.install(str(results_dir), method)
    except Exception:
        pass
```

Nothing else in the repository was touched. **The fork (`/weka/s226137394/diff-dagger`)
was not edited at all** — it is patched at runtime, only when the flag is set.

### 4.3 KAG token contribution — MEASURED, not approximated
`distil/scripts/measure_kag_tokens_pusht.py` (already run; output
`distil/results/_compute/kag_tokens_pusht.json`):

* Tokenizer = **the serving tokenizer**: the HF tokenizer vLLM itself loads from
  `/weka/s226137394/models/Qwen3-32B`. Not tiktoken, not an estimate.
* KAG doc = `pool_rl_robo/p4/kag/PushT-v1.kag.txt` (4 777 chars raw → 5 035 chars
  after `format_kag_for_prompt`).
* **`kag_tokens = 1433` tokens per KAG-carrying call** (exact tokenization of the
  verbatim block `_kag_block(format_kag_for_prompt(KAG))` that the pipeline splices
  into every analysis / prescription / cross-episode / aggregator prompt).
* Cross-check by paired prompt-token diff (same prompt rendered with and without
  the block): analysis **1431**, prescription **1471** — the spread is each
  builder's short KAG-conditional instruction line, ≈3 %.
* Per-round KAG cost = `kag_calls × 1433`, where `kag_calls` is counted from the
  run's own telemetry (`kag_in_prompt == true`). **VLM prompts carry no KAG.**

### 4.4 Row builder — `distil/scripts/build_pusht_d5_row.py`
Turns the two `d5_events.jsonl` files into
`distil/results/_compute/PushT_image_d5.json`. Definitions:

* **DISEIL total s/round** = `round_start(1)` → end of the last span of round 1
  (rollout + analysis + collect + retrain + held-out eval).
* **DISEIL reasoning-only s/round** = `Σ analyze_and_prescribe` + `Σ
  collect_prescribed_demos` — rollout analysis + VLM + reasoning LLM +
  prescription + feasibility. Reported **separately** from the total, because the
  round's wall-clock is dominated by the from-scratch retrain that **both** arms
  pay (established on the running Door job). The per-stage breakdown
  (`rollout_and_detect`, `train_policy`, `evaluate_heldout`, …) is emitted too.
* **Baseline total s/round** = first `iil_episode` after `collect_initial_demos`
  → end of that round's `evaluate_heldout`. The bootstrap demo collection is logged
  as its own span and **excluded from both arms**.
* **Overhead ×** = DISEIL total s/round ÷ Baseline total s/round.
* **VLM tok/round** = Σ(prompt+completion) over `stage=="VLM"`.
  **LLM tok/round** = Σ over `Reasoning` + `Plain`.
  **Reasoning-LLM tok/round** = Σ over `Reasoning` of
  `completion_tokens − tokens(visible completion_text)` (the stripped `<think>`
  block, still billed in `output_tokens`), tokenized with the serving tokenizer.
  If `transformers` is unavailable the field is emitted literally as
  `UNMEASURED`, never guessed.
* SafeDAgger token fields are `0` **by construction** — the arm makes no LLM call
  (`run_pool_rl_robo.sh` sets `--no-llm` and starts no vLLM server for it).

---

## 5. What is NOT yet measured

Every timing and token number for this row is **UNMEASURED until the two jobs above
run** — none is stated anywhere in this document or in any output file. The only
already-measured quantities are:

* `kag_tokens = 1433` per KAG-carrying call (§4.3), and
* the historical DISEIL round-minus-retrain time from the live study
  (`run_111/p4_subtask/.../iteration_002/iteration_record.json`,
  `elapsed_seconds = 2127.8`), quoted only as an order-of-magnitude sanity anchor
  for the queue time — it is **not** the D5 number.

## 6. Files

| path | status |
|---|---|
| `.../pool_rl_robo/telemetry_d5.py` | new |
| `.../pool_rl_robo/config_pusht_image_d5.yaml` | new |
| `.../pool_rl_robo/orchestrator/_common.py` | +11 lines, env-gated, non-breaking |
| `distil/scripts/run_pusht_d5.sbatch` | new (the sbatch wrapper) |
| `distil/scripts/measure_kag_tokens_pusht.py` | new (already run) |
| `distil/scripts/build_pusht_d5_row.py` | new |
| `distil/results/_compute/kag_tokens_pusht.json` | new (measured output) |
