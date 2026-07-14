# PROVENANCE — DISEIL re-run on OpenRouter (cluster HPC-A)

Cluster **HPC-A**. Cells owned here: **Door (state)**, **Door (image)**, **Push-T (state)**,
**Push-T (image)** — DISEIL arm only, 5 seeds each, B = 20, D = 1.
Lift and Wipe are HPC-B's and are not run here. No baselines are re-run.

Every number in `cells/*.json` traces to a run that completed on this cluster. Nothing is
estimated, extrapolated or hand-filled. A cell that cannot be run is marked **UNRUN** with
the reason, never invented.

---

## 1. The API — OpenRouter, and only OpenRouter

| | |
|---|---|
| Endpoint | `https://openrouter.ai/api/v1` (`OPENAI_BASE_URL`) |
| Key | `OPENROUTER_API_KEY` (`sk-or-…`), from the gitignored repo-root `.env`. Never printed, never committed. |
| VLM | `qwen/qwen3-vl-30b-a3b-instruct` |
| Reasoning / prescription LLM | `qwen/qwen3-32b` |

No OpenAI API is involved. **This is a real hazard, not a formality:** the repo `.env` also
carries a genuine OpenAI `sk-proj-…` key in `OPENAI_API_KEY`, and both codebases fall back to
it. See §4.

**Model change vs. the original Push-T runs.** The previous Push-T runs served
**`Qwen3-VL-32B`** and **`Qwen3-32B`** from *local vLLM* servers. OpenRouter does not offer a
`qwen3-vl-32b`; the vision model here is **`qwen/qwen3-vl-30b-a3b-instruct`** (a 30B A3B MoE
instruct model), as specified for this re-run. The reasoning model `qwen/qwen3-32b` is the same
model, now hosted. Door was already an OpenRouter cell, so its models are unchanged.

---

## 2. Protocol, confirmed against the code (not against the report)

| Quantity | Value | Where it is fixed in code |
|---|---|---|
| Budget **B** | **20** acquired demos | `distil/run.py:216` (`--budget`, default 20) → `final_demos = n_init + budget` (`run.py:292`), stop at `distil/p4/loop.py:90`. Push-T: `budget: 20` in the re-run configs. |
| Demos/round **D** | **1** | Door: hardcoded — `distil/p4/loop.py:148-158` breaks on the first successful demo. Push-T: `demos_per_round: 1`. |
| **Retrain cadence** | **every 1 demo** | Door: `train_and_calibrate` runs at the top of *every* round (`distil/p4/loop.py:70-72`); with D=1 that is one retrain per acquired demo. Push-T: `nd_retrain: 1` ("retrain from scratch after every demo") in every shipped Push-T study config. **Not changed.** |
| Held-out eval | **fixed 100 episodes, same seeds every round** | Door: `eval_episodes=100`, `eval_seed_base=5_000_000` (`distil/config.py:42-43`), passed as a loop-invariant `base_seed` (`p4/loop.py:74-76`). Push-T: `heldout_n: 100`, `heldout_seed_base=7777`. |
| Initial demos **Ni** (excluded from budget) | Door **4**; Push-T **20** | Door: `distil/config.py:119` (`num_init_demos=4`); the shared bootstrap on disk is `Door_{state,image}_ni4.pkl`. Push-T: `initial_demos: 20` — **must stay 20**, it is the bootstrap the shared `init_ckpt.pth` was trained on. |
| Arm | DISEIL | Door: `--ablation full`. Push-T: `--methods p4_subtask`. |

**Push-T arm = `p4_subtask`.** Confirmed from the report's own provenance doc
(`build/d5_pusht_plan.md`: *"DISEIL is `p4_subtask`"*). `p4_top3` is the superseded arm.
⚠️ The orchestrator **silently drops** unknown method names (`_common.py:349`), so
`--methods full` would run *nothing* and still exit 0.

### Round-0 sanity check (independent confirmation of Ni and the bootstrap)
Door (state) re-run round-0 held-out success: **0.450, 0.400, 0.420** (seeds 1–3).
`05_progress.md:158` states the Door (state) round-0 band as **0.41–0.47**. The re-run
reproduces it, which independently confirms Ni=4 and the shared bootstrap.

### Two report-side errors found (the CODE is right; the REPORT needs fixing)
1. **Retrain cadence.** `05_progress.md:150` says the robot tasks retrain *"once every fourth
   acquired demonstration"* and that the policy is *"refreshed five times over the budget"*.
   The code retrains after **every** demo (20 retrains over the budget), in both codebases.
   Confirmed by the author. No hyperparameter was changed; the report prose is wrong.
2. **Initial demonstrations.** `05_progress.md:154` says the initial set *"holds twenty
   demonstrations for every task"*. The code uses per-task Ni: Lift 8, Wipe 12, **Door 4**,
   GridWorld 20 (`distil/config.py`). `05_progress.md:158` in the same section states the
   swept values (…4 for Door…), so the section contradicts itself. Push-T is genuinely 20.

Neither was "fixed" here. The re-run uses the code's values, which are the values that
produced the published numbers.

---

## 3. Changes made, and why (nothing that touches the method)

| # | Change | Why |
|---|---|---|
| 1 | `distil/run.py`: `_strict_llm()` + `_assert_openrouter()`; abort when `make_llm()` returns `None`. | Fail-fast. See §4. |
| 2 | `distil/p4/loop.py`: under `DISEIL_STRICT_LLM=1`, an LLM exception **re-raises** instead of falling back to the geometric planner. | A run that loses the LLM part-way is invalid; it must not finish and write a plausible result. |
| 3 | `pool_rl_robo/p4_subtask/pipeline.py`: under `DISEIL_STRICT_LLM=1`, assert base_url + key shape and make **one live LLM call** before the loop starts; abort on failure. | The fork swallows every LLM error and would still write a curve with `n_queries: 0` and **no fallback flag**. |
| 4 | `pool_rl_robo/envs/env_setup.py`: `RESULTS_ROOT` honours `POOL_RESULTS_ROOT`. | Write the re-run into its own namespace so published results cannot be clobbered. Unset ⇒ byte-identical to before. |
| 5 | New `config_pusht_state_or20.yaml`, `config_pusht_image_or20.yaml`. | Copies of the study configs with **two** knobs moved: `budget 100→20` (the CoC B=20) and `target_sr 1.0→1.01`. |
| 6 | New `distil/scripts/run_pusht_openrouter.sbatch`. | `run_pool_rl_robo.sh` **unconditionally** exports `OPENAI_BASE_URL=http://127.0.0.1:<proxy>/v1` (line 167) for any `p4_*` method and demands 3 GPUs. It had to be bypassed, not reused. |

**`target_sr 1.0 → 1.01` is the one protocol-motivated knob.** The CoC protocol spends the
whole fixed budget. The early-stop test is `sr >= target_sr` with `sr ∈ [0,1]`, so `1.0` is
*not* unreachable — a 60/60 rollout satisfies it (this is the documented "60/60 fluke").
`1.01` is unreachable, so the stop can never fire. Every finished run is checked for having
actually acquired 20 demonstrations.

No hyperparameter of the method was touched: descriptor, clustering, cluster memory
(γ=0.6, σ=0.06, λ=1.0), diversity, SELECT/BRIDGE, KAG, retry caps, screen/rollout episodes,
epochs and the retrain cadence are all as shipped.

**Nothing that changes the local-vLLM Push-T path was altered**, so HPC-B is unaffected: the
strict guard, the results-root override and the smoke flag are all opt-in via env vars that
default to the previous behaviour.

---

## 4. Fail-fast: why a misconfigured API would otherwise have gone unnoticed

Both codebases degrade **silently** to a deterministic no-LLM fallback and still write a
plausible result. All 20 runs could have returned fallback-only numbers that are not DISEIL.

* `distil/p4/llm.py:337-340` — `make_llm()` returns `None` if `OPENAI_BASE_URL` is unset.
  `distil/run.py` then logged *"UNAVAILABLE … geometric fallback"* and **carried on**.
* `distil/p4/llm.py:50-51` — the key falls back to `OPENAI_API_KEY`. With the real OpenAI
  `sk-proj-…` key present in `.env`, a missing `OPENROUTER_API_KEY` would have sent Qwen
  model names to **api.openai.com**.
* The Push-T fork (`diffdagger/main_analysis/llm_clients.py:59`) reads `OPENAI_API_KEY`
  **only** — it has no `OPENROUTER_API_KEY` fallback. Left alone it would have sent the
  **OpenAI** key to OpenRouter (401), and the fork swallows that: the run finishes with
  `n_queries: 0`, `stopped_reason: "no_progress"`, and **no fallback flag anywhere**.

Guards now in place (all under `DISEIL_STRICT_LLM=1`, set by both launchers):
1. Assert `OPENAI_BASE_URL` is exactly `https://openrouter.ai/api/v1`.
2. Assert an OpenRouter key is present (Push-T additionally asserts it is `sk-or-…`, since
   the fork reads `OPENAI_API_KEY`, which the launcher sets to the OpenRouter key).
3. Push-T makes **one live LLM call before the loop starts** and aborts unless it returns
   usable text with non-zero tokens.
4. An LLM failure **during** a run aborts it rather than falling back.

### Pre-flight evidence (both proven before any real job was submitted)
* **Transport.** OpenRouter serves `POST /v1/responses` — HTTP 200. This matters: the Push-T
  fork speaks the **Responses API** (`client.responses.create`), not chat-completions. Probed
  through the fork's own `VLMClient` / `ReasoningClient` / `PlainClient`: the VLM correctly
  described a synthetic frame ("blue T-shaped block … red square pusher to the lower right"),
  the reasoner returned 575 completion tokens, and the strict-JSON aggregator returned clean
  JSON with no `<think>` leakage. **No proxy/shim is needed**, and Push-T DISEIL therefore
  needs **1 GPU, not 3**.
* **Loop, on a GPU compute node** (job 111026, Door state): guard asserted OpenRouter; VLM
  2,165 tokens / analysis+decision 4,948 tokens; a real prescription (`SELECT`,
  `CONFIDENCE: 85`); KAG present (`kag_calls: 3`). Compute nodes do reach OpenRouter.

---

## 5. Submitted jobs

Results namespaces (both new; neither existed before — no published result is on the write path):
* Door: `distil/results/rerun_openrouter/Door/<modality>/full/seed<N>/`
* Push-T: `…/pool_rl_robo/results/rerun_openrouter/PushT-v1/run_<id>/p4_subtask/`

### Door (state) — 5/5 submitted
```
sbatch --job-name=or_Door_state_s<N> \
  --partition=gpu-large,gpu --qos=batch-long --time=1-00:00:00 --mem=32G \
  --export=ALL,CONDA_ENV=diffdagger,DISEIL_STRICT_LLM=1,TASK=Door,MODALITY=state,\
ABLATION=full,SEED=<N>,BUDGET=20,\
OUTPUT_DIR=/weka/s226137394/DmNfull/distil/results/rerun_openrouter/Door/state/full/seed<N>,\
BOOTSTRAP_DIR=/weka/s226137394/DmNfull/distil/results/shared_bootstrap \
  distil/scripts/run_distil.sbatch
```
`CONDA_ENV=diffdagger` is required: `run_distil.sbatch` defaults to a `distil` env that does
not exist on HPC-A. `NUM_INIT` is deliberately unset so the per-task Ni (Door = 4) is used.
The bootstrap is the published `shared_bootstrap` (`Door_state_ni4.pkl`), so the re-run starts
from the same initial demonstrations as the cited runs.

| seed | job ID |
|---|---|
| 1 | 111027 |
| 2 | 111030 |
| 3 | 111031 |
| 4 | 111032 |
| 5 | 111033 |

Smokes: 111025 (Door, /tmp — output lost, compute-node-local), 111026 (Door, /weka — the one
quoted in §4), 111034 (Push-T state).

*(Remaining cells are appended as they are submitted.)*

### Door (state) — COMPLETE, 5/5 valid
Jobs 111027 / 111030 / 111031 / 111032 / 111033 — all COMPLETED (03:01–05:21 wall each).
**96.6 ± 1.3** over 5 seeds. **0 fallback rounds.** Seed 2 acquired 16/20 (policy saturated
at SR 0.96 — no usable failures for 4 consecutive rounds, remaining budget unspendable; the
LLM ran throughout). `saturation_patience` was not touched. Published Door (state) seed 4
shows the same behaviour at 11/20, so this is method behaviour, not a re-run artefact.
