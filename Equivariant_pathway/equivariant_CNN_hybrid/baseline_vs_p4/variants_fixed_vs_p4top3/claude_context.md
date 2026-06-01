# claude_context.md — porting the baseline-vs-P4 comparison to a new policy

> **NOTE for this folder:** `variants_fixed_vs_p4top3/` is itself a WORKED
> EXAMPLE of the port described below — a fork of
> `baseline_vs_p4_sequential_batch` that swaps the *methods* (not the
> policy): `baseline_fixed`, `baseline_random`, `p4_top3` on a single
> FIXED 30-layout pool per run. For what THIS suite specifically does, read
> the header of `claude_memory.md` in this folder. The generic porting
> guide below is unchanged and still the reference for adapting to a new
> policy/baseline; treat its "three methods" as the canonical template, and
> see this suite for how to add method variants (parameterize
> `selection/baseline_dagger.py`, add a `p4/pipeline_*.py`, retarget
> `config.yaml` / `workspace.METHOD_DIR_NAMES` / `aggregate.py` /
> `nb_plot.py` / the smoke launcher).

You (a Claude Code session in some *other* folder) are about to rebuild
the demonstration-acquisition comparison that was first built for an
`EquivariantCNNHybridPolicy` on a 5×5 maze. This file is the transferable
knowledge: the experiment design, the pipeline architecture, what to swap
vs reuse, the open-source-model serving recipe, and **every error already
paid for** so you don't re-discover them.

The original, fully-working reference lives at:
`/weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/baseline_vs_p4_sequential_batch/`
— read its `claude_memory.md` for project-internal detail. Copy/adapt from
it; don't import across experiment folders.

---

## 1. The experiment in one paragraph

We train a **policy** by behaviour cloning on a small set of expert
demonstrations, then iteratively add a fixed **budget** of extra
corrective demonstrations and ask: *which demonstration-acquisition
strategy reaches a target held-out success rate with the fewest extra
demos?* It is a **sample-efficiency** comparison. Three strategies share
the exact same per-round failure pool so the comparison is apples-to-apples
within each round.

Production scale (reference): 10 runs × 3 methods, BUDGET=15 extra demos on
top of 20 initial BFS demos, TARGET_SR=0.90 on a 200-layout held-out set,
max_rounds=50.

The headline framing the user wants: **lead with sample-efficiency, state
results honestly, own ties, never overclaim.** (In the reference sweep the
result was a tie: baseline ≈ P4-sequential, P4-batch behind.)

---

## 2. The three methods being compared

All three run the same loop (§3); they differ only in *how the next
corrective demo(s) are chosen*:

1. **baseline (DAgger, highest-loss).** Roll out every layout in the
   round's correction pool, rank failures by per-step BCE loss (descending)
   with `(n_steps, jitter)` tie-break, pick the **single** highest-loss
   episode, save a BFS-correction demo for it, fine-tune. One demo/round.

2. **p4_sequential.** An LLM pipeline (vision + reasoning + aggregator)
   prescribes **exactly one** layout per round — the most informative one —
   with a prompt directive saying "exactly one." One demo/round.

3. **p4_batch.** The same LLM pipeline prescribes **N layouts at once**
   covering all current failure modes, hard-capped to remaining budget.

"P4" = the upstream LLM analysis pipeline (`pipeline/` in the repo:
`vlm_analyser`, `reasoning`, `knowledge_fetcher`/KAG, `aggregator`). It
looks at rendered failure frames + trajectories and prescribes corrective
layouts with a forced "corridor" pathway.

---

## 3. The round loop (the canonical algorithm — confirmed with the user)

This is the design the user explicitly wants; replicate it for every
method. Per **run** (a run = one independent seed):

```
bootstrap: train policy on the initial N demos; eval on held-out set
round r = 1..max_rounds:
    if extras_added >= budget:        -> stop "budget_exhausted"
    sample a FRESH correction pool of `correction_n` layouts
        (blocked from training + held-out + all PRIOR rounds in THIS run)
    roll out the policy on the pool; find the failures
    choose corrective demo(s):
        baseline -> rank by loss, pick top 1
        p4_*     -> LLM prescribes 1 (seq) or many (batch, capped to budget)
    collect BFS/A* corrective demo(s) for the chosen layout(s)
    fine-tune the policy on D_old ∪ D_new (replay buffer, warm-start)
    eval on the held-out set
    if heldout_sr >= target_sr:       -> stop "target_hit"
    if extras_added >= budget:        -> stop "budget_exhausted"
    # a 0-demo round does NOT stop the run (see Gotcha L1)
```

**Critical loop semantics (paid-for lessons):**

- **Per-round pool rotation.** Sample a fresh pool every round (don't reuse
  one fixed pool for the whole run). A fixed pool saturates to 100% in a few
  rounds → `correction_sr=1.0` → no failures → loop stops early with budget
  unspent. Seed the pool deterministically, e.g.
  `91_000_000 + run_id*1000 + round_idx`, blocked from train+heldout+prior
  rounds of the same run. The three methods call the *same* idempotent
  "ensure pool for (run, round)" helper so they share the round's pool.

- **A 0-demo round must NOT terminate the run** (Gotcha L1). Only
  `target_hit`, `budget_exhausted`, or `max_rounds` are terminal. If a round
  collects 0 demos (LLM prescribed nothing usable / pool happened to have no
  failures), continue — the next round samples a fresh pool. Guard against an
  LLM stuck forever with a `max_consecutive_empty` counter (default 8 →
  `no_progress`).

- **Replay-buffer fine-tune, not retrain-from-scratch.** Each round build
  D_old (cumulative minus this round's new demos) and D_new (this round's),
  compose a `WeightedRandomSampler` so each minibatch is ~`replay_mix` from
  D_old and ~`(1-replay_mix)` from D_new, with a `replay_mix_floor` so
  neither side vanishes. Warm-start weights from the last checkpoint
  (optimizer/scheduler NOT restored — match upstream `train.py`).

---

## 4. Repository topology & what to build for a NEW policy

The reference suite is a **self-contained folder** that imports repo-wide
helpers read-only and never edits upstream. For a new policy, create a
sibling suite folder next to that policy's code and mirror this structure:

```
<your_policy>/baseline_vs_p4/<your_suite>/
├── config.yaml                 # ONE source of truth for all knobs
├── submit_one.sh               # SBATCH, one job = one run_id (OpenAI/closed-model cluster)
├── submit_one_qwen.sh          # SBATCH for the open-source (vLLM) cluster — see §6
├── submit_all.sh               # launcher: loops sbatch ${SUBMIT_SCRIPT} per run
├── submit_smoke.sh             # launcher: 2 short jobs for wiring tests
├── submit_aggregate.sh         # SBATCH for cross-run aggregation
├── orchestrator/
│   ├── workspace.py            # per-run path resolver, run-id discovery
│   ├── bootstrap.py            # shared init demos + initial checkpoint (idempotent)
│   └── run_one.py              # CLI entry per run: bootstrap -> methods -> run_summary.json
├── layouts/
│   ├── layout_setup.py         # ensure_correction_layouts_for_round(...) — the rotation primitive
│   └── contamination.py        # cross-run / intra-run overlap audit
├── selection/
│   ├── rank.py                 # (-loss, -n_steps, jitter) ranker
│   └── baseline_dagger.py      # baseline loop  ** see L1: give it the same continue-on-empty fix **
├── p4/
│   ├── prompts.py              # self-awareness + mode directive + infeasible-feedback blocks
│   ├── runner.py               # wraps the upstream LLM pipeline; reads model-name env overrides
│   ├── demo_collector.py       # corridor-aware demo recorder + infeasibility_reason()
│   ├── _p4_common.py           # shared round loop (seq & batch differ only in addendum + cap)
│   ├── pipeline_seq.py         # thin wrapper, mode="sequential"
│   └── pipeline_batch.py       # thin wrapper, mode="batch"
├── corridor/blocker.py         # parse "(r,c)->(r,c)" steps, wall off non-corridor, A* feasibility
├── trainer/finetune_replay.py  # WeightedRandomSampler replay trainer, --resume warm-start
├── logging_ext/                # training_log.csv, compression_log.csv, prescription_overlap.json
├── aggregation/aggregate.py    # contamination check + sr_vs_demos / sr_vs_training figures
├── qwen/proxy.py               # OpenAI-compatible router → two vLLM backends (see §6)
├── nb_plot.py + inspect_results.ipynb   # self-contained plotting (see §9)
└── results/  slurm_logs/  logs/
```

### Policy-specific (SWAP these) vs reusable (KEEP these)

**Swap for a new policy** (the only parts that know the network):
- The policy model + dataset + trainer modules you import (reference imports
  `equivariant_CNN_hybrid.{model,dataset,train}` →
  `EquivariantCNNHybridPolicy`, `HybridDemoDataset`, `collate_fixed_size`).
  Point `trainer/finetune_replay.py` and the rollout/eval helpers at your
  new policy/dataset instead.
- The rollout + eval entry (`rollout_test` equivalent) for your policy.
- `baseline_dagger.py`'s loss computation if your policy's loss differs.

**Keep as-is / copy verbatim** (policy-agnostic):
- The whole round-loop structure & stopping logic (§3).
- Per-round pool rotation + contamination audit.
- The replay-buffer fine-tuning scheme.
- The entire P4 LLM pipeline + corridor blocking + demo collector — **as
  long as the task is still the 5×5 maze.** If you change the *task*, the
  prompts (`prompts.py`), corridor parser, and `vlm_analyser` rendering are
  maze-specific and need rework; the orchestration/serving still transfer.
- All of §6 (open-source serving) and §7 (gotchas) — these are
  infrastructure, fully transferable.

> **Rule the user enforces:** never edit upstream/shared code in place. If
> an upstream helper needs changing, COPY it into your suite and adapt the
> copy. Ask before a non-trivial copy.

---

## 5. The P4 LLM pipeline — what to know

- It is the upstream `pipeline/` package, driven via
  `Equivariant_pathway._analysis_common.run_profile_analysis(...)` with a
  profile yaml (reference uses `p4_vlm_reasoning_kag_cross_plain_llm.yaml`)
  and `extra_overrides`.
- Phases: **B** = per-failure VLM analysis + reasoning + KAG (one set of
  LLM calls per failed episode); **C** = aggregator clusters failures and
  emits the final prescription JSON (`failure_clusters`,
  `demonstration_prescriptions`, `total_demonstrations_needed`).
- Per-episode output carries a `<<<FINAL_REC>>> ... <<<END_FINAL_REC>>>`
  block (corridor, steps, n_demos). The aggregator output is strict JSON.
- Model names come from the profile's `llm.model` / `llm.vlm_model`,
  overridable via `extra_overrides["llm"]`. On the open-source cluster the
  suite injects `LLM_MODEL_NAME` / `VLM_MODEL_NAME` from env.
- **It uses the OpenAI Responses API** (`client.responses.create`, POST
  `/v1/responses`) — NOT chat completions. This matters enormously for
  serving (Gotcha S4).
- The client is built once at import with no `base_url`, so it reads
  `OPENAI_BASE_URL` from env — set that BEFORE importing pipeline modules
  (Gotcha S8).

---

## 6. Serving open-source models (the Qwen cluster) — full recipe

The closed-model path just needs `OPENAI_API_KEY`/`OPENAI_BASE_URL`. The
open-source path serves two local models behind a tiny router so the
pipeline code is byte-identical.

### Cluster facts (Weka / `/weka/s226137394/...`)
- Partitions: `gpu` (a100/40G) and `gpu-large` (h100/80G, h200/141G).
- Conda envs:
  - `maze` — has the pipeline/orchestrator/torch/pandas/matplotlib but **no
    vllm**. This is the env the sbatch activates for the orchestrator+proxy.
  - `vllm_embed` — vllm 0.17.1 (+ install `bitsandbytes`). Use its python as
    `VLLM_PYTHON` for the vLLM servers.
  - `vllm_env` — vllm 0.16.0 (fallback). `vllm_env_new` is broken.
- Models: `/weka/s226137394/models/{Qwen3-32B,Qwen3-VL-32B,...}` (raw bf16
  shards; vLLM bnb-quantizes on the fly).

### Architecture
```
[pipeline] openai.OpenAI(base_url=$OPENAI_BASE_URL).responses.create(model=...)
    │
    ▼  qwen/proxy.py  (CPU, port = base+2)  routes by body["model"]
    ├──→ vLLM Qwen3-VL-32B  on GPU 0  (port base+1)   handles model=qwen3-vl-32b
    └──→ vLLM Qwen3-32B     on GPU 1  (port base+0)   handles everything else
```

### The proxy MUST do four things (each is a paid-for fix — §7 S4/S5/S6)
1. Route `POST /v1/chat/completions`, `POST /v1/completions`, **`POST
   /v1/responses`**, `GET /v1/responses/{id}`, `GET /v1/models`,
   `GET /healthz` — by `body["model"]`.
2. **Inject `detail:"auto"`** on every `input_image` item lacking it
   (openai SDK 2.30-pipeline vs 2.24-vLLM schema drift).
3. **Strip `<think>...</think>`** from `/v1/responses` JSON responses
   (Qwen3 is a reasoning model; think blocks break the aggregator's
   `json.loads`).
4. **Cap `max_output_tokens`** to ~8192 (env `PROXY_MAX_OUTPUT_TOKENS`); the
   pipeline always asks for 16384, which overflows the context.

### The sbatch (`submit_one_qwen.sh`) MUST
- `#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=2` (NOT `--gpus=2`).
- Launch the two vLLM servers with `"${VLLM_PYTHON}"` (not the activated
  `maze` python), each pinned via `CUDA_VISIBLE_DEVICES=0|1`, with
  `--quantization bitsandbytes --load-format bitsandbytes
   --max-model-len 40960 --enforce-eager`.
- Derive **unique per-job ports** from `SLURM_JOB_ID`
  (`20000 + (JOB_ID % 12000)*3`), not hardcoded 8000/8001/8002.
- Set `OPENAI_BASE_URL=http://127.0.0.1:<proxy_port>/v1` and
  `OPENAI_API_KEY=local` before launching the orchestrator.
- Set robustness knobs to avoid the retry death-spiral (§7 S7):
  `export OAI_SDK_TIMEOUT=900 OAI_MAX_IN_FLIGHT=2 OAI_SDK_MAX_RETRIES=2`.
- Probe `/v1/models` on both backends + `/healthz` on the proxy before
  starting; `trap EXIT` to tear everything down.

---

## 7. The complete gotcha catalog (do not re-discover)

### Cluster / SLURM
- **C1 `$SLURM_SUBMIT_DIR`.** SLURM runs the script from a spool dir, so
  `${BASH_SOURCE[0]}` mis-resolves and relative `mkdir` fails with
  Permission denied. Resolve `SCRIPT_DIR` from `$SLURM_SUBMIT_DIR` when set.
- **C2 launchers are `bash`, not `sbatch`.** `submit_all.sh`/`submit_smoke.sh`
  call sbatch themselves; `sbatch`-ing them wastes a slot.
- **C3 `--gpus=N` can split across nodes.** The `.batch` step then sees
  fewer GPUs than requested → second vLLM crashes with
  `NVMLError_InvalidArgument` / "0 active drivers". Use `--nodes=1
  --gpus-per-node=N`. Verify `sacct -j <id> --format=NodeList` is one node.
- **C4 unique ports per job.** Up to 4 jobs co-locate on an 8-GPU node;
  fixed ports → `OSError: [Errno 98] Address already in use`. Derive from
  `SLURM_JOB_ID`.
- **C5 sbatch `--export` truncates comma values.** `METHODS=a,b,c` becomes
  `a`. Encode commas as `+` on send (`${METHODS//,/+}`), decode on receive
  (`${METHODS//+/,}`).
- **C6 aggregate `--constraint`.** Don't carry a closed-cluster constraint
  like `gpu-l40s|gpu-v100` to a cluster without those features → `sbatch:
  error: Invalid feature specification`. Aggregation is CPU-only; drop the
  constraint.

### Conda / deps
- **D1 `maze` has no vllm.** Launch vLLM with `VLLM_PYTHON`
  (`vllm_embed`/`vllm_env`), keep orchestrator+proxy on `maze`. Symptom:
  `ModuleNotFoundError: No module named 'vllm'` in `vllm_*_*.log`.
- **D2 `bitsandbytes`** must be `pip install`ed into the `VLLM_PYTHON` env
  (the `--quantization bitsandbytes` path).

### Serving / API
- **S4 Responses API, not chat.** Pipeline uses `client.responses.create`
  → `/v1/responses`. Proxy must route it or every call 404s and the retry
  wrapper burns ~25 min/call. Symptom: `[vlm-retry] ... 404 {'detail':'Not
  Found'}`.
- **S5 `input_image.detail` schema drift.** openai 2.30 (pipeline) omits
  `detail`; openai 2.24 (vLLM) requires it. Proxy injects `detail:"auto"`.
  Symptom: `400 - 188 validation errors ... Input should be a valid string`.
- **S6 Qwen3 `<think>` blocks break the aggregator.** Strip
  `<think>...</think>` from `/v1/responses` responses at the proxy. Symptom:
  `prescribed=0 ... collected=0` every round despite 200s;
  `full_output.json` `raw_output` starts with `<think>`. (vLLM 0.17.1 has no
  `--reasoning-parser` and Responses API hardcodes chat-template kwargs, so
  request-side `enable_thinking:false` isn't available — strip on response.)
- **S7 retry death-spiral.** Slow generation (enforce-eager + bnb, ~35 tok/s)
  + 16384-token requests + a 120s client timeout → requests time out, the
  SDK retries, but vLLM doesn't cancel the originals → 8–11 pile up, throughput
  per request collapses, more timeouts. One unlucky run can spin for >16h
  while peers finish. Fix: `OAI_SDK_TIMEOUT=900`, `OAI_MAX_IN_FLIGHT=2`,
  `OAI_SDK_MAX_RETRIES=2`, and the `PROXY_MAX_OUTPUT_TOKENS=8192` cap.
  Symptom: hundreds of `Request timed out` in one run's `.out`.
- **S8 set `OPENAI_BASE_URL` before importing the pipeline** (module-level
  client built at import time).
- **S9 vLLM can't hot-swap weights** (60–120s/load). Use two persistent
  servers, not load/unload per call.

### Context-length budget (Qwen3-32B, three knobs that MUST agree)
- **B1 model hard ceiling is 40960** (`max_position_embeddings`). vLLM
  refuses `max_model_len > 40960` (RoPE NaN risk). Set `VLLM_MAX_MODEL_LEN=40960`.
- **B2 default 262144 OOMs the KV cache** (~64 GiB needed vs ~50 free).
- **B3 output must be bounded.** Pipeline asks for 16384 output; cap to 8192
  at the proxy so input has room. (`max_output_tokens > max_model_len` also
  errors if max_model_len is set too low.)
- **B4 input (failure count) must be bounded.** The aggregator concatenates
  one summary per failure; at `correction_n=50` a round hit 23 failures →
  32769 input tokens → overflow even with the 8192 output cap
  (40960−8192=32768, over by 1). **Use `correction_n=20`** (caps failures at
  20). The pool is re-sampled each round, so 20/round still covers broadly.
  This is the real lever — set it in `config.yaml`, `run_one.py` fallbacks,
  and `layout_setup.py`.

### Loop / logic
- **L1 don't stop on a 0-demo round.** Old code returned `no_new_demos` the
  first empty round, leaving budget unspent and the curve truncated —
  handicapping that method. Only `target_hit`/`budget_exhausted`/`max_rounds`
  are terminal; count consecutive empties and bail at a generous cap
  (`no_progress`). **Apply this to baseline_dagger.py too**, not just P4 —
  in the reference sweep only P4 got the fix, so baseline early-stopped in 2
  of 9 runs (a real asymmetry; fix both for a clean comparison).
- **L2 feed infeasible corridors back to the LLM.** When the A*/BFS
  feasibility check rejects a prescribed corridor (steps on fire / not
  4-adjacent / out of bounds / wrong endpoints / disconnected), record
  start/goal/fires/steps + a human-readable reason and inject a "don't
  repeat these" block into the next round's reasoning AND aggregator
  prompts. (See `demo_collector.infeasibility_reason` +
  `prompts.infeasible_feedback_block`.)
- **L3 layout-yaml keys are inconsistent.** Probe `("test_layouts",
  "training_layouts", "heldout_test_layouts", "layouts")` when reading.

---

## 8. Canonical knobs (`config.yaml`, the reference values)

```
budget: 15          target_sr: 0.90      max_rounds: 50      max_steps: 60
heldout_n: 200      initial_demos: 20    initial_epochs: 500 correction_n: 20
seed: 0 (per-run seed = seed + run_id)
methods: [baseline, p4_sequential, p4_batch]
trainer: finetune_epochs 20, lr 5e-5, batch 64, weight_decay 1e-4,
         replay_mix 0.5, replay_mix_floor 0.2, val_frac 0.1
p4.sequential.demos_per_round: 1      p4.batch.demos_per_round: null (LLM decides, capped)
corridor_blocking: true
```
Every knob has an env var → CLI flag on `run_one`; the sbatch only emits a
flag when its env var is set, so unset values inherit `config.yaml`.

Open-source-cluster-only env: `VLLM_PYTHON`, `VLLM_MAX_MODEL_LEN`(=40960),
`PROXY_MAX_OUTPUT_TOKENS`(=8192), `OAI_SDK_TIMEOUT`/`OAI_MAX_IN_FLIGHT`/
`OAI_SDK_MAX_RETRIES`, `LLM_MODEL_PATH/NAME`, `VLM_MODEL_PATH/NAME`,
`VLLM_QUANTIZATION`/`VLLM_LOAD_FORMAT`(=bitsandbytes).

---

## 9. How to run, verify, and inspect

### Run (from the suite dir)
```bash
# wiring smoke (2 short jobs)
SUBMIT_SCRIPT=submit_one_qwen.sh bash submit_smoke.sh
# production sweep + chained aggregation
AGGREGATE_AFTER=1 SUBMIT_SCRIPT=submit_one_qwen.sh bash submit_all.sh
# aggregation by hand (CPU only, from the REPO ROOT for the module path)
python3 -u -m <pkg.path>.aggregation.aggregate --budget 15 --target_sr 0.90
```
Smoke gate is **content, not exit code**: a P4 round must show
`prescribed>=1 ... collected>=1`, proxy must show `POST /v1/responses ...
status=200` for both `qwen3-32b` and `qwen3-vl-32b`, and `vllm_llm_*.log`
must not contain `Address already in use` or the max_model_len error.

### Per-run green-flag checklist
- job `COMPLETED 0:0`, single-node allocation, all proxy calls 200.
- each method's `learning_curve.json` populated; no degenerate p4 (every p4
  method prescribed real demos; sum of `n_new_demos` > 0).
- `stopped_reason` ∈ {target_hit, budget_exhausted, no_progress, max_rounds}.

### Cross-run audit (aggregate.py writes `results/aggregate/`)
- `contamination_report.json`: **`correction_pool_intra_run_overlap` MUST be
  empty** (rotation worked). `demo_cross_run_overlap == initial_demos` (e.g.
  20) is EXPECTED — it's the shared init set, NOT contamination; the report
  may say `status: violation` for this benign reason, so read the specific
  field, not just the status.
- figures: `sr_vs_demos_added.png`, `sr_vs_training_rounds.png`,
  `compression_dist.png`; `summary.json` has per-method final-SR mean±std.

### Inspect interactively
`nb_plot.py` + `inspect_results.ipynb`: every notebook cell is
self-contained (re-imports `nb_plot`, runs one plot). Functions:
`single_run(N)`, `all_runs(METHOD)`, `compare(M1,M2,...)`, and wrappers
`baseline_vs_seq()/seq_vs_batch()/baseline_vs_batch()/compare_all()`. Needs
a kernel with numpy+matplotlib (+pandas for the table); `maze` has them but
needs `ipykernel` installed + registered to appear as a Jupyter kernel.

---

## 10. Things the user cares about (carry these reflexes over)
- Reviewer-defensible framing: lead efficiency, own ties, don't overclaim.
- Terse responses; don't restate the diff.
- The user will push back and explicitly invites disagreement when you have
  a good reason — surface real downsides, don't silently comply.
- Don't run production sweeps casually — that's real compute. Default to
  "I'll write/fix it; you queue the job," and use smoke tests for wiring.
- Don't edit upstream/shared code; copy into the suite and adapt.
- If you want a cleaner Qwen-vs-GPT comparison and faster sweeps, consider
  disabling Qwen thinking outright (`/no_think`) instead of stripping it —
  it removes the token-budget pressure and the death-spiral risk, at the
  cost of testing that `/no_think` is honored on this cluster (untested so
  far). Keep model config identical across all runs of one sweep.
```
