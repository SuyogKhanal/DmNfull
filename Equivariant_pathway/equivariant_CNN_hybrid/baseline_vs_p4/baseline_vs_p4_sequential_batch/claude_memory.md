# Claude memory — baseline_vs_p4_sequential_batch

You (a future Claude Code session) are picking up an in-flight research
project. Read this file end-to-end before suggesting changes. It contains
the user's intent, the design decisions made so far, the constraints we
operate under, and the gotchas we've already paid for.

## The user

- PhD student. Username `s226137394`, repo lives at
  `/vast/s226137394/DmN/DmNfull/` on the primary cluster and
  `/weka/s226137394/DmNfull/` on the open-source-model cluster.
- Comfortable with bash + SLURM + Jupyter; will push back when an idea
  feels wrong; explicitly invites you to disagree when you have good
  reason.
- **Wants reviewer-defensible framing**: results stated honestly, no
  overclaiming. Lead with sample-efficiency, own any accuracy tie.
- Terse responses; don't summarise what they can read in the diff.
- Conda env on both clusters: `maze`
  (`/home/s226137394/.conda/envs/maze/bin/python`).

## The experimental question

Compare three demonstration-acquisition policies on a 5×5 maze
behaviour-cloning task with an `EquivariantCNNHybridPolicy`:

1. **baseline** — DAgger. Each round: roll out every layout in the
   round's correction pool, rank failures by per-step BCE loss
   (descending) with `(n_steps, jitter)` tie-break, pick **one**
   highest-loss episode, save a BFS-correction demo for it, fine-tune,
   eval on heldout.
2. **p4_sequential** — LLM prescribes **one** layout per round (the
   most informative one), via the upstream P4 pipeline (vision +
   reasoning + aggregator) with a prompt directive saying "exactly
   one".
3. **p4_batch** — LLM prescribes N layouts at once covering all current
   failure modes, capped to remaining budget.

For each run, all three methods share the round's correction pool so
the comparison is apples-to-apples within round.

Production sweep: 10 runs × 3 methods, BUDGET=15 extra demos on top of
the initial 20 BFS demos, TARGET_SR=0.90 heldout, max_rounds=50.

The first sweep on the OpenAI cluster completed; the next sweep is
moving to the Qwen open-source cluster.

## Repository topology

Two distinct things live under `Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/`:

- `baseline_vs_p4/` — the **upstream** suite, owned by the broader
  repo. Includes `baseline_budget.py`, `p4_budget.py`, `run_one.py`,
  `aggregate_chart.py`, `swatch.sh`, `layout_setup.py`, the runs from
  prior projects (`runs/`, `runs_baseline_nb/`, etc.). **We do not
  modify anything in here.** We import its helpers read-only.
- `baseline_vs_p4/baseline_vs_p4_sequential_batch/` — **our suite**.
  Everything you build/change lives in here.

If something upstream needs to change, **copy** it into our suite and
adapt the copy. Never edit upstream.

Other repo-wide deps we import read-only (do not copy, do not modify):
- `Equivariant_pathway.layout_sampler` —
  `sample_layouts`, `_load_blocked_signatures`, `_signature`, `write_yaml`.
- `Equivariant_pathway.expert` — `AStarExpert`, `build_grid`,
  `compute_distance_map`, `optimal_action_mask`. Used by the corridor
  blocker.
- `Equivariant_pathway.equivariant_CNN_hybrid.{model, dataset, train}`
  — `EquivariantCNNHybridPolicy`, `HybridDemoDataset`,
  `collate_fixed_size`. Used by the replay trainer.
- `Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.{baseline_budget, p4_budget}`
  — we reuse their lower-level helpers (`_rollout_with_loss`,
  `_save_corrective_demo`, `_eval_heldout`, `_rollout`, `_read_sr`,
  `_count_demos`, `_load_correction_layouts`, `_load_model`,
  `_find_deviation_step`, `REASONING_ADDENDUM_BASE`,
  `AGGREGATOR_ADDENDUM_BASE`, `_budget_addendum`).
- `Equivariant_pathway.collect_demos` — `_build_forced_env`,
  `_record_one`. Used by the P4 corridor-aware demo collector.
- `Equivariant_pathway._analysis_common` — `run_profile_analysis`.
  This is what wires the upstream P4 pipeline; we call it with our own
  `extra_overrides`.
- `pipeline/` — upstream LLM pipeline (`_oai_retry.py`,
  `reasoning.py`, `aggregator.py`, `vlm_analyser.py`, etc.). Uses
  `from openai import OpenAI` with no hardcoded base_url, so vLLM's
  OpenAI-compatible server is a drop-in via `OPENAI_BASE_URL`. Model
  names come from `configs/experiment_config.yaml`'s `llm.model` /
  `llm.vlm_model`, overridable through `extra_overrides`.

## What each module in our suite does

```
baseline_vs_p4_sequential_batch/
├── config.yaml                   # ONE place for all knobs; everything else inherits
├── run_all.sh                    # local-bash launcher (no SLURM, no longer the prod path)
├── submit_one.sh                 # SBATCH for the OpenAI cluster, one job = one run_id
├── submit_one_qwen.sh            # SBATCH for the Qwen cluster (gpus=2, vLLM + proxy)
├── submit_all.sh                 # launcher: for i in 1..N_RUNS; sbatch ${SUBMIT_SCRIPT}
├── submit_smoke.sh               # launcher: submits 2 short jobs (run_id 99, 100)
├── submit_aggregate.sh           # SBATCH for the cross-run aggregation step
├── claude_memory.md              # this file
│
├── orchestrator/
│   ├── workspace.py              # per-run path resolver, mkdir helpers
│   ├── bootstrap.py              # upstream-shared init demos + checkpoint, idempotent
│   └── run_one.py                # CLI entry per run: bootstrap → methods → run_summary.json
│
├── layouts/
│   ├── layout_setup.py           # ensure_correction_layouts_for_round(...) — the rotation primitive
│   └── contamination.py          # cross-run check; reports intra-run and cross-run overlaps separately
│
├── selection/
│   ├── rank.py                   # (-policy_loss, -n_steps, jitter) ranker
│   └── baseline_dagger.py        # baseline run loop; calls finetune_replay subprocess
│
├── p4/
│   ├── prompts.py                # self-awareness block + mode directive on top of upstream addendums
│   ├── runner.py                 # wraps run_profile_analysis; reads LLM_MODEL_NAME / VLM_MODEL_NAME
│   ├── demo_collector.py         # corridor-aware demo recorder using AStarExpert on masked grid
│   ├── _p4_common.py             # shared round-loop body (sequential & batch differ only in addendum + cap)
│   ├── pipeline_seq.py           # thin wrapper, mode="sequential"
│   └── pipeline_batch.py         # thin wrapper, mode="batch"
│
├── corridor/
│   ├── blocker.py                # parse "(r,c)->(r,c)" steps, mask non-corridor FREE cells, AStarExpert
│   └── expert_constrained.py     # re-export of blocker's public helpers
│
├── trainer/
│   └── finetune_replay.py        # WeightedRandomSampler over D_old ∪ D_new, --resume warm-start
│
├── logging_ext/                  # name avoids shadowing stdlib `logging`
│   ├── training_log.py           # CSV: round, demos_added_total, training_rounds_total, heldout_sr, ...
│   ├── compression_log.py        # CSV: round, n_failures_observed, n_layouts_prescribed, ratio, infeasible
│   └── prescription_overlap.py   # JSON: per-round LLM reasoning text (first 3 runs only by default)
│
├── aggregation/
│   └── aggregate.py              # runs contamination check, builds sr_vs_demos + sr_vs_training plots
│
├── qwen/
│   ├── proxy.py                  # FastAPI router 8002 → 8000/8001 by body["model"]
│   └── __init__.py
│
├── results/                      # output root (gitignored)
│   ├── run_{1..N}/
│   │   ├── shared/round_NNN/correction_layouts.yaml   # rotated pool, one yaml per round
│   │   ├── shared/init_demos/                          # copy of upstream shared init
│   │   ├── shared/init_checkpoints/
│   │   ├── baseline/{demos,checkpoints,results}
│   │   ├── p4_sequential/{demos,checkpoints,results}
│   │   ├── p4_batch/{demos,checkpoints,results}
│   │   └── run_summary.json
│   └── aggregate/
│       ├── summary.json
│       ├── contamination_report.json
│       ├── compression_summary.csv
│       └── figures/{sr_vs_demos_added.png, sr_vs_training_rounds.png, compression_dist.png}
│
├── slurm_logs/                   # SBATCH --output / --error land here
└── logs/                         # per-run mirror of orchestrator stdout via tee
```

## Key design decisions and why

### Per-round correction-pool rotation

**Problem solved:** Originally the correction pool was sampled once per
run and reused every round. The policy hit 100% on that fixed pool
within ~5 rounds → `correction_sr = 1.0` → no failures → loop stopped
with `no_new_demos` even though the policy was nowhere near 100% on the
broader layout space. The 15-demo budget was never exhausted.

**Fix:** At the top of every round, sample 50 fresh layouts (default
`correction_n=50`) blocked from training (20) + heldout (200) + every
prior round in **this** run. Within one round, baseline /
p4_sequential / p4_batch all call
`ensure_correction_layouts_for_round(run_id, round_idx, ...)` with the
same `(run_id, round_idx)`; the function is idempotent so the
second/third caller reads the cached yaml → apples-to-apples comparison.

**Seed:** `91_000_000 + run_id * 1000 + round_idx`. Distinct across all
(run, round) pairs in any realistic sweep.

**File layout:**
`results/run_{id}/shared/round_{NNN:03d}/correction_layouts.yaml`. Each
round also writes a `layout_setup_report.json` next to its yaml.

**Verification primitive:**
`layouts.contamination.cross_run_check` walks every per-round pool and
reports `correction_pool_intra_run_overlap` (must be empty if the
blocked-set logic worked) and `correction_pool_cross_run_overlap`
(visibility only — rounds at the same index in two different runs may
collide by chance; we surface it but don't fail).

### Replay-buffer fine-tuning (not retrain-from-scratch)

`trainer/finetune_replay.py` is the only trainer in the suite.
Subprocess CLI mirroring upstream `train.py`. It builds **two**
`HybridDemoDataset` instances (D_old = cumulative pool minus new demos,
D_new = just this round's new demos), then composes a
`WeightedRandomSampler` so each minibatch is in expectation:

```
~ replay_mix * batch_size       samples drawn from D_old
~ (1 - replay_mix) * batch_size samples drawn from D_new
```

`replay_mix_floor` clamps both sides so neither vanishes (default 0.2;
target mix default 0.5). `--resume` warm-starts model weights from
`last_/best_hybrid_policy.pth` — optimizer/scheduler state are NOT
restored (upstream `train.py` is the same; we match its semantics).

### P4 loop: continue-on-empty + infeasible-corridor feedback (2026-05-22)

Two coupled changes in `p4/_p4_common.py` (+ `p4/prompts.py`,
`p4/demo_collector.py`), prompted by run_7 of the first real Qwen sweep
where p4_sequential stopped at `no_new_demos` with 13/15 budget unused
and heldout 0.615:

1. **A 0-demo round no longer ends the run.** Old code returned
   `stopped_reason="no_new_demos"` the first time a round collected 0
   demos. That was a fixed-pool-era assumption; with per-round rotation a
   0-demo round just means that round's prescription was infeasible/empty.
   Now the loop only terminates on `target_hit`, `budget_exhausted`, or
   `max_rounds`; a 0-demo round increments `consecutive_empty` and only
   bails after `max_consecutive_empty` (config key, default 8) in a row
   (`stopped_reason="no_progress"`). Intended design: keep sampling fresh
   pools until target or budget.
2. **Infeasible corridors are fed back to the LLM.** When the A*/BFS
   feasibility checker rejects a prescribed corridor, we record
   start/goal/fires/steps + a human reason
   (`demo_collector.infeasibility_reason`: on-fire, non-adjacent,
   out-of-bounds, wrong endpoints, or disconnected). The running
   `infeasible_memo` is rendered by `prompts.infeasible_feedback_block`
   and appended to BOTH reasoning and aggregator addendums in subsequent
   rounds ("you prescribed these infeasible pathways; don't repeat").
   The `reason` is also persisted in each round's `collect_summary.json`.

### Highest-loss selection rule (baseline)

`selection/rank.py::rank_failures` sorts by `(-policy_loss, -n_steps,
jitter)`. Baseline picks exactly one demo per round (top of the rank).
The ranking is persisted to
`results/run_{id}/baseline/results/round_NNN/ranking.json` so you can
audit which episode was chosen and what runners-up looked like.

### Corridor blocking

When the LLM prescribes a `steps: (r,c)->(r,c)->...` pathway,
`corridor/blocker.py` walls off every FREE cell *outside* the corridor
before building the A* expert. The constrained expert is then forced to
walk exactly the prescribed path. Infeasible corridors (e.g. step
through fire / disconnected) are logged in `compression_log.csv` as
`n_corridor_infeasible` and the round records zero demos collected.
Toggle via `CORRIDOR_BLOCKING=true|false` (default true).

Self-test passes: `python -m
…baseline_vs_p4_sequential_batch.corridor.blocker`.

### Qwen serving (two vLLM servers + tiny proxy)

The user explicitly chose `--gpus=2` after we discussed memory and
swap costs. Reasoning:

- Qwen3-32B at 8-bit ≈ 32 GB weights + ~10 GB activations + KV cache.
  Qwen3-VL-32B is roughly the same. Two persistent vLLM servers on
  one H100 (80 GB) would be tight; on H200 (141 GB) comfortable.
  Two GPUs sidesteps the question.
- Hot-swapping is not viable: vLLM is a persistent server; killing and
  restarting takes 60-120 s per model swap; the upstream pipeline
  interleaves VLM and LLM calls per failure → swap-per-call would cost
  hours per round.

Architecture:

```
[upstream pipeline]
    │ openai.OpenAI(base_url=$OPENAI_BASE_URL).chat.completions.create(model=...)
    ▼
qwen/proxy.py  (port 8002, CPU)
    │ routes by body["model"]
    ├──→ port 8001  vLLM serving Qwen3-VL-32B on GPU 0   (handles vlm_analyser)
    └──→ port 8000  vLLM serving Qwen3-32B    on GPU 1   (handles reasoning + aggregator + plain LLM)
```

The pipeline asks for `model=qwen3-32b` or `model=qwen3-vl-32b`; the
proxy matches and forwards transparently (streams response, headers
preserved). `p4/runner.py` reads `LLM_MODEL_NAME` and `VLM_MODEL_NAME`
from env and injects them into `extra_overrides["llm"]` so the upstream
config defaults get overridden cleanly.

`submit_one_qwen.sh` starts both vLLM servers in background, starts the
proxy, probes `/v1/models` on all three with a 10-min timeout, then
launches the same `orchestrator.run_one` the OpenAI cluster uses. `trap
EXIT` tears everything down.

## Knobs

`config.yaml` holds the canonical defaults. Every env var below maps to
a CLI flag on `orchestrator.run_one`; `submit_one.sh` and
`submit_one_qwen.sh` translate env vars → CLI flags only when the var
is actually set, so unset values inherit from `config.yaml`.

| env var                  | CLI flag on run_one              | config.yaml key                    |
|--------------------------|----------------------------------|------------------------------------|
| `RUN_ID`                 | `--run_id` (required)            | n/a                                |
| `CONFIG`                 | `--config`                       | n/a                                |
| `METHODS`                | `--methods`                      | `methods`                          |
| `BUDGET`                 | `--budget`                       | `budget`                           |
| `TARGET_SR`              | `--target_sr`                    | `target_sr`                        |
| `CORRECTION_N`           | `--correction_n`                 | `correction_n`                     |
| `HELDOUT_N`              | `--heldout_n`                    | `heldout_n`                        |
| `INITIAL_DEMOS`          | `--initial_demos`                | `initial_demos`                    |
| `INITIAL_EPOCHS`         | `--initial_epochs`               | `initial_epochs`                   |
| `MAX_ROUNDS`             | `--max_rounds`                   | `max_rounds`                       |
| `MAX_STEPS`              | `--max_steps`                    | `max_steps`                        |
| `SEED`                   | `--seed`                         | `seed` (per-run seed = SEED+RUN_ID)|
| `BASELINE_ROUND_EPOCHS`  | `--baseline_finetune_epochs`     | `baseline_finetune_epochs`         |
| `ROUND_EPOCHS`           | `--p4_finetune_epochs`           | `p4_finetune_epochs`               |
| (none, shared default)   | `--finetune_epochs`              | `trainer.finetune_epochs`          |
| `LR`                     | `--lr`                           | `trainer.lr`                       |
| `BATCH_SIZE`             | `--batch_size`                   | `trainer.batch_size`               |
| `WEIGHT_DECAY`           | `--weight_decay`                 | `trainer.weight_decay`             |
| `REPLAY_MIX`             | `--replay_mix`                   | `trainer.replay_mix`               |
| `REPLAY_MIX_FLOOR`       | `--replay_mix_floor`             | `trainer.replay_mix_floor`         |
| `CORRIDOR_BLOCKING`      | `--corridor_blocking`            | `corridor_blocking`                |

`submit_all.sh` extras: `N_RUNS` (default 10), `SUBMIT_SCRIPT`
(default `submit_one.sh`), `AGGREGATE_AFTER` (set to `1` to chain a
dependent aggregation job).

Qwen-cluster-only env vars (read by `submit_one_qwen.sh`):
`LLM_MODEL_PATH`, `VLM_MODEL_PATH`, `LLM_MODEL_NAME`, `VLM_MODEL_NAME`,
`VLLM_LLM_PORT`, `VLLM_VLM_PORT`, `PROXY_PORT`, `VLLM_QUANTIZATION`,
`VLLM_LOAD_FORMAT`, `VLLM_GPU_MEM_UTIL`, `VLLM_DTYPE`,
`VLLM_READY_TIMEOUT_SEC`, `VLLM_EXTRA_FLAGS`, `QWEN_CONDA_ENV`.

## How to run

All commands assume you're in
`…/baseline_vs_p4_sequential_batch/`.

```bash
# === Smoke test (2 jobs, all 3 methods, tiny budget) ===
bash submit_smoke.sh                                   # OpenAI cluster
SUBMIT_SCRIPT=submit_one_qwen.sh bash submit_smoke.sh  # Qwen cluster

# === Production sweep (10 runs × 3 methods) ===
bash submit_all.sh                                     # OpenAI cluster
SUBMIT_SCRIPT=submit_one_qwen.sh bash submit_all.sh    # Qwen cluster

# === Production sweep with auto-aggregation chained on afterok ===
AGGREGATE_AFTER=1 bash submit_all.sh

# === Aggregation by hand (after a sweep finishes) ===
python3 -u -m \
  Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.baseline_vs_p4_sequential_batch.aggregation.aggregate \
  --budget 15 --target_sr 0.90
# Or via SLURM:
sbatch submit_aggregate.sh

# === Override knobs for a one-off sweep ===
BUDGET=20 CORRECTION_N=60 BASELINE_ROUND_EPOCHS=80 ROUND_EPOCHS=300 \
bash submit_all.sh
```

**Important:** `submit_smoke.sh`, `submit_all.sh`, and
`submit_aggregate.sh` (the chained version) are **launchers**, not job
scripts. The intended invocation is `bash …`, not `sbatch …`. They were
made robust to either after a user hit the SLURM-spool-dir issue (see
gotchas), but `bash` is the cleaner path.

## Verification after every sweep

```bash
# (a) Per-round pools actually differ (per-run rotation worked)
ls results/run_99/shared/round_*/correction_layouts.yaml | wc -l
python3 -c "
import yaml, hashlib, pathlib
sigs = [hashlib.sha256(open(p, 'rb').read()).hexdigest()[:10]
        for p in sorted(pathlib.Path('results/run_99/shared').glob('round_*/correction_layouts.yaml'))]
print('unique pool hashes:', len(set(sigs)), '/', len(sigs))
"

# (b) Each method fine-tuned at least once
for m in baseline p4_sequential p4_batch; do
  awk -F, 'NR==1{next} END{print FILENAME, $3}' \
    results/run_99/$m/results/training_log.csv
done

# (c) Contamination report — intra-run MUST be empty
python3 -u -m \
  Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.baseline_vs_p4_sequential_batch.aggregation.aggregate \
  --budget 15 --target_sr 0.90
python3 -m json.tool < results/aggregate/contamination_report.json
# Expect: status="ok", correction_pool_intra_run_overlap = []

# (d) Saturation regression: read history; correction_sr can be 1.0 in
#     a single round (the pool was fresh) but stopped_reason must NOT
#     be "no_new_demos" while extra_demos < budget.
```

The aggregation produces three figures under
`results/aggregate/figures/` and a `summary.json` with per-(run, method)
final SR + extras-to-target + stopped_reason.

## Gotchas (already paid for — do not re-discover)

1. **SLURM working directory.** SLURM copies the submitted script into
   `/var/spool/slurm/d/jobNNNN/slurm_script` and runs it from there;
   `${BASH_SOURCE[0]}` resolves to the spool dir, so any `mkdir`
   relative to that fails with `Permission denied`. **Always prefer
   `$SLURM_SUBMIT_DIR`** (set by SLURM to the dir sbatch was invoked
   from) when resolving the script dir:
   ```bash
   if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
       SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
   else
       SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   fi
   ```
   Applied to `submit_one.sh`, `submit_one_qwen.sh`,
   `submit_aggregate.sh`, `submit_smoke.sh`, `submit_all.sh`.

2. **Don't `sbatch` the launchers.** `submit_smoke.sh`, `submit_all.sh`
   are launchers — they call `sbatch` for the real job script
   (`submit_one*.sh`). Running `sbatch submit_smoke.sh` enqueues the
   launcher itself as a job (with no SBATCH header it ends up using
   account defaults), wasting a slot. Use `bash submit_smoke.sh` from
   the login node.

3. **Layout yaml top-level keys are inconsistent.** `write_yaml` writes
   under `heldout_test_layouts`. `_load_correction_layouts` and
   `_signatures_from_yaml` try multiple keys in order:
   `("test_layouts", "training_layouts", "heldout_test_layouts",
   "layouts")`. If you're reading these yamls directly, probe all four
   — don't assume one.

4. **vLLM is persistent.** It cannot hot-swap weights. Don't propose
   "load model A, do call, unload, load model B" — the load takes
   60-120s per model. Either two persistent servers (current design),
   one model handling everything, or batched-then-swapped (requires
   pipeline restructuring).

5. **The pipeline's `_CLIENT` is module-level.** Upstream constructs
   `openai.OpenAI()` once at import time with no `base_url` arg, so it
   reads `OPENAI_BASE_URL` from env. That env var must be set **before**
   the first import of upstream pipeline modules. The Qwen sbatch sets
   it before invoking python -m run_one, which is correct.

6. **Demo overlaps across runs are expected for `p4_batch` in the
   first sweep.** That sweep was collected before per-round rotation;
   all rounds shared one correction pool, so every run's p4_batch
   demos overlap heavily with every other run's p4_batch demos. Don't
   panic — the new code's rotation fixes this for future sweeps.

7. **Two `_persist_curve` functions exist** —
   `selection.baseline_dagger._persist_curve` and
   `p4._p4_common._persist_curve`. Both write
   `learning_curve.json` with the same schema (`history` list,
   per-row `correction_yaml` path, `correction_yaml_dir` at the top
   pointing at the run's shared dir).

8. **`maze` env does not have vllm.** The Qwen sbatch activates the
   `maze` conda env (correct — that's where the orchestrator/pipeline
   code lives) but the two vLLM server launches need a separate
   interpreter. `submit_one_qwen.sh` reads `VLLM_PYTHON`, default
   `/home/s226137394/.conda/envs/vllm_embed/bin/python` (vllm 0.17.1).
   `vllm_env` (vllm 0.16.0) is an alternative. Symptom when this is
   wrong: `vllm_llm_*.log` and `vllm_vlm_*.log` contain
   `ModuleNotFoundError: No module named 'vllm'` and the job dies at
   the 600s readiness probe.

9. **`bitsandbytes` must be installed in `VLLM_PYTHON`'s env.** Script
   passes `--quantization bitsandbytes --load-format bitsandbytes` and
   the model dirs hold raw bf16 shards; vLLM does on-the-fly 8-bit
   quant. Install: `/home/s226137394/.conda/envs/vllm_embed/bin/pip
   install bitsandbytes`.

10. **`--gpus=2` can split across nodes; use `--nodes=1
    --gpus-per-node=2` instead.** With the bare `--gpus=2` directive on
    the `gpu-large` partition, SLURM happily allocates the two GPUs on
    two different nodes (e.g. h100-m-01 + h100-m-03), but the `.batch`
    step runs on only one node and sees only ONE GPU. Symptom: VLM
    (CVD=0) loads fine, LLM (CVD=1) crashes with
    `vllm.third_party.pynvml.NVMLError_InvalidArgument` during model
    architecture inspection plus an earlier "0 active driver(s) found"
    triton notice. Verify via `sacct -j <jobid> --format=NodeList` —
    `NodeList` should be a single node.

11. **vLLM default `max_model_len` is 262144 for Qwen3.** That needs
    ~64 GiB KV cache per server — exceeds the ~50 GiB free after the
    bnb-quantized 20 GiB model fits in 80 GB H100. Cap via
    `VLLM_MAX_MODEL_LEN` (default **65536** in `submit_one_qwen.sh`).
    Three failure modes seen while tuning this:
    - Too high (262144): VLM log `ValueError: ... max seq len (262144),
      64.0 GiB KV cache needed > available 49.49 GiB`.
    - Too low for output (8192): `max_output_tokens=16384 cannot be
      greater than max_model_len=8192` (reasoning step wants 16384 out).
    - Too low for input+output (32768): the **aggregator** at
      production `correction_n=50` sends ~16k input and requests 16384
      output → `400 ... You passed 16385 input tokens and requested
      16384 output tokens ... context length is only 32768`. The
      aggregator prompt scales with failure count, so smoke
      (correction_n≈8) never triggered it; only the full sweep did.
    - Too high (65536): vLLM REFUSES to start —
      `User-specified max_model_len (65536) is greater than the derived
      max_model_len (max_position_embeddings=40960)`. Qwen3-32B's hard
      ceiling is **40960**; can't exceed it without
      `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (risks RoPE NaN — don't).
    **Resolution (3 knobs together):**
    1. `VLLM_MAX_MODEL_LEN=40960` (the model's hard max).
    2. Proxy caps `max_output_tokens` to 8192 (env
       `PROXY_MAX_OUTPUT_TOKENS`) — the pipeline always asks for 16384.
    3. `correction_n=20` (NOT 50) — this is the real lever. The
       aggregator concatenates one summary per failure; at
       correction_n=50 a round had 23 failures → 32769 input tokens,
       which overflowed even with the 8192 output cap (32769 > 40960-8192
       = 32768, over by 1). correction_n=20 caps failures at 20 → ~28k
       input worst case, fits comfortably. Hardcoded to 20 in
       `config.yaml`, `orchestrator/run_one.py` (3 fallbacks), and
       `layouts/layout_setup.py`. The per-round pool is freshly sampled,
       so 20/round still covers broad layout space across the run.
    All three are load-bearing: model caps at 40960, output must be
    bounded, and the input (failure count) must be bounded too.

12. **Upstream pipeline uses the Responses API, not Chat Completions.**
    `pipeline/vlm_analyser.py`, `pipeline/aggregator.py`,
    `pipeline/reasoning.py`, and `pipeline/knowledge_fetcher.py` all
    call `client.responses.create(...)`, which POSTs to
    `/v1/responses`. The Qwen proxy MUST route this path (it does, as
    of the fix on 2026-05-21). If the proxy is missing the route, every
    LLM/VLM call returns FastAPI's default `404 {"detail": "Not Found"}`
    and the upstream `_oai_retry` wrapper interprets it as a transient
    error, burning 13 retries × ~120s ≈ 25 min per call before giving
    up. vLLM 0.17.1 serves `/v1/responses` on the backend; the proxy
    just needs to forward.

14. **`sbatch --export` truncates METHODS at commas.** `sbatch
    --export=ALL,METHODS=baseline,p4_sequential,...` parses each comma
    as a variable separator — so METHODS becomes just `baseline` and
    `p4_sequential` is interpreted as an empty `p4_sequential=`
    variable. `submit_smoke.sh` and `submit_all.sh` now encode commas
    in METHODS as `+` on the way out; `submit_one.sh` and
    `submit_one_qwen.sh` decode `+` back to `,` on the way in. Symptom
    before the fix: `run_summary.json` lists fewer methods than the
    user requested, and the CLI line in the .out log shows only the
    first one.

15. **Qwen3 is a reasoning model — `<think>` blocks break the
    aggregator.** Qwen3-32B/Qwen3-VL-32B emit `<think>...</think>`
    inline in their response text. The upstream aggregator
    (`pipeline/aggregator.py` ~line 446) does a strict `json.loads`
    after only stripping ```` ```json ```` fences; a leading `<think>`
    makes the parse fail → it falls back to `{"raw_output": raw}` → no
    `failure_clusters` → `total_demonstrations_needed: null` →
    **0 layouts prescribed → 0 demos → flat heldout_sr**. The
    per-episode FINAL_REC regex happens to survive (it finds the block
    after the think), but the Phase-C aggregator does not. GPT on the
    OpenAI cluster never emitted think tokens, so this only bites the
    Qwen path. Symptom: `prescribed=0 kept=0 collected=0` every round
    despite all proxy calls returning 200, and
    `phase_c.parsed_prescription.raw_output` in `full_output.json`
    starting with `<think>`. Fix: `qwen/proxy.py` strips
    `<think>...</think>` from `/v1/responses` JSON responses before
    returning (`_strip_think_recursive`). vLLM 0.17.1 has no reasoning
    parser available and the Responses API hardcodes its chat-template
    kwargs (protocol.py:271), so neither a `--reasoning-parser` flag
    nor a request-body `chat_template_kwargs:{enable_thinking:false}`
    is available — response-side stripping is the working fix.
    Alternative if a cleaner GPT-comparable comparison is wanted:
    disable thinking via `/no_think` in prompts (untested on this
    cluster).

16. **vLLM/proxy ports must be unique per job, not hardcoded.** A
    `gpu-large` node has 8 GPUs; each job takes 2 (`--gpus-per-node=2`),
    so up to 4 jobs co-locate on one node. Hardcoded ports
    8000/8001/8002 collide across co-located jobs and with orphaned
    vLLM servers from cancelled jobs → `OSError: [Errno 98] Address
    already in use` in `vllm_llm_*.log`, server never binds, job spins
    doing nothing until cancelled. `submit_one_qwen.sh` now derives a
    per-job 3-port block from `SLURM_JOB_ID`:
    `_PORT_BASE=20000 + (JOB_ID % 12000)*3`, ports = base, base+1,
    base+2. Job ids are unique & monotonic so co-located/sequential
    jobs never share a block. Still env-overridable. Don't revert to
    fixed ports — it only "works" when jobs happen not to co-locate
    (which is luck, not correctness).

13. **openai SDK 2.24 vs 2.30 schema drift on `input_image.detail`.**
    The `maze` env has openai 2.30 (used by the pipeline); the
    `vllm_embed` env has openai 2.24 (used by vLLM's pydantic
    validation). In 2.24, `ResponseInputImageParam.detail` is a
    `Required[Literal["low","high","auto"]]`. In 2.30 it added
    "original" and the pipeline omits the field entirely (OpenAI's
    hosted API defaults to "auto"). vLLM rejects with `400 - 188
    validation errors ... 'Input should be a valid string'` (the union
    cascade — first it tries `input: str`, then each variant of the
    list type, all failing because of missing `detail`). The proxy
    (`qwen/proxy.py::_patch_responses_payload`) now injects `detail:
    "auto"` on every `input_image` content item that lacks one before
    forwarding. Don't remove this patch unless both envs are upgraded
    to compatible openai SDK versions.

## Open items / what's NOT built yet

1. **`inspect_results.ipynb`** — the user asked whether I was planning
   to create one. I said yes but deferred to make room for the
   rotation + Qwen-cluster work. When the user comes back to this,
   build a notebook under
   `…/baseline_vs_p4_sequential_batch/inspect_results.ipynb` that:
   - Loads all `run_*/<method>/results/learning_curve.json` and
     `training_log.csv` + `compression_log.csv`.
   - Reads `aggregate/contamination_report.json` and prints a one-line
     PASS/FAIL banner.
   - Renders the headline curves inline (sr vs demos_added, sr vs
     training_rounds) with mean ± std and per-run thin lines.
   - Has a `EXCLUDE_RUNS = {2}`-style filter that re-plots without
     re-running anything.
   - Drills into per-run ranking + per-round LLM reasoning (from
     `prescription_overlap.json`) for the first 3 runs.
   - Computes per-method "extras_to_target" stats + a paired Wilcoxon
     comparing the methods.
   - The mental model: `aggregate.py` is the headless version of the
     same numbers; the notebook is where the user iterates on
     framing/filters.

2. **Qwen cluster smoke has not run end-to-end yet.** As of this
   writing, the user just hit a SLURM cwd bug submitting the smoke job
   on the Qwen cluster; the bug is fixed (item 1 above), the smoke is
   being re-queued.

3. **Paper-framing helper.** The user has memory entries indicating
   they want results stated honestly (lead efficiency, own ties). A
   small helper inside the notebook to auto-print the framing diff
   would be useful but isn't built.

## Things the next Claude should NOT do

1. **Do not edit anything outside `baseline_vs_p4_sequential_batch/`.**
   If something upstream needs to change, **copy** it into our suite
   and adapt the copy. Ask the user first if the copy is non-trivial.
2. **Do not introduce backwards-compat shims.** When you change a
   function signature, just change the callers. The suite is internal;
   we're not maintaining stable APIs.
3. **Do not add tests, docstrings, or comments that weren't asked
   for.** The user values terse code. Add a comment only when the WHY
   is non-obvious.
4. **Do not silently swallow user pushback.** When the user proposes
   something that has real downsides (e.g. "swap models per call"),
   say so clearly and offer alternatives. The user explicitly invited
   this.
5. **Do not run production sweeps from a Claude session.** That burns
   real compute. Smoke tests on small budgets are fine if the user
   asks. Always default to "I'll write/fix it; you queue the job."

## User memory snippets (from `~/.claude/projects/.../memory/MEMORY.md`)

- "User wants honest reviewer-defensible framing."
- "run_02 was degenerate in the prior project; user has paired-test
  framing reflexes from that experience."
- "Canonical baseline notebook is `baseline_only_notebook_v2.ipynb`"
  (that's an artifact of a different earlier project; not directly
  relevant here but explains why the user thinks in terms of notebook
  inspection cells).

## File-finder reference (the things you'll touch most)

- Per-round pool sampling: [layouts/layout_setup.py](layouts/layout_setup.py)
- Cross-run contamination: [layouts/contamination.py](layouts/contamination.py)
- Replay trainer (subprocess): [trainer/finetune_replay.py](trainer/finetune_replay.py)
- Baseline DAgger round loop: [selection/baseline_dagger.py](selection/baseline_dagger.py)
- P4 shared round loop: [p4/_p4_common.py](p4/_p4_common.py)
- P4 LLM wrapper + model-name overrides: [p4/runner.py](p4/runner.py)
- Self-awareness prompts: [p4/prompts.py](p4/prompts.py)
- Corridor blocking: [corridor/blocker.py](corridor/blocker.py)
- Per-run orchestration: [orchestrator/run_one.py](orchestrator/run_one.py)
- Aggregation + figures: [aggregation/aggregate.py](aggregation/aggregate.py)
- Qwen proxy: [qwen/proxy.py](qwen/proxy.py)
- SBATCH (OpenAI): [submit_one.sh](submit_one.sh)
- SBATCH (Qwen): [submit_one_qwen.sh](submit_one_qwen.sh)
- Launcher: [submit_all.sh](submit_all.sh)
- Smoke launcher: [submit_smoke.sh](submit_smoke.sh)
- Single config: [config.yaml](config.yaml)
