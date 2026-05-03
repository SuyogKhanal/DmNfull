# DmN — LLM-guided DAgger for a 5×5 dynamic maze

A vision-conditioned diffusion policy is trained to navigate a randomised 5×5
maze from a randomised start to a randomised goal while avoiding randomised
fire tiles. When the policy fails, **an LLM pipeline (VLM → KAG → RAG → Reasoning
→ TKF → Aggregator)** analyses the failures and prescribes specific maze
**layouts** (start, goal, fire positions) the human (or a BFS-based rule-based
expert) should demonstrate next. The corrective demos go back into training,
the model retrains on the *aggregated* dataset, and the loop repeats until the
target success rate is reached.

The end-goal is to compare profiles **p1 → p6** (each adding one more
component on top of the previous) on a *fair footing*: same baseline
checkpoint, same baseline demos, same ablation seed, isolated per-profile RAG
banks. **The hypothesis** — which this codebase is built to test — is that one
LLM-driven pass that looks at *all* failures together prescribes a smaller,
better targeted demo set than DAgger's per-failure interventions.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [The 6 ablation profiles](#the-6-ablation-profiles)
- [Running individual components](#running-individual-components)
  - [Train the diffusion policy](#train-the-diffusion-policy)
  - [Record demos with pygame](#record-demos-with-pygame)
  - [Rule-based BFS expert](#rule-based-bfs-expert)
  - [Run one profile pipeline (Phase A + B + C)](#run-one-profile-pipeline-phase-a--b--c)
  - [Run the full ablation suite (all profiles, one rollout)](#run-the-full-ablation-suite-all-profiles-one-rollout)
  - [Active loop (interactive)](#active-loop-interactive)
  - [One-shot round (headless)](#one-shot-round-headless)
  - [Baselines (naive DAgger / diff-DAgger)](#baselines-naive-dagger--diff-dagger)
  - [Dashboard](#dashboard)
- [Headless ablation studies on Slurm](#headless-ablation-studies-on-slurm)
- [Statistical comparison: McNemar + paired t-test (P4 / P5 / P6)](#statistical-comparison-mcnemar--paired-t-test-p4--p5--p6)
- [Email notifications via Gmail API](#email-notifications-via-gmail-api)
- [What gets saved where](#what-gets-saved-where)
- [Glossary of components](#glossary-of-components)

---

## Repository layout

```
DmNfull/
├── baselines/              # naive_dagger.py, diff_dagger.py — comparison baselines
├── checkpoints/            # best_model.pth, best_model_ema.pth, baseline.pth (you create this)
├── configs/
│   ├── experiment_config.yaml      # master config (rollout / pipeline / kag / rag / tkf / llm / tracking)
│   ├── maze_layouts.py             # named maze grids ("multimodal" is the primary)
│   └── ablation_profiles/
│       ├── p1_vlm_plain_llm.yaml
│       ├── p2_vlm_reasoning_plain_llm.yaml
│       ├── p3_vlm_reasoning_cross_plain_llm.yaml
│       ├── p4_vlm_reasoning_kag_cross_plain_llm.yaml
│       ├── p5_vlm_reasoning_kag_rag_cross_plain_llm.yaml
│       ├── p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm.yaml
│       └── ... (legacy profiles for one-off experiments)
├── dashboard/              # Gradio UI for inspecting a saved suite run
├── demos/                  # Baseline demos (top-level *.json) + per-profile collected demos
│   ├── *.json              # baseline (read-only by active_loop)
│   └── active_loop/<profile>/<loop_id>/round_<N>/*.json
├── envs/                   # MazeNavEnv (gym-style)
├── knowledge/              # kag_maze_knowledge.json
├── model/                  # diffusion_policy.py
├── pipeline/
│   ├── pipeline_runner.py          # full run: rollout (Phase A) + per-failure (B) + aggregator (C)
│   ├── rollout.py                  # Phase A
│   ├── vlm_analyser.py             # VLM (cached per (rollout_id, episode_id))
│   ├── kag_loader.py               # KAG knowledge graph
│   ├── rag_bank.py                 # FAISS-backed RAG, owner_run_id-isolated
│   ├── reasoning.py                # analysis + grounded prescription (FINAL_REC)
│   ├── knowledge_fetcher.py        # TKF (top-k demo coverage)
│   ├── aggregator.py               # cross-episode reasoning + structured prescription + recommended_layouts
│   └── response_cache.py           # disk cache for VLM only
├── results/
│   ├── runs/                       # one-off run_pipeline outputs
│   ├── ablations/<run_id>/         # full suite outputs (p1..p6)
│   └── active_loop/<profile>/      # active-loop logs and per-round artefacts
├── scripts/
│   ├── train_diffusion.py          # trainer (tqdm-instrumented)
│   ├── play_maze.py                # interactive demo recorder + layout-queue driver
│   ├── rule_based_collector.py     # headless BFS expert demo collector
│   ├── run_pipeline.py             # one profile, one rollout, full pipeline
│   ├── run_ablation_suite.py       # ALL profiles over one shared rollout
│   ├── run_round.py                # ONE round of the active loop, headless (slurm-friendly)
│   ├── active_loop.py              # interactive multi-round driver (human or bfs expert)
│   ├── run_baseline.py             # naive DAgger / diff-DAgger
│   ├── notify.py                   # Gmail API notifier
│   └── slurm/
│       └── round.sh                # sbatch template for headless rounds
├── credentials.json        # (you provide) Google OAuth client for Gmail notifier
├── token.json              # (auto-created by `python scripts/notify.py --setup`)
├── environment.yml         # conda env spec
└── README.md               # this file
```

---

## Setup

### 1. Conda environment

```bash
conda env create -f environment.yml
conda activate maze
```

The env pulls in PyTorch + CUDA, `transformers`, `faiss-cpu`, `crewai`,
`gradio`, `pygame`, `opencv-python`, `tqdm`, plus the Google API libs
(`google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`)
needed for email notifications.

### 2. OpenAI API key

Create `.env` in the project root:

```dotenv
OPENAI_API_KEY=sk-...
```

Every script loads `.env` via `python-dotenv`. The pipeline uses the model
configured under `configs/experiment_config.yaml -> llm.model`; change there
if you want a different OpenAI model.

### 3. Pin a baseline checkpoint (one time, before any active loops)

After your initial training on the seed `demos/*.json`:

```bash
cp checkpoints/best_model.pth checkpoints/baseline.pth
# and the EMA, if you have it:
cp checkpoints/best_model_ema.pth checkpoints/baseline_ema.pth
```

Every active-loop run starts by copying `baseline.pth → best_model.pth`, so
profiles p1..p6 are guaranteed to begin from identical weights.

### 4. (Optional) Gmail notifier setup

See [Email notifications via Gmail API](#email-notifications-via-gmail-api).

---

## The 6 ablation profiles

Each profile adds exactly **one** new component on top of the previous one,
so the difference between consecutive profiles isolates the effect of that
component:

| Profile | Adds | Configured flags |
| --- | --- | --- |
| **p1** | VLM + Plain LLM | `use_vlm`, `use_plain_llm` |
| **p2** | + Reasoning (per-episode) | `use_reasoning` |
| **p3** | + Cross-episode aggregator | `use_cross_episode_reasoning` |
| **p4** | + KAG (knowledge graph) | `use_kag` |
| **p5** | + RAG (similar-failure retrieval) | `use_rag` |
| **p6** | + TKF (top-k demo coverage) | `use_tkf` |

YAMLs live under `configs/ablation_profiles/`. The full set of legacy single-
component profiles (`vlm_only.yaml`, `no_kag.yaml`, etc.) is still available
for one-off experiments.

---

## Running individual components

### Train the diffusion policy

```bash
# Train from scratch on demos/*.json
python scripts/train_diffusion.py

# Resume from checkpoints/best_model.pth
python scripts/train_diffusion.py --resume

# Train on multiple demo sources (used by active_loop)
python scripts/train_diffusion.py --resume \
    --demo_paths "demos/*.json,demos/active_loop/p3/loop_<id>/**/*.json"
```

Flags: `--demo_dir <dir>`, `--demo_paths <comma-globs>` (overrides `--demo_dir`,
recurses when the pattern contains `**`), `--checkpoint_dir`, `--epochs`,
`--batch_size`, `--lr`, `--resume`. Outputs go to `checkpoints/best_model.pth`,
`best_model_ema.pth`, `best_model_meta.json`, and `training_log.json`.

Live progress: outer epoch tqdm + inner per-batch tqdm + every-10-epoch summary.

### Record demos with pygame

The `play_maze.py` script has **three modes**:

```bash
# 1) Free-roam — random layouts each episode (legacy)
python scripts/play_maze.py

# 2) Single forced layout
python scripts/play_maze.py \
    --start 0,0 --goal 4,4 --fires "2,1;2,2;3,3" \
    --demo_dir demos/extra --layout_id manual_001 --single_episode

# 3) Layout-queue driver — read recommended_layouts.json and walk every layout
python scripts/play_maze.py \
    --layouts-from results/active_loop/p3/loop_<id>/round_<N>/recommended_layouts.json
```

Keys: arrow keys / drag to move, **S** save (auto-advance to next layout in queue
mode), **N** skip the current layout (queue mode only), **R** retry the same
layout, **F** toggle fire mode (free-roam only), **Q** / Esc quit.

### Rule-based BFS expert

When no human is available (e.g. a headless slurm job), the BFS collector
generates demos for every prescribed layout autonomously: build the env on
that exact layout, plan a fire-free shortest path with BFS, execute it, and
save the demo in the same JSON shape `play_maze` produces.

```bash
# Collect demos for an entire round of prescriptions
python scripts/rule_based_collector.py \
    --layouts-from results/active_loop/p3/loop_<id>/round_<N>/recommended_layouts.json

# Override the demo_dir embedded in the JSON
python scripts/rule_based_collector.py \
    --layouts-from <path> --demo_dir demos/some/other/dir

# Skip layouts that BFS finds unsolvable (no fire-free path) instead of erroring
python scripts/rule_based_collector.py --layouts-from <path> --skip-unsolvable
```

A `bfs_collection_summary.json` is written next to the demos with counts and
per-layout outcomes.

### Run one profile pipeline (Phase A + B + C)

`run_pipeline.py` is the single-shot version: roll out N episodes, run the
selected profile's pipeline, save to `results/runs/run_<ts>/`.

```bash
# Default: full system from configs/experiment_config.yaml
python scripts/run_pipeline.py

# Override with an ablation profile
python scripts/run_pipeline.py --ablation configs/ablation_profiles/p3_vlm_reasoning_cross_plain_llm.yaml

# CLI overrides take precedence
python scripts/run_pipeline.py --n_episodes 10 --seed 42 --tag debug
```

### Run the full ablation suite (all profiles, one rollout)

`run_ablation_suite.py` is the **headless ablation entry point**: it runs ONE
shared rollout (Phase A), then re-runs Phase B + Phase C for each profile in
sequence on that same rollout. This is the cheapest way to compare p1..p6
because the expensive rollout phase isn't repeated.

```bash
# All 6 profiles
python scripts/run_ablation_suite.py

# A subset
python scripts/run_ablation_suite.py --profiles p3,p5,p6

# Override episode count / seed for the shared rollout
python scripts/run_ablation_suite.py --n-episodes 10 --seed 42

# Force every per-profile LLM call to re-run instead of hitting the cache
python scripts/run_ablation_suite.py --force-rerun-cache

# Open the Gradio dashboard automatically when the suite finishes
python scripts/run_ablation_suite.py --launch-dashboard
```

Output structure:

```
results/ablations/run_<ts>/
├── suite_manifest.json
├── rollout/                          # the shared Phase A
│   ├── full_output.json
│   └── episodes/episode_<i>/...
├── p1_vlm_plain_llm/
│   ├── full_output.json              # Phase B + C reusing the rollout
│   ├── ablation_summary.json
│   ├── rag_bank/                     # owner-id-isolated, see pipeline/rag_bank.py
│   └── episodes/episode_<i>/         # per-episode VLM/KAG/RAG/reasoning/TKF/prescription
└── suite_summary_<ts>.json           # SR / failures / demos / cache stats per profile
```

For headless slurm execution, just wrap this in an sbatch — see
[Headless ablation studies on Slurm](#headless-ablation-studies-on-slurm).

### Active loop (interactive)

`active_loop.py` is the **multi-round driver**: it trains, evaluates, prompts
you to record demos for the LLM-prescribed layouts (or runs the BFS expert
silently), then loops. Demos for profile `pX`'s loop are isolated under
`demos/active_loop/pX/<loop_id>/`, so other profiles' loops never contaminate
this one. Each round retrains on the **full aggregated dataset** (baseline +
this profile's collected demos), defending against catastrophic forgetting.

```bash
# Most common: human expert, target SR 0.9
python scripts/active_loop.py --profile p3

# Use the BFS rule-based expert instead — fully autonomous, no pygame
python scripts/active_loop.py --profile p3 --expert bfs --no-demo-prompt

# Override episode count / target / max rounds
python scripts/active_loop.py --profile p3 --rounds 15 --target-sr 0.85 --n_episodes 10
```

Key flags:

- `--profile p1..p6` (required) or full profile name.
- `--expert {human,bfs}` — human (default, launches pygame per layout) or
  `bfs` (headless rule-based collector).
- `--rounds N` — max rounds (default 20).
- `--target-sr 0.9` — stop early if reached.
- `--baseline-checkpoint checkpoints/baseline.pth` — pinned weights restored
  at loop start. `--skip-baseline-restore` to opt out.
- `--skip-train-first-round` — don't retrain before round 1 (use the
  freshly restored baseline weights as-is).
- `--no-pygame` — for `--expert human` dry-runs; layouts saved but pygame
  never opens.
- `--no-demo-prompt` — skip the "press ENTER after recording" prompts.

### One-shot round (headless)

`run_round.py` is the **headless slurm-friendly counterpart** to active_loop:
it runs ONE round and exits. Loop / round resolution is automatic — pass
`--loop-id` to attach to an existing loop, otherwise it picks the latest or
creates a new one.

```bash
# First round of a fresh loop for p3, BFS expert
python scripts/run_round.py --profile p3 --expert bfs

# Attach to an existing loop and run the next round
python scripts/run_round.py --profile p3 --loop-id loop_20260501_120000 --expert bfs

# Force a brand-new loop
python scripts/run_round.py --profile p3 --new-loop --expert bfs

# Auto-loop inside one process: keeps running rounds until target SR or
# --max-rounds is reached. Requires --expert bfs (no human possible).
python scripts/run_round.py --profile p3 --expert bfs --auto-loop --max-rounds 20

# Re-run the LLM pipeline on the existing checkpoint (no training)
python scripts/run_round.py --profile p3 --skip-train

# Email notifications at every milestone (see notifier section)
python scripts/run_round.py --profile p3 --expert bfs --notify-to you@example.com
```

### Baselines (naive DAgger / diff-DAgger)

```bash
# Naive DAgger: BFS-expert demo for every failed episode
python scripts/run_baseline.py --baseline naive

# Diff-DAgger: only intervene on episodes that match a heuristic
python scripts/run_baseline.py --baseline diff
```

These produce comparison numbers for the central thesis ("one LLM pass beats
per-failure intervention").

### Dashboard

```bash
python dashboard/app.py
```

A Gradio UI auto-loads the most recent suite under `results/ablations/`.
Top: failure-episode selector + frame thumbnails. Six tabs (P1..P6) each with
sub-tabs for VLM / KAG / RAG / Reasoning / TKF / Per-Episode Prescription /
Cross-Episode Reasoning / Final Prescription. Disabled components are shown
as grayed-out notices so you can see what each profile excluded.

---

## Headless ablation studies on Slurm

The headless workflow is designed so you **never need to leave your laptop
on**. Slurm owns the compute, the BFS rule-based expert plays the prescribed
layouts autonomously, and Gmail notifications keep you informed at every
milestone.

### Two modes

#### A. One round per slurm job (then resubmit)

Best when you want a human to record demos between rounds.

```bash
# round 1 — laptop can be off
PROFILE=p3 NEW_LOOP=1 EXPERT=human NOTIFY_TO=you@example.com \
    sbatch scripts/slurm/round.sh

# email arrives -> sync the recommended_layouts.json locally -> record demos:
python scripts/play_maze.py \
    --layouts-from results/active_loop/p3/loop_<id>/round_1/recommended_layouts.json

# round 2 — same loop:
PROFILE=p3 LOOP_ID=loop_<id> EXPERT=human NOTIFY_TO=you@example.com \
    sbatch scripts/slurm/round.sh
```

#### B. Auto-loop inside one slurm job (BFS expert)

Best for fully autonomous runs — the loop trains, evals, BFS-collects,
retrains, evals, … inside the SAME job until target SR or `MAX_ROUNDS`.

```bash
PROFILE=p3 EXPERT=bfs AUTO_LOOP=1 MAX_ROUNDS=15 \
    NOTIFY_TO=you@example.com TARGET_SR=0.9 \
    sbatch scripts/slurm/round.sh
```

Use longer `--time=` directives for auto-loop (edit `scripts/slurm/round.sh`).

### All available env vars for `slurm/round.sh`

| Var | Default | Meaning |
| --- | --- | --- |
| `PROFILE` | `p1` | Profile alias (p1..p6) or full name. |
| `LOOP_ID` | (auto) | Reuse existing loop. Empty → run_round picks latest, or creates one. |
| `NEW_LOOP` | `0` | `1` forces a fresh `loop_<timestamp>/` directory. |
| `ROUND` | (auto) | Explicit round number. Empty → next unused. |
| `N_EPISODES` | (config default) | Override `rollout.n_episodes`. |
| `SEED` | (config default) | Override `rollout.seed`. |
| `TARGET_SR` | `0.90` | Recorded; auto-loop stops when reached. |
| `BASELINE_CKPT` | `checkpoints/baseline.pth` | Restored at round 1 of new loops. |
| `SKIP_TRAIN` | `0` | `1` skips training (run pipeline only). |
| `SKIP_BASELINE_RESTORE` | `0` | `1` keeps current `checkpoints/` as-is. |
| `EXPERT` | `bfs` | `bfs` / `human` / `none`. |
| `AUTO_LOOP` | `0` | `1` chains rounds until target / max. |
| `MAX_ROUNDS` | `20` | Cap on auto-loop rounds inside one job. |
| `NOTIFY_TO` | (unset) | Empty disables email. |
| `EXTRA_ARGS` | (empty) | Extra flags appended verbatim to `run_round.py`. |

The job writes `slurm_logs/<jobname>_<jobid>.{out,err}` and (when
`NOTIFY_TO` is set) emails the last 400 lines of the .out at the end so you
can read everything that happened on the cluster from your inbox.

### Headless ablation across all 6 profiles

The cheapest cross-profile comparison is one sbatch wrapping
`run_ablation_suite.py` (one shared rollout, six Phase B+C runs):

```bash
cat > scripts/slurm/ablation.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=dmn_ablation
#SBATCH --partition=gpu
#SBATCH --qos=batch-short
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=you@example.com
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

module purge
module load Anaconda3
source /home/$USER/.bashrc
conda activate maze
cd /vast/$USER/DmN/DmNfull/
mkdir -p slurm_logs
python -u scripts/run_ablation_suite.py --n-episodes 10 --seed 42
EOF

sbatch scripts/slurm/ablation.sh
```

Output lands under `results/ablations/run_<ts>/` and is ready for
`python dashboard/app.py`.

To run **one full active loop per profile** in parallel slurm jobs (each
profile autonomous via BFS), submit one job per profile from the same
working tree. `scripts/slurm/round.sh` already auto-isolates each profile
under `checkpoints/${PROFILE}/` (see `CHECKPOINT_DIR` default at the top of
that script), so demos, RAG banks, AND checkpoints are all per-profile —
parallel jobs do not trample each other.

```bash
for p in p4 p5 p6; do
    PROFILE=$p EXPERT=bfs AUTO_LOOP=1 MAX_ROUNDS=5 \
        NEW_LOOP=1 TARGET_SR=0.9 \
        NOTIFY_TO=you@example.com \
        sbatch --job-name=dmn_${p} scripts/slurm/round.sh
done
```

What's isolated:

| Resource | Path | Isolated by |
| --- | --- | --- |
| Demos | `demos/active_loop/<profile>/<loop_id>/` | `active_loop.profile_loop_demos_dir` |
| RAG bank | `results/active_loop/<profile>/<loop_id>/round_<N>/rag_bank/` | `run_eval` (per-round wipe) |
| Training output | `checkpoints/<profile>/best_model{,_ema}.pth` | `slurm/round.sh` sets `--checkpoint-dir checkpoints/${PROFILE}`; `train_diffusion.py` writes there |
| Eval load | reads `checkpoints/<profile>/best_model_ema.pth` | `run_eval` overrides `cfg.rollout.checkpoint_path` to the same dir |

The eval-load row is the fix on this branch: previously `run_eval` left
`cfg.rollout.checkpoint_path` at the master-config default
(`checkpoints/best_model_ema.pth`, the *global* file), so the per-round
pipeline was reading whichever profile last touched the global path
rather than the just-trained per-profile EMA. `run_eval` now mirrors
`run_train`'s `--checkpoint-dir` flow, pinning the eval to the same
per-profile directory.

After each profile's job finishes, snapshot its final EMA checkpoint into
a stable per-profile filename so the McNemar evaluator can name them
explicitly:

```bash
for p in p4 p5 p6; do
    cp checkpoints/${p}/best_model_ema.pth \
       checkpoints/${p}_final_ema.pth
done
```

---

## Statistical comparison: McNemar + paired t-test (P4 / P5 / P6)

Once each profile has finished its active loop, two independent tests
answer two different questions:

1. **McNemar (episode-level, paired binary).** Did P_a's *trained policy*
   outperform P_b's on the same held-out episodes? Outcome per episode is
   success ∈ {0, 1}; same `(seed_base + episode_idx)` schedule across
   profiles so episode_id `i` is the same start/goal/fire layout in every
   profile. Discordant pairs (b, c) drive the test.
2. **Paired t-test (round-level, prescription quality).** Did P_a's *LLM
   prescriptions* propose higher-quality layouts than P_b's during the
   active-loop rounds themselves? Per round per profile we record
   `save_rate = n_saved / n_prescribed` (solvable layouts) and
   `mean_steps = mean(len(actions))` across saved demos (richness). One
   paired observation per round.

Both tests Bonferroni-correct across the two pairs `(p4,p5)` and `(p5,p6)`
within their metric, so corrected α = 0.025.

### Recommended: chain everything off the training jobs in slurm

The end-to-end workflow is three sbatches, each chained off the previous
via `--dependency=afterok`:

1. **Three training jobs** (parallel, one per profile) — your existing
   `scripts/slurm/round.sh` pattern, just with `--parsable` so we capture
   the job IDs.
2. **One McNemar eval array job** (3 array tasks, parallel) — each task
   evaluates one profile on its own GPU. Snapshots that profile's final
   EMA checkpoint into a per-array-job directory and writes
   `per_episode_success_policy.json` under `results/mcnemar/run_<id>/`.
3. **One McNemar analysis job** (CPU, fast) — runs `mcnemar_analysis.py`
   on the three success vectors and `prescription_quality_analysis.py`
   over the active-loop rounds. Auto-discovers the latest eval run dir
   and the latest `loop_log.json` per profile.

```bash
# 1) Train P4/P5/P6 in parallel.
JID4=$(PROFILE=p4 EXPERT=bfs AUTO_LOOP=1 MAX_ROUNDS=5 NEW_LOOP=1 \
       TARGET_SR=0.9 NOTIFY_TO=you@example.com \
       sbatch --parsable --job-name=dmn_p4 scripts/slurm/round.sh)
JID5=$(PROFILE=p5 EXPERT=bfs AUTO_LOOP=1 MAX_ROUNDS=5 NEW_LOOP=1 \
       TARGET_SR=0.9 NOTIFY_TO=you@example.com \
       sbatch --parsable --job-name=dmn_p5 scripts/slurm/round.sh)
JID6=$(PROFILE=p6 EXPERT=bfs AUTO_LOOP=1 MAX_ROUNDS=5 NEW_LOOP=1 \
       TARGET_SR=0.9 NOTIFY_TO=you@example.com \
       sbatch --parsable --job-name=dmn_p6 scripts/slurm/round.sh)

# 2) Per-profile per-episode eval as a 3-task array, runs in parallel.
EVAL_JID=$(N_EPISODES=500 SEED_BASE=0 NOTIFY_TO=you@example.com \
    sbatch --parsable \
           --dependency=afterok:${JID4}:${JID5}:${JID6} \
           scripts/slurm/mcnemar_eval.sh)

# 3) McNemar + prescription-quality analysis once the array has finished.
sbatch --dependency=afterok:${EVAL_JID} \
       NOTIFY_TO=you@example.com \
       scripts/slurm/mcnemar_analysis.sh
```

Why three jobs:

- Training jobs already isolate `checkpoints/${PROFILE}/`, so they run
  in true parallel.
- The McNemar eval is the slow part (500 episodes × DDPM denoise per
  step). Splitting it into a 3-task array means each profile gets its
  own GPU and the wall-clock is ~1/3 of the serial version, modulo
  cluster scheduling.
- The analysis step is ~seconds, needs no GPU, and needs all three
  per-profile success files. Running it as a separate CPU-only job
  keeps the GPU array job from sitting idle on it, and `afterok:<array_job_id>`
  is satisfied only when every array task exits 0 — so we never run
  the test on a partial set of profiles.

Snapshot layout written by `mcnemar_eval.sh` (so a later training round
can't mutate the weights you tested against):

```
checkpoints/snapshots/<array_job_id>/
├── p4/best_model_ema.pth   + best_model.pth + best_model_meta.json
├── p5/...
└── p6/...
```

Outputs land in `results/mcnemar/run_<array_job_id>/`:
`per_episode_success_policy.json` per profile (written by the array
tasks), then `mcnemar_results.json` and
`prescription_quality_results.json` (written by the analysis job).

Env vars for `mcnemar_eval.sh` (the array job):

| Var | Default | Meaning |
| --- | --- | --- |
| `N_EPISODES` | `500` | Episodes per profile for the per-episode eval. |
| `SEED_BASE` | `0` | First env seed; episodes use `SEED_BASE..SEED_BASE+N-1`. |
| `NOTIFY_TO` | (unset) | Empty disables the per-task end-of-job mail. |
| `PROJECT_ROOT` | `/vast/$USER/DmN/DmNfull` | Working tree to `cd` into. |

Env vars for `mcnemar_analysis.sh` (the analysis job):

| Var | Default | Meaning |
| --- | --- | --- |
| `MCNEMAR_RUN_DIR` | (latest under `results/mcnemar/`) | Pin a specific eval run dir. |
| `P4_LOOP_LOG` / `P5_LOOP_LOG` / `P6_LOOP_LOG` | (auto) | Override the auto-detected latest `loop_log.json`. |
| `NOTIFY_TO` | (unset) | Empty disables the SLURM_END mail. |
| `PROJECT_ROOT` | `/vast/$USER/DmN/DmNfull` | Working tree to `cd` into. |

### Manual / interactive equivalent (skip slurm)

Each step the slurm wrapper does is its own callable script, so you can
run them by hand if you'd rather not queue a job.

#### 1. Snapshot the per-profile checkpoints (one-time, after training)

```bash
mkdir -p checkpoints/snapshots/manual
for p in p4 p5 p6; do
    mkdir -p checkpoints/snapshots/manual/${p}
    cp checkpoints/${p}/best_model_ema.pth   checkpoints/snapshots/manual/${p}/
    cp checkpoints/${p}/best_model.pth       checkpoints/snapshots/manual/${p}/  2>/dev/null || true
    cp checkpoints/${p}/best_model_meta.json checkpoints/snapshots/manual/${p}/  2>/dev/null || true
done
```

The meta sidecar matters: `pipeline/rollout._load_meta()` looks for
`best_model_meta.json` next to the checkpoint and falls back to
defaults if it isn't there, which can build the policy with the wrong
horizons.

#### 2. Per-episode policy evaluation (input to McNemar)

```bash
python scripts/run_mcnemar_eval.py \
    --p4-ckpt checkpoints/snapshots/manual/p4/best_model_ema.pth \
    --p5-ckpt checkpoints/snapshots/manual/p5/best_model_ema.pth \
    --p6-ckpt checkpoints/snapshots/manual/p6/best_model_ema.pth \
    --n-episodes 500 --seed-base 0
```

Loads each profile's checkpoint, rolls out N episodes per profile on
`seeds 0..N-1`, reseeds `torch / cuda / numpy / random` per episode
(locks DDPM denoise noise so any per-episode SR difference is
attributable to the checkpoints, not RNG drift), and writes one
`per_episode_success_policy.json` per profile under
`results/mcnemar/run_<ts>/<profile>/`.

#### 3. McNemar paired test on the success vectors

```bash
python scripts/mcnemar_analysis.py \
    --results-dir results/mcnemar/run_<ts>
```

Builds the 2×2 paired table for `(p4, p5)` and `(p5, p6)` and runs
`statsmodels.stats.contingency_tables.mcnemar` (exact binomial when
`b + c < 25`, continuity-corrected χ² otherwise). Prints both
uncorrected (α = 0.05) and Bonferroni (α = 0.025) decisions, and writes
`mcnemar_results.json` next to the inputs.

#### 4. Paired t-test on prescription quality (across active-loop rounds)

```bash
python scripts/prescription_quality_analysis.py \
    --p4-loop-log results/active_loop/p4_vlm_reasoning_kag_cross_plain_llm/loop_<id>/loop_log.json \
    --p5-loop-log results/active_loop/p5_vlm_reasoning_kag_rag_cross_plain_llm/loop_<id>/loop_log.json \
    --p6-loop-log results/active_loop/p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm/loop_<id>/loop_log.json
```

For each round per profile, reads `recommended_layouts.json` for
`n_prescribed`, counts `demo_*.json` files in the round's `demo_dir` for
`n_saved` and uses each demo's `len(actions)` for `mean_steps`. The
`bfs_collection_summary.json` (whether at the round-level legacy path or
the loop-level post-fix path) is loaded as a cross-reference; the file
count is canonical and the script warns if the summary disagrees.

If profiles ran a different number of rounds, the script truncates each
profile to `min(n_rounds_per_profile)` and prints a WARNING. Output:
`prescription_quality_results.json` next to the P4 loop log.

### Reading the results

For each of the (P4 vs P5) and (P5 vs P6) comparisons you'll get:

| Test | Reports | Significance threshold |
| --- | --- | --- |
| McNemar | b, c, χ² / exact stat, p-value | Bonferroni α = 0.025 |
| Paired t-test (save_rate) | mean ± std per profile, t, p-value | Bonferroni α = 0.025 |
| Paired t-test (mean_steps) | mean ± std per profile, t, p-value | Bonferroni α = 0.025 |

A profile "wins" a comparison when the corresponding `Δ > 0` and the
Bonferroni-corrected decision is SIGNIFICANT.

---

## Email notifications via Gmail API

The notifier uses Google's Gmail API (OAuth2) with `credentials.json` you
provide.

### One-time setup

1. Drop your **Desktop OAuth client** credentials into the project root as
   `credentials.json`. Create one at
   <https://console.cloud.google.com/apis/credentials>.
2. Enable the Gmail API on that project at
   <https://console.cloud.google.com/apis/library/gmail.googleapis.com>.
3. On a machine with a browser, run:
   ```bash
   python scripts/notify.py --setup
   ```
   A browser window opens, you authorise the `gmail.send` scope, and
   `token.json` is written next to `credentials.json`.
4. Copy **both** `credentials.json` and `token.json` to the HPC project root.
   Tokens auto-refresh; you only repeat the browser step if the refresh
   token gets revoked.

### Use it from the CLI

```bash
# Plain text mail
python scripts/notify.py --to you@example.com --subject "training done" --body "loss=0.012"

# Mail the tail of a slurm .out
python scripts/notify.py --to you@example.com \
    --subject "round 3 done" --body-file slurm_logs/dmn_round_42.out --tail 200

# Structured event with JSON payload
python scripts/notify.py --to you@example.com \
    --event TRAIN_DONE --payload-json '{"loss":0.012,"epochs":5000}'

# Attach a file
python scripts/notify.py --to you@example.com --subject "layouts" \
    --body "see attachment" --attach results/active_loop/p3/loop_x/round_1/recommended_layouts.json
```

### What gets emailed during an active loop / slurm run

When `NOTIFY_TO` is set (env var or `--notify-to`), `run_round.py` sends one
email at every milestone:

| Event | When |
| --- | --- |
| `LOOP_INVOCATION` | run_round starts (loop_id, starting_round, expert, …) |
| `ROUND_START` | round begins (profile, loop_id, round, expert) |
| `TRAIN_DONE` | training finishes (loss, elapsed, dataset size) |
| `EVAL_DONE` | rollout + Phase B + Phase C done (SR, n_failures, n_clusters, n_layouts) — `recommended_layouts.json` is attached |
| `DEMO_COLLECTION_DONE` | BFS expert finished (n_saved, n_skipped, demo_dir) |
| `ROUND_END` | round wrap-up (SR, target_hit, profile_demos_total) |
| `AUTO_LOOP_DONE` | --auto-loop terminates (rounds_done, final_round, target_hit, final_sr) |
| `SLURM_START` / `SLURM_END` | slurm job start / end (sent by `slurm/round.sh`; SLURM_END includes the last 400 lines of the .out) |

Notifier failures are best-effort: an email outage will not kill your job.

---

## What gets saved where

```
results/runs/run_<ts>/                   # single profile pipeline (run_pipeline.py)
├── full_output.json
├── ablation_summary.json
├── config_used.yaml
├── rag_bank/                            # owner_run_id == run_<ts>
└── episodes/episode_<i>/
    ├── episode_data.json
    ├── frames/
    ├── vlm_report.txt
    ├── kag_context.txt
    ├── rag_retrieved.txt
    ├── reasoning.txt
    ├── tkf_result.json
    └── final_prescription.txt

results/ablations/run_<ts>/              # full suite (run_ablation_suite.py)
├── suite_manifest.json
├── rollout/                             # shared Phase A
└── <profile>/                           # one dir per profile, full Phase B + C output

results/active_loop/<profile>/           # active loop / run_round.py
├── loop_log.json                        # cumulative per-round entries across loops
├── learning_curve.png                   # SR vs round
└── <loop_id>/
    ├── loop_manifest.json
    ├── loop_summary.json                # rounds_to_target, demos collected, ...
    └── round_<N>/
        ├── full_output.json             # the full pipeline output for this round
        ├── prescriptions.json           # parsed aggregator output (clusters, n_demos, ...)
        ├── recommended_layouts.json     # what play_maze --layouts-from / BFS collector reads
        ├── metrics.json
        ├── config_used.yaml
        └── rag_bank/                    # round-isolated RAG bank

demos/                                   # baseline demos (top-level *.json) + active_loop/...
└── active_loop/<profile>/<loop_id>/round_<N>/*.json
```

---

## Glossary of components

- **VLM** — vision-language describe-the-end-frame check, cached per
  `(rollout_id, episode_id)` so all profiles in one ablation suite share the
  VLM call without polluting other suites.
- **KAG** — knowledge-augmented generation. The graph in
  `knowledge/kag_maze_knowledge.json` defines corridor regions
  (`left_edge`, `top_edge`, `right_edge`, `bottom_edge`, `central_mixed`)
  and a failure-mode taxonomy. Injected with **HIGH priority** into the
  reasoning + prescription + aggregator prompts so corridor names come from
  KAG.
- **RAG** — FAISS index of past failure embeddings (CLIP + summary). The
  bank is **owner-run-id-isolated**: profile p5 cannot retrieve from
  profile p4's bank even if directories were shared. Retrieved cases must
  be cited by rank in the prescription's `demo_variations` or `rationale`.
- **Reasoning (analysis pass)** — produces sections 1-5 of root-cause
  analysis. **Forbidden** from emitting `<<<FINAL_REC>>>`.
- **Reasoning (prescription pass)** — receives analysis + KAG + RAG + TKF
  and **derives** `<<<FINAL_REC>>>` under STRICT GROUNDING RULES (corridor
  must come from KAG, demo_variations must cite RAG cases, n_demos must
  reflect TKF gap, path must respect the corridor and avoid fires).
- **TKF (Top-K knowledge fetcher)** — checks demo coverage for the analysis-
  proposed corridor; verdict `FOUND / PARTIAL / NOT_FOUND`. Drives n_demos.
- **Aggregator** — clusters per-episode FINAL_RECs, produces a structured
  JSON with `failure_clusters`, `demonstration_prescriptions[]` (each with
  validated `recommended_layouts`: `[{start_pos, goal_pos, fire_positions,
  n_repetitions, rationale}]`), `total_demonstrations_needed`, and
  `confidence`.

---

## Quick reference

```bash
# Inspect everything for a finished suite
python dashboard/app.py

# Sanity-check the pipeline on a tiny rollout
python scripts/run_pipeline.py --n_episodes 3 --tag smoke

# Compare all 6 profiles on the same rollout (cheap, ~one rollout cost)
python scripts/run_ablation_suite.py --n-episodes 10 --seed 42

# Fully autonomous active loop in one slurm job (BFS expert, email notifications)
PROFILE=p3 EXPERT=bfs AUTO_LOOP=1 MAX_ROUNDS=15 \
    NOTIFY_TO=you@example.com NEW_LOOP=1 \
    sbatch scripts/slurm/round.sh

# Human-in-the-loop active loop on your laptop
python scripts/active_loop.py --profile p3 --expert human

# Record prescribed demos for a specific round (auto-walks the layout queue)
python scripts/play_maze.py \
    --layouts-from results/active_loop/p3/loop_<id>/round_1/recommended_layouts.json
```