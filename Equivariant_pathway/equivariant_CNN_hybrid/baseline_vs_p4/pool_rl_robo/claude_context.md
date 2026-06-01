# claude_context.md — `pool_rl_robo` suite

A self-contained sibling of `pool_x_selector/` (same depth under
`baseline_vs_p4/`) that ports the §18 **P4-LLM vs 5 DAgger-family IIL baselines**
comparison from the 5×5 maze to **continuous-control RL robotics**. Read this
end-to-end before changing anything.

Motivation (maze §19): the 5×5 maze saturated (~0.90) and demonstration
*selection* could not separate the methods; the recommended next step was a
harder, non-saturating environment. These MuJoCo locomotion + Fetch manipulation
tasks are that regime.

---

## 1. The user / cluster
- PhD student `s226137394`, repo `/weka/s226137394/DmNfull/` on the Qwen/vLLM
  cluster. Email `s226137394@deakin.edu.au`. Terse, action-mode; invites
  disagreement. Reviewer-defensible A* framing.
- **Conda env: `pool_rl_robo`** — a CLONE of `maze` + the RL stack
  (stable_baselines3 2.8, sb3_contrib 2.8, gymnasium_robotics 1.4.2,
  huggingface_sb3, numpy 2.x). `maze` itself is UNTOUCHED.
- **GOTCHA (load-bearing):** `~/.bashrc` prepends `maze/bin` to `PATH`, so after
  `conda activate pool_rl_robo` a bare `python`/`pip` STILL resolves to the
  **maze** env. ALWAYS use the explicit interpreter
  `/home/s226137394/.conda/envs/pool_rl_robo/bin/python` (the submit scripts set
  `PYBIN` to it). Installing into the clone with a bare `pip` silently corrupts
  `maze`.

## 2. The experiment in one paragraph
Compare **6 demonstration-acquisition methods** — **P4-LLM** (an LLM picks which
novice-visited state to request an expert demo for) vs **SafeDAgger\***,
**DropoutDAgger**, **EnsembleDAgger**, **ThriftyDAgger**, **Stagger** (the 5 IIL
rules from `pool_x_selector/baseline_implementations_guide.md`) — on **5
environments** (HalfCheetah-v4, Hopper-v4, Walker2d-v4, FetchReach-v4,
FetchPickAndPlace-v4), seeds 42..46. DAgger active loop, **1 expert
query/round** (budget=15), so the **reward-vs-#queries** curve is the common
yardstick. The expert is a pretrained SB3/SB3-contrib policy from HuggingFace;
the novice is a small MLP (BC). Fetch (Dict obs) is flattened
`observation+achieved_goal+desired_goal` (the "MultiInputPolicy" requirement).

**Run model (NO job arrays — user preference):** one SLURM job = one SEED,
running all 5 envs **sequentially** under a single Qwen3-32B vLLM
(`submit_one_seed.sh`). 5 seeds = **5 separate jobs** (`submit_5seeds_5jobs.sh`,
seeds 42..46 → run_0..run_4), then a chained cross-seed aggregate
(`aggregate.py`, mean ± std over seeds).

**Reading the metrics:** locomotion (HalfCheetah/Hopper/Walker2d) → **mean
reward** (forward velocity; positive). Fetch (FetchReach/FetchPickAndPlace) →
**success_rate** is the real metric; reward is SPARSE (−1/step until success), so
even a *perfect* expert has a NEGATIVE return (FetchPickAndPlace expert ≈ −17 at
sr=1.0). Negative reward on Fetch ≠ broken. The numbers in the result tables are
the NOVICE (post-DAgger), not the expert (experts are verified by the smoke gate:
all 5 ✓, Fetch sr=1.0).

**Demonstration / eval budget:** 5 seed expert demos (BC the round-0 novice) + 15
budget (1 expert demonstration/round) = up to 20 expert *trajectories* (each a
full expert rollout from the chosen state → many (s,a) pairs). Held-out eval =
20 fresh-seed episodes/round. There is no maze-style layout correction pool — the
per-round candidate "pool" is the states the novice visits in 1 rollout
(~1000 for MuJoCo, 50 for Fetch); the selector picks 1.

## 2b. What we're trying to do — intended P4-LLM architecture vs CURRENT build

**Goal.** Show that **P4-LLM** (LLM-guided demonstration *compression/prescription*)
is more sample-efficient than the 5 interactive-IL (IIL) baselines: instead of
the expert intervening per a hand-rule, P4 lets failures accumulate, **compresses
the top-K failures**, and asks an LLM to prescribe **one** corrective
demonstration that covers them — cutting the extra expert demos the baselines spend.

**The 5 IIL baselines** (from `pool_x_selector/baseline_implementations_guide.md`;
each = a DAgger query rule, expert provides the correct continuation from the
queried state): **SafeDAgger\*** (action-discrepancy > τ), **DropoutDAgger**
(MC-dropout ball/prob test), **EnsembleDAgger** (ensemble doubt OR discrepancy),
**ThriftyDAgger** (novelty OR success-Q risk, budget-calibrated), **Stagger**
(one random sample/round). See `selection/iil_baselines.py`.

**Intended full P4-LLM pipeline** (the maze `pool_x_selector` P4_top3 architecture):
roll out the novice → take the **top-3 failures** → a **VLM** ingests failure
frames (start / first-mistake / end) + the **trajectories** → a **reasoning
ANALYSIS pass** → a separate **PRESCRIPTION pass** → a **planner LLM** reads the
prescription and **prescribes ONE environment configuration** (compressing the 3
failures into 1) → that config is **loaded into the simulator** → the **expert
solves it** → that single demonstration is added. Compression: 3 failures → 1
demo. P4 is compared to the IIL baselines on the shared reward-vs-#queries
yardstick.

**CURRENT IMPLEMENTATION (honest gap — this is the lightweight version).**
The P4 here is a **simplified, TEXT-ONLY, SINGLE-PASS selector**. It does NOT
(yet) implement: the **VLM / image frames** (no vision at all), the **two-pass
analysis+prescription** split, a **planner that prescribes a NEW env
configuration loaded into the simulator**, or the explicit **top-3→1
compression**. What it does: build one text prompt = the env's **KAG** doc
(`p4/kag/{env}.md`) + a table of the **≤30 highest-discrepancy visited states**
(numeric summary) → the LLM **selects ONE existing visited state** → the expert is
rolled out from that state to episode end = the corrective demo. So it tests
*"LLM-guided state selection,"* a PROXY for the full pipeline (`p4/prompts.py`,
`p4/selector.py`, `p4/pipeline_p4.py`).

**Open design question to build the full pipeline in continuous control.** In the
maze the planner prescribes a *layout* (start/goal/fires/corridor). For
continuous control there is no equivalent discrete "configuration": for **Fetch**
it could be a prescribed **goal/object placement**; for **MuJoCo locomotion**
it'd be an **initial state (qpos/qvel)**, which is not a natural thing for an LLM
to author. Porting the full prescribe-and-load architecture requires settling
this (and adding a VLM + the two reasoning passes). This is the main TODO if we
want the *real* P4, not the selection proxy.

## 3. The 6 methods → kinds
`p4_llm` (LLM selector), `safe_dagger`→safe, `dropout_dagger`→dropout,
`ensemble_dagger`→ensemble, `thrifty_dagger`→thrifty, `stagger`→stagger. Dispatch
table: `selection/iil_baselines.py::KIND_OF`. Method dir names:
`orchestrator/workspace.py::METHOD_DIR_NAMES`. **Keep these in lockstep** with
`config*.yaml::methods` and the run drivers' defaults.

## 4. Topology (mirrors pool_x_selector)
```
pool_rl_robo/
├── config.yaml / config_baselines.yaml   knobs (P4 / 5 IIL)
├── smoke_test.py                          PHASE-1 gate (5 experts load+play)
├── run_experiment.py                      convenience single-env all-6 driver
├── model.py                               MLPPolicy (local; no upstream policy to borrow)
├── envs/   env_setup.py (paths+register+resolve+obs/act/success)  experts.py (EXPERTS+ENV_PROMPTS+load)
├── selection/  iil_baselines.py (6 rules + pick_one + run_dagger)  rollout.py (env-walk)
│              uncertainty.py (Ensemble, MC-dropout)  success_q.py (Thrifty risk)  rank.py
├── p4/   pipeline_p4.py (wrapper)  prompts.py  selector.py  runner.py (QwenClient)
├── trainer/finetune_replay.py             in-process warm-started BC (train_bc)
├── logging_ext/  training_log.py  compression_log.py
├── orchestrator/  workspace.py  bootstrap.py  run_one.py (P4)  run_baselines.py (5 IIL)  _common.py
├── aggregation/aggregate.py                cross-env summary + combined {ENV}_run{id}.json
├── qwen/proxy.py                           BORROWED verbatim from pool_x_selector (generic OpenAI shim)
└── submit_*.sh   PREFERRED (no arrays): submit_one_seed.sh (1 job=1 seed=5 envs seq+vLLM),
                  submit_5seeds_5jobs.sh (5 jobs, seeds 42-46). submit_smoke.sh (GPU smoke gate),
                  submit_aggregate.sh. [legacy array variants kept: submit_one_qwen.sh / submit_all.sh
                  / submit_seed_sweep.sh / submit_baseline*.sh — user prefers the per-seed jobs]
```
Borrowed/external: `qwen/proxy.py` (copied verbatim). Unlike the maze, the
domain "expert/model" are NOT upstream modules — the expert is the SB3 library +
HF checkpoints (`envs/experts.py`) and the novice is the local `model.py`. The
heavy maze `pipeline/` LLM package is NOT reused: P4 here is text-only
(`p4/runner.py` calls `/v1/chat/completions`, strips Qwen3 `<think>`, parses
strict JSON, falls back on failure).

## 5. Expert-repo corrections (Phase-1 smoke surfaced these — do NOT revert)
`envs/experts.py::EXPERTS`. The originally-specified ids 404'd:
- MuJoCo: `sb3/tqc-{Env}-v3` (no `-v4` repos exist); run fine on the `-v4` envs.
- FetchPickAndPlace: `IntelliGrow/FetchPickAndPlace-v4`, file
  `sac-FetchPickAndPlace-v4.zip` (native v4, SAC).
- FetchReach: the sb3 v1 TQC is INCOMPATIBLE (17-dim vs gymnasium's 16-dim obs;
  needs mujoco_py) → use `kuds/fetch-reach-dense-tqc` (TQC on FetchReachDense-v4,
  same 16-dim space; reaches the goal on the sparse FetchReach-v4 env).
`load_expert` overrides obs/action spaces via `custom_objects` (old-Gym unpickle),
blanks `replay_buffer_kwargs` (drops removed HER `online_sampling`), and passes
`env=` (HER models). numpy 2.x is required for the IntelliGrow zip.

## 6. Protocol details / faithfulness caveats (report; do NOT silently "fix")
- 1 query/round across all methods → reward-vs-#queries is comparable; the IIL
  switch rules are applied at the selected-state granularity.
- Each round aggregates a FULL expert sub-trajectory from the selected state
  (`rollout.expert_demo_from_state` replays the novice prefix, then rolls the
  expert to episode end) — meaningful learning signal; budget axis stays = rounds.
- ThriftyDAgger risk Q: Fetch uses real `info['is_success']`; locomotion uses a
  survival proxy (`env_setup.episode_success`); HalfCheetah never terminates so
  its risk saturates to 0 (Thrifty ≈ EnsembleDAgger there).
- MC-dropout is FULL-network here (fidelity improvement over the maze fusion-head-only).
- EnsembleDAgger round-1: `pick_one` picks the highest-discrepancy queryable
  state even at doubt≈0 (do NOT add a `score>0`-only filter — deadlocks).
- `tau`/`sigma` are raw-action L2 distances (env-scale-dependent) — calibrate per
  env; the `pick_one` fallback keeps the comparison running if a gate is mis-set.
- P4-LLM falls back to highest-discrepancy on ANY LLM failure (so jobs never
  crash on the LLM path).

## 7. How to run  (PREFERRED: per-seed jobs, NO arrays)
```
PYBIN=/home/s226137394/.conda/envs/pool_rl_robo/bin/python
$PYBIN smoke_test.py --predownload            # cache the 5 experts (login node, once)
# GPU smoke gate — sbatch it, wait, analyze, THEN launch production:
sbatch submit_smoke.sh                         # experts + live LLM/KAG token-budget + 1-env live pipeline
# One seed = ONE job = all 5 envs sequential (one Qwen3-32B vLLM serves them):
sbatch submit_one_seed.sh                      # seed 42 -> run_0
sbatch --export=ALL,SEED=43,RUN_ID=1 submit_one_seed.sh   # any single seed
# Full 5 seeds = 5 SEPARATE jobs (seeds 42..46) + chained cross-seed aggregate:
bash submit_5seeds_5jobs.sh
# aggregate by hand:  $PYBIN -m <pkg>.aggregation.aggregate   (run from repo root)
```
Each (env,seed) writes `results/{ENV}/run_{seed_idx}/{method}/results/learning_curve.json`
(+ `training_log.csv`, P4 `compression_log.csv`) and a combined
`results/{ENV}_run{seed_idx}.json`; `aggregate.py` rolls up `summary.json` with
**mean ± std over seeds**. `--nodes=1 --gpus-per-node=2`,
`--constraint=gpu-h100|gpu-h200`. Validation loop (user preference): sbatch the
GPU smoke → wait → analyze → then launch the per-seed production jobs.

## 8. Verification green-flags
1. `bash submit_smoke.sh` → 5/5 ✓ (sane rewards; Fetch success_rate>0).
2. `results/{ENV}/run_0/{method}/results/learning_curve.json` exists for all 6
   methods per env; `history` non-empty; reward rises with queries.
3. P4 used the LLM: `slurm_logs/vllm_*.log` reached startup; the run log shows P4
   selections without the fallback firing on clean calls.
4. `aggregation/aggregate.py` → `results/aggregate/summary.json` + per-env
   `results/{ENV}_run0.json`.

## 9. Never do
- Never install into the clone with a bare `pip` (PATH gotcha → corrupts maze);
  use the explicit interpreter.
- Never `sbatch` the bash launchers (they call sbatch internally).
- Never use `--gpus=N` (splits across nodes); use `--nodes=1 --gpus-per-node=N`.
- Never revert the expert-repo corrections (§5) or the load_expert fixes.
- Never run production from a Claude session without explicit user approval.
