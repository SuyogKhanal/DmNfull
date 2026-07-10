# HANDOFF — DISTIL consolidation + matrix (state of play + 2-HPC split)

This is the live ledger for continuing the DISTIL run on a second HPC (or a fresh
session). It reflects exactly what is built, smoked, running, and still to build.
Source of truth for jobs = `distil/RUN_STATE.md`. Bring-up = `SETUP_HPC2.md`.

## 1. What is DONE (built + smoked green)

The consolidated module lives at **`DmNfull/distil/`** (one self-contained package;
`python -m distil.run ...`). All of the following are smoke-verified end-to-end
(OpenRouter VLM+analysis+decision+CONFIDENCE, prompts+KAG+telemetry written):

- **Robot state tasks** Lift / Wipe / Door (robosuite UR5e, diffusion policy) — GPU.
- **GridWorld state** (p4m-equivariant escnn classifier, BFS expert, entropy
  self-uncertainty) — CPU/GPU, `distil/gridworld/`.
- **OpenRouter LLM client** (`distil/p4/llm.py`): Qwen3-VL + Qwen3-32B, reasoning-token
  budgets, `<think>` strip, task-aware prompts (GridWorld navigation-framed).
- **All ablation flags** real switches (`distil/config.py` ABLATIONS): Tier-1
  `memory_off` (Eq-9 λ→0), `allocation_random` (Stagger control), `clustering_off`
  (single group), `decision_heuristic`/`fallback_only` (no LLM), `vlm_off`, `kag_off`,
  `bridge_off`, `near_dominant_off`, `fixed_k3`, `llm_effort_low`. Wired + tested.
- **Image-modality policy core** (`distil/models/image_encoder.py` spatial-softmax, no
  R3M) + `DiffusionPolicy(modality="image")` + `DemoDataset` image windows — trains,
  calibrates, acts (synthetic-verified). **NOT yet wired** into the per-step render
  loop (see build item B1).
- **`aggregate.py`** (master table + Tier-4 diagnostics + Tier-5 sign test) — tested.
- **`matrix.py`** (cell enumeration, priority tiers, RUN_STATE, GPU/CPU routing).
- **Portability**: `PORTABILITY.md` + pins (`requirements.txt`/`environment.yml`).
  robosuite = public commit `85abee22`; escnn `1.0.11`.

## 2. What is RUNNING on HPC-A (this cluster) — PHASE-DISCIPLINED

Phase 1 (consolidate + one smoke) is done. We are now in a MEASURED Phase-2 validation,
NOT a full blast: HPC-A runs a **seed-1 validation** of GridWorld's allocation thesis
(full + 5 Tier-1 knockouts, seed 1) + Lift-state (HPC-A's cheap share). Only after the
seed-1 set validates the full budget=20 loop do we expand GridWorld to 5 seeds. The
GridWorld full-seed1 gate runs on **CPU** (GridWorld needs no GPU → zero GPU contention).

## 3. 2-HPC SPLIT — the actual assignment (whole cells only, 08_..md golden rule 6)

Cost-balanced (COST: GridWorld 1, Lift 3, Door 4, Wipe 6). Each cluster owns WHOLE
`(task, modality)` cells — never split one cell's arms/seeds across clusters (bootstrap
fairness). **This is the correction: do NOT run HPC-B's cells on HPC-A.**

| cluster | STATE cells it owns | later (image / PushT) |
|---|---|---|
| **HPC-A** (weka, this session) | **GridWorld**, **Lift** | GridWorld-image, Lift-image |
| **HPC-B** (shared-nothing) | **Wipe** (long pole), **Door** | Wipe-image, Door-image, **PushT** (state+image) |

### HPC-B, do this (bring up per `SETUP_HPC2.md`, then run YOUR cells only):
```bash
git clone git@github.com:SuyogKhanal/DmNfull.git && cd DmNfull
git checkout stackcube-hybrid-plugcharger-handoff && export DISTIL_ROOT="$PWD"
conda env create -f distil/environment.yml -n distil && conda activate distil   # robosuite @85abee22 pin
cp distil/.env.example .env   # add OPENROUTER_API_KEY (HPC-B's own)
# generate each owned cell's bootstrap LOCALLY (byte-identical, deterministic), then run all arms:
for T in Wipe Door; do
  python -m distil.run --make-bootstrap --task $T --modality state \
     --bootstrap-dir distil/results/shared_bootstrap/${T}_state
  for A in full memory_off allocation_random clustering_off decision_heuristic vlm_off; do
    for S in 1 2 3 4 5; do
      sbatch --export=ALL,TASK=$T,MODALITY=state,ABLATION=$A,SEED=$S,BUDGET=20,\
OUTPUT_DIR=distil/results/$T/state/$A/seed$S,\
BOOTSTRAP_DIR=distil/results/shared_bootstrap/${T}_state distil/scripts/run_distil.sbatch
    done
  done
done
# push ONLY the light leaves back (never checkpoints/telemetry/frames):
git add distil/results/**/result.json distil/results/**/config.yaml distil/RUN_STATE.md
git commit -m "HPC-B: Wipe+Door state results" && git push origin <results-branch>
```
Aggregation runs on HPC-A after `git pull`ing HPC-B's leaves (`python -m distil.aggregate`).

## 4. What is STILL TO BUILD (precise specs from the source study)

### B1. Image-modality render threading (robot tasks)  [medium]
Policy/dataset already support images. Remaining: capture `env.render(height=S,width=S)`
per step and carry frames into the demo/rollout obs. Files + guards (image code runs
only when `modality=="image"`, state path byte-identical):
- `distil/envs.py`: build with `offscreen=True`, `render_camera=TASK_CAMERA[task]`
  (Lift=agentview, Door/Wipe=frontview) for image runs.
- `distil/collect.py` `rollout_expert` + `distil/p4/collect.py` `_run_expert_to_done`:
  capture a frame aligned with each recorded state → `traj["image"]` (L,H,W,3 uint8).
- `distil/eval.py` `make_obs_seq`/`policy_rollout` + `distil/p4/screen.py`: maintain an
  image deque, feed `{"image":…}` to the policy.
- `distil/config.py`: add `camera/image_size(84)/keypoints(32)`; env factory offscreen.
- Descriptor/VLM UNCHANGED (geometric, 02_..md #4). Uncertainty UNCHANGED (v-pred loss).
- Smoke: `--task Lift --modality image --budget 1 --smoke` on a GPU (EGL).
- GridWorld image = swap the equivariant policy for the vendored RGB CNN
  (`distil/gridworld/rgb_policy.py`) conditioned on the 80×80 bird-eye; descriptor stays
  geometric.

### B2. Baseline gate family  [medium — copy + rewire]
The 6 baselines already exist in `pool_rl_robo/selection/{iil_baselines.py,uncertainty.py,
success_q.py}` with PAPER-EXACT HPs (tau=0.1, N=10, p=0.9, chi=0.05, M=5, alpha=0.99).
Port those 3 files into `distil/baselines/`, rewiring the 3 fork imports to distil
equivalents: dataset append → `distil/dataset.py`; `train_policy` → `distil/train.py
train_diffusion_policy`; `evaluate_heldout` → `distil/eval.py evaluate_policy`; collect →
`distil/collect.py`. Gate rules: Safe = ‖a_nov−a_exp‖>tau; Dropout = N MC draws, fire when
frac-in-τ-ball < p; Ensemble = M members, doubt>chi OR disc>tau; Thrifty = novelty(doubt)
OR risk(1−Q_ψ), thresholds = pool quantiles; Stagger = uniform-random recorded failure
(GridWorld-only; robot random control = the `allocation_random` ablation). Diff-DAgger is
already `distil/diffdagger.py`. Route arms via `run.py` (ablation ∈ BASELINES → baseline
loop). `aggregate.py` already expects `{diffdagger,safe,dropout,ensemble,thrifty,stagger}`.

### B3. PushT-v2 (the primary task)  [large — vendor the ManiSkill fork]
PushT lives in the fork `/weka/s226137394/diff-dagger` (vendored `mani_skill 3.0.0b7`,
NOT pip). The DISTIL loop for PushT ALREADY EXISTS as `pool_rl_robo/p4_subtask/`
(silhouette-k + Eq-9 memory + SELECT/BRIDGE Eq-10 + infeasibility loop). Port =
**vendor into `distil/pusht/`**:
- Env: `mani_skill` fork subtree + `PushTEnv2` (`envs/tasks/tabletop/push_t.py:567`,
  id `PushT-v2`, thresh **0.70**, goal_z_rot **π/2**, horizon 250) + reposition envs
  `PushT-Start-v0`/`PushT-Subtask-v0`. Runs in the `diffdagger` conda env (sapien 3.0.0b1).
- Expert: PPO `PushTHard_98_cw_202503033128.pth` + `model/RL.py` + `agents/RL_agent.py`
  (`MultipleExperts`, obs_dim 29, action_dim 7, clockwise-only → infeasibility re-prescribe).
- DISTIL: all of `p4_subtask/{descriptor,clustering,memory,planner,subtask_entry,collect,
  kag_bounds}` + KAG `p4/kag/PushT-v1.json`. **Fix bug**: `semantic_map.py:16-17`
  `GOAL_ZROT=(5/3)π` is the v1 value — must be **π/2** for v2.
- Policy: fork `DiffDAggerPolicy` (ConditionalUnet1D; R3M for image) OR reuse distil's
  DiffusionPolicy with the state/image encoder.
- **Portability**: whole fork subtree is a PRIVATE fork → vendor it (not a public pin);
  rewrite the `env_setup.py` `/weka` symlink + `sys.path` hack to package-relative;
  strip the hardcoded `/weka/.../PushTEnv` dataset path + conda python paths.

## 5. Aggregate + stats (run on HPC-A)
```bash
python -m distil.aggregate --results-dir distil/results --out distil/results/_agg
# -> master_table.csv, summary.md (headline SR, ablation Δ-vs-full, sign test), xlsx
```
Pull HPC-B's `result.json` leaves via git first; never commit checkpoints/telemetry/frames.

## 6. Ledger discipline
Update `distil/RUN_STATE.md` (matrix.py writes it) as jobs are submitted/finish; commit it
so both clusters + the aggregator share one source of truth. Reconcile against `squeue` at
session start.
