# 04 — Codebase map (the 4 scattered repos) → ONE clean module

The supervisor's complaint: code scattered across four repos, not reproducible. Consolidate into
**one module** (suggested: a `distil/` python package in this folder or a single subfolder).

## Where things currently live (what to take from each)
1. **`/weka/s226137394/diff-dagger-ur5`** — *the cleanest starting point*. A self-contained
   robosuite (UR5e) engine with the full DISTIL loop already ported, built last session:
   - `diffdagger_rs/p4/loop.py` — the round loop (screen → cluster → memory → LLM decide →
     collect → retrain → eval). **Copy this structure; apply the `02_...md` design changes.**
   - `diffdagger_rs/p4/{descriptor,clustering,diversity,memory,parse}.py` — the compression core
     (geometric descriptor, silhouette-k clustering, cluster memory Eq 9, FPS context set,
     SELECT/BRIDGE parse). **Task-agnostic via a duck-typed descriptor — reuse.**
   - `diffdagger_rs/p4/{collect,bridge,bounds}.py` — SELECT (replay-to-t\* + expert takeover) and
     BRIDGE (place object + expert solve) primitives; per-task placement bounds.
   - `diffdagger_rs/p4/{llm,prompts,kag}.py` + `kag/*.json` — the multi-stage LLM client
     (VLM+Reasoning), the prompts, the KAG renderer + graphs. **Swap the vLLM client for the
     OpenRouter client (`02_...md` #3).**
   - `diffdagger_rs/{envs,experts,diffdagger,eval,config}.py` — env wrappers (Lift/Door/Wipe),
     scripted experts, the diffusion policy + train/calibrate, eval, config + the
     **byte-identical shared-bootstrap** (`make-bootstrap`).
   - `scripts/` — sbatch + `analyze_comparison.py` + `launch_definitive.sh`.
   - Branch `p4-llm-vlm-hybrid` on GitHub `SuyogKhanal/diff-dagger-ur5`.
2. **`/weka/s226137394/diff-dagger`** — the **ManiSkill fork**: the PushT env (**`PushT-v2` /
   `PushTEnv2`**, thresh 0.70, goal `π/2` — NOT v1), the clockwise PPO expert
   (`PushTHard_98_cw`), and the original multi-stage LLM pipeline
   (`diffdagger/main_pipeline/pipeline.py` `LLMGuidedDAggerPipeline`,
   `diffdagger/main_analysis/llm_clients.py`). Custom PushT envs under `env_restart/custom_envs/`.
3. **`/weka/s226137394/DmNfull/.../pool_rl_robo`** — the **suite** that drove PushT/StackCube:
   `p4/` (select_arm, prompts, kag/, runner), `p4_subtask/` (the V3-hybrid SubtaskPlanner:
   SELECT/BRIDGE decision + infeasibility re-prescribe loop), `envs/`, `configs/`,
   `orchestrator/`, and **GridWorld** (confirm location — see `03_...md`). The `qwen/proxy.py`
   (drop it — OpenRouter replaces the local proxy).
4. **`/weka/s226137394/DmNfull`** — the paper repo (this folder lives here). GitHub
   `SuyogKhanal/DmNfull`.

## Suggested consolidated structure (`distil/`)
```
distil/
  envs/           # one adapter per task; state + image variants; the NEW horizons (03_...md)
                  #   gridworld.py, lift.py, door.py, wipe.py, pusht.py  (PushT = PushT-v2)
  experts/        # scripted oracle / motion-planner / PPO per task; SEPARATE takeover budget
  policy/         # diffusion policy (state + image encoder); train_from_scratch; uncertainty
  descriptor.py   # Eq-7 geometric descriptor per task (GEOMETRIC even for image runs; no R3M)
  cluster.py      # silhouette-k (Eq 8) + pick_dominant + near-dominant constraint (flag)
  memory.py       # cluster memory Eq 9 (gamma,sigma,lambda) — flag lambda=0 to ablate
  context.py      # context set S (kappa=3): forced rep + worst-peak + FPS fill  (flags)
  collect.py      # SELECT (takeover at t_flag) / BRIDGE (place + solve) + INFEASIBILITY loop
  llm/            # OpenRouter client; stages: vlm -> analysis -> decision(SELECT/BRIDGE)+CONFIDENCE
  kag/            # per-task KAG json + renderer  (copied into each run's output folder)
  prompts.py      # all prompts (copied into each run's output folder)
  loop.py         # the DISTIL round loop, honoring every flag in 05_...md and change in 02_...md
  bootstrap.py    # byte-identical shared bootstrap per (task, seed)
  config.py       # one dataclass/yaml per run; every ablation = one flag
  run.py          # CLI: python -m distil.run --task Wipe --modality image --ablation memory_off --seed 3
  aggregate.py    # results/ -> master table + Tier-4 diagnostics + sign test
```

## Consolidation checklist
- [ ] Single env registry covering GridWorld/Lift/Wipe/Door/PushT × {state, image} with the
      **increased horizons** and a **decoupled expert-takeover budget** (`02_...md` #2).
- [ ] Descriptor is **geometric for all modalities** (no R3M); image only changes the *policy*
      encoder, not the descriptor/clustering (`02_...md` #4).
- [ ] Clustering/takeover uses the **first-threshold-crossing** step, not argmax (`02_...md` #1).
- [ ] LLM = **OpenRouter** client, multi-stage, with a **confidence score** in the decision
      (`02_...md` #3, #6). No vLLM, no H100/H200.
- [ ] **Infeasibility re-prescribe loop** wired into `collect` (`02_...md` #5).
- [ ] Every ablation flag from `05_...md` is a real switch; `full DISTIL` = all defaults.
- [ ] Byte-identical shared bootstrap shared by every arm of a (task, seed).
- [ ] Prompts + KAG written into each run's output dir (golden rule 3).
- [ ] One smoke (1 round, tiny budget, 1 GPU, 1 cheap OpenRouter call) per task before the matrix.
