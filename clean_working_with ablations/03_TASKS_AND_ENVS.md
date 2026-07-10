# 03 — Tasks, environments, experts, horizons (state + image)

Five tasks × two modalities (state, image) = the 10 rows of the headline table
(`DISTIL_ablation_preview.xlsx` GT_SR/GT_InfoGain). Success = `info["success"]` in every env.

| task | modality | env / id | horizon (CURRENT → **NEW**) | success | expert | code |
|---|---|---|---|---|---|---|
| **GridWorld 5×5** | state (14-d) + image (80×80×3) | `MazeNavEnv` (not gym-registered; layout name = task id, default `multimodal`) | 200 (keep) | reach GOAL tile (step on FIRE = fail) | `AStarExpert` = BFS shortest-path, **multi-label** optimal-action mask (preserves multi-modality; `multimodal` layout has two equal 8-step routes) | `DmNfull/envs/maze_env.py`, `configs/maze_layouts.py`, `Equivariant_pathway/expert.py`; policies `CNN_pathway/model.py` (image CNN) + `Equivariant_pathway/model.py` (state MLP) |
| **Lift** | state (23-d); **image = TODO** | robosuite UR5e | `env=200, expert=200` → **raise (see §NEW)** | cube lifted above table | scripted motion-planner oracle | `diff-dagger-ur5/diffdagger_rs/{envs,experts}.py` |
| **Door** | state (20-d); **image = TODO** | robosuite UR5e, `use_latch=False` | `env=300, expert=300` → **raise** | hinge > 0.3 rad | scripted oracle (closed-loop hinge feedback) | same |
| **Wipe** | state (28-d + nearest-K markers); **image = TODO** | robosuite UR5e, WipingGripper (0-DOF) | `env=500, expert=500` → **raise/keep + decouple** | all dirt markers wiped | scripted oracle | same |
| **PushT** | state + image | **`PushT-v2` / `PushTEnv2`** (ManiSkill fork) — NOT v1 | env `250`, **expert takeover budget `120` (separate)** | intersection ≥ **0.70**, goal_z_rot **π/2** | **clockwise-only PPO `PushTHard_98_cw`** (capability gap: some rotations need CCW → infeasibility loop re-prescribes) | fork `/weka/s226137394/diff-dagger`, suite `pool_rl_robo` |

## Two things to BUILD (they don't exist yet)
1. **Image variants for Lift / Wipe / Door.** `diffdagger_rs/config.py` TASKS are **state-only**;
   robosuite runs never used camera obs. For the image rows you must add a **camera-obs +
   image-encoder path** to the diffusion policy (offscreen render → visual encoder; the render
   plumbing exists — `envs.py` supports `offscreen=True`, camera `frontview` for Door/Wipe,
   `agentview` for Lift). **The descriptor/clustering stays GEOMETRIC (no R3M) — only the policy
   encoder changes** (`02_...md` #4).
2. **GridWorld into the consolidated pipeline.** GridWorld lives in the `DmNfull` root maze
   pipeline (separate from the robot pipelines). Port `MazeNavEnv` + `AStarExpert` into the
   consolidated `distil/envs/gridworld.py` with both obs heads (state 14-d, image 80×80×3) and
   the geometric descriptor (agent cell, signed goal offset, progress, Manhattan-dist).

## §NEW — horizons + expert budget (apply `02_...md` #1, #2)
- The current Lift/Wipe/Door horizons were **too small** given the finding that the expert's
  post-takeover budget was `horizon − t*` (Lift t*≈192/200 → ~8 steps). Two coupled fixes:
  - **Take over at `t_flag` = first threshold crossing**, not the argmax peak (much earlier).
  - **Give the expert a fixed, generous takeover budget** (e.g. the full task horizon), decoupled
    from `horizon − t_flag` — exactly like PushT's separate 120-step budget. Then raising env
    horizons is a safety margin, not the primary fix.
- Pick concrete numbers when you implement (suggest: Lift 250, Door 350, Wipe 500; expert budget
  = full horizon after takeover). Verify each expert can solve from `t_flag` within the budget in
  a smoke before the matrix.

## Descriptor features per task (Eq 7 — GEOMETRIC, all modalities)
- **Robot (Lift/Door/Wipe/PushT):** `φ = [x, y, sinθ, cosθ, ρ=t*/T, δ=eef-contact-dist]` (6-D,
  quaternion-free), anchored at `t_flag`. Per-task specifics already in
  `diff-dagger-ur5/diffdagger_rs/p4/descriptor.py` (Lift cube pose+grasp; Door frame pose+hinge;
  Wipe dirt-centroid+coverage) — reuse, but anchor at `t_flag` not the peak.
- **GridWorld:** agent cell, signed offset to goal, progress ρ, Manhattan distance-to-goal.
- **Image runs:** SAME geometric φ (NOT R3M). Only the policy sees pixels.

## Env identity gotchas (do not repeat past mistakes)
- **PushT is `PushT-v2`** (thresh 0.70, goal π/2), even though the results dir is *labelled*
  `PushT-v1`. The held-out evaluator is v1 (thresh 0.90, goal 5π/3) — a goal mismatch between
  scoring and takeover env; keep both straight (it caused a failed local repro last time).
- robosuite `reset(seed)` = `_reseed_placement(env, seed)` then `env.reset()` (hard_reset=False),
  and `step` passes actions through untransformed — reproduce scenes that way.
