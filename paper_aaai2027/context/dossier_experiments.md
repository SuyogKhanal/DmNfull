# Experimental-setup dossier — PACE interactive-imitation-learning

Authoritative experimental protocol for the AAAI-2027 submission. Method under test:
**PACE** (Perceive -> Assess -> Choose -> Execute). All numeric results
ship as `\PH{...}` placeholders (see `results_placeholders.md`).

**Paper scope: EXACTLY five tasks — T1 Toy grid, T2 Push, T3 Lift, T4 Wipe, T5 Door.**
No other task is mentioned anywhere in the paper.

---

## A. Tasks

### T1 — Toy 5x5 grid navigation  (IMPLEMENTED)
- **Sim / benchmark:** custom Gymnasium env `MazeNavEnv` — `/weka/s226137394/DmNfull/envs/maze_env.py`.
  Suite driver: `pool_x_selector` (grid-navigation IIL comparison).
- **Robot / embodiment:** none — a point agent on a discrete 5x5 tile grid
  (`configs/maze_layouts.py`, PRIMARY layout `medium_1`: open 5x5, start (0,0),
  goal (4,4), two fire tiles at (2,1)/(2,2) that block the straight middle path;
  two equally-optimal 8-step routes — top vs bottom — so the task is genuinely
  multi-modal). Tiles: 0 free, 1 wall, 2 fire (terminal, -10), 3 goal (terminal, +10).
- **Observation (two modalities):**
  - STATE: 14-d float vector = agent (row,col)/(gs-1) [2] + goal (row,col)/(gs-1) [2]
    + 3x3 local tile patch flattened [9] + steps-remaining fraction [1].
    (For the equivariant policy the grid is re-encoded as a 5-channel one-hot map:
    0=agent,1=goal,2=fire,3=wall,4=free.)
  - IMAGE: 80x80x3 uint8 bird's-eye render (integer-aligned cells).
- **Action space:** `Discrete(4)` — UP / DOWN / LEFT / RIGHT.
- **Expert:** **A\*/BFS shortest-path oracle** (`Equivariant_pathway/expert.py`).
  Emits a length-4 *multi-label optimal-action mask* — every action that reduces
  BFS shortest-path-to-goal by 1 is a positive, so both optimal routes are taught
  (multi-modality preserved). Walls + fire are impassable.
- **Episode horizon:** env `MAX_STEPS`=200; rollout cap `max_steps`=60.

### T2 — Push = ManiSkill PushT  (IMPLEMENTED, both modalities)
- **Sim / benchmark:** ManiSkill3 (vendored via the Diff-DAgger fork at
  `external/diff_dagger` -> `/weka/s226137394/diff-dagger`). Suite driver:
  `pool_rl_robo`. Gym id `PushT-v2` (the fork's trained variant; suite id `PushT-v1`).
- **Robot / embodiment:** **Franka Panda** with a `panda_stick` end-effector
  (no gripper). Task: push a T-shaped block to a fixed goal T-pose; success =
  movable T overlaps goal T within tolerance.
- **Observation (two modalities), obs_horizon = 1:**
  - STATE (`pusht_state.yaml`): `obs_mode=state_dict`; policy obs_keys
    `[agent_qpos, extra_tcp_pose, extra_obj_pose]`, **proprio_dim = 21**
    (privileged T pose visible to the policy).
  - IMAGE (`configs/hydra/pusht_image.yaml`): `obs_mode=rgb`, base_camera **256x256**
    RGB; policy obs_keys `[agent_qpos, extra_tcp_pose, sensor_data_base_camera_rgb]`,
    proprio_dim = 14 (robot joints + stick TCP only; the T pose is NOT a policy
    input — the policy must read it from the image). Encoder obs_dim =
    512*(1+spatial_softmax)*n_views + proprio = 512*2*1 + 14 = **1038**.
- **Action space:** `rel_joint_pos`, action_dim = **9** (env control_mode
  `pd_joint_pos`). max_episode_steps = 250.
- **Expert:** **PPO** policy (`agents.RL_agent.MultipleExperts` -> `model.RL.PPO`,
  obs_dim 29, action_dim 7, ckpt `PushTHard_98...pth`; expert acts in joint_pos,
  success_mode). Same privileged expert for BOTH modalities, so demos are identical
  and the only moving axis between the state and image studies is the observation.

### T3 — Lift = RoboSuite / UR5e  (UPCOMING — NOT in code)
- **Sim / benchmark:** RoboSuite. **Robot / embodiment: UR5e arm.**
- **Intended obs / action:** state + image DIFFUSION policy (same backbone family
  as T2); lift an object off a table. Exact state dims / image size / action space
  TBD when authored.
- **Expert:** motion-planner (RoboSuite scripted / MP oracle).
- **Status:** NO implementation exists. Grep across the repo finds `Lift`, `UR5`,
  `robosuite` only in `paper_aaai2027/` (workflow + placeholders) — never in any
  `.py`. Placeholders stand in until the run lands.

### T4 — Wipe = RoboSuite / UR5  (UPCOMING — NOT in code)
- **Sim / benchmark:** RoboSuite. **Robot / embodiment: UR5 arm.** Task: wipe a
  spot off a table.
- state + image DIFFUSION policy; expert = motion-planner. Exact dims TBD.
- **Status:** NO implementation exists (only paper-side placeholders; the single
  code hit for "Wipe" is an unrelated `--wipe` CLI flag in `p4_only/pipeline.py`).

### T5 — Door open = RoboSuite / UR5  (UPCOMING — NOT in code)
- **Sim / benchmark:** RoboSuite. **Robot / embodiment: UR5 arm.** Task: open a door.
- state + image DIFFUSION policy; expert = motion-planner. Exact dims TBD.
- **Status:** NO implementation exists (only paper-side placeholders).

> **Implemented vs upcoming:** T1 and T2 are fully implemented and runnable today
> (grid-nav suite + ManiSkill PushT state/image). T3/T4/T5 (RoboSuite/UR5) are
> UPCOMING — no simulator wiring, policy config, expert, or results yet.

---

## B. Policy architectures (per task x modality)

### Toy STATE — equivariant MLP  (p4m-equivariant net)
- `EquivariantUNetPolicy` (`Equivariant_pathway/model.py`, escnn). Group
  `flipRot2dOnR2(N=4)` = **p4m / D4** (8-fold: 4 rotations x flip). Three stacked
  equivariant 3x3 R2Conv blocks (regular-rep channels (16,32,64); InnerBatchNorm +
  ReLU; last block dilation=2), then a 1x1 Conv2d head -> per-cell action logits;
  forward gathers the logits at the agent cell via the agent-channel one-hot mask.
  Input 5-channel one-hot grid map; output 4 logits. (The 8-fold symmetry is the
  toy-state inductive bias vs the plain CNN.)
- NOTE on code shape: the runnable toy suite trains an
  `EquivariantCNNHybridPolicy` (equivariant branch + a small RGB-CNN context branch
  + fusion MLP). For the paper the two toy modalities are reported as the two
  branches / two policy variants: **STATE = equivariant MLP**, **IMAGE = plain CNN**.

### Toy IMAGE — plain CNN
- `RGBCNNPolicy` (`CNN_pathway/rgb_cnn_policy.py`). Stacked _ConvBlock (Conv3x3 ->
  GroupNorm -> GELU, x2) + MaxPool2d, channels (32,64,128,256); AdaptiveAvgPool2d(1)
  -> 256-d feat -> MLP head (Linear 256->128 -> GELU -> Dropout -> Linear ->4).
  Non-equivariant. Input (B,3,80,80) RGB. Adaptive pool makes it grid-size-invariant.

### T2–T5 STATE and IMAGE — Diffusion Policy (shared backbone, all methods)
Every robot-task method (PACE, Diff-DAgger, and the IIL baselines) shares ONE
diffusion-policy architecture from the fork (`agents.diffusion_policy.DiffDAggerPolicy`);
only the demonstration-acquisition rule differs (apples-to-apples).
- **Denoiser:** `model.unet1d.ConditionalUnet1D` — 1-D conditional U-Net,
  diffusion_step_embed_dim 256, down_dims [256,512,1024], kernel_size 5, n_groups 8.
- **Horizons:** obs_horizon = **1**, pred_horizon = **32**, action_horizon = 2.
- **Diffusion:** DDIM scheduler, num_train_timesteps 16, **num_inference_steps 16**,
  beta 1e-4->0.02 squaredcos_cap_v2, prediction_type **v_prediction**.
- **Optim:** AdamW lr 3e-4, wd 0.01, cosine schedule, 500 warmup, train_bs 64.
- **STATE encoder:** MLP over proprio (proprio_dim, PushT = 21); no vision.
- **IMAGE encoder:** **R3M (ResNet-18)** vision encoder, **finetuned end-to-end**
  (frozen_encoder=false), BN->GroupNorm, **spatial-softmax** keypoints; obs_dim 1038
  for PushT. (Frozen global-512 R3M plateaus ~0.30 on tight alignment; finetune +
  spatial-softmax is the paper image recipe.)
- **DAgger query internals (used by Diff-DAgger & the P4/IIL detectors):** alpha 0.99
  diffusion-loss quantile, patience K=1, batch_multiplier (loss oversampling).
- **T3/T4/T5:** same diffusion-policy family (state MLP encoder / image R3M encoder);
  concrete configs authored when the RoboSuite tasks land.

---

## C. Active-loop (interactive-IL) protocol

Common structure for every task: bootstrap an initial policy from a few expert
demos, then repeat {roll out -> query rule flags a failure -> collect ONE new
expert demo -> retrain -> held-out eval} until a stop condition. **1 query / round
(hard cap: exactly 1 demo added per round)** for every method — the fair
sample-efficiency axis.

### Toy (pool_x_selector / baseline_vs_p4, `config.yml` + `config.yaml`)
- **Initial demos:** 20 A\*/BFS demos -> initial policy (shared across all methods
  and runs via a cached global bootstrap). initial_epochs 500.
- **Per-round rollout / pool:** a size-20 correction pool per round (`correction_n`;
  fixed-pool sampled once, or rotate = fresh 20/round). One demo prescribed/round.
- **Retrain:** **from scratch** each round (`train_from_scratch: true`; P4
  round_epochs 500, baseline_round_epochs 100). A replay-buffer fine-tuner variant
  exists (finetune_epochs 20, lr 5e-5, replay_mix 0.5) but the primary toy protocol
  is from-scratch.
- **Held-out eval:** 200 fixed held-out layouts (`heldout_n: 200`) evaluated every
  round (fixed held-out layout set on disk; same set across methods/runs).
- **Budget cap:** 15 extra demos on top of the initial 20 (`budget: 15`),
  max_rounds 50, max_steps 60.
- **Target SR:** 0.90.
- **Stop reasons:** target_sr reached | budget exhausted | max_rounds |
  (rotate pool) max_consecutive_empty zero-demo rounds.

### Robot tasks (pool_rl_robo, `config.yaml`) — T2 today; T3/T4/T5 same protocol
- **Initial demos:** 20 expert demos -> shared BC warm-start diffusion policy
  (`initial_demos: 20`, initial_epochs 300). ONE shared init per seed for all arms.
- **Per-round rollout:** discover a failure via each method's query rule
  (P4 failure-discovery rollout 60 episodes/round; Diff-DAgger native
  diffusion-loss CDF alpha=0.99, K=1). demos_per_round = 1.
- **Retrain:** the diffusion policy retrains **FROM SCRATCH every nd_retrain=4
  demos** (Diff-DAgger paper §III-C2), round_epochs 200, max_train_steps 30000 —
  shared cadence across all methods.
- **Held-out eval:** 100 held-out episodes per checkpoint (`heldout_n: 100`,
  vectorized eval_num_envs 20, heldout_seed_base **7777** — fixed), max_steps 200.
- **Budget cap:** 100 additional expert demos (`budget: 100`), max_rounds 100,
  max_episodes_per_arm 400.
- **Target SR:** 0.90 (early-stop and record the query count when hit).
- **Stop reasons:** target_sr reached | budget exhausted | max_rounds |
  max_episodes_per_arm.

---

## D. Metrics logged

Per (task x modality x method x seed), from `learning_curve.json`:
- **heldout success rate (`sr`)** — final held-out SR (toy: /200; robot: /100),
  reported mean±std over seeds. Best per column bold.
- **number of expert queries (`q`)** — expert queries (= demos added) to reach the
  0.90 target SR; if never reached, report the full budget. A secondary cost axis
  (`ExpertCallCounter`: total `get_action` / `move_to_next_goal` calls) is logged
  for auditability but `q` = demos-added is the headline query metric.
- **cumulative demos** — running demo count (`cum_demos` / `n_demos`), x-axis of
  the learning curve.
- **learning-curve AUC / demo-efficiency (`eff`)** — area under the SR-vs-#demos
  curve (higher = more sample-efficient). Reported per task x modality.
- **coverage (`cov`)** — state/space coverage of the collected demos (grid-cell /
  start-state coverage toy; workspace/pose coverage robot).

### `learning_curve.json` schemas

Toy (`pool_x_selector` / baseline_vs_p4), per method dir:
```json
{
  "method": "baseline_only_budget_hybrid", "run_index": 4,
  "budget": 15, "target_sr": 0.9,
  "demo_dir": "...", "checkpoint_dir": "...",
  "correction_yaml": "...", "heldout_yaml": "...",
  "history": [
    {"round": 0, "cum_demos": 20, "extra_demos": 0, "budget_remaining": 15,
     "heldout_sr": 0.41, "heldout_n_successes": 82, "heldout_n_episodes": 200,
     "correction_sr": null, "n_failures_in_pool": null, "n_picked": 0, "n_new_demos": 0},
    {"round": 1, "cum_demos": 35, "extra_demos": 15, "budget_remaining": 0,
     "heldout_sr": 0.785, "heldout_n_successes": 157, "heldout_n_episodes": 200,
     "correction_sr": 0.45, "n_failures_in_pool": 22, "n_picked": 15, "n_new_demos": 15}
  ]
}
```

Robot (`pool_rl_robo`), suite-normalized (converted from the fork schema):
```json
{
  "method": "p4_top3", "env": "PushT-v1", "seed": 1, "budget": 100,
  "backbone": "diffusion",
  "history": [
    {"round": 0, "n_queries": 0, "success_rate": 0.6, "demos_added": 0,
     "n_demos": 20, "stop_reason": null}
  ],
  "final_performance": {"success_rate": 0.9, "n_queries": 42},
  "stopped_reason": "target_reached"
}
```
Fork-native per-round keys (source, before conversion): `round`, `expert_calls`,
`heldout_success_rate`, `demos_added`, `cumulative_demos`, `stop_reason`.

Aggregate/figure artefacts: `results/curve.json` (`{n_points, x, y, points}` mean
SR-vs-demos), per-task×modality learning-curve PDFs, P4-component ablation, and
qualitative failure->prescription panels.

---

## E. Seeds

**5 seeds** per (task x modality x method) cell; every metric reported as
mean±std over the 5 seeds. (Suite `config.yaml` files default `n_runs`/`seed` to
larger sweeps for development; the paper reports **5 seeds**, per-run seed =
base_seed + run_id, with the SAME shared bootstrap per seed across all methods so
the only difference between arms is the demo-acquisition rule.)

---

## F. Method roster (rows of every results table)
- **PACE (ours)** — Perceive->Partition->Prioritize->Prescribe.
- **SafeDAgger**, **DropoutDAgger**, **EnsembleDAgger**, **ThriftyDAgger** — published IIL baselines.
- **Random ("stagger")** — uniform-random 1-demo/round control (NOT a published method).
- **Diff-DAgger** — robot tasks ONLY (T2–T5), NOT on the toy grid.
</content>
</invoke>
