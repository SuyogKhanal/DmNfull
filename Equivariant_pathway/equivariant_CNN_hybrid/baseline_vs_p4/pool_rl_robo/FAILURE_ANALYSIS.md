# FAILURE_ANALYSIS.md — why the first pool_rl_robo environment set was abandoned

The first build of this suite ported the maze "P4-LLM vs IIL-baselines" comparison
to continuous control using **gym/MuJoCo locomotion + Fetch manipulation** envs,
with SB3/HuggingFace experts and an MLP (locomotion) / R3M-image diffusion (Fetch)
novice. It ran to completion but produced **no usable comparison**: either the wrong
metric (reward, not task success) or no learning at all. This file is the paper
record. Numbers below are from the `run_0` production curves and the smoke logs
(`results/*_run0.json`, `logs/*.log`) of that build.

## What broke, per environment

| Env | Type | Failure mode | run_0 final |
|---|---|---|---|
| **HalfCheetah-v4** | MuJoCo locomotion | No success metric (reward-only); BC-from-few-demos reward is noisy & non-monotonic; methods don't separate. `ensemble_dagger` even collapsed. | mean_reward 1157→2176; `ensemble` **−21** (collapse); `sr=None` |
| **Hopper-v4** | MuJoCo locomotion | Same: reward-only, high variance round-to-round, no success rate, no clean sample-efficiency signal. | mean_reward 1198–3140 (no ordering); `sr=None` |
| **Walker2d-v4** | MuJoCo locomotion | Same: reward-only, noisy; demonstration *selection* unmeasurable. | mean_reward 1093–1863; `sr=None` |
| **FetchReach-v4** | Sparse-reward manip | Trivial reach task; saturates/noisy at moderate SR; too easy to separate methods, and reward is the sparse −1/step floor so the headline metric is confusing. | sr 0.30–0.65 (no separation) |
| **FetchPickAndPlace-v4** | Sparse-reward manip | **Novice never learns.** R3M-image diffusion BC on 5–8 demos + light fine-tune stays at sr≈0 every round; reward pinned at the −50 sparse floor. P4 prescribed solvable configs (`solved=True`) yet eval `sr` stayed 0.0 — so there is no learning signal to separate methods. | **sr 0.0–0.05** (all methods); reward −50 |

## Root-cause verdict

- **Not** an installation, expert-availability, or LLM-pipeline bug — the experts
  loaded (smoke gate passed), demos were collected, P4's LLM prescribed and the
  expert solved prescribed configs. The failure is **task/learning-regime**:
  - **Locomotion (HalfCheetah/Hopper/Walker2d)** is the wrong task family for this
    study — there is no binary task-success metric, reward is dense/noisy, and
    "which demonstration to acquire" cannot be read off a noisy reward curve. The
    new design constraint (NO RL locomotion) reflects this.
  - **Fetch** is the right family (sparse success) but the wrong difficulty/backbone
    pairing: FetchReach is too easy (saturates, no separation), and the
    FetchPickAndPlace image-diffusion novice never crossed sr>0 on the tiny demo
    budget — every method degenerates to sr≈0, so the comparison is vacuous.

**Conclusion → move to ManiSkill robotic manipulation** (StackCube-v1, PickCube-v1,
PushT-v1, PlugCharger-v1) with a **shared Diffusion-Policy backbone** and an
**automated motion-planner / PPO expert**. These give (a) a real binary success
metric, (b) tasks hard enough that demonstration acquisition gates success, and
(c) a policy class strong enough to actually learn from the prescribed demos — the
regime where P4-LLM's selection can reproducibly beat the IIL baselines. The old
environment layer (`envs/`, MLP/R3M policies, MuJoCo/Fetch experts) is removed in
favour of the ManiSkill stack vendored in the user's Diff-DAgger fork.
