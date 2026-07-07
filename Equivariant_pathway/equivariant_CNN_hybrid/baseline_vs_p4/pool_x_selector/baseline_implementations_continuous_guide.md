
```markdown
# Baseline Implementation Guide: DAgger-Family Methods for Continuous Robotic Manipulation
# (Continuous Action Space Version — pool_rl_robo/)

> **Do NOT use the old `baseline_implementations_guide.md` for this suite.**
> This document supersedes it for pool_rl_robo/. The old guide was written
> for a discrete grid-world action space. This guide is scoped to continuous
> robotic manipulation with ManiSkill tasks (StackCube-v1, PickCube-v1,
> PushT-v1, PlugCharger-v1), obs_mode="state", control_mode="pd_ee_delta_pose",
> and a Franka Panda arm with action ∈ ℝ⁷ (end-effector delta pose + gripper).

---

## Overview

This document provides algorithmically precise descriptions of five interactive
imitation learning (IIL) baselines to compare against P4-LLM (p4_top3) in the
continuous action space setting. The methods are **SafeDAgger**, **DropoutDAgger**,
**EnsembleDAgger**, **ThriftyDAgger**, and **Stagger**. A sixth baseline,
**Diff-DAgger**, is sourced directly from the cloned Diff-DAgger repo and is
not specified here — use it as-is from `external/diff_dagger/`.

The coding LLM must implement each method faithfully using the exact decision
rules, training procedures, and switching criteria described below.

---

## Scope and Constraints

- **State space:** S ⊆ ℝⁿ (ManiSkill `obs_mode="state"` vectors; n varies by task)
- **Action space:** A ⊆ ℝ⁷ (pd_ee_delta_pose: [Δx, Δy, Δz, Δqx, Δqy, Δqz, gripper])
- **Episode horizon:** T = 200 steps
- **Expert:** ManiSkill motion planner (StackCube, PickCube, PlugCharger) or
  HDF5 replay (PushT). Interface: `expert.query(obs) → action ∈ ℝ⁷`
- **Policy backbone:** Diffusion Policy from `external/diff_dagger/` for ALL
  methods. MLP backbones described in original papers are replaced by Diffusion
  Policy for fair comparison. See "Policy Backbone Note" below.
- **Demo budget:** initial_demos=50, budget=100, demos_per_round=1 (all methods)
- **Target SR:** 0.90 (stop early if reached; record query count)

---

## Policy Backbone Note (Critical — Read Before Implementing Any Baseline)

The original papers for SafeDAgger, DropoutDAgger, EnsembleDAgger, and
ThriftyDAgger used MLP policies. In this suite ALL methods use Diffusion Policy
as the novice policy π_nov for fairness. The following adaptations apply:

- **DropoutDAgger:** MC-Dropout is applied to the linear layers of the diffusion
  policy's noise prediction network (the denoising U-Net). Enable dropout at
  BOTH training and inference time. Run N forward passes with different dropout
  masks to obtain the action distribution.

- **EnsembleDAgger:** Train M independent diffusion policy instances on the same
  aggregated dataset D, each with a different random seed. Ensemble variance is
  computed over the M denoised action predictions at inference.

- **ThriftyDAgger:** Use ensemble variance from M diffusion policy instances as
  the novelty estimator. The goal-conditioned Q-function risk estimator uses a
  separate MLP — it does NOT need to be a diffusion model. The Q-function takes
  (s_t, a_t) ∈ ℝ^{n+7} and outputs a scalar success probability.

- **SafeDAgger and Stagger:** No architecture changes needed — these methods
  only use the policy's mean action output, which is unchanged.

---

## Common Setup (Shared Across All Baselines)

All five baselines share the following DAgger outer loop:

```
Algorithm: DAgger (Base — Continuous Manipulation)
Input: Expert policy π_exp, Diffusion Policy novice π_nov
       Initial dataset D_0 ← {50 motion-planner trajectories}
Initialize: π_nov,0 ← BC on D_0

Phase 2 loop (round i = 1 to max_rounds=100):
    1. Roll out π_nov,i for T=200 steps → trajectory τ_i
    2. Apply decision rule DR(·) to determine correction demo
    3. Request expert correction: call expert.query(obs) for relevant states
    4. Add ONE new trajectory to D: D ← D ∪ {τ_correction}
    5. Fine-tune π_nov,i+1 on D
    6. Evaluate on heldout_n=100 episodes → record SR
    7. If SR ≥ 0.90: stop early, record round i as query_count_to_90
```

**Key shared definitions:**
- `π_nov(s)` — novice policy mean action output (denoised action from diffusion)
- `π_exp(s)` — expert policy output (motion planner or HDF5 replay)
- `‖·‖₂` — L2 norm over the 7-dimensional action vector
- `D` — aggregated dataset of (state, expert-action) trajectory pairs
- `demos_per_round = 1` — exactly one trajectory added per round, all methods

---

## Baseline 1: SafeDAgger

**Paper:** Zhang & Cho, "Query-Efficient Imitation Learning for End-to-End
Autonomous Driving," AAAI 2017. arXiv:1605.06450.

**Core Idea:** At each timestep, compare the novice's action to the expert's
action. If the L2 discrepancy exceeds threshold τ, the expert intervenes and
that (state, expert-action) pair is recorded. This identifies the specific
trajectory segment where the novice diverges from the expert.

**In the continuous manipulation setting:** τ is defined in pd_ee_delta_pose
space (ℝ⁷). A reasonable default is τ=0.1, but this should be tuned per task
since action magnitudes differ between translation and rotation components.

### Decision Rule

```
DR_SafeDAgger(s_t, τ):
    a_nov = π_nov(s_t)
    a_exp = π_exp(s_t)
    if ‖a_nov − a_exp‖₂ > τ:
        return INTERVENE  # expert acts; record (s_t, a_exp)
    else:
        return AUTONOMOUS  # novice acts
```

### How One Demo Is Selected Per Round

Run the current novice policy for one full episode. Identify the timestep t*
of first intervention (argmin_t {t : DR=INTERVENE}). Collect the expert
trajectory from t* to episode end. This is the ONE correction demo added this
round.

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `τ` | 0.1 | Action discrepancy threshold in ℝ⁷ |

### Applicable Environments (this suite)

- StackCube-v1  (grasp + stack, action ∈ ℝ⁷, T=200)
- PickCube-v1   (grasp + transport, sparse reward, T=200)
- PushT-v1      (non-prehensile push, no gripper action, T=200)
- PlugCharger-v1 (contact-rich insertion, tight tolerances, T=200)

---

## Baseline 2: DropoutDAgger

**Paper:** Menda, Driggs-Campbell & Kochenderfer, "DropoutDAgger: A Bayesian
Approach to Safe Imitation Learning," arXiv:1709.06166, 2017.

**Core Idea:** Use MC-Dropout on the Diffusion Policy noise prediction network
to obtain N action samples at each timestep. If a sufficient fraction p of
the samples fall within a ball of radius τ around the expert's action, the
novice acts. Otherwise, the expert intervenes. This uses Bayesian uncertainty
to decide when the novice is confident enough to act autonomously.

**In the continuous manipulation setting:** MC-Dropout is applied to the
linear (dense) layers of the diffusion policy's U-Net denoiser. Enable dropout
at BOTH training and inference. Each of the N forward passes uses a different
randomly sampled dropout mask, producing N distinct denoised action samples.

### Decision Rule

```
DR_DropoutDAgger(s_t, τ, p, N):
    # N forward passes through diffusion policy with dropout active
    {a_j}_{j=1..N} ← [π_nov(s_t) with dropout mask j]
    a_exp = π_exp(s_t)
    p_hat = (1/N) * Σ_{j=1..N} 1{‖a_exp − a_j‖₂ ≤ τ}
    if p_hat ≥ p:
        return AUTONOMOUS  # mean action: (1/N) Σ a_j
    else:
        return INTERVENE   # expert acts; record (s_t, a_exp)
```

### How One Demo Is Selected Per Round

Same as SafeDAgger: run full episode, identify first intervention timestep t*,
collect expert trajectory from t* to end. One trajectory added per round.

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `τ` | 0.1 | Ball radius around expert action |
| `p` | 0.9 | Probability threshold for autonomous action |
| `N` | 10 | Number of MC-Dropout forward passes per timestep |
| `d` | 0.1 | Dropout probability per linear layer |

### Key Properties

- If d=0: reduces to SafeDAgger* (deterministic threshold)
- If τ=0: expert labels everything (Behaviour Cloning)
- If p=0: novice always acts (no expert queries during rollout)

### Applicable Environments (this suite)

All four tasks. PushT-v1 note: gripper dimension is inactive — uncertainty
should be evaluated over the 6 active action dimensions only.

---

## Baseline 3: EnsembleDAgger

**Paper:** Menda, Driggs-Campbell & Kochenderfer, "EnsembleDAgger: A Bayesian
Approach to Safe Imitation Learning," IROS 2019. arXiv:1807.08364.

**Core Idea:** Train M independent Diffusion Policy instances. At each timestep,
the novice acts only if BOTH: (1) the mean ensemble action is close to the
expert action (discrepancy rule), AND (2) the ensemble variance is below a
threshold χ (doubt rule). This provides more reliable uncertainty estimates
than single-model dropout.

**In the continuous manipulation setting:** Each ensemble member is a full
Diffusion Policy trained independently on the same aggregated dataset D with
a different random seed. The ensemble mean and variance are computed over the
M denoised action predictions at inference (no dropout needed).

### Decision Rule

```
DR_EnsembleDAgger(s_t, τ, χ):
    # M forward passes through M independent diffusion policy instances
    {a_m}_{m=1..M} ← [π_nov_m(s_t) for m in 1..M]
    a_bar = (1/M) * Σ_m a_m            # ensemble mean
    σ² = (1/(M-1)) * Σ_m ‖a_m − a_bar‖₂²  # ensemble variance (scalar)
    a_exp = π_exp(s_t)
    tau_hat = ‖a_bar − a_exp‖₂         # discrepancy
    if tau_hat ≤ τ AND σ² ≤ χ:
        return AUTONOMOUS  # act with mean action a_bar
    else:
        return INTERVENE   # expert acts; record (s_t, a_exp)
```

### How One Demo Is Selected Per Round

Same as SafeDAgger: run full episode, first intervention at t*, expert
trajectory from t* to end. One trajectory added per round.

### Training Note

After each round, ALL M ensemble members are retrained on the updated D.
Use different random seeds for each member. Do not share weights between
ensemble members at any point.

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `M` | 5 | Number of independent Diffusion Policy instances |
| `τ` | 0.1 | Discrepancy threshold (mean vs. expert action) |
| `χ` | 0.05 | Doubt threshold (ensemble variance) |

### Key Property

As χ → ∞: reduces to SafeDAgger* (only discrepancy matters).
The original paper finds the doubt rule (variance-only) Pareto-dominates
the discrepancy rule alone — set χ conservatively.

---

## Baseline 4: ThriftyDAgger

**Paper:** Hoque, Balakrishna, Novoseller, Wilcox, Brown & Goldberg,
"ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive
Imitation Learning," CoRL 2021. arXiv:2109.08273.

**Core Idea:** Intervene when the current state is either: (1) novel (outside
the distribution of states the novice has seen), OR (2) risky (the novice is
unlikely to succeed from here). Thresholds are auto-calibrated from a budget
parameter α_h (target intervention fraction). This is the most expensive
baseline to implement but the most principled.

**In the continuous manipulation setting:**
- Novelty estimator: ensemble variance from M Diffusion Policy instances
  (same M instances used for EnsembleDAgger if running both — share the
  ensemble to avoid re-training M policies twice per round)
- Risk estimator: a goal-conditioned Q-function (separate MLP, not a diffusion
  model) trained on the aggregated dataset D. Input: (s_t, a_t) ∈ ℝ^{n+7},
  output: scalar ∈ [0,1] (estimated probability of reaching the goal set G)
- Goal set G: success condition as defined by ManiSkill's `info["success"]`

### Component 1: Novelty Estimation

```
Novelty(s_t) = (1/(M-1)) * Σ_m ‖π_nov_m(s_t) − a_bar‖₂²
             = ensemble variance of the M diffusion policy action predictions
```

High variance ↔ novel state (outside training distribution).

### Component 2: Risk Estimation

Define the success Q-function:
```
Q(s_t, a_t) = E[Σ_{t'=t}^{T} γ^{t'-t} · 1_G(s_{t'}) | s_t, a_t]
```
where 1_G(s) = 1 if `info["success"]` is True at state s.

Train Q̂ by minimising:
```
J_Q = (1/2) * (Q̂(s_t,a_t) − (1_G(s_t) + (1−1_G(s_t)) · γ · Q̂(s_{t+1}, π_nov(s_{t+1}))))²
```

Risk at timestep t:
```
Risk(s_t, a_t) = 1 − Q̂(s_t, a_t)
```

### Component 3: Switching Policy

```
Intervene (autonomous → expert) if:
    Novelty(s_t) > δ_h   OR   Risk(s_t, π_nov(s_t)) > β_h

Cede (expert → autonomous) if:
    ‖π_nov(s_t) − π_exp(s_t)‖₂ < δ_r   AND   Risk(s_t, π_nov(s_t)) < β_r

Note: β_r < β_h  (harder to exit expert mode than to enter it)
```

### Component 4: Budget-Based Threshold Calibration

Given desired intervention rate α_h (e.g. 0.10 = 10% of timesteps):
```
β_h = (1−α_h)-quantile of {Risk(s, π_nov(s))} over previously visited states
δ_h = (1−α_h)-quantile of {Novelty(s)} over previously visited states
δ_r = mean ‖π_nov(s) − π_exp(s)‖₂ on expert-visited states after retraining
β_r = median {Risk(s, π_nov(s))} over previously visited states
```

Recompute these four thresholds after every round.

### How One Demo Is Selected Per Round

Run the novice until first intervention. Collect expert trajectory from that
point to episode end (or until cede condition is met). This is the ONE
correction trajectory added this round.

### Full Algorithm

```
Algorithm: ThriftyDAgger (Continuous Manipulation)
Init: π_nov,0 ← BC on D_0 (50 initial demos)
      Q̂ ← train on D_0
      Set β_h, β_r, δ_h, δ_r from α_h

For round i = 1 to max_rounds:
    mode ← AUTONOMOUS
    For t = 1 to T:
        if mode = AUTONOMOUS:
            if Intervene(s_t, δ_h, β_h):
                mode ← EXPERT
            else:
                execute a_t = π_nov(s_t); log (s_t, a_t, s_{t+1})
        if mode = EXPERT:
            a_t = π_exp(s_t); log (s_t, a_t, s_{t+1}); add to D
            if Cede(s_t, δ_r, β_r):
                mode ← AUTONOMOUS
    D ← D ∪ {expert segments from this episode}  [cap at 1 trajectory]
    Retrain π_nov,i+1 on D
    Update Q̂ on D
    Recompute β_h, β_r, δ_h, δ_r
    Evaluate SR; if SR ≥ 0.90: stop
```

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `α_h` | 0.10 | Target expert intervention fraction of timesteps |
| `γ` | 0.99 | Q-function discount factor |
| `M` | 5 | Ensemble size for novelty estimator (share with EnsembleDAgger) |

---

## Baseline 5: Stagger

**Paper:** Swamy, Choudhury, Bagnell & Wu, "Interactive and Hybrid Imitation
Learning: Provably Beating Behavior Cloning," NeurIPS 2024. arXiv:2412.07057.

**Core Idea:** Query the expert for EXACTLY ONE state-action pair per round,
chosen uniformly at random from the current rollout trajectory. This minimal
interaction schedule has a clean theoretical guarantee: interactive IL with
one sample per round provably beats BC when the environment has low recovery
cost. Stagger serves as the theoretical lower-bound baseline.

**In the continuous manipulation setting:** Stagger is a natural fit because
this suite already caps all methods at demos_per_round=1. Stagger's defining
property — one sample per round — is structurally identical to the budget
constraint imposed across all 7 methods. It is the simplest possible IIL
method and sets the theoretical floor for what any one-query-per-round method
can achieve.

**No uncertainty metric, no risk metric, no threshold tuning.** The query
schedule is purely round-based.

### Algorithm

```
Algorithm: Stagger (Continuous Manipulation)
Init: π_nov,0 ← BC on D_0 (50 initial demos)

For round i = 1 to max_rounds:
    1. Roll out π_nov,i for T=200 steps → trajectory τ_i = {s_1,...,s_T}
    2. Sample ONE state uniformly: s* ~ Uniform(τ_i)
    3. Query expert: a* = π_exp(s*)
    4. Add (s*, a*) to D  [this is a single (state, action) pair, not a
       full trajectory — see note below]
    5. Fine-tune π_nov,i+1 on D
    6. Evaluate SR; if SR ≥ 0.90: stop
```

**Implementation note on "one sample" vs. "one trajectory":**
The original Stagger paper adds a single (state, action) pair per round.
However, for consistent comparison in this suite where all other methods add
one full expert trajectory per round, use the following adaptation:

  Stagger (adapted): sample s*, then collect the expert trajectory from s*
  to episode end (T − t* steps). This is still ONE expert episode segment
  (the minimal correction from the sampled failure point), keeping budget
  comparable to other methods.

If you want to be strictly faithful to the paper's single-sample protocol for
an ablation, set a config flag `stagger_single_sample: true` to switch
between the two variants.

### Hyperparameters

None. The protocol is fully determined by `max_rounds` and `demos_per_round=1`.

### Theoretical Context

Let C_rec = recovery cost (expected steps to return to a nominal trajectory).
Stagger provably achieves suboptimality that scales with C_rec rather than
the compounding error rate of BC. For manipulation tasks where recovery is
cheap (e.g. PickCube — the arm can simply re-approach), Stagger should
outperform BC and approach the performance of more sophisticated methods.
For tasks where recovery is hard (e.g. PlugCharger — precise re-alignment
is expensive), Stagger's advantage over BC narrows.

---

## Evaluation Protocol

For all 7 methods (p4_top3, diff_dagger, and the five above), report:

1. **Success Rate (SR) vs. Expert Queries** — primary plot: SR curve over
   the 100-query Phase 2 budget. X-axis = cumulative expert queries (0–100),
   Y-axis = SR on heldout_n=100 evaluation episodes.

2. **Queries to 90% SR** — the number of Phase 2 queries needed to first
   reach target_sr=0.90. If a method never reaches 0.90 within 100 queries,
   report as ">100". This is the primary efficiency metric.

3. **Final SR at budget exhaustion** — SR at query 100 for methods that did
   not reach 0.90.

4. **Phase 1 cost** — number of initial demos required to reach 50% SR from
   the 50 warm-start demos. (All methods share the same 50 initial demos,
   so Phase 1 cost should be identical across methods — report as a sanity
   check.)

5. **Expert intervention fraction** (ThriftyDAgger only) — fraction of
   timesteps per episode where the expert was in control during training.

When plotting across environments, normalise by expert queries (not wall-clock
time) since motion planner speed varies by task.

---

## Implementation Checklist

- [ ] All five baselines share the same DAgger outer loop (Phase 2: roll out →
      decision rule → 1 demo added → fine-tune → evaluate)
- [ ] Only the decision rule / query criterion differs between baselines
- [ ] Diffusion Policy backbone used for ALL methods (no MLP novice policies)
- [ ] DropoutDAgger: dropout enabled at BOTH training and inference time
- [ ] EnsembleDAgger: M independent diffusion policy instances, different seeds,
      retrained on D after every round
- [ ] ThriftyDAgger: novelty from ensemble variance; risk from separate Q-MLP;
      thresholds auto-calibrated from α_h after every round; Q-function updated
      every round
- [ ] Stagger: no uncertainty metric; uniform state sampling; one expert segment
      per round
- [ ] demos_per_round=1 enforced for ALL methods — if any method adds >1
      trajectory in a round, raise BudgetViolationError
- [ ] target_sr=0.90 early-stopping checked after every round for all methods
- [ ] State/action dimensions NOT hardcoded — read from env.observation_space
      and env.action_space at runtime
- [ ] All methods tested on at least StackCube-v1 before running SLURM array
```

***



```markdown
---

## Baseline 6: Diff-DAgger

**Source:** SungWookLee-HCI/Diff-DAgger (GitHub). Cloned to
`external/diff_dagger/` in STEP 1. Install with `pip install -e .`.

**Paper:** Lee et al., "Diff-DAgger: Uncertainty Estimation with Diffusion
Policy for Robotic Manipulation," arXiv:2410.14868, 2024.

**Core Idea:** Diff-DAgger uses the **denoising loss of a Diffusion Policy
as a direct uncertainty signal** to decide when to query the expert. During
a rollout, the policy computes its denoising loss at each timestep. When
the loss exceeds a threshold λ, the current state is flagged as high-uncertainty
and the expert is queried. This is the only baseline in this suite where the
uncertainty signal is derived natively from the diffusion process itself —
no separate ensemble, no MC-Dropout, no risk Q-function.

**Relationship to P4-LLM:** Diff-DAgger and P4-LLM share the same uncertainty
trigger (peak diffusion loss). The difference is what each method does with
that signal:
  - Diff-DAgger: queries the expert immediately at the high-loss timestep and
    adds the resulting (state, expert-action) correction to the dataset.
  - P4-LLM (p4_top3): passes the high-loss frame to the VLM → Reasoning LLM
    → Config LLM pipeline to generate a targeted correction episode from a
    prescribed initial configuration. The expert then solves that prescribed
    episode, not the original rollout state.

This makes Diff-DAgger the most direct ablation of P4-LLM — same loss signal,
different correction strategy.

### Decision Rule

```
DR_DiffDAgger(s_t, λ):
    a_nov, L_t ← π_nov(s_t)   # diffusion policy forward pass returns
                                # both the denoised action and denoising loss
    if L_t > λ:
        return INTERVENE       # expert acts; record (s_t, π_exp(s_t))
    else:
        return AUTONOMOUS      # novice acts with a_nov
```

### How One Demo Is Selected Per Round

Run the current novice policy for one full episode. Record the denoising loss
at every timestep. Identify t* = argmax_t L_t (peak loss timestep). Query the
expert from t* to episode end. This one expert segment is the correction demo
added this round.

This mirrors how P4-LLM uses the high_loss_frame — the same t* is the trigger.
The only difference is that Diff-DAgger queries the expert in the original
rollout context, while P4-LLM prescribes a new initial configuration.

### Implementation Instructions

**Do NOT reimplement Diff-DAgger.** Use the existing implementation from the
cloned repo:

```
external/diff_dagger/
```

After cloning and installing (`pip install -e .`), confirm it exposes:
  - The diffusion policy training loop (used as shared backbone for all methods)
  - The DiffDAgger decision rule (used only for the diff_dagger method)
  - Any other IIL baselines already implemented in the repo (safe_dagger,
    thrifty_dagger, etc.) — use those directly if present

If the repo's DiffDAgger implementation does not support ManiSkill environments
out of the box, write a thin adapter in `pool_rl_robo/baselines/diff_dagger_adapter.py`
that:
  1. Wraps the ManiSkill env to match whatever interface Diff-DAgger expects
  2. Exposes `expert.query(obs) → action` using the ManiSkill motion planner
  3. Does NOT modify any files inside `external/diff_dagger/`

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `λ` | Repo default | Denoising loss threshold for expert query |

Use the threshold value from the Diff-DAgger repo's default config. Do not
tune it separately — use whatever the authors set as the default for
manipulation tasks.

### Applicable Environments (this suite)

All four tasks. The denoising loss signal is environment-agnostic — it depends
only on how well the diffusion policy can denoise its own action predictions,
which is meaningful in all continuous manipulation settings.

### Why Diff-DAgger Is the Critical Baseline

Diff-DAgger is the single most important baseline for P4-LLM because:
1. It uses the identical uncertainty trigger (peak diffusion loss at t*)
2. It uses the identical policy backbone (Diffusion Policy)
3. The only difference is the correction strategy

If P4-LLM outperforms Diff-DAgger, it means the VLM → Reasoning LLM →
Config LLM prescription step adds value beyond simply querying the expert
at the point of failure. This is the central empirical claim of pool_rl_robo/.
```

***

Also update the **Implementation Checklist** at the bottom of the guide to add this entry:

```markdown
- [ ] Diff-DAgger sourced directly from external/diff_dagger/ — NOT reimplemented
- [ ] If ManiSkill adapter is needed, it lives in
      pool_rl_robo/baselines/diff_dagger_adapter.py and does NOT modify
      any files inside external/diff_dagger/
- [ ] Diff-DAgger threshold λ uses the repo's default — not separately tuned
- [ ] Confirm that the Diff-DAgger denoising loss is accessible per-timestep
      during rollout (needed to identify t*) before proceeding to STEP 4
```

And update the **Comparison Summary table** (add as the last row before Evaluation Protocol):

```markdown
| **Diff-DAgger** | Peak diffusion denoising loss > λ | Native diffusion loss (no ensemble/dropout) | Threshold λ (repo default) | Lee et al., arXiv 2024 |
```


That is the complete baseline_implementations_continuous_guide.md. The key differences from the old discrete guide are:

Scope locked to ManiSkill + pd_ee_delta_pose — no MuJoCo locomotion, no grid world, no Dubins Car

Diffusion Policy backbone specified for all five methods with exact adaptation instructions for MC-Dropout and ensemble variants

demos_per_round=1 / target_sr=0.90 enforced throughout and consistent with the prompt

correction_n never appears — removed entirely as that concept does not exist here

Stagger adapted with a note on single-sample vs. single-trajectory variants so the coding LLM knows which to use and how to add an ablation flag