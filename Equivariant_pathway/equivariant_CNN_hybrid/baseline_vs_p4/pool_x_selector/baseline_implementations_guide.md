# Baseline Implementation Guide: DAgger-Family Methods for Comparison with P4-LLM

## Overview

This document provides environment-agnostic, algorithmically precise descriptions of five interactive imitation learning (IIL) baselines derived directly from their original papers. These methods are **SafeDAgger**, **DropoutDAgger**, **EnsembleDAgger**, **ThriftyDAgger**, and **Stagger**. The coding LLM is expected to implement each method faithfully, using the exact decision rules, training procedures, and switching criteria described below.

The method being compared against these baselines is referred to as **P4-LLM** throughout this document.

> **Environment Agnosticism Note:** These baselines apply to any environment with a state space `S`, action space `A`, and time horizon `T`. Concrete examples are provided across multiple domains — robotic manipulation (peg insertion, cable routing), continuous locomotion control (MuJoCo HalfCheetah), and autonomous navigation (Dubins Car) — to prevent over-specialization. The implementation must generalize across these settings.

---

## Common Setup (Shared Across All Baselines)

All five baselines share the following base DAgger structure:

```
Algorithm: DAgger (Base)
Input: Expert policy π_exp, supervised learner (novice policy π_nov)
Initialize: Dataset D ← ∅, initialize π_nov,0
For epoch i = 1 to K:
    Sample T-step trajectory using decision rule DR(·)
    Collect D_i = {(s, π_exp(s))} for all states s visited
    Aggregate: D ← D ∪ D_i
    Train π_nov,i+1 on D via supervised learning
```

The **decision rule DR(·)** is what differentiates each baseline. At every timestep `t`, DR decides whether to execute the novice's action `a_nov,t` or the expert's action `a_exp,t`.

**Key shared definitions:**
- `π_nov(s)` — novice policy output (mean action)
- `π_exp(s)` — expert policy output
- `‖ · ‖` — L2 norm unless otherwise specified
- `τ` — action discrepancy threshold (method-specific)
- `D` — aggregated dataset of (state, expert-action) pairs

---

## Baseline 1: SafeDAgger

**Paper:** Zhang & Cho, "Query-Efficient Imitation Learning for End-to-End Autonomous Driving," AAAI 2017. arXiv:1605.06450.

**Core Idea:** Train a secondary "safety policy" that predicts whether the novice's action will deviate from the expert's action by more than a threshold `τ`. Query the expert only when this predicted deviation exceeds `τ`. This reduces unnecessary expert queries while addressing the covariate shift problem of vanilla DAgger.

### Algorithm

```
Algorithm: SafeDAgger Decision Rule (SafeDAgger*)
Input: Observation o_t, threshold τ
1. a_nov,t ← π_nov(o_t)
2. a_exp,t ← π_exp(o_t)
3. if ‖a_nov,t − a_exp,t‖₂ ≤ τ:
       return a_nov,t   # novice acts
   else:
       return a_exp,t   # expert intervenes
```

> **Note on two variants:** The paper distinguishes SafeDAgger* (the oracle version above, which directly compares actions) from SafeDAgger (which trains a deep classifier to *predict* whether the deviation will exceed τ, thereby avoiding the need to always call the expert). For baseline comparison purposes, implement **SafeDAgger*** unless the expert is unavailable at inference time.

### Training Procedure

```
Initialize D ← ∅, π_nov,0
For epoch i = 1 to K:
    Run episode with DR_SafeDAgger(o_t, τ) at each step
    Collect D_i = {(s, π_exp(s))} for all visited states
    D ← D ∪ D_i
    Train π_nov,i+1 on D
```

### Hyperparameters
| Parameter | Role |
|-----------|------|
| `τ` | Action discrepancy threshold; the novice acts if deviation from expert is below this value |
| `K` | Number of DAgger epochs |
| `T` | Maximum episode length |

### Applicable Environments
- Continuous control (e.g., MuJoCo locomotion: HalfCheetah, Hopper)
- Autonomous driving (e.g., CARLA simulator)
- Robotic manipulation (e.g., peg insertion, grasping)

---

## Baseline 2: DropoutDAgger

**Paper:** Menda, Driggs-Campbell & Kochenderfer, "DropoutDAgger: A Bayesian Approach to Safe Imitation Learning," arXiv:1709.06166, 2017.

**Core Idea:** Train the novice policy as a Bayesian neural network using Monte Carlo Dropout. At each timestep, query the network `N` times with randomly sampled dropout masks to obtain a distribution over actions. The novice acts only if a sufficient fraction `p` of the sampled actions fall within a ball of radius `τ` around the expert's action. Otherwise, the expert intervenes.

### Algorithm

```
Algorithm: DropoutDAgger Decision Rule
Input: Observation o_t, thresholds τ (ball radius), p (probability threshold), N (dropout samples)
1. Sample N actions: {a_nov,t,j}_{j=1..N} ← π_nov(o_t)  # N forward passes with dropout
2. a_exp,t ← π_exp(o_t)
3. p_hat ← (1/N) * Σ_{j=1..N} 1{‖a_exp,t − a_nov,t,j‖₂ ≤ τ}
4. if p_hat ≥ p:
       return (1/N) * Σ_{j=1..N} a_nov,t,j   # novice acts (mean action)
   else:
       return a_exp,t                           # expert intervenes
```

### Key Properties
- If dropout probability `d = 0`, DropoutDAgger reduces to SafeDAgger*.
- If `τ = 0`, reduces to Behavior Cloning (novice never acts).
- If `p = 0`, expert only labels but never controls the system during rollout.

### Training Procedure

```
Initialize D ← ∅, π_nov,0 (trained with dropout at every weight layer, dropout probability d)
For epoch i = 1 to K:
    Run episode with DR_DropoutDAgger(o_t, τ, p, N) at each step
    Collect D_i = {(s, π_exp(s))} for all visited states
    D ← D ∪ D_i
    Train π_nov,i+1 on D (with dropout enabled during both training and inference)
```

### Hyperparameters
| Parameter | Role |
|-----------|------|
| `τ` | Ball radius around expert action for probability estimation |
| `p` | Probability threshold; novice acts if ≥ p fraction of dropout samples are within ball |
| `N` | Number of dropout forward passes per timestep |
| `d` | Dropout probability at each weight layer |

### Applicable Environments
- Continuous control (MuJoCo HalfCheetah-v1; observations ∈ ℝ²⁰, actions ∈ ℝ⁶)
- Navigation (Dubins Car with Lidar observations; noisy, high aleatoric uncertainty)
- Any environment with high uncertainty where conservative expert queries are needed

---

## Baseline 3: EnsembleDAgger

**Paper:** Menda, Driggs-Campbell & Kochenderfer, "EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning," IROS 2019. arXiv:1807.08364.

**Core Idea:** Replace MC-Dropout with an ensemble of `M` neural networks. The novice acts only if two conditions are jointly satisfied: (1) the mean novice action is close to the expert action (discrepancy rule), AND (2) the variance of the ensemble's predicted actions is below a threshold (doubt rule). This provides a more reliable uncertainty estimate than dropout.

### Algorithm

```
Algorithm: EnsembleDAgger Decision Rule
Input: Observation o_t, discrepancy threshold τ, doubt threshold χ
1. Compute mean and variance from ensemble:
   a_bar_nov,t, σ²_nov,t ← π_nov(o_t)   # π_nov is an ensemble of M networks
2. a_exp,t ← π_exp(o_t)
3. tau_hat ← ‖a_bar_nov,t − a_exp,t‖₂   # discrepancy
4. chi_hat ← σ²_nov,t                    # doubt (ensemble variance)
5. if tau_hat ≤ τ AND chi_hat ≤ χ:
       return a_bar_nov,t   # novice acts (mean action)
   else:
       return a_exp,t       # expert intervenes
```

> **Key insight:** As χ → ∞, EnsembleDAgger reduces to SafeDAgger* (only discrepancy matters). As τ → ∞, only doubt matters. The paper finds the doubt rule (variance-only) Pareto-dominates the discrepancy rule alone.

### Training Procedure

```
Initialize D ← ∅
Initialize ensemble of M neural networks {f_1, ..., f_M}, each trained independently
For epoch i = 1 to K:
    Run episode with DR_EnsembleDAgger(o_t, τ, χ) at each step
    Collect D_i = {(s, π_exp(s))} for all visited states
    D ← D ∪ D_i
    Train each ensemble member f_m on D independently (with different random seeds / bootstrap samples)
Novice policy π_nov: mean action = (1/M) Σ_m f_m(s); variance = (1/(M-1)) Σ_m (f_m(s) - mean)²
```

### Hyperparameters
| Parameter | Role |
|-----------|------|
| `M` | Number of ensemble members (original paper uses M=10 for inverted pendulum, M=5 for HalfCheetah) |
| `τ` | Discrepancy threshold for mean action vs. expert action |
| `χ` | Doubt threshold for ensemble variance |

### Applicable Environments
- Continuous control (MuJoCo HalfCheetah-v1)
- Stabilization tasks (Inverted Pendulum: state ∈ ℝ², action ∈ ℝ¹)
- Robotic manipulation (high-dimensional state/action spaces)

---

## Baseline 4: ThriftyDAgger

**Paper:** Hoque, Balakrishna, Novoseller, Wilcox, Brown & Goldberg, "ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning," CoRL 2021. arXiv:2109.08273.

**Core Idea:** Use two complementary intervention criteria — **state novelty** (is the state outside the distribution of previously seen states?) and **task risk** (is the probability of successfully completing the task from this state low?) — to decide when to call the expert. A budget parameter α_h specifies the desired fraction of timesteps with interventions; thresholds are set automatically to hit this budget.

### Formal Problem Statement

Minimize surrogate loss:
```
J(π_r) = Σ_{t=1}^{T} E_{s_t ~ d^{π_r}_t} [L(π_r(s_t), π_h(s_t))]
```
subject to supervisor burden B(π) ≤ Γ_b, where:
```
B(π) = C(π) · (L + I(π))
```
- `C(π)` = expected number of context switches (autonomous → supervisor)
- `I(π)` = expected number of supervisor actions per intervention
- `L` = latency of a context switch (in timesteps)

### Component 1: Novelty Estimation

```
Novelty(s) = Var_{ensemble} [π_r(s)]
           = average variance of action vector across ensemble members at state s
```
Train an ensemble of policies on bootstrapped samples of supervisor trajectories. High variance = novel state.

### Component 2: Risk Estimation

Define a success Q-function:
```
Q^{π_r}_G(s_t, a_t) = E[Σ_{t'=t}^{∞} γ^{t'-t} 1_G(s_{t'}) | s_t, a_t]
```
where `1_G(s)` = 1 if state s is in goal set G, 0 otherwise.

Train a Q-function approximator Q̂^{π_r}_{φ,G} by minimizing:
```
J_Q(s_t, a_t, s_{t+1}; φ) = (1/2) * (Q̂(s_t, a_t) − (1_G(s_t) + (1 − 1_G(s_t)) · γ · Q̂(s_{t+1}, π_r(s_{t+1}))))²
```

Risk is then:
```
Risk^{π_r}(s, a) = 1 − Q̂^{π_r}_{φ,G}(s, a)
```

### Component 3: Switching Policy

**Intervene (autonomous → supervisor) if:**
```
Intervene(s_t, δ_h, β_h) = TRUE iff:
    Novelty(s_t) > δ_h    (state is sufficiently novel)
    OR
    Risk^{π_r}(s_t, π_r(s_t)) > β_h    (task completion probability is low)
```

**Cede control back (supervisor → autonomous) if:**
```
Cede(s_t, δ_r, β_r) = TRUE iff:
    ‖π_r(s_t) − π_h(s_t)‖²₂ < δ_r    (robot and human actions agree)
    AND
    Risk^{π_r}(s_t, π_r(s_t)) < β_r    (risk is low)
```
Note: β_r < β_h (asymmetric thresholds — stricter to exit supervisor mode than to enter it).

### Component 4: Threshold Setting from Budget α_h

Given desired intervention rate α_h ∈ [0, 1]:
```
β_h = (1 − α_h)-quantile of {Risk^{π_r}(s, π_r(s))} over previously visited states
δ_h = (1 − α_h)-quantile of {Novelty(s)} over previously visited states
δ_r = mean action discrepancy on supervisor-visited states after π_r is trained
β_r = median of {Risk^{π_r}(s, π_r(s))} over previously visited states
```

### Full ThriftyDAgger Algorithm

```
Algorithm: ThriftyDAgger
Input: Supervisor π_h, desired budget α_h, offline demos D_h (N_demo trajectories)
1. Initialize π_r via Behavior Cloning on D_h
2. Collect initial dataset D_r from π_r rollout
3. Initialize Q̂^{π_r}_{φ,G} by optimizing J_Q on D_r ∪ D_h
4. Set thresholds β_h, β_r, δ_h, δ_r from α_h (Section 4.4)
5. For episode e = 1 to N:
      For timestep t = 1 to T:
          if in autonomous mode:
              if Intervene(s_t, δ_h, β_h):
                  switch to supervisor mode
              else:
                  execute a_t = π_r(s_t); add (s_t, a_t, s_{t+1}) to D_r
          if in supervisor mode:
              execute a_t = π_h(s_t); add (s_t, a_t, s_{t+1}) to D_h
              if Cede(s_t, δ_r, β_r):
                  switch to autonomous mode
      After episode e:
          Update π_r via supervised learning on D_h
          Update Q̂^{π_r}_{φ,G} on D_r ∪ D_h
          Recompute thresholds β_h, β_r, δ_h, δ_r from α_h
```

### Hyperparameters
| Parameter | Role |
|-----------|------|
| `α_h` | Target fraction of timesteps with human intervention (e.g., 0.01 = 1%) |
| `γ` | Discount factor for Q-function |
| `N` | Number of training episodes |
| `T` | Max timesteps per episode |
| `N_demo` | Number of offline demonstrations to initialize BC |

### Applicable Environments
- Long-horizon robotic manipulation (peg insertion, Robosuite; T=175 steps)
- Physical robot control from image observations (cable routing, da Vinci robot; 64×64×3 RGB)
- Multi-robot fleet supervision (3-robot concurrent supervision)
- Any task with a goal set G and binary success/failure signal

---

## Baseline 5: Stagger

**Paper:** Swamy, Choudhury, Bagnell & Wu, "Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning," NeurIPS 2024. arXiv:2412.07057.

**Core Idea:** Stagger is a theoretically-motivated variant of DAgger in which only **one state-action pair per round** is queried from the expert and used to update the policy. This "one sample per round" structure enables a clean theoretical proof that interactive IL provably beats Behavior Cloning when the environment has low recovery cost — i.e., when it is easy for the agent to return to a safe state after a mistake.

> **Important:** Stagger's trigger for choosing which state to query is **not** loss-based. It is a scheduled single-sample protocol. The theoretical contribution is the guarantee itself, not an uncertainty or risk criterion.

### Algorithm

```
Algorithm: Stagger
Input: Expert policy π_exp, environment MDP M
Initialize: Policy π_0 (e.g., via Behavior Cloning on an initial dataset D_0)
For round i = 1, 2, ..., K:
    1. Roll out current policy π_i in environment M for T steps → trajectory τ_i
    2. Select one state s* from τ_i (e.g., uniformly at random, or first visited state)
    3. Query expert: a* = π_exp(s*)
    4. Add single sample (s*, a*) to dataset D
    5. Update policy: π_{i+1} ← train on D (e.g., via gradient step or full retraining)
Return π_K
```

### Why One Sample Per Round?

The one-sample structure allows the analysis to treat each round as a no-regret online learning step. The key theoretical result is:

- Let `C_rec` = recovery cost (expected cost of returning to a nominal trajectory after deviating)
- Stagger provably achieves regret scaling with `C_rec`, rather than the compounding error rate of BC
- When `C_rec` is low (e.g., navigation tasks where recovery is cheap), Stagger's suboptimality gap over BC is bounded and diminishes with K

### Practical Notes for Implementation

- The state selection in step 2 can be uniform random, last-visited, or highest-uncertainty; the theory holds regardless
- After K rounds, the final policy π_K has been trained on K expert-labeled samples
- Unlike ThriftyDAgger or SafeDAgger, there is **no risk metric, no novelty metric, and no uncertainty threshold** — the query schedule is purely round-based
- Stagger is computationally cheap: only one expert call per round

### Comparison Context
- Stagger is a theoretical baseline demonstrating that *any* interactive method (even one-sample-per-round) can provably beat BC
- For empirical comparison with P4-LLM, report performance as a function of number of expert queries (rounds K) to make results comparable across all five baselines

### Applicable Environments
- Environments with low recovery cost (e.g., grid-world navigation, car racing, simple manipulation)
- MuJoCo locomotion tasks (HalfCheetah, Ant, Hopper)
- Language-conditioned tasks (CLEVR, instruction following in simulated environments)
- Any sequential decision-making MDP where BC is a valid comparison

---

## Comparison Summary

| Baseline | Expert Query Trigger | Uncertainty Method | Budget Control | Key Paper |
|----------|---------------------|--------------------|----------------|-----------|
| **SafeDAgger** | Action discrepancy > τ | None (deterministic threshold) | Manual τ tuning | Zhang & Cho, AAAI 2017 |
| **DropoutDAgger** | Dropout distribution outside ball | MC-Dropout (Bayesian NN) | Manual τ, p tuning | Menda et al., arXiv 2017 |
| **EnsembleDAgger** | High ensemble variance OR discrepancy | Neural ensemble variance | Manual τ, χ tuning | Menda et al., IROS 2019 |
| **ThriftyDAgger** | Novel state OR high task failure risk | Ensemble novelty + Q-function risk | Auto-calibrated from α_h budget | Hoque et al., CoRL 2021 |
| **Stagger** | Fixed schedule (1 sample per round) | None | None (round-based) | Swamy et al., NeurIPS 2024 |

---

## Evaluation Protocol

For all baselines and P4-LLM, report the following metrics:

1. **Autonomous Success Rate** — success rate when deployed with no expert in the loop after training
2. **Intervention-Aided Success Rate** — success rate when expert is available at deployment
3. **Number of Expert Interventions (per episode)** — total number of times control switched to expert
4. **Number of Expert Actions (per episode)** — total expert-controlled timesteps
5. **Number of Robot/Agent Actions (per episode)** — total autonomous timesteps
6. **Expert Query Budget** — total expert queries consumed across all training rounds

When comparing across environments, always report metrics relative to the **total number of expert queries used**, as this is the primary efficiency axis across all five methods.

---

## Implementation Checklist for the Coding LLM

- [ ] All five baselines share the same DAgger outer loop (data aggregation + retraining)
- [ ] Only the decision rule DR(·) differs between baselines
- [ ] SafeDAgger: implement both SafeDAgger* (oracle) and full SafeDAgger (with classifier) if expert is unavailable at inference
- [ ] DropoutDAgger: enable dropout at **both training and inference** time (MC-Dropout)
- [ ] EnsembleDAgger: train M independent networks on the same dataset D; variance is computed across ensemble predictions
- [ ] ThriftyDAgger: implement both a novelty estimator (ensemble variance) and a risk estimator (goal-conditioned Q-function); implement budget-based threshold calibration
- [ ] Stagger: no uncertainty metric needed; implement the one-sample-per-round protocol exactly
- [ ] Do not hard-code environment-specific state/action dimensions; use configurable input/output sizes
- [ ] All methods should be tested on at least two different environments to verify generalization

