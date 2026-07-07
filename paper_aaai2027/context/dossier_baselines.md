# Dossier — Interactive-Imitation-Learning Baselines (query rules, exact)

**Scope.** These are the demonstration-acquisition baselines compared against our
method **PACE** (Perceive → Assess → Choose → Execute). The paper's
five tasks: **T1** Toy 5×5 grid navigation (`pool_x_selector`; state→equivariant
MLP, image→plain CNN); **T2** Push = ManiSkill PushT (`pool_rl_robo`; state- and
image-based diffusion policy); **T3** Lift, **T4** Wipe, **T5** Door-open
(RoboSuite UR5/UR5e, state+image diffusion policy; upcoming). Baselines:
**SafeDAgger**, **DropoutDAgger**, **EnsembleDAgger**, **ThriftyDAgger**, a
uniform-random control (**Stagger** — random, *not* a published method), and
(robot tasks only) **Diff-DAgger**.

**Notation.** Novice policy π_θ, expert π\*, state s, action a. Two code paths:
- **Discrete-maze / T1** (`pool_x_selector/selection/{iil_baselines,uncertainty,success_q}.py`):
  a **per-layout, pool-scored, one-demo-per-round** rotate-pool DAgger loop. Each
  method rolls the whole correction pool, scores every layout, and picks exactly
  ONE via a `queryable` gate + argmax(score); the gate is that method's query
  inequality (a fallback picks top-score when nothing is queryable so no round is
  wasted). Corrective demo = BFS from the first deviation. `a` is one of 4
  discrete moves; the "expert" is the A\* optimal-action set.
- **Continuous-diffusion / T2–T5** (`pool_rl_robo/selection/{iil_baselines,uncertainty,success_q}.py`):
  a **per-step, inline-intervention** DAgger loop (`_run_one_episode_iil`,
  mirrors the fork's `diffdagger_baseline._run_one_episode`). The novice is the
  fork diffusion policy π_θ; roll step by step; **the FIRST step whose rule fires
  is the intervention point t\***, the expert takes over inline and finishes the
  episode; that successful expert trajectory is the ONE corrective demo for the
  round. Retrain FROM SCRATCH every `nd_retrain` demos; early-stop at `target_sr`.
  All signals live in **joint-delta (`rel_joint_pos`) space** where both π_θ's
  first executed step `a_nov = action_seq[:, obs_horizon-1, :num_joints]` and the
  expert's `clip_and_scale`-d + wrap-to-[-π,π] delta `a_exp` are comparable, so
  `‖a_nov − a_exp‖₂` is consistent.

Files:
- `.../pool_x_selector/selection/iil_baselines.py` (maze; `run`, `_finalize_scores`, `_pick`)
- `.../pool_x_selector/selection/uncertainty.py` (maze; discrepancy, ensemble, MC-dropout)
- `.../pool_x_selector/selection/success_q.py` (maze; ThriftyDAgger success-Q)
- `.../pool_x_selector/selection/baseline_dagger.py` (maze; plain highest-loss / random DAgger driver)
- `.../pool_rl_robo/selection/iil_baselines.py` (robot; unified per-step arm + native Diff-DAgger)
- `.../pool_rl_robo/selection/{uncertainty,success_q}.py` (robot signals)
- fork `diffdagger/agents/diffusion_policy.py` (native Diff-DAgger CDF rule)
- fork `diffdagger/util/cdf.py` (`CDF.get_quantile`)
- fork `diffdagger/main_pipeline/diffdagger_baseline.py` (fork-native run-to-budget driver)

---

## 1. SafeDAgger (`safe`)

**Idea.** Query when the novice's action disagrees "too much" with the expert.

**Maze (T1).** Per layout compute the **action-discrepancy** = fraction of
rollout steps whose executed action is NOT in the A\* expert's optimal set (a step
also counts as off if the agent is unreachable, i.e. the optimal mask is empty):

    d_safe(layout) = ( 1/N ) · Σ_{t=1..N}  1[ opt_t.sum()==0  OR  opt_t[a_t] < 0.5 ]

  where `opt_t = expert.optimal_actions(traj_t)` (A\* over the BFS distance map).

- **Query rule:** `queryable ⟺ d_safe > tau_safe`.
- **score = d_safe** (argmax picks the most-discrepant layout).
- Default **`tau_safe = 0.10`**.
- (`uncertainty.action_discrepancy`; `_finalize_scores` kind=="safe".)

**Robot (T2–T5).** Per step, in joint-delta space:

    d = ‖ a_nov − a_exp ‖₂ ,      query ⟺ d > tau

- First step with `d > tau` is t\*; **score = d**.
- Default **`tau = 0.1`** (`hp.get("tau", 0.1)`).
- (`_decide_query`, kind=="safe".)

> Faithfulness note: this is **SafeDAgger\*** — the classic SafeDAgger trains a
> learned *safety classifier* predicting the discrepancy; here the discrepancy is
> computed directly against the (A\* / motion-planner) expert as an oracle. Same
> decision variable, oracle safety signal instead of a learned one.

---

## 2. DropoutDAgger (`dropout`)

**Idea.** Estimate epistemic uncertainty by N stochastic forward passes; query
when too few of them agree with the expert (fall inside the expert "ball").

**Maze (T1).** Load the shared checkpoint into a policy with a >0 **fusion-head
dropout** `dropout_d` and keep `nn.Dropout` **active at inference** (`.train()` on
Dropout only; rest in eval). The executed trajectory is driven by the **clean
(dropout-off) argmax**; at every visited state additionally draw **N** stochastic
(dropout-on) argmax samples. Per step:

    p̂(t) = ( fraction of the N samples whose action ∈ expert optimal mask )
    mean_disagreement = mean_t ( 1 − p̂(t) )
    frac_low          = fraction of steps with  p̂(t) < p_thresh

- **Query rule (faithful DropoutDAgger):** `queryable ⟺ frac_low > 0`, i.e. the
  expert intervenes whenever ANY step has in-ball fraction `p̂(t) < p_thresh`.
- **score = mean_disagreement** (ranks the queryable layouts).
- Defaults **`mc_N = 16`**, **`dropout_d = 0.2`**, **`p_thresh = 0.5`**
  (`tau_drop = 0.5` is declared but unused by the current rule).
- (`uncertainty.load_dropout_model`, `mc_dropout_rollout`, `dropout_uncertainty`.)

> Architectural caveat (reported, not fixed): the shared checkpoint has a single
> `nn.Dropout` in the fusion head only (CNN + equivariant backbone have none), so
> MC-dropout samples stochasticity only from the fusion head — retraining with
> backbone dropout would break the apples-to-apples warm-start with P4.

**Robot (T2–T5).** A diffusion policy is already stochastic (fresh Gaussian prior
per `get_action`), so N `get_action` calls give N distinct denoised action
samples — that IS the MC-dropout signal here (`mc_dropout_deltas =
action_sample_deltas`; the fork ships a UNet dropout but disables it in forward
and the fork is never edited). Per step, with expert delta `a_exp` and N draws
`{a^(1)…a^(N)}` in joint-delta space:

    within = ( 1/N ) · Σ_i  1[ ‖ a^(i) − a_exp ‖₂ ≤ tau ]        # in-τ-ball fraction p̂
    query ⟺ within < p

- **score = 1 − within**.
- Defaults **`N = 10`**, **`tau = 0.1`**, **`p = 0.9`** (`hp.get`).
- (`_decide_query`, kind=="dropout"; `uncertainty.action_sample_deltas`.)

---

## 3. EnsembleDAgger (`ensemble`)

**Idea.** M independently-trained policy members; query on **disagreement
(doubt)** OR on **mean discrepancy** vs the expert. Doubt is the novelty/epistemic
term, discrepancy the safety term.

**Maze (T1).** M members fine-tuned independently each round (replay fine-tuner
called M times, distinct seed `base_seed + 101·m`, per-member checkpoint dir);
rollout under the **ensemble-mean-action** policy (argmax of the mean logits).
Two signals over the T-step mean trajectory:

    doubt            = mean_t ( 1 − max_a count_t[a] / M )          # per-step action disagreement
    mean_discrepancy = mean over members of ( that member's frac. of off-optimal-mask steps )

- **Query rule:** `queryable ⟺ (mean_discrepancy > tau_ens) OR (doubt > sigma_ens)`.
- **score = doubt** (argmax; ties → higher policy_loss / n_steps).
- Defaults **`M = 5`**, **`tau_ens = 0.10`**, **`sigma_ens = 0.10`**;
  eval mode default `ensemble_eval_mode = "mean"` (held-out SR computed under the
  ensemble-mean policy; `"member0"` delegates to the single-policy evaluator).
- Load-bearing detail: in early rounds members are identical so `doubt==0`
  everywhere; a failure is then queryable only via the discrepancy arm with
  score==0, and `_pick` **must still pick it** (by highest loss) so a demo is
  collected and members can diverge on the next fine-tune (else deadlock).
- (`uncertainty.{finetune_ensemble,ensemble_rollout,ensemble_doubt,ensemble_mean_discrepancy}`.)

**Robot (T2–T5).** M members retrained from scratch each round with distinct seeds
(`seed + 101·(mi+1)`); at each step take each member's first joint-delta →
`member_deltas` (M×J), `a_bar = mean`, `doubt = mean over joint-dims of
var_over_members`:

    doubt     = mean_j Var_m( member_deltas[m, j] )
    mean_disc = ( 1/M ) · Σ_m  ‖ member_deltas[m] − a_exp ‖₂
    query ⟺ (mean_disc > tau) OR (doubt > chi)

- **score = doubt**.
- Defaults **`M = 5`**, **`tau = 0.1`**, **`chi = 0.05`** (`hp.get`).
- (`_decide_query`, kind=="ensemble"; `uncertainty.member_deltas`.)

> Faithfulness note: the *acting/evaluated* policy is the single novice π_θ (not
> the ensemble mean) on the robot side — the M members feed only the doubt/disc
> SIGNAL. This is cleaner for held-out eval (one scored policy). Members are
> retrained with distinct seeds so doubt is meaningful from round 1.

---

## 4. ThriftyDAgger (`thrifty`)

**Idea.** Query on **novelty** (= ensemble doubt, the EnsembleDAgger term) OR on
**task risk** `risk = 1 − Q(s,a)`, where Q is a goal-conditioned success value
predicting P(reach goal | s, a). Thresholds are **budget-α auto-calibrated
quantiles** (higher α ⇒ query more sparingly).

**Query rule (both code paths):**

    query ⟺ (doubt > delta_h)  OR  (risk > beta_h),      risk = 1 − Q(s, a)

**Success-Q training (both paths).** Small MLP predicting the **logit** of
P(reach goal | s, a); apply σ for the probability. Trained by BCE-with-logits on
a **Monte-Carlo success-to-go** target (NOT a Bellman backup — used only to rank):

    y_t = 1[success] · gamma^(n − 1 − t)

(1.0 at the last pre-terminal state of a successful trajectory, geometric decay
backwards; 0 for a failed trajectory). Since only successful expert demos enter
the dataset, stored demos have success=1 ⇒ `y_t = gamma^(steps-to-end)`.
Optimizer AdamW (`lr=q_lr`, `weight_decay=1e-4`), batch 256, BCEWithLogits.

- **Maze:** features = flattened 5-channel grid encoding (goal channel makes it
  goal-conditioned) ⧺ one-hot(action). Transitions = cumulative corrective demos
  (success=1) + this round's pool rollouts (success/failure). Net warm-started
  across rounds at `<method_root>/success_q/q_net.pth`. Risk per layout =
  `agg_t (1 − Q(s_t, a_t))` with `thrifty_risk_agg ∈ {mean, max}` (default
  **`mean`**). (`success_q.{SuccessQNet,build_transitions,train_success_q,q_values_for_steps}`.)
  - Defaults: **`gamma = 0.95`**, **`q_epochs = 50`**, **`q_lr = 1e-3`**,
    hidden 128, `thrifty_risk_agg = "mean"`.
- **Robot:** features = concat of expert `obs_keys` (sorted, flattened) ⧺ joint
  action; target as above over the fork replay buffer per episode. **Best-effort**:
  if the buffer can't be read into (obs, action, y) it returns None and
  ThriftyDAgger **degrades to novelty-only** (documented caveat).
  - Defaults: **`gamma = 0.99`**, **`q_epochs = 50`**, hidden 128, AdamW `lr=1e-3`.

**Quantile auto-calibration of the thresholds.**
- **Maze:** over the pool's per-layout novelty/risk, take the **(1−α_h)-quantile**:
  `delta_h = quantile(novelties, 1−alpha_h)`, `beta_h = quantile(risks, 1−alpha_h)`
  (layout-level reduction of the paper's per-state quantiles). combined
  normalized **score = norm(doubt) + norm(risk)**. Default **`alpha_h = 0.10`**
  (so thresholds are the 0.90-quantile). (`_finalize_scores`, kind=="thrifty".)
- **Robot:** `thresholds = {"delta_h": 0.0, "beta_h": 1.0}` initialized in
  `run_iil_arm`; with `beta_h = 1.0` the risk arm is effectively inert (risk ∈
  [0,1]) so on the robot side ThriftyDAgger fires primarily on **novelty
  (doubt > 0)** unless overridden — the paper-faithful per-state quantile
  recalibration is present in the maze path; the robot path uses the fixed
  conservative init. **score = max(doubt, risk)**. (`_decide_query`, kind=="thrifty".)
  - Robot defaults visible in code: `M = 5`, `gamma = 0.99`, `q_epochs = 50`.

---

## 5. Uniform-random control — "Stagger" (`stagger`)

**Not a published method** — a random control ("one sample per round"), roughly
the existing random-rotate control; documented as a floor, never as a baseline
method.

**Maze (T1).** Among the pool's **failures only**, draw a per-failure uniform
score (successes get score 0):

    queryable ⟺ (not success);   score = U(0,1) if failure else 0

argmax of per-failure uniform draws = a uniformly random failure ⇒ exactly one
random failure corrected per round. (`_finalize_scores`, kind=="stagger".)

**Robot (T2–T5).** Pre-pick ONE uniform-random visited step per episode
`stagger_t = randint(0, max_steps−1)`; query ⟺ `timestep ≥ stagger_t` (score =
timestep). The expert takes over at that random step. (`_run_one_episode_iil`,
kind=="stagger".)

---

## 6. Diff-DAgger (robot tasks only — T2–T5)

**Idea.** The diffusion policy's **native** query rule: intervene when the
model's own **diffusion (denoising) loss** on the state–action pair is
persistently in the tail of its training-set loss distribution.

**Per-step signal (fork `diffusion_policy.get_action(..., dagger=True)`).**
Compute the average diffusion loss `L(s, a_nov)` on the normalized obs + predicted
action (`get_avg_diffusion_loss_ndata`, averaged over `num_per_batch` noise draws,
optionally repeated across all training timesteps × `batch_multiplier`). A **CDF**
of training-set diffusion losses is built once (`CDF` over `get_stats_from_dataset`)
and the query **threshold** is its **α-quantile**:

    diffusion_loss_threshold = CDF.get_quantile(alpha)
        where CDF.get_quantile(q) = sorted_losses[ int(len(sorted_losses)·q) ]

Then a sliding window (`deque`, maxlen = `patience_window`) of per-step violations
`1[ L > diffusion_loss_threshold ]` is kept, and:

    query ⟺  Σ (violations in the last patience_window steps)  ≥  patience

i.e. **CDF(diffusion_loss) > α  for K = `patience` (consecutive/windowed) steps.**
With `patience_window = patience = K`, this is exactly "the last K steps were all
above the α-quantile."

- Defaults: **`alpha = 0.99`** (the α-quantile threshold; `cfg.policy.get("alpha",
  0.99)`), **`patience = 1`** in the unified early-stopping arm
  (`patience_window = patience`); the fork's stand-alone `PolicyConfig` default is
  `patience = 2`. `batch_multiplier`, `num_per_batch` carried from the fork policy
  config. Every from-scratch retrain **recalibrates** the threshold at α from the
  current dataset's loss CDF (`update_diffusion_threshold`).
- **score = diffusion_loss** (logged as `picked_score`).

**Two drivers (apples-to-apples note).**
- Unified arm `run_iil_arm` with `kind="diff"` (in `pool_rl_robo/.../iil_baselines.py`,
  `KIND_OF["diff_dagger"]="diff"`): runs the native rule but **early-stops at
  `target_sr`** like every other method — the fix for the "Diff-DAgger consumed 43
  demos" asymmetry (the fork's own baseline had no target stop, only
  `intervention_total ≥ budget_total`).
- Fork-native `diffdagger_baseline.run_diffdagger_baseline` (run-to-budget,
  `budget_total` default 10, `nd_retrain` retrain cadence, `max_episodes` backstop)
  — used when reproducing the fork's original consume-full-budget behavior.

---

## Shared loop hyperparameters (defaults)

- **Maze one-demo-per-round DAgger** (`iil_baselines.run`): `budget`, `target_sr`,
  `max_rounds`, `max_steps`, `correction_n`, replay-mix fine-tune (`finetune_epochs`,
  `lr`, `batch_size`, `weight_decay`, `replay_mix`, `replay_mix_floor`),
  `max_consecutive_empty = 8` (rotate-pool: a zero-demo round does not stop the run;
  cap consecutive empties). Stop reasons: `initial>=target` / `target_hit` /
  `budget_exhausted` / `no_progress` / `max_rounds`. Per-method HP defaults as
  listed above (`tau_safe=0.10`, `mc_N=16`, `dropout_d=0.2`, `p_thresh=0.5`,
  `M=5`, `tau_ens=0.10`, `sigma_ens=0.10`, `alpha_h=0.10`, `gamma=0.95`,
  `q_epochs=50`, `q_lr=1e-3`, `thrifty_risk_agg="mean"`).
- **Maze plain DAgger driver** (`baseline_dagger.run`): pick ONE failure/round by
  highest per-step BCE loss (`selection="highest_loss"`) or uniform-random
  (`selection="random"`); fixed-pool or rotate-pool. This is the vanilla
  DAgger-with-loss-ranking reference the IIL query rules are layered on top of.
- **Robot arm** (`run_iil_arm` / orchestrator `_common.py`): `budget` default 100,
  `target_sr` default 0.90, `nd_retrain` default 4, `num_initial_demos` 50,
  `max_episodes` (`max_episodes_per_arm`) default 400, `M` default 5. Retrain
  FROM SCRATCH every `nd_retrain` demos; an episode counts as a demo only if it
  succeeded, intervened, and `expert_steps ≥ 10` and `≤ cfg.expert.max_episode_steps`.
  Smoke overrides: `M=2`, `dropout N=3`, `thrifty q_epochs=5`.
