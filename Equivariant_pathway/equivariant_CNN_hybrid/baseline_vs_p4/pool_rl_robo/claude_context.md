# claude_context.md — single-file onboarding for `pool_rl_robo`

**Read this top-to-bottom at the start of a session.** It is the complete context for
the **A\* study: P4-LLM-select vs Diff-DAgger** on ManiSkill manipulation tasks, plus
the exact recipe to run the comparison on a NEW environment. Fill in the box below and
tell Claude "run the next experiment per claude_context.md".

---

## ▶ ACTIVE TASK (next session)

```
TARGET ENV:   StackCube-v1
GOAL:         Enable StackCube, then ONE-RUN (single seed) comparison of
              diff_dagger vs p4_select(diffusion-loss variant). NOT 5 seeds yet —
              just confirm the pipeline works end-to-end on StackCube for one run.
DESIGN:       p4_select uses Diff-DAgger-style DIFFUSION-LOSS detection (NOT action-
              discrepancy) + LLM selection. See "## NEXT TASK — StackCube one-run".
```

**Claude, do `## NEXT TASK — StackCube one-run` below, honoring `## INVARIANTS` and the
general `## HOW TO RUN A NEW ENVIRONMENT` recipe. Smoke before the one run.** Only
PushT-v1 has run end-to-end; StackCube is a first-time bring-up.

---

## TL;DR — what this is and where it stands

- **Study:** does an LLM that *selects which failure to correct* (P4-LLM-select, built on
  SafeDAgger detection + on-policy expert correction) reach a target success rate with
  **fewer expert demonstrations** than **Diff-DAgger** (native diffusion-loss query)?
  Shared Diffusion-Policy backbone; only the demo-acquisition rule differs.
- **DONE — PushT-v1 (n=5 seeds, jobs 98949–98953, all COMPLETED):** both methods reach
  **100% success**; **P4-LLM-select gets there with ~1.8× fewer demonstrations**
  (31.0 ± 10.9 vs 55.2 ± 12.4 demos to 100%, **winning all 5/5 seeds**). Demos→90%:
  12.0 ± 2.9 vs 15.8 ± 5.2. Figure + summary in
  `results/aggregate/astar/astar_PushT-v1_{p4select_vs_diffdagger.png,summary.json}`.
- **NEXT:** repeat on StackCube-v1, then other envs (PickCube-v1, PlugCharger-v1).
- The harness survived **two rounds of adversarial multi-agent audit**; the fairness
  invariants in `## INVARIANTS` are load-bearing — keep them or the comparison breaks.

---

## HOW TO RUN A NEW ENVIRONMENT  (the recipe)

Everything is driven by the `ENV` shell var; results land in
`results/<ENV>/run_<seed>/{diff_dagger,p4_select}/`. Aggregation is per-env, so envs
never contaminate each other.

**0. Use the right interpreter.** Bare `python`/`pip` hit the `maze` env (bashrc PATH).
Always: `/home/s226137394/.conda/envs/diffdagger/bin/python`. From repo root
`/weka/s226137394/DmNfull` set `PYTHONPATH=/weka/s226137394/DmNfull`.

**1. (Recommended) Create the KAG for the env** so the LLM selector is grounded.
`p4/kag/<ENV>.json` (only `PushT-v1.json` exists today). Mirror that file's schema
(geometry, objects, success condition, failure taxonomy, reasoning implications). If
missing, the run still works but the selector sees only the one-line `task_description`
from `envs/env_setup.py::TASKS` (degraded grounding, not a crash).

**2. SMOKE FIRST (mandatory for any non-PushT env — first end-to-end bring-up).**
Validates the motion-planner expert, shared bootstrap, both arms, `total_expert_calls`
logging, and `LLM selector ON`. Tiny budget (~15–25 min on a 3-GPU H100 node):
```bash
cd .../pool_rl_robo
ENV="StackCube-v1" METHODS="diff_dagger+p4_select" SEED="1" RUN_ID="950" \
  CONFIG="$(pwd)/config_astar_smoke.yaml" \
  sbatch --array=1 --gpus-per-node=3 --constraint=gpu-h100 --time=02:00:00 \
         --job-name="prr_smoke_${ENV}" run_pool_rl_robo.sh
```
Watch `slurm_logs/pool_rl_<jobid>_1.out` + `logs/<ENV>_run950_<jobid>.log`. PASS =
bootstrap writes `init_ckpt.pth`; both arms write `learning_curve.json` with non-null
`total_expert_calls` and a sane `stopped_reason`; log shows `LLM selector ON`; no
`Traceback`/proxy errors. **If the motion-planner expert errors here, fix that before the
5-seed launch** (this is the most likely first-run failure for a new task).

**3. Launch the 5 seeds** (one isolated sbatch job per seed; both methods share each
job's bootstrap):
```bash
ENV="StackCube-v1" bash submit_astar_5seeds.sh        # seeds 1–5 → run_1..run_5
```
(Each job uses `config_astar.yaml`, 3 GPUs, H100, 10-day walltime, nd_retrain=1,
target_sr=1.0, budget=100.)

**4. Monitor & gate.** Each seed ~12–21 h. Watch `stopped_reason`: `target_hit` = reached
100% (clean). If any **diff_dagger** seed shows `max_episodes` with `<100` demos, the
episode backstop truncated it → raise `max_episodes_per_arm` in `config_astar.yaml` and
re-run that seed (PushT didn't need this; harder envs might).

**5. Aggregate** when seeds finish (handles partial data + honest censoring):
```bash
PYTHONPATH=/weka/s226137394/DmNfull /home/s226137394/.conda/envs/diffdagger/bin/python \
  -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.aggregation.aggregate_astar \
  --env StackCube-v1 --runs 1,2,3,4,5
# → results/aggregate/astar/astar_StackCube-v1_{p4select_vs_diffdagger.png,summary.json}
```
Headline = demos-to-target (final `n_queries`, since both stop at `target_hit`);
secondary = demos-to-90% and the (disclosed, non-comparable) `total_expert_calls`.

> Expect harder envs may NOT reach 100% — then they run to budget=100 or plateau
> (`max_episodes`), and the metric becomes demos-to-90% / final-SR-at-budget. The
> aggregator right-censors non-reachers honestly (reports `n_reached`, `censored_seeds`).

---

## EXPERTS PER TASK — and the motion-planner gap (READ before Stack/Pick/Plug)

**Expert source is paper-faithful** (verified vs arXiv 2410.14868 Table I):
- **Pushing (PushT)** → trained **PPO** expert (`MultipleExperts`); native per-state
  `get_action`. **WIRED + VALIDATED** (the only validated task). Paper: "two experts
  trained using PPO"; our "1 Expert" = single PPO agent.
- **Stacking / PickCube / Plugging** → ManiSkill **panda motion planners**
  (`solveStackCube`/`solvePickCube`/`solvePlugCharger`), already vendored at
  `mani_skill/examples/motionplanning/panda/solutions/` and imported by
  `MotionPlannerExpert.__init__`. Paper uses a "rule-based RRT motion planner" for
  stacking/plugging — same thing. So you do NOT need to find/train experts; they exist.

**THE GAP (this is the real work for a non-PushT run):** `envs/experts.py::MotionPlannerExpert`
implements ONLY `solve()` — an OPEN-LOOP whole-episode planner that `env.reset()`s first.
The arms also call (and it is MISSING): `move_to_next_goal(...)` (the corrective DEMO, used
by BOTH methods), `get_action(obs)` (per-state π_exp for p4_select's action-discrepancy),
and `generate_stationary_action()`. Control mode is **compatible** (planner + policy both
`pd_joint_pos`) — nothing to fix there; `follow_path` already emits per-step `[qpos,gripper]`.

**DECISION (made 2026-06-08): p4_select on motion-planner tasks uses DIFFUSION-LOSS detection
(Option 2), NOT action-discrepancy.** Consequence: **NEITHER method needs the expert's per-state
`get_action`** — the per-state-replan problem is AVOIDED. Both detect via the policy's own
diffusion loss and need the expert ONLY for the corrective demo (`move_to_next_goal`). Bonus:
the comparison gets *cleaner* — both arms share the SAME detection signal, differing only in
WHICH failure is corrected (diff_dagger: first state whose CDF(loss)>α-quantile for K steps;
p4_select: LLM picks among the top-3 highest-peak-loss FAILED candidate rollouts).

So the ONLY expert-adapter work is **`move_to_next_goal`** (+ a trivial
`generate_stationary_action`). `get_action` can stay unimplemented for motion-planner tasks.

**HARDEST PART (budget the session here):** `move_to_next_goal` must re-plan from a NON-RESET
mid-episode state. `solve()` hardwires `env.reset(seed)` first — wrong for a DAgger
intervention, which must continue from where the policy left off. Refactor to build
`PandaArmMotionPlanningSolver` against the live env / plan from the current robot qpos without
reset, capture `follow_path`'s per-step `[qpos,gripper]` actions → demo TensorDict in the
PushT schema, convert `pd_joint_pos`→`rel_joint_pos`. But it's needed only ONCE per
intervention (not per step), so it's far cheaper than per-state planning. Everything else
(control mode — already `pd_joint_pos`-compatible — imports, demo-packing) is mechanical.
(`experts.py`'s docstring advertises `get_action`/`move_to_next_goal` as if implemented — they
are NOT; fix the docstring too.)

## NEXT TASK — StackCube one-run (diffusion-loss p4_select)

Goal: enable StackCube and run **ONE seed** comparing `diff_dagger` vs `p4_select`
(diffusion-loss variant). Confirm it works end-to-end; 5-seed averaging comes later.

**Build (4 pieces):**
1. **`MotionPlannerExpert.move_to_next_goal(...)`** (envs/experts.py) — re-plan from the
   LIVE mid-episode state (NOT `solve()`, which resets), capture `follow_path`'s per-step
   `[qpos,gripper]` actions + obs into the SAME demo TensorDict schema the PushT
   `MultipleExperts.move_to_next_goal` produces; convert `pd_joint_pos`→`rel_joint_pos`.
   Also add `generate_stationary_action()` (hold = current qpos+gripper). Fix the
   misleading docstring. (This is the only real engineering — see EXPERTS section.)
2. **StackCube env wiring** (envs/env_setup.py / maniskill_env.py) — StackCube-v1 →
   `expert_kind="motionplanner"`, `control_mode="pd_joint_pos"`, same
   `VariousActionSpaceWrapper`/obs stack as PushT. Author its Hydra cfg if missing.
   (Optional) author `p4/kag/StackCube-v1.json` for selector grounding.
3. **p4_select diffusion-loss detection** (p4/select_arm.py) — swap `_safe_rollout_one`'s
   per-step signal from expert action-discrepancy (`_expert_delta`/`expert.get_action`) to
   the policy's OWN diffusion loss via `policy.get_action(dagger=True, return_dict=True)`
   (the loss/query the diff arm already uses). `t_star` = argmax diffusion-loss step;
   candidate ranking + top-3 = highest PEAK diffusion loss among FAILED candidates; LLM
   selects which to correct; correction = replay prefix to `t_star` then
   `expert.move_to_next_goal` (motion planner). NO `expert.get_action` anywhere. Gate this
   on `expert_kind=="motionplanner"` (keep PushT's action-discrepancy path intact) OR make
   diffusion-loss the default for both — your call, but document it.
4. **Config** (e.g. `config_stack.yaml`) — match the paper's **StackCube** Table IV row
   (NOT pushing's): `initial_demos: 20`, `budget: 60` (= N_f), `diff_dagger: {alpha: 0.99,
   patience: 2, batch_multiplier: 32}` (K=2, N_b=512 for StackCube), `nd_retrain: 1`
   (both arms), `target_sr: 1.0` (StackCube likely won't hit 100% → runs to budget=60;
   censoring handles it), `p4_select.n_cand: 6`, `methods: ["diff_dagger","p4_select"]`,
   `eval_num_envs: 10`, `heldout_n: 100`, `max_episodes_per_arm: 5000`.

**Run order:** (a) login-node import/syntax check; (b) **GPU smoke** (tiny budget, both
methods, ENV=StackCube-v1) — assert the planner yields a non-empty demo from a mid-episode
state, action shapes match the policy's training tensors, both arms log a curve +
`total_expert_calls`, `LLM selector ON`; (c) **submit ONE seed** (single sbatch job,
SEED=1 RUN_ID=1, NOT `submit_astar_5seeds.sh`), 3 GPUs, H100; (d) aggregate with
`aggregate_astar --env StackCube-v1 --runs 1` (the per-round CSV + per-seed plot + the
mean/curve all work for one run).

## INVARIANTS — fairness rules that MUST hold (two audits enforced these)

1. **Shared bootstrap:** both arms in a seed load the SAME
   `run_<seed>/shared_baselines/init_ckpt.pth` (one bootstrap per job).
2. **`nd_retrain=1` for BOTH** (retrain from scratch every demo); `target_sr=1.0`
   (stop only at budget=100 OR 100%; 90% is read off the curve, not a stop point).
3. **Both arms add demos ONLY from genuine failures.** `p4/select_arm.py` selects among
   FAILED candidates and `if not fails: continue` (skips all-success rounds). The old
   `fails or cands` fallback let the expert "correct" a SUCCESS from a reset state — an
   easy demo diff_dagger can't make. **Do not reintroduce it.**
4. **Episode backstop `max_episodes_per_arm` (5000)** is anti-infinite-loop only; it must
   never truncate a method before its 100-demo budget. A seed that hits it is a genuine
   plateau, labeled `stopped_reason=max_episodes` and right-censored in aggregation.
5. **Identical held-out eval** for both arms (same `heldout_seed_base=7777`,
   `heldout_n`, frozen-policy protocol). Screening/rollout seeds (0,1,2,…) are disjoint
   from eval seeds.
6. **Primary metric = demonstrations added (`n_queries`).** `total_expert_calls` is a
   SECONDARY, method-specific cost (p4_select screens with ~hundreds of `get_action`
   calls/demo; diff_dagger's native query ≈1/demo) — report it but label it
   non-comparable.

---

## HARNESS MAP

- **Engine = the user's Diff-DAgger fork**, read-only at `external/diff_dagger` →
  `/weka/s226137394/diff-dagger`. Vendors ManiSkill3 (3.0.0b7), the V-objective CNN-UNet
  diffusion policy, PushT PPO expert + panda motion planners, the Diff-DAgger query rule,
  the Qwen P4 VLM→reason→prescribe pipeline, the FastAPI proxy, `_bootstrap_shared_init`,
  `train_policy`, `evaluate_heldout`. **Never edit the fork; import it.**
  `envs/env_setup.bootstrap_fork_path()` puts it on `sys.path` (and binds fork subpkgs in
  `sys.modules` to dodge the `DmNfull/model` shadowing).
- **The two arms under comparison:**
  - `selection/iil_baselines.py::run_iil_arm` (kind `"diff"` = diff_dagger; also the 5
    IIL baselines). Native diffusion-loss CDF query at alpha=0.99.
  - `p4/select_arm.py::run_p4_select_arm` (p4_select): roll `n_cand` candidates, take
    top-3 highest-discrepancy FAILURES, LLM picks one (`_llm_select`, text-only), expert
    corrects on-policy from `t_star`.
- **Orchestration:** `orchestrator/_common.py` (`run_suite`→`run_method` dispatch,
  `resolve_knobs`, `build_cfg`, the `ExpertCallCounter` proxy). `orchestrator/run_one.py`
  (`METHOD_SPEC`), `orchestrator/workspace.py` (`METHOD_DIR_NAMES`).
- **Aggregation:** `aggregation/aggregate_astar.py` (this study) — mean±std curves,
  queries-to-threshold, censoring, expert-call axis. (`aggregation/aggregate.py` is the
  older cross-env/7-method one.)
- **Configs:** `config_astar.yaml` (the 5-seed run — single source of truth),
  `config_astar_smoke.yaml` (tiny validation). `config.yaml` is the older 7-method config.
- **Launchers:** `submit_astar_5seeds.sh` (calls `sbatch` 5×, `+`-encoded
  `METHODS="diff_dagger+p4_select"`), `run_pool_rl_robo.sh` (the SLURM script — decodes
  `+`→`,`, sets `NEED_LLM=1` for p4_select, boots vLLM+proxy, runs the orchestrator).
- **Env id map:** `envs/env_setup.py::TASKS`. PushT-v1 → fork `PushT-v2` +
  `PushT-Start-v0` reposition env + PPO expert. Stack/Pick/Plug → same id, `motionplanner`
  expert (`envs/experts.py::MotionPlannerExpert` wrapping the panda solutions). Switching
  env = set `ENV` (the array index in `run_pool_rl_robo.sh::ENVS` is overridden by `ENV`).

## CLUSTER / ENV

- **Conda:** orchestrator `diffdagger` (`/home/s226137394/.conda/envs/diffdagger/bin/python`,
  sapien 3.0.0b1 / torch 2.4.1+cu121); vLLM servers `vllm_embed`; proxy `maze`.
- **3 GPUs/node:** GPU0 Qwen3-VL-32B (vision), GPU1 Qwen3-32B (text), GPU2 orchestrator
  (ManiSkill GPU sim + diffusion). p4_select needs the LLM → 3 GPUs; a diff-only run
  could use 1 GPU + `--no-llm`.
- **SLURM:** partition `gpu-large`, `qos=batch-long` (10-day max), always
  `--nodes=1 --gpus-per-node=N` (never `--gpus=N` — can split across nodes),
  `--constraint=gpu-h100` (h200 nodes hit SAPIEN Vulkan device-lost on render).

## DIFF-DAGGER PAPER FAITHFULNESS (arXiv 2410.14868, verified)

Our diff_dagger is the paper's own algorithm with matching hyperparameters — it is NOT
undertrained. Pushing(1-Expert) Table IV: N_i=20, N_f=100, N_d=4, α=0.99, K=1, N_b=512.
Ours: initial_demos=20 (=N_i), budget=100 (=N_f), α=0.99, patience K=1, native fork query
rule (`get_action(dagger=True)`, CDF(loss)>α-quantile). batch_multiplier=32 → N_b=16*32=512
(matches the paper; PushT seeds 98949-53 used bm=8→N_b=128, self-consistent but 4× noisier).
We differ only in nd_retrain=1 (vs paper N_d=4 — we retrain MORE often, benign) applied to
BOTH arms. **We are STATE-based**; compare ONLY to the paper's STATE column: Diff-DAgger
Pushing(1-Expert) **State = 0.96 @ 100 demos** (image=0.87/0.94 is NOT our setting). Our
diff_dagger reaches 1.0 at ~46–75 demos — consistent with / better than the paper, so the
p4_select win is over a faithful, well-trained Diff-DAgger. The paper's "ND=40" is its
Table II *50%-success failure-prediction* count (not a convergence budget); "N_d=4" is the
per-round intervention cadence — neither is a performance number.

## CAVEATS TO DISCLOSE IN THE PAPER

1. `total_expert_calls` is a method-specific cost (oracle screening for p4_select vs
   implicit loss-query for diff_dagger), reported as a secondary, non-comparable axis;
   the comparable budget is demonstrations added. With a human expert the p4_select
   screening would be a learned safety classifier (à la SafeDAgger), not labeling effort.
2. Any seed with `stopped_reason=max_episodes` is a plateau backstop (right-censored at
   its final demo count), not a successful budget exhaustion.
3. Action discrepancy lives in joint-delta space (`selection/iil_baselines.py`).
4. The LLM selector falls back to the highest-discrepancy candidate (index 0) on LLM
   failure; quantify how often this fires from the logs.
5. PushT used the fork's PPO expert + PushT-v2/PushT-Start-v0 remap; other tasks use panda
   motion-planner experts (different demo source — note it).

## HISTORY (so you don't repeat killed runs)

- Live/final: jobs **98949–98953** = PushT seeds 1–5, all COMPLETED (the result above).
- Killed (do NOT use their data): 98921–98925 (audit found episode-cap + aggregation
  confounds) and 98930–98934 (audit found the reset-state-demo confound). 98928/run_950 =
  a smoke. run_0/900/901 = older single-seed artifacts (pre-audit; run_901's p4_select may
  have run LLM-OFF due to a `submit_p4select.sh` export bug — don't trust it).
- See `FAILURE_ANALYSIS.md` for why the old gym/MuJoCo/Fetch env set was abandoned.

## NEVER DO

- Never edit the fork (`external/diff_dagger`); import/adapt into the suite.
- Never bare-`pip` into a cloned env (PATH gotcha corrupts `maze`).
- Never `--gpus=N` (splits across nodes); use `--nodes=1 --gpus-per-node=N`.
- Never launch the 5-seed run on a new env without a passing smoke first.
- Never reintroduce the `fails or cands` fallback in `select_arm.py` (invariant #3).
