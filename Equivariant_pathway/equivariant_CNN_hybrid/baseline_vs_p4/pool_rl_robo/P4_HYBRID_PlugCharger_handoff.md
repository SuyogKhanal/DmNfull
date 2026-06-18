# P4-LLM **V3 Hybrid** → **PlugCharger** handoff (context.md)

This is the execution brief for replicating the **P4-LLM V3 Hybrid vs Diff-DAgger**
head-to-head on the **PlugCharger-v1** task, written for a fresh session. It mirrors
the StackCube handoff (`V3_HYBRID_explained_and_StackCube_handoff.md`) and folds in
**everything learned bringing the hybrid up on StackCube** (the costly lessons —
don't relearn them). A ready-to-paste NEW-SESSION PROMPT is at the very bottom.

Read this whole file before acting. Verify every anchor against the LIVE code before
editing — the anchors are a map, not gospel (this bit us repeatedly on StackCube).

---
---

# PART 1 — What the V3 Hybrid is + how it has gone so far (read first)

### The method (one paragraph)
Each round: roll the current diffusion policy to find failures → build a small
geometric descriptor per failure → **cluster** them into types → pick the dominant
type (with cross-round **memory** so coverage rotates) → the **LLM decides SELECT or
BRIDGE** → collect **one** demo accordingly → retrain → eval on a frozen held-out set
→ stop at target or budget.
- **SELECT ep#** → re-run that exact failed episode and let the motion-planner expert
  correct it on-policy from the divergence point (faithful DAgger correction).
- **BRIDGE ep#,ep#** → prescribe ONE new middle-ground scene between 2–3 cited
  failures and have the motion-planner solve it (the creative "compression").
Budget unit = one *successful* demo; empty/infeasible attempts are budget-free.

### Results so far (be honest with Suyog about the bar)
- **PushT**: the hybrid **won decisively** — demos-to-100%: 31 vs 46 (seed 1), 46 vs 48
  (seed 2); first P4 variant to reach 100%; pure-select ablation plateaued at 0.92.
- **StackCube** (just finished, 2 seeds, budget 60, warm-start 200 demos → init SR
  ~0.55): a **modest win, NOT a blowout.** Diff-DAgger tops ~0.64–0.65; the hybrid is
  ahead on the noise-robust metrics (rolling-mean SR, peak ~0.69, area-under-curve,
  demos-to-0.60), **clearly on seed 2, a wash on seed 1**. The single-eval noise band
  (±~0.10 at 100 eps) is wider than the ~0.04–0.05 margins, so it's "consistent-but-
  modest," not statistically clean. Compression fired (~19% BRIDGE). The global
  failure clustering is real (silhouette 0.24 vs shuffled-null 0.17, p=0.000); the
  per-round clustering is NOT testable (only ~6 failures/round vs PushT's ~34).
- **Lesson for PlugCharger expectations**: motion-planner tasks are noisy, slow, and
  saturate low. Expect a subtle result, plan for more eval episodes / smoothing, and
  **tell Suyog honestly if bring-up doesn't learn** — we cannot beat Diff-DAgger on
  PlugCharger until Diff-DAgger works on PlugCharger.

---
---

# PART 2 — PlugCharger technical handoff (execute this)

## 2.0 ⚠️ READ FIRST — PlugCharger is essentially UN-WIRED (more than StackCube was)
Verified 2026-06-19 against the live repo. PlugCharger-v1 has only:
- a **catalog entry** in `envs/env_setup.py:121-131` (`expert_kind="motionplanner"`,
  `reposition_env_id=None`, `hydra_cfg="diffdagger/config/sim/plugcharger_state.yaml"`
  marked "authored later"),
- a **KAG json** `p4/kag/PlugCharger-v1.json`.

It is MISSING (you must create all of these):
1. **Hydra config** — no `configs/hydra/plugcharger_state.yaml` (only `stackcube_state.yaml`
   exists). Author the suite-local PlugCharger Hydra cfg (mirror `stackcube_state.yaml`).
2. **Motion-planner plan** — `experts.py::_PLAN_REGISTRY` has only `StackCube-v1`. You
   must write `_plan_plugcharger` (port the fork's
   `mani_skill/examples/motionplanning/panda/solutions/plug_charger.py::solve` —
   grasp(obb)→reach→close→pre_insert(refine)→insert via `move_to_pose_with_screw`).
3. **Reposition env** — `reposition_env_id=None`. Write `PlugCharger-Start-v0` (mirror
   `envs/stackcube_start.py`) that honours prescribed charger + receptacle poses on
   reset (needed for BRIDGE).
4. **Normalizers** — no PlugCharger normalizers in `assets/`. Generate them (adapt
   `tools/gen_stackcube_normalizers.py`).
5. **Bootstrap / sweep / any results** — none. Run the bootstrap-size sweep
   (adapt `tools/sweep_stackcube_bootstrap.py`) to find the warm-start N.

**So PlugCharger is a FULL bring-up, harder than StackCube** (contact-rich peg
insertion, success = charger inserted within **5 mm / 0.2 rad** — the tightest
tolerance of the suite). Budget bring-up time accordingly.

## 2.1 Replicate the Diff-DAgger PAPER faithfully (Suyog's explicit ask)
Use the Diff-DAgger paper's **Table IV "Plugging" row** (source of truth — it's in
`/weka/s226137394/diff-dagger/claude_context_diffdagger.md §5`), the fork's
PlugCharger env, and the fork's motion-planner solution as the expert:

| Param | Plugging value | Note |
|---|---|---|
| Action space | **Abs Pose** (`ee_pose_6d`) | **NOT rel_joint_pos** (StackCube/PushT used joint). Big contract change. |
| To / Tp / Ta | 1 / **64** / 8 | obs horizon 1; predict 64, execute 8 |
| Pred target / Td | V-obj / **64** | 64 denoising steps (contact-rich; StackCube used 16) |
| N_i (initial_demos) | 20 (paper) | but **warm-start higher per the sweep** — see 2.4 |
| N_f (budget) | **100** | (StackCube used 60) |
| N_d (retrain cadence) | **8** | paper retrains every 8; see fairness note below |
| α / K / N_b | 0.99 / **2** / 512 | K=2 filters fluctuations |

Training: 300 epochs, `max_train_steps=30000`, batch 64, lr 1e-4 AdamW, from scratch
every N_d (paper §6 — fresh init + fresh optimizer; NO warm-start of the policy).

**Expert** = the fork motion-planner (`solutions/plug_charger.py`), the same family
the paper uses. **Everything data-path is the fork's** (its PlugCharger env, dataset,
diffusion policy) — never edit the fork; author StackCube-style suite-local configs/
envs that resolve fork-first.

## 2.2 THE ENGINE FACT (unchanged from StackCube)
PlugCharger is a **motion-planner** task ⇒ it uses the **suite-native engine**
`p4/select_arm.py::run_p4_top3_arm`, NOT the fork's PushT-hardcoded
`LLMGuidedDAggerPipeline`. `_common.py:198-224` already dispatches `p4_top3` +
`expert_kind=="motionplanner"` to `run_p4_top3_arm`. `p4_subtask` is PushT-only
(would crash). The V3 hybrid is **already implemented inside `run_p4_top3_arm`** for
StackCube — PlugCharger is mostly a **re-parameterization**, not a rewrite.

## 2.3 What already EXISTS (reuse) vs what you WRITE for PlugCharger
**Already built + validated on StackCube (reuse, mostly verbatim):**
- `p4/stackcube_hybrid.py` — the hybrid planner: `StackCubeFailureDescriptor`,
  `cluster_stackcube`, `CubeLayoutSpec`, `StackCubeHybridPlanner` (SELECT/BRIDGE via
  the prescriber's `rationale` tag + `extra_addendum`; feasibility gate; escalation
  to SELECT). Reuses the task-agnostic core `p4_subtask/{clustering,memory,telemetry}`.
- `p4/select_arm.py` inline injection in `run_p4_top3_arm`: `_stackcube_descriptor`
  (replay candidate→t*→read poses), hybrid-aware `_prescribe_and_collect`
  (SELECT→`_correct_onpolicy_from`, BRIDGE→`_collect_prescribed_demo`), gated by
  `hp["subtask"].collect=="hybrid"`, threaded through `_common.py`. **TEXT-ONLY default
  + a `use_vlm` flag; eval env forced HEADLESS** (`make_eval_env`) — see render lesson 2.5.
- `run_pool_rl_robo.sh` **`SKIP_VLM=1`** path (2-GPU, LLM+orch, no VLM) — see 2.5.
- Figure suite `results/aggregate/figures/make_stackcube_figures.py` (6 figs) +
  `make_stackcube_headtohead.py` + `tools/reeval_final.py` (tighter eval).

**Best approach for PlugCharger**: GENERALIZE the StackCube hybrid into a task-agnostic
core rather than copy-paste. The SELECT/BRIDGE planner logic, clustering, memory,
feasibility gate, and the `_prescribe_and_collect` injection are **geometry-agnostic**.
Only these differ and must be written:
1. **`PlugChargerFailureDescriptor`** — feature ≈ `[charger_x, charger_y,
   charger→socket dx, dy, grasp_state, insertion_progress(t*/T)]` (+ maybe insertion
   angle error). Built by replaying a `select_arm` candidate (`exec_actions`→`t_star`)
   and snapshotting `charger.pose`, `receptacle`/`goal_pose`, `agent.is_grasping(charger)`.
   Use `.cpu().numpy()` on GPU pose tensors (GPU-sim is mandatory — see 2.6).
2. **PlugCharger kag-bounds** — charger/socket xy ranges from `p4/kag/PlugCharger-v1.json`
   + the env's spawn region; the BRIDGE layout (charger pose + receptacle pose) is
   clamped by `PlugCharger-Start-v0.set_prescription`.
3. **`PlugChargerLayoutSpec`** (or reuse a generic `LayoutSpec`): SELECT carries the
   candidate; BRIDGE carries the prescribed charger+receptacle poses.
4. **Planner binding** — the cube-specific bits of `StackCubeHybridPlanner`
   (`feat6`, `decision_addendum` prompt text, `_select_feasible` = charger graspable/
   on-table, layout extraction) re-expressed for charger/socket geometry.
5. **`_plan_plugcharger`** in `experts.py` + register in `_PLAN_REGISTRY` (port the
   fork solve with GPU-sim fixes).
6. **`PlugCharger-Start-v0`** reposition env + wire `reposition_env_id` in `env_setup.py`.
7. **`plugcharger_state.yaml`** (suite Hydra cfg) + a **PlugCharger prescriber** prompt
   (adapt `p4/prompts.py` cube-prescription → charger-prescription) + a hybrid config
   `config_plug_hybrid.yaml` (clone the Plugging Table IV row + a `p4.subtask` block).
8. **PlugCharger descriptor builder** in `select_arm.py` (`_plugcharger_descriptor`),
   parameterized so `run_p4_top3_arm` picks the right descriptor by `suite_env_id`.

## 2.4 Fairness invariants you MUST preserve (same as StackCube)
- Shared bootstrap reused **byte-identically** via `P4_REUSE_INIT_CKPT` (never rebuild —
  GPU-nondeterministic). `_common.py:399-414`; written to `run_<seed>/shared_baselines/init_ckpt.pth`.
- **Both arms identical config** — same N_d, N_f, α, K, action space, eval. The only
  difference is demo-acquisition. (Decide N_d: paper says 8 for Plugging; our prior
  comparisons used 1. **Pick ONE and use it for BOTH arms.** Recommend matching the
  paper — N_d=8 — since Suyog wants paper-faithful; note it makes each arm ~8× cheaper
  to retrain but coarser.)
- Demos ONLY from genuine failures (`if not fails: continue`) — never the `fails or
  cands` fallback. `select_arm.py` guard preserved.
- Identical held-out eval (`heldout_seed_base=7777`, frozen, screening seeds disjoint).
  **Use ≥200–400 eval episodes** (not 100) given the noise lesson — bake it into the config.
- Primary metric = SR-vs-demos curve + demos-to-threshold (PlugCharger won't hit 100%;
  pick a threshold both reach). `total_expert_calls` secondary.
- Never edit the fork.

## 2.5 RENDER + GPU lessons (these cost a full day on StackCube — don't repeat)
- **VLM rendering deadlocks** when many SAPIEN/Vulkan render contexts coexist. Fix:
  `make_eval_env` forces the eval env **headless** (`render_mode=None`) — it never
  renders, and that drops the contention to the working 2-context config. Already done;
  keep it.
- **Rendering is h100-only** (h200 hits "Vulkan device-lost"). The hybrid (which renders
  for the VLM) must run on **`--constraint=gpu-h100`** if `use_vlm: true`.
- **The hybrid defaults to TEXT-ONLY** (`use_vlm: false`) — robust and sufficient (the
  decision is driven by geometry, not VLM frames). For PlugCharger, **start text-only**;
  only enable the VLM (3-GPU, h100-only) if Suyog wants exact PushT parity (it cost a
  re-validation cycle on StackCube and gave no decision-quality benefit).
- **GPU sizing**: text-only hybrid = 2 GPUs (`SKIP_VLM=1`, LLM+orch); VLM hybrid = 3
  GPUs (VLM+LLM+orch). Diff-DAgger baseline = 1 GPU, no LLM. Submit per
  `submit_stack_hybrid.sh` (clone → `submit_plug_hybrid.sh`).
- **Cluster tip**: gpu-large (h100/h200) is often saturated for ≥3-GPU jobs; the a100
  `gpu` partition is frequently idle and OK for **non-rendering** jobs (baselines,
  re-eval, text-only with bnb-4bit Qwen3-32B fitting in 40GB). Rendering jobs → h100.

## 2.6 GPU-sim motion-planner gotchas (CONFIRMED on StackCube; apply to PlugCharger)
- The fork panda solves were written for CPU sim and **CRASH on GPU sim**: bare
  `.numpy()` on cuda tensors → use `.cpu().numpy()`. `physx_cpu` backend is BROKEN in
  this fork → **GPU sim is mandatory** (`sim_backend="gpu"`).
- `get_actor_obb(actor)` reads a STALE CPU pose under GPU sim → grasp targets the world
  origin (grabs air). Build the grasp from `charger.pose.p`/`.q` directly (the
  StackCube fix; the plug solve uses `compute_grasp_info_by_obb` — replace with a
  live-pose grasp). Verify with `tools/diag_*` before trusting demos.
- Settle after release so static checks register (GPU residual velocity). Demos run
  ~150–250 steps; set env `max_episode_steps` ~200–300, `expert.max_episode_steps` ~300.

## 2.7 The correct work order (DO NOT skip to "smoke + 2 jobs")
1. **Wire PlugCharger bring-up**: hydra cfg, `_plan_plugcharger` (GPU-sim-fixed),
   normalizers, reposition env. Validate the motion-planner demo SUCCESS rate first
   (StackCube planner hit 91%; PlugCharger is harder — if the planner can't reliably
   insert, nothing downstream works — tell Suyog).
2. **Bootstrap-size sweep** (adapt `sweep_stackcube_bootstrap.py`): find the warm-start
   N giving a climbable init SR on the SAME held-out eval. PlugCharger may need more
   demos (tight task). If SR stays ~0 at all N, bring-up is the blocker — report it.
3. **Run Diff-DAgger PlugCharger ≥2 seeds to budget** (paper-faithful) as the baseline;
   save the bootstraps. Mark `PlugCharger-v1` wired once it learns.
4. **Generalize/port the hybrid** (2.3 items 1–8). Unit-test the planner logic offline
   (mirror `tools/test_hybrid_logic.py` → 26/26 passed for StackCube).
5. **Smoke** (low budget/few epochs) on a GPU — verify SELECT + BRIDGE both fire, a
   demo is collected, no render hang, descriptors build. (Suyog's standing rule: ALWAYS
   smoke before a full run.)
6. **Submit exactly two full hybrid jobs** (same 2 seeds, reusing the diff_dagger
   bootstraps via `P4_REUSE_INIT_CKPT`). Monitor; report SR-vs-demos curves +
   demos-to-threshold + SELECT/BRIDGE mix. Then build the 6 figures (clone
   `make_stackcube_figures.py` → `make_plugcharger_figures.py`).

## 2.8 Key file:line anchors (verify against live code)
- Dispatch: `orchestrator/_common.py:198-224` (p4_top3 motionplanner → select_arm),
  `:399-414` (P4_REUSE_INIT_CKPT), hp threading `:211-218` (add `subtask`).
- Hybrid engine: `p4/select_arm.py::run_p4_top3_arm` (~494+): `_stackcube_descriptor`,
  `_hybrid_text_analysis`, `_prescribe_and_collect` (use_vlm gate + SELECT/BRIDGE
  dispatch), `_correct_onpolicy_from` (SELECT), `_collect_prescribed_demo` (BRIDGE),
  `if not fails: continue` guard. Planner: `p4/stackcube_hybrid.py`.
- Reposition + prescription template: `envs/stackcube_start.py` (`set_prescription`).
- Motion-planner expert template: `envs/experts.py` (`MotionPlannerExpert`,
  `_plan_stackcube`, `_PLAN_REGISTRY`); fork solve to port:
  `/weka/s226137394/diff-dagger/mani_skill/examples/motionplanning/panda/solutions/plug_charger.py`.
- Fork PlugCharger env (success = dist≤5e-3 & angle≤0.2):
  `/weka/s226137394/diff-dagger/mani_skill/envs/tasks/tabletop/plug_charger.py` (`@register_env max_episode_steps=200`, `evaluate()`).
- Paper Table IV: `/weka/s226137394/diff-dagger/claude_context_diffdagger.md §5`.
- StackCube bring-up contracts (the data-path schema to replicate): memory
  `stackcube-bringup-contracts.md` + `stackcube-bringup-rootcause.md`; configs
  `config_stack_hybrid.yaml`, `config_stack_p4top3.yaml`, `tools/gen_stackcube_normalizers.py`,
  `tools/sweep_stackcube_bootstrap.py`, `submit_stack_hybrid.sh`.

## 2.9 Be honest with Suyog
- Bring-up first; PlugCharger may not learn (tight tolerances). If the motion-planner
  can't insert reliably, or BC from N demos stays at ~0 SR, say so plainly with the
  sweep numbers — don't fake a comparison in a vacuous regime.
- The StackCube result was a **modest, noise-limited win**, not a PushT blowout. Set
  the same expectation for PlugCharger and report the noise-robust metrics + the honest
  caveats (eval-noise band, demos-to-threshold fragility).

---
---

# NEW-SESSION PROMPT (copy-paste into a fresh chat in this project)

> We just finished the StackCube head-to-head (P4-LLM **V3 hybrid** vs Diff-DAgger):
> a **modest win** — the hybrid is ahead on the noise-robust metrics (rolling-mean SR,
> peak, area-under-curve, demos-to-0.60), clearly on seed 2, a wash on seed 1; not the
> clean PushT blowout, because StackCube is noisy/slow/low-saturating. I now want the
> **same V3 hybrid methodology applied to PlugCharger-v1**, as a clean apples-to-apples
> head-to-head vs Diff-DAgger, then **two production jobs on two seeds — only after a
> passing low-budget/few-epoch smoke.**
>
> **Replicate the Diff-DAgger ORIGINAL PAPER faithfully for PlugCharger**: use the
> paper's Table IV **"Plugging"** row (Abs Pose `ee_pose_6d`, Tp=64, Ta=8, Td=64,
> Ni=20, Nf=100, Nd=8, α=0.99, K=2, Nb=512 — in
> `/weka/s226137394/diff-dagger/claude_context_diffdagger.md §5`), the fork's
> PlugCharger env, and the fork's motion-planner solution
> (`mani_skill/examples/motionplanning/panda/solutions/plug_charger.py`) as the expert.
> Work fully on the basis of their repository — **never edit the fork**; author
> suite-local configs/envs that resolve fork-first.
>
> **Start by fully reading**
> `Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/P4_HYBRID_PlugCharger_handoff.md`
> and the memories `stackcube-bringup-rootcause.md`, `stackcube-bringup-contracts.md`,
> `p4_subtask_method.md`.
>
> **Critical (don't skip):** PlugCharger is **un-wired** — no Hydra config, no
> motion-planner plan (`_PLAN_REGISTRY` has only StackCube), no reposition env
> (`reposition_env_id=None`), no normalizers, no bootstrap. So this is a **full
> bring-up first**, harder than StackCube (contact-rich insertion, success within
> 5 mm/0.2 rad). The hybrid engine **already exists** for StackCube
> (`p4/stackcube_hybrid.py` + the inline injection in `p4/select_arm.py::run_p4_top3_arm`,
> text-only by default with a `use_vlm` flag, headless eval, `SKIP_VLM` 2-GPU path) —
> PlugCharger is a **re-parameterization**: write a PlugCharger descriptor (charger/
> socket geometry), kag-bounds, `_plan_plugcharger` (port the fork solve with the
> GPU-sim fixes — `.cpu().numpy()`, grasp from live `charger.pose` not stale obb),
> `PlugCharger-Start-v0` reposition env, `plugcharger_state.yaml`, a charger prescriber
> prompt, and `config_plug_hybrid.yaml`. Then: bring-up → Diff-DAgger baseline (2 seeds,
> finished) → port hybrid → smoke → 2 full jobs reusing those bootstraps via
> `P4_REUSE_INIT_CKPT`. Preserve all fairness invariants (§2.4: byte-identical
> bootstrap, both arms identical config incl. Nd, fails-only, frozen 7777 eval — use
> ≥200 eval eps given the noise). Heed the render lesson (§2.5: VLM rendering is the
> deadlock risk — default text-only; VLM only on h100, 3-GPU, if exact parity is wanted)
> and the GPU-sim gotchas (§2.6). **Be honest with me if bring-up is the real blocker —
> we can't beat Diff-DAgger on PlugCharger until Diff-DAgger works on PlugCharger.**
> Verify every code anchor against the live code before editing.
