# P4-LLM **V3 Hybrid** — Plain explanation + StackCube handoff

This file has **two parts**:

1. **PART 1 — For a human (Suyog):** what the V3 hybrid is, in plain Q&A, including the
   questions a supervisor is likely to ask. Read this top to bottom; no code needed.
2. **PART 2 — For the LLM (next session):** the technical context to replicate the V3 hybrid
   architecture on the **StackCube** task. This is a handoff brief, written to be executed.

A ready-to-paste **NEW-SESSION PROMPT** is at the very bottom.

---
---

# PART 1 — V3 Hybrid explained (human Q&A)

### Q1. In one sentence, what is the V3 hybrid?
It's our P4-LLM method where, each round, an LLM looks at the robot policy's recent failures,
groups them into *types*, and then **decides** the smartest way to spend one expert
demonstration: either **fix one real failure exactly where it happened** (SELECT), or
**invent one new "middle-ground" scene that covers several similar failures at once** (BRIDGE).

### Q2. What is Diff-DAgger (the thing we're trying to beat), in plain terms?
Diff-DAgger is the classical method. It watches the policy run, and the moment the policy looks
unsure, the expert takes over *right there* and finishes the task. That correction becomes a
training demo. It's precise, but it has **no memory of failure types** — if ten failures are
really the same mistake, it still fixes them one-at-a-time, spending ten demos.

### Q3. Why is our method potentially better? What is "compression"?
Because the LLM groups failures into types and can fix a whole *type* with **one** demo instead
of one demo per instance. That's the "compression" — *fewer demonstrations to reach the same
skill*. Our whole research aim is: **can an LLM compress corrections and be more sample-efficient
than classical selection-based DAgger?**

### Q4. We tried three versions. What were they and why did we land on V3?
- **v1 (synthesis):** the LLM invented brand-new scenes near the failures. Problem: those scenes
  are slightly *fake* — the robot is tested on natural scenes it actually encounters, so demos in
  invented scenes don't transfer well. It **tied** Diff-DAgger and plateaued.
- **v2 (pure selection):** the LLM only picks a *real* recorded failure, and the expert corrects
  it on-policy (just like Diff-DAgger's internals). This is faithful and strong, but near the end
  it **gets stuck**: when the policy is ~92% good, the failures it finds are flaky, and when we
  re-run them to make a demo the (now-good) policy solves them — so there's nothing to correct and
  progress stalls.
- **v3 (hybrid):** give the LLM **both** tools and let it choose each round. Real-failure
  correction when one failure is representative (SELECT); a new middle-ground scene when several
  failures are spread out and one bridging demo covers them all (BRIDGE). This is what the
  supervisor asked for: **LLM *recommendation* + compression, with the leniency to do either.**

### Q5. What does "SELECT vs BRIDGE" actually mean, concretely?
- **SELECT ep#** — "Failure #7 is a good representative of this whole group. Re-run that exact
  scene and let the expert fix it from the point the policy went wrong." (a real, on-policy demo)
- **BRIDGE ep#,ep#** — "Failures #3 and #9 are at opposite ends of the same mistake. Don't fix
  either one; instead set up a scene **in between them** and demonstrate from there — one demo
  that teaches both." (the LLM's creative compression)

### Q6. Did we beat Diff-DAgger on PushT? (the result)
**Yes — on both seeds, on the headline metric (demonstrations needed to reach 100% success).**

| Seed | V3 hybrid | Diff-DAgger | Winner |
|---|---|---|---|
| 1 | **31 demos → 100%** | 46 demos → 100% | **Hybrid (1.48× fewer)** |
| 2 | **46 demos → 100%** | 48 demos → 100% | **Hybrid** |

Mean: **38.5 vs 47 demos** (~18% more sample-efficient). It is also the **first** P4-LLM variant
to ever reach 100% on PushT — the old method saturated around 85% and never crossed 90%.

### Q7. Be honest — where does Diff-DAgger still do better?
Diff-DAgger reaches **90%** slightly *faster* (it's very precise early). Our method spends a few
early demos on bridges, so it's a touch slower to 90% — but it's much faster through the brutal
**90%→100% endgame**, which is the part that actually defines "task solved" and the point both
methods stop at. So: **Diff-DAgger sprints early; we finish first.**

### Q8. What is the ablation, and why does it matter for the story?
We ran **pure-selection (v2)** alongside the hybrid on the same seeds. On seed 2, pure-selection
**got stuck at 92%** and never finished. On that *exact same seed*, the hybrid reached 100% — and
the only difference is the **BRIDGE** option. So the ablation proves the LLM's compression/leniency
isn't decoration — **it's what carries the method through the endgame that pure selection can't.**

### Q9. Is the comparison fair (apples-to-apples)?
Yes, and this is enforced: both methods start from the **byte-identical** initial policy
(same `init_ckpt.pth`), retrain from scratch the same way after every demo, are scored on the
**same frozen held-out set** (100 episodes, fixed seeds), and "one demo" means the same thing for
both (one *successful* expert demonstration; failed/skipped attempts cost nothing). The only
difference between the arms is **how the next demo is chosen.**

---

## Supervisor-question rehearsal (likely pushback + the answer)

**"Isn't this just p4_select with extra steps?"**
No. p4_select picks 1 of 3 failures from a 3-line text list. The hybrid reasons over the *whole
failure census* grouped into types (with cluster geometry + memory of what it already covered) and
can **also create a bridging scene** — which p4_select cannot. And critically, on seed 2 pure
selection *failed* (stuck at 0.92) where the hybrid succeeded.

**"Is the BRIDGE just the old failed synthesis (v1/p4_top3)?"**
Same *mechanism* (a prescribed scene), but used **surgically and rarely, by LLM choice**, not for
every demo. v1 synthesized *every* demo and plateaued; the hybrid bridges only when it judges a
group is spread out, and anchors the bridge on a real failure's configuration. The data shows this
targeted use *helps* (it's what cracked seed 2's endgame) whereas blanket synthesis *hurt*.

**"Why slower to 90% but faster to 100%?"**
Bridges trade a little early precision for *coverage*. Early on, precise single corrections (what
Diff-DAgger and pure-select do) raise SR fastest. But near the top, the remaining failures are
diverse and flaky; a bridging demo that covers a spread of them is what finally closes the gap.
Net: we win the metric that matters (demos-to-100%).

**"How do you know the clustering is real and not noise?"**
We don't cluster raw high-dimensional state — we cluster a small (6-D) physically-meaningful
signature (where the object was, its orientation, how far into the task it failed, contact
distance). On real PushT data this grouped 34 failures into 4 types with 28 in one dominant type —
and it's all logged per round, so it's auditable, not assumed.

**"Where could it break / what's the weakness?"**
The endgame "re-roll solves it" stall (pure-selection hit it). The hybrid mitigates it via bridges,
but on a harder task it could still slow down. The clean future fix: near convergence, correct from
the *recorded* failure state instead of re-rolling.

**"Is it rigged in our favor?"**
No — same bootstrap, same eval, same budget definition, same retrain. If anything the held-out eval
(natural scenes) is *harder* for us than for Diff-DAgger, because our bridges are slightly
off-distribution; we win anyway.

---
---

# PART 2 — LLM Context-Sharing Section (PushT → StackCube)

**You (the LLM in the next session) are being asked to replicate the V3 hybrid on StackCube.**
Read this whole section before acting. It encodes a costly recon so you don't repeat it.

## 2.0 ⚠️ READ FIRST — StackCube is NOT in the same state PushT was

A 3-agent recon (2026-06-14) established:

1. **No finished Diff-DAgger StackCube baseline exists.** `results/StackCube-v1/run_1/diff_dagger`
   is **truncated at round 14, final SR 0.04**; `run_950` is a **budget=3 smoke, SR 0.0**. You have
   **nothing valid to compare the hybrid against.**
2. **StackCube does not currently learn.** Both bootstraps have **init_sr = 0.0**, and the one real
   run plateaus at **SR ≤ 0.04**. A "beat Diff-DAgger" comparison in this regime is **vacuous** —
   neither method leaves the floor.
3. `StackCube-v1` is **`wired=False`** in `envs/env_setup.py` (still flagged unverified).
4. `assets/stackcube/` (the demo `dataset_dir`) is **empty** — fresh runs have no demos unless you
   reuse an `init_ckpt.pth` via `P4_REUSE_INIT_CKPT` or regenerate motion-planner demos.
5. There is **no StackCube V3-hybrid config** (`config_p4_hybrid.yaml`/`config_p4_subtask.yaml` are
   PushT-only).

**Therefore the order of work is bring-up FIRST, comparison SECOND.** Do NOT jump to "smoke + 2
hybrid jobs." See §2.5 for the correct sequence. Be honest with the user if bring-up is hard:
beating Diff-DAgger requires Diff-DAgger to actually work on StackCube first.

## 2.1 The architecture you're replicating (recap of what won on PushT)

Per round: roll the current diffusion policy to find failures → build a small geometric descriptor
per failure → **cluster** them into types → pick the dominant type (with cross-round **memory** so
coverage rotates) → the LLM **decides SELECT or BRIDGE** → collect **one** demo accordingly →
retrain from scratch → eval on frozen held-out → stop at target or budget. Budget unit = one
*successful* demo; empty/infeasible attempts are budget-free. A rollout-SR target hit must be
**confirmed by the held-out eval** before stopping (this fixed a fluke early-stop on PushT).

## 2.2 THE CRITICAL ENGINE FACT (do not get this wrong)

There are **two P4 execution engines** in this repo:

- **PushT engine** = the fork's `LLMGuidedDAggerPipeline` (PushT-hardcoded: tee prescription,
  `meta.json` from `HighLossImageSaver`). The V3 hybrid hooks into it via fork hooks A/B/C/D and the
  `p4_subtask/` package. **This is PushT-only.**
- **StackCube engine** = suite-native **`p4/select_arm.py`** (`run_p4_top3_arm` /
  `run_p4_select_arm`). The fork pipeline is NOT used. There is **no `meta.json`** — a failure is an
  in-memory candidate `{seed, exec_actions, t_star, peak_disc, success, n_steps}` and its state is
  reconstructed by **replaying `exec_actions` to `t_star`** and reading live env state.

**`p4_subtask` will crash on StackCube as written** (`_common.py:233-243` dispatches `p4_subtask`
unconditionally to the fork pipeline; `p4_subtask/pipeline.py:46` hard-raises without a
`PushT-Subtask-v0` reposition env). So for StackCube you do **NOT** reuse the fork-pipeline path —
you implement the hybrid **inside `p4/select_arm.py::run_p4_top3_arm`**.

## 2.3 What's reusable VERBATIM vs what you must write

**Reuse verbatim (task-agnostic — the "compression core"):**
- `p4_subtask/clustering.py` (silhouette k-sweep + sklearn/numpy fallback + `pick_dominant`)
- `p4_subtask/diversity.py` (`farthest_point_select`)
- `p4_subtask/memory.py` (`CentroidMemory` recency rotation — 2-D xy centroid; cubeA xy works)
- `p4_subtask/telemetry.py`
- `p4_subtask/planner.py::SubtaskPlanner._parse_choice` (regex for `SELECT epN` / `BRIDGE epN,epN`)
- The **on-policy correction primitives already running for StackCube**:
  `p4/select_arm.py::_safe_rollout_one` (`:101-161`, diffusion-loss detector via
  `_policy_loss_seq` `:75-98`) and `_correct_onpolicy_from` (`:164-189`). For StackCube the expert
  is `MotionPlannerExpert`, whose `move_to_next_goal` **re-plans from the live state with no reset**
  (`experts.py:211-213`) — i.e. the SELECT primitive is *already implemented*.
- The **BRIDGE collector already exists**: `select_arm.py::_collect_prescribed_demo` (`:461-491`)
  uses `StackCube-Start-v0.set_prescription(cubeA_xyz, cubeB_xyz, cubeA_zrot, cubeB_zrot)`
  (`stackcube_start.py:61-83`, clamps X[-0.15,0.15] Y[-0.25,0.25] z=0.02, MIN_SEP 0.05) then the
  motion planner solves it; `CubePrescriptionClass` validates poses (`p4/prescription.py`).

**You must write (StackCube variants):**
1. **`descriptor_stackcube.py`** — a `StackCubeFailureDescriptor`. Feature ≈
   `[cubeA_x, cubeA_y, cubeB_x, cubeB_y, (A−B offset dx,dy or dist), grasp_state, progress=t*/T]`.
   Built by **replaying a `select_arm` candidate (`exec_actions`→`t_star`) and snapshotting**
   `cubeA.pose`, `cubeB.pose`, `agent.qpos`, `is_grasping(cubeA)` (or read the 30-D state_dict obs
   keys `extra_cubeA_pose`/`extra_cubeB_pose`/`agent_qpos`). **NOT a meta.json parser.** Carry
   `seed` = candidate seed (scene identity for on-policy re-roll).
2. **`kag_bounds_stackcube.py`** (or parameterize `kag_bounds.py`) — cube xy/z ranges + MIN_SEP,
   from `p4/kag/StackCube-v1.json` + `stackcube_start.py` constants.
3. **A `CubeLayoutSpec`** (replaces `ResetSpec`): `mode ∈ {onpolicy_correction, bridge}`; for BRIDGE,
   a middle-ground `cubeA_xyz`/`cubeB_xyz` from 2–3 cited failures (bounded shift, `set_prescription`
   clamps). No stick/TCP geometry.
4. **A StackCube planner binding** — subclass/parameterize `SubtaskPlanner` so `_pose_dist`,
   `_build_context` (prompt text), and `reset_spec_for` use **cube** geometry and emit
   `CubeLayoutSpec`. Inherit unchanged: `_compute_round`, `select_candidates`, `_parse_choice`,
   `_pick_member_for_correction`, `note_collect`, the memory/diversity/cluster core.
5. **Inline injection in `select_arm.py::run_p4_top3_arm`** (per-round block `:650-677`), NOT fork
   hooks: after `fails = [...]`, build StackCube descriptors → `planner.select_candidates(...)`
   (hook A) → `planner.round_context(rnd)` into the prescriber prompt (hook B) →
   `planner.reset_spec_for(cmd, rnd)` dispatch (hook C): `onpolicy_correction` →
   `_correct_onpolicy_from(env, expert, policy, cfg, chosen_candidate)`; `bridge` →
   `set_prescription` + `_collect_prescribed_demo`. Then `planner.note_collect(...)`. Map hook D
   (heldout-confirm before stop) onto the existing `if sr >= target_sr` check. **Leave
   `run_p4_select_arm` untouched** (it stays the pure-SELECT ablation).
6. **A StackCube hybrid config** — clone `config_stack.yaml` (StackCube budget/hyperparams:
   `initial_demos=20`, `budget=60`, `target_sr=1.0`, `nd_retrain=1`, diff_dagger `alpha=0.99
   patience=2 batch_multiplier=32`) + add a `p4.subtask`/`collect: hybrid` block. Method name: keep
   it routing through `run_p4_top3_arm`'s motion-planner branch (the cleanest is to add the
   hybrid behind a config flag read inside `run_p4_top3_arm`, so `p4_top3` on StackCube becomes the
   hybrid when `subtask.collect: hybrid` is set — OR add a proper `p4_subtask` motion-planner
   branch in `_common.py:233` that calls a new `run_p4_subtask_motionplanner_arm`). Decide and keep
   it apples-to-apples.

## 2.4 Fairness invariants you MUST preserve (from claude_context.md, two audits)
- Shared bootstrap reused exactly (`P4_REUSE_INIT_CKPT`); never rebuild (GPU-nondeterministic).
- `nd_retrain=1`, `target_sr=1.0` (stop only at 100% or budget; 90% is read off the curve).
- Demos ONLY from genuine failures (`if not fails: continue`). **Never** reintroduce the
  `fails or cands` fallback (lets the expert "correct" a success — an easy demo Diff-DAgger can't
  make).
- Identical held-out eval (`heldout_seed_base=7777`, frozen). Screening seeds disjoint from eval.
- Budget unit = one *successful* demo; empty/infeasible/skipped attempts are budget-free.
- Primary metric = demonstrations added (`n_queries`); `total_expert_calls` is secondary.
- **Never edit the fork** for StackCube (the StackCube engine is suite-side anyway).

## 2.5 The correct work order for the next session
1. **Diagnose StackCube bring-up.** Why is SR ~0? (expert demo quality, dataset, training, the
   `assets/stackcube` empty-dir issue, proprio_dim=30 contract.) Tools exist:
   `tools/diag_stackcube_planner.py`, `tools/sweep_stackcube_bootstrap.py`,
   `tools/test_stackcube_render.py`. Get diff_dagger SR meaningfully **> 0** with a small run.
2. **Run Diff-DAgger StackCube for ≥2 seeds to full budget=60**, finished, as the baseline. Mark
   `StackCube-v1` `wired=True` once verified. Save the bootstraps.
3. **Archive any existing StackCube method results** (`run_*/p4_top3`, `run_*/p4_select`, stale
   diff_dagger smokes) to `results/StackCube-v1/_archived_*` before fresh runs.
4. **Implement the StackCube V3 hybrid** (§2.3 items 1–6). Unit-test the pure logic offline.
5. **Minimal smoke** (low budget e.g. 1–2, few epochs) — verify SELECT and BRIDGE both fire and a
   demo is collected, no errors. Per the user's standing rule: *always* smoke with low budget + few
   epochs for a fast result before any full run.
6. **Submit exactly two full hybrid jobs** (same 2 seeds as the diff_dagger baseline, reusing those
   bootstraps via `P4_REUSE_INIT_CKPT`). Monitor. Report the q90/q100 head-to-head + the
   SELECT/BRIDGE mix.

## 2.6 Key file:line anchors
- Dispatch: `orchestrator/_common.py:198-231` (p4_top3 → motionplanner branch),
  `:233-243` (p4_subtask — **no motionplanner branch; PushT-only, will crash on StackCube**).
- SELECT primitive: `p4/select_arm.py:164-189` (`_correct_onpolicy_from`), used in
  `run_p4_select_arm:334-369`.
- BRIDGE primitive: `p4/select_arm.py:461-491` (`_collect_prescribed_demo`),
  `_prescribe_and_collect:604-648`, per-round hybrid attach point `run_p4_top3_arm:650-677`.
- Diffusion detector: `p4/select_arm.py:75-98` (`_policy_loss_seq`), `_safe_rollout_one:101-161`,
  `_calibrate_diffusion:529-547`.
- Reposition env + prescription: `envs/stackcube_start.py:48-117` (`set_prescription:61-83`).
- Motion-planner expert (live-state replan, NO reset): `envs/experts.py:156-213`,
  `_plan_stackcube:264-318`.
- Task spec: `envs/env_setup.py:100-110` (StackCube; `expert_kind="motionplanner"`,
  `reposition_env_id="StackCube-Start-v0"`, `wired=False`).
- Config + contracts: `configs/hydra/stackcube_state.yaml` (proprio_dim=30, action_dim=8,
  rel_joint_pos), `config_stack.yaml`, `config_stack_p4top3.yaml`, `p4/kag/StackCube-v1.json`.
- The V3 hybrid to port from: `p4_subtask/{planner,descriptor,subtask_entry,collect,clustering,
  diversity,memory,telemetry}.py`; PushT result + design in the memory note `p4_subtask_method.md`
  and this file's Part 1.

## 2.7 PushT result (the bar you're replicating the *approach* of, not the numbers)
V3 hybrid beat Diff-DAgger on demos-to-100%: **31 vs 46** (seed 1), **46 vs 48** (seed 2); first P4
variant to reach 1.0; pure-select ablation plateaued at 0.92 on seed 2 (→ BRIDGE is necessary).
Jobs: hybrid `run_11`/`run_12`, pure-select `run_1`/`run_2`, v1 archived
(`results/PushT-v1/_archived_p4_subtask_v1/`).

---
---

# NEW-SESSION PROMPT (copy-paste this into a fresh chat in the same project)

> We just won PushT with our P4-LLM **V3 hybrid** (it beat Diff-DAgger on demos-to-100%: 31 vs 46
> and 46 vs 48 across two seeds — first P4 variant to reach 100%; the pure-selection ablation
> plateaued at 0.92, proving the LLM's BRIDGE option is what cracks the endgame). I now want the
> **same V3 hybrid architecture to beat Diff-DAgger on the StackCube task**, as a clean
> apples-to-apples comparison (V3 hybrid vs Diff-DAgger), then **two production jobs on two seeds**
> — but only after a passing low-budget/few-epoch smoke.
>
> **Start by fully reading**
> `Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo/V3_HYBRID_explained_and_StackCube_handoff.md`
> — especially **PART 2 (the LLM Context-Sharing Section)** and the memory note
> `p4_subtask_method.md`.
>
> **Critical things that doc establishes (don't skip):** StackCube uses the suite-native
> `p4/select_arm.py` engine, NOT the fork pipeline the PushT hybrid hooks into — so this is a real
> port (new cube descriptor, cube-layout BRIDGE spec, inline injection into `run_p4_top3_arm`), and
> the on-policy SELECT + cube-prescription BRIDGE primitives **already exist** in `select_arm.py`.
> AND — **StackCube is not validated yet**: there is no finished Diff-DAgger StackCube baseline
> (only a truncated SR-0.04 run and an SR-0.0 smoke), the bootstraps start at init_sr 0.0, and the
> task currently doesn't learn. So the order is **bring-up first** (get Diff-DAgger StackCube to
> actually learn + finish 2 seeds), **then** port the hybrid, **then** smoke, **then** the two full
> jobs reusing those exact bootstraps via `P4_REUSE_INIT_CKPT`. Preserve all fairness invariants in
> §2.4. Archive any stale StackCube results first. Verify everything against the live code before
> editing — the doc's anchors are a map, not gospel. Be honest with me if bring-up is the real
> blocker; we can't beat Diff-DAgger on StackCube until Diff-DAgger works on StackCube.
