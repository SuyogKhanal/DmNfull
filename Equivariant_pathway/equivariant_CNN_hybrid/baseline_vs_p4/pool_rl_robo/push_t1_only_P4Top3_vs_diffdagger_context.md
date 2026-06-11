# push_t1_only_P4Top3_vs_diffdagger_context.md

**PRIMARY context document for the PushT-v1 — `p4_top3` (LLM-prescription) vs Diff-DAgger
equal-to-equal comparison.** Read this top-to-bottom to resume. It is self-contained:
everything PushT-relevant from the older `claude_context.md` (suite/harness, cluster,
fairness invariants, paper faithfulness) and `claude_p4_top3_vs_diff-dagger_context.md`
(the experiment design + implementation) has been merged here. All file:line anchors below
were **re-verified against the live code on 2026-06-11** (a 6-agent verification pass) — they
are current, not the stale anchors from the old docs.

> Scope: **PushT-v1 only**, **p4_top3 vs diff_dagger only**. StackCube/PickCube/PlugCharger
> bring-up, the motion-planner-expert work, and the older 7-method suite were intentionally
> dropped. If you need those, see the original `claude_context.md`.

---

## ▶ STATE (update each session)

```
RUN TARGET:   PushT-v1 — p4_top3 vs the EXISTING diff_dagger seeds 1-5 (run_1..run_5)
STAGE:        FULL 5-SEED p4_top3 RUN IS LIVE (jobs 99591/99593/99594/99595/99596), all RUNNING
AS OF:        2026-06-11
LAST:         smoke (99400) → 1-seed validation (99591, fixed the high-loss-frame bug) →
              fired all 5 seeds reusing each seed's diff_dagger bootstrap. Verified all 7
              code changes are present in the current tree.
FINDING:      p4_top3 is UNDERPERFORMING diff_dagger on SR (plateauing ~0.85-0.91 vs
              diff_dagger's 1.0) — as predicted. The win to report is COVERAGE, not SR.
NEXT:         (a) let all 5 finish (~2026-06-12); (b) aggregate p4_top3 vs diff_dagger
              (queries-to-90%, final-SR + the coverage heatmap); (c) optional Phase B
              (per-env configurable prompts) — currently a zero-diff no-op (prompts_dir: null).
```

---

## 1. What this experiment is

Compare a **P4-LLM prescription** demo-acquisition rule against **Diff-DAgger**, sharing the
SAME diffusion-policy backbone and the SAME per-seed bootstrap, so ONLY the rule that
chooses the next demonstration differs:

- **Diff-DAgger** (baseline, the reference seeds): native diffusion-loss CDF query
  (`get_action(dagger=True)`, CDF(loss) > α-quantile). Corrects failures **in place** —
  the expert finishes the exact episode where the current policy diverged.
- **p4_top3** (this study): **VLM → reasoning analysis → reasoning prescription → plain LLM
  → prescribe a NEW object pose**; the expert solves that prescribed scene → ONE demo. The
  LLM **generates new scenarios** rather than correcting existing ones.

**Headline metric** = demonstrations to reach a held-out success threshold, on a FROZEN
`evaluate_heldout` (seed 7777, 100 eps) that is **IDENTICAL** for both arms — so the learning
curves are directly comparable. **Budget unit = one *successful* demo** (empty / infeasible /
failed attempts are budget-free).

p4_top3's intended differentiator is **coverage**: its prescriptions scatter the tee across
the table, exploring far more of the object-pose space than diff_dagger's in-place
corrections ever visit. That is the figure the sibling `p4_select` study could not produce.

### p4_top3 vs p4_select — do NOT conflate them
There are **two** separate P4-LLM studies on PushT. Keep them distinct:
- **`p4_select`** (a *different*, already-finished study): SafeDAgger/diffusion-loss detection
  + the LLM **SELECTS which existing failure** to correct on-policy. It **WON**: demos-to-90%
  **12.0 ± 2.9 vs diff_dagger 15.8 ± 5.2**, demos-to-100% **31.0 vs 55.2**, both reach 1.0,
  winning 5/5 seeds. Aggregate: `results/aggregate/astar/astar_PushT-v1_summary.json`.
- **`p4_top3`** (THIS study, currently running): the LLM **PRESCRIBES a new pose** (generates
  scenarios). It is **underperforming** diff_dagger on SR (see §6). These are NOT the same
  result — never cite p4_select's win as if it were p4_top3's.

> ⚠️ Terminology trap: the live jobs are named `prr_p4t3_sN` and ARE p4_top3. A verification
> agent once mislabeled them "p4-select" from log text — the SLURM job names are authoritative.

---

## 2. CRITICAL architecture fact — the engine is the FORK

The live p4_top3 engine is the **Diff-DAgger fork** `main_pipeline`
(`LLMGuidedDAggerPipeline.run_budget_cycle` in
`/weka/s226137394/diff-dagger/diffdagger/main_pipeline/pipeline.py`). The suite
`pool_rl_robo/p4/{vlm,analysis,prescription,config_generator}.py` modules are **dead
reference code** (never imported by the running path). **All pipeline/engine changes go in
the fork** — the user owns it and the supervisor reviewed it. Suite-side files that DO run:
`p4/pipeline.py` (thin wrapper that configures + calls the fork), `p4/kag.py` (KAG injection),
`orchestrator/_common.py` (dispatch + bootstrap reuse).

**Never edit the fork bootstrap** (`_bootstrap_shared_init`) and **never** put bootstrap-reuse
logic inside the fork — reuse is **suite-side** (a subagent once added a `P4_REUSE` guard to
the fork's `comparison_harness.py`; it was reverted).

---

## 3. Per-round target flow (sequential mode → 1 config / 1 demo)

```
VLM (3 frames: start / highest-loss / end, for each of the top-3 highest-loss FAILED episodes)
  → reasoning ANALYSIS pass        (KAG injected)
  → reasoning PRESCRIPTION pass    (KAG injected)
  → plain LLM emits ONE compressed scene config
  → INFEASIBILITY CHECK: load the config, run the PPO expert.
        • expert solves it  → that solve IS the demo (feasibility test + demo in one)
        • expert can't solve → bank the pose into infeasible_history, re-prescribe
                               (analysis + prescription + "INFEASIBLE-CONFIG, redo" chain),
                               capped by infeasible_attempts (5)
  → EMPTY prescription → re-prescribe with a "MANDATORY-PRESCRIPTION" addendum,
                         capped by represcribe_attempts (5); NEVER skip the round.
KAG (per-env object spawn-box + arm/TCP bounds) grounds analysis + prescription + aggregator.
```

"top-3" = the **3 worst FAILED episodes** per round (`max_failures_per_round=3`). "frames=1"
= **one** highest-loss frame **per episode** (`top_k_high_loss=1`) → start + peak + end. These
are different knobs.

---

## 4. Implemented changes — VERIFIED against the live tree (2026-06-11)

All present and current. Anchors are post-verification (the old doc's anchors were stale).

### Suite side

1. **Exact-bootstrap reuse** — `orchestrator/_common.py:373-389` (inside `run_suite`; the old
   doc said "~line 338" — that line now holds the unrelated p4_select parallel-split branch).
   If env var `P4_REUSE_INIT_CKPT` points at an existing `init_ckpt.pth`, load it verbatim,
   read `init_sr` from sibling `init_meta.json`, and **SKIP** `_bootstrap_shared_init`
   (`_common.py:381-388`, imported at `:312`). Rebuild is GPU-nondeterministic → wouldn't
   byte-match diff_dagger, so reuse is mandatory for equal-to-equal.
   - `run_pool_rl_robo.sh:75` exports/echoes `P4_REUSE_INIT_CKPT` when set.

2. **resolve_knobs p4 retry caps** — `orchestrator/_common.py:66-68`:
   `p4_represcribe_attempts` (default 5), `p4_infeasible_attempts` (default 5),
   `p4_prompts_dir` (default None) — all from `scfg['p4']`.

3. **KAG injection fix** — `p4/kag.py`, `kag_text_path` (`:85-91`): removed the PushT
   special-case that returned the fork's generic `kag_document.txt`. EVERY env (incl.
   PushT-v1) now renders its own `p4/kag/<env>.json` → cached `<env>.kag.txt`
   (`load_kag_text → format_kag_context(load_kag_graph(env))`). This is why the supervisor
   saw "KAG not available" — the per-env bounds weren't reaching the LLM.
   - ⚠️ **Stale code to ignore/clean:** the module-level docstring `p4/kag.py:9-12` still
     describes the OLD "PushT reuses the fork's kag_document.txt" behavior, and `_FORK_PUSHT_KAG`
     (`:23`) is now **unused**. Harmless, but misleading — fix when convenient.

4. **Frames=1** — `p4/pipeline.py:163`: `pcfg.analyzer.frames.top_k_high_loss = 1`.

5. **Retry caps threaded into the fork cfg** — `p4/pipeline.py:166-167`:
   `pcfg.budget.represcribe_attempts` / `.infeasible_attempts` from `k`.

6. **Prompt-override plumbing (suite half, dormant)** — `p4/pipeline.py`:
   `_PROMPT_KEY_TO_ATTR` (`:66-75`, 8 keys → fork `*_override` attrs);
   `_apply_prompt_overrides(pcfg, suite_env_id, prompts_dir)` (`:78`) **no-ops when
   prompts_dir is falsy** (`:82-83`), else loads `<dir>/<env>.yaml` (fallback `default.yaml`,
   `:88-90`); invoked at `:169` via `k.get("p4_prompts_dir")`. Run uses `prompts_dir: null` →
   pure no-op → fork-default prompts → zero-diff. (Fork half = Phase B, §8.)

### Fork side (`/weka/s226137394/diff-dagger/diffdagger/`)

7. **BudgetConfig caps** — `main_pipeline/config.py:189-190`:
   `represcribe_attempts: int = 5`, `infeasible_attempts: int = 5` (after
   `max_consecutive_empty`).

8. **Empty-retry, not skip** — `main_pipeline/pipeline.py` budget loop:
   `_prescribe_once()` closure (`:1154-1158`, also banks `_last_dropped` into
   `infeasible_history`); empty-retry `while` (`:1172`, cap from `b.represcribe_attempts` at
   `:1166`) with the `_MUST` "MANDATORY-PRESCRIPTION" addendum (`:1167-1170`); restores
   `reasoning_add`/`aggregator_add` after (`:1181-1182`).

9. **Infeasibility loop** — `main_pipeline/pipeline.py`: `_collect_prescribed_demos`
   (`:1232`) wrapped in a `while` (`:1235`, cap from `b.infeasible_attempts` at `:1227`); on
   zero successful demos, bank the poses into `infeasible_history` (init `:1075`; append
   `:1237-1242`), build `infeasible_feedback_block` (imported `:940-942`) + the `_REDO`
   "INFEASIBLE-CONFIG" addendum (`:1228-1231`), re-prescribe (`:1250`), re-cap to `eff_cap`
   (`:1255`; `eff_cap` set `:1214`).
   - **Budget parity:** `budget_used += n_ok` (`:1268`), `n_ok = #successful demos` (`:1261`)
     → empty/infeasible/failed attempts are budget-free → "budget = one successful demo"
     holds identically to diff_dagger.
   - AST-parse of `pipeline.py` + `config.py` = **PARSE_OK** (verified with the diffdagger
     interpreter).

10. **High-loss-frame bug fix (CRITICAL)** — `util/high_loss_image_saver.py`. The original
    saver only appended frames with `loss > threshold`; with a reused policy the threshold was
    stale-scaled (peak losses ~0.003 ≪ 0.73) and the percentile fallback was broken
    (`batch["obs"]` KeyError) → **the VLM received NO high-loss frames**. Fix: always track a
    per-episode peak frame and fall back to it.
    - `__init__`: `self._peak_frame: Optional[dict] = None` (`:158`); reset in `start_episode`
      (`:183`).
    - `log_step`: build `entry` (`:207-214`); update `_peak_frame` when
      `frame is not None and (peak is None or loss > peak.loss)` (`:219-221`) — BEFORE the
      `if loss > self._threshold` append.
    - `end_episode`: right after the `if self._episode_id is None: return` guard, if no
      high-loss frames but a peak exists, `self._high_loss_frames = [self._peak_frame]`
      (`:244-252`).
    - Validated on real data: the 1-seed run then saved ~60 high-loss frames and the VLM
      received e.g. `t0072_loss0.0047.png`.

### KAG documents
- **PushT-v1** (`p4/kag/PushT-v1.json`, ~6151 chars): keys `meta/nodes/edges/reasoning`;
  contains object spawn-box `ws_tee` (`:17`) + arm/TCP `ws_tcp` (`:18`) bounds. Renders+injects.
  Tee bounds: `x[-0.20,0.20] y[-0.25,0.05] z=0.021`.
- (Other envs' KAGs exist — PickCube/PlugCharger/StackCube — but are out of scope here.)

---

## 5. CURRENT LIVE STATUS — 5 seeds running (2026-06-11)

All 5 RUNNING; each reuses `results/PushT-v1/run_<seed>/shared_baselines/init_ckpt.pth` via
`P4_REUSE_INIT_CKPT` and writes to `run_<seed>/p4_top3/`. Started 2026-06-10; ~26 min/demo
(LLM chain + infeasibility retries + from-scratch retrain + eval) → ~40 h/seed for budget 100
→ all should finish ~2026-06-12.

| Seed | Job | Elapsed | Demos | Last held-out SR | Max so far | q90 | Retry/infeasible events |
|---|---|---|---|---|---|---|---|
| 1 | 99591 | 1d10h | 71 | 0.81 | 0.88 | not yet | INFEASIBLE×10 |
| 2 | 99593 | 1d10h | 73 | 0.82 | 0.86 | not yet | INFEASIBLE×10, re-prescribe×2 |
| 3 | 99594 | 1d03h | 60 | 0.81 | **0.91** | 51 demos | INFEASIBLE×12 |
| 4 | 99595 | 1d00h | 54 | 0.75 | 0.85 | not yet | INFEASIBLE×8 |
| 5 | 99596 | 16h | 2 | 0.56 | 0.56 | not yet | (just started) |

**The new machinery is firing in production:** the infeasibility loop fired 8–12×/seed
(LLM prescribed configs the PPO expert couldn't solve → correctly rejected + re-prescribed),
and the empty-retry fired on seed 2 (×2). No `MANDATORY`/empty-retry path was hit elsewhere,
and no `vk::DeviceLost`/proxy/import errors.

**Live progress is read from logs**, not the suite `learning_curve.json` (which is only
written at job completion). Parse `logs/PushT-v1_run<seed>_<job>.log` for
`[Budget] round N: heldout_sr=X calls=C` lines.

---

## 6. The honest finding (and the caveat that predicted it)

**On PushT, Diff-DAgger wins decisively on SR and sample-efficiency:**
- **Diff-DAgger (all 5 seeds finished):** final SR = **1.00 every seed**; q90 (demos-to-90%)
  = 19, 10, 22, 17, 11 → **mean ≈ 15.8**.
- **p4_top3 (live):** plateauing **~0.85–0.91**; only seed 3 momentarily touched 0.91 (at 51
  demos — ~3× slower than diff_dagger's 16) before regressing to 0.81. Seeds 1/2/4 have spent
  54–73 demos and still haven't reached 0.90. This matches the historical **~0.82–0.83**
  plateau (run_0: 0.82@62; run_900: 0.83@37).

**Why this is expected, not a bug:** prescribing *new* (often OOD) object configs is inherently
less sample-efficient for a *natural-distribution* held-out eval than diff_dagger correcting
on-policy exactly where the current policy fails. On an easy task like PushT, in-place
correction just wins. The fixes (KAG bounds → in-bounds prescriptions; infeasibility loop →
only solvable configs become demos; empty-retry → no wasted rounds) made p4_top3 *functional
and improving* (0.52 → 0.85–0.91) but did not close the SR gap.

**So the paper story for p4_top3 is COVERAGE, not SR** — its prescriptions explore far more of
the tee-pose space than diff_dagger, with an honestly-disclosed SR/efficiency trade-off. Build
the coverage heatmap (§9) as the headline figure for p4_top3, alongside the honest SR curve.

---

## 7. Equal-to-equal guardrails / fairness invariants (must hold)

These are load-bearing — two adversarial audits enforced the suite-wide versions; the subset
that applies to p4_top3 vs diff_dagger:

1. **Shared bootstrap, reused exactly.** Each seed's p4_top3 arm loads the SAME
   `run_<seed>/shared_baselines/init_ckpt.pth` that diff_dagger used (via
   `P4_REUSE_INIT_CKPT`). **Never rebuild** — rebuild is GPU-nondeterministic.
2. **`nd_retrain=1` for BOTH** (retrain from scratch every demo); **`target_sr=1.0`** (stop
   only at 100% or budget=100; 90% is read off the curve, not a stop point).
3. **Budget unit = one *successful* demo.** Empty/infeasible/failed attempts are budget-free
   (`budget_used += n_ok`) → retries don't buy free progress.
4. **Identical held-out eval** for both arms: `heldout_seed_base=7777`, `heldout_n=100`,
   frozen-policy protocol. Eval seeds are disjoint from rollout/screening seeds.
5. **Episode backstop `max_episodes_per_arm=5000`** is anti-infinite-loop only; it must never
   truncate a method before its 100-demo budget. A seed that hits it is a genuine plateau
   (`stopped_reason=max_episodes`, right-censored in aggregation).
6. **`prompts_dir: null`** for the comparison run → fork-default prompts → zero-diff vs the
   conditions diff_dagger's prescription chain would have used.
7. **Primary metric = demonstrations added (`n_queries`).** `total_expert_calls` is a
   secondary, method-specific cost (disclose, label non-comparable).

---

## 8. REMAINING — Phase B: per-env configurable prompts (optional, deferred)

The supervisor's "prompts in config" ask. The **suite half is already written** (`§4.6`); the
**fork half is NOT**. Do this only when nothing is mid-run at fork import, and keep the
comparison run at `prompts_dir: null` (so it stays zero-diff). To implement:
- **Fork:** add `*_override` fields to `AnalyzerConfig` (`main_analysis/config.py`, mirroring
  the existing `prompt_addendum_*`); in `stage2_prescriptive.py` (run_per_episode_reasoning /
  run_episode_prescription / run_cross_episode_reasoning) and `stage3_prescriptive.py`
  (run_prescriptive_stage3) use `sys = override or _DEFAULT`; thread overrides from
  `pipeline.py::_analyze_and_prescribe` (4 call sites) + the addendum assembly.
- **Suite:** author `p4/prompts/{default,PushT-v1}.yaml`. Ship `PushT-v1.yaml` as a
  **verbatim** copy of the fork's current system prompts + addenda so enabling the feature is
  zero-diff. Set `config_astar_p4top3.yaml: p4.prompts_dir: p4/prompts` to turn it on.

The 8 suite keys (`_PROMPT_KEY_TO_ATTR`, `p4/pipeline.py:66-75`):
`analysis_system, prescription_system, cross_system, aggregator_system, vlm_system,
reasoning_addendum, aggregator_addendum, sequential_addendum`.

---

## 9. Aggregation + next steps

When all 5 p4_top3 seeds finish:
1. **SR / queries curves** — `aggregation/aggregate_astar.py` currently compares
   `["p4_select","diff_dagger"]`; point/add `p4_top3` for this comparison. Produces mean±std
   curves, queries-to-threshold, honest censoring, and the expert-call axis. Headline =
   demos-to-90% and final-SR-at-budget (p4_top3 will NOT reach 1.0 → it runs to budget=100 or
   plateaus; the aggregator right-censors non-reachers, reporting `n_reached`/`censored_seeds`).
2. **Coverage figure (p4_top3's intended win)** — `aggregation/coverage_heatmap.py`
   (+ `render_env_topdown.py`, `reconstruct_tee_starts.py`): per-round CSV, 5-seed average,
   on-env gradient overlay (Gaussian-smoothed, NOT raw pixels — a prior user request),
   start-vs-trajectory. Works for any method dir; run it on `run_<seed>/p4_top3/`.
3. Results land in `results/PushT-v1/run_<seed>/p4_top3/results/learning_curve.json`
   (alongside `diff_dagger/`). Aggregate output → `results/aggregate/astar/`.

---

## 10. RUN RECIPE (equal-to-equal) — for re-launches

```bash
cd /weka/s226137394/DmNfull/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo
# SMOKE (1 round, throwaway RUN_ID=990):
ENV=PushT-v1 METHODS=p4_top3 SEED=1 RUN_ID=990 SMOKE=1 \
  P4_REUSE_INIT_CKPT="$(pwd)/results/PushT-v1/run_1/shared_baselines/init_ckpt.pth" \
  CONFIG="$(pwd)/config_astar_p4top3.yaml" \
  sbatch --array=1 --nodes=1 --gpus-per-node=3 --constraint=gpu-h100 --time=02:00:00 \
         --job-name=prr_p4t3_smoke run_pool_rl_robo.sh

# FULL 5 SEEDS (each reuses ITS diff_dagger bootstrap → byte-identical start):
for S in 1 2 3 4 5; do
  ENV=PushT-v1 METHODS=p4_top3 SEED=$S RUN_ID=$S \
    P4_REUSE_INIT_CKPT="$(pwd)/results/PushT-v1/run_${S}/shared_baselines/init_ckpt.pth" \
    CONFIG="$(pwd)/config_astar_p4top3.yaml" \
    sbatch --array=1 --nodes=1 --gpus-per-node=3 --constraint=gpu-h100 --time=10-00:00:00 \
           --job-name="prr_p4t3_s${S}" run_pool_rl_robo.sh
done   # ← always validate seed 1 ALONE first, then fire the rest
```
3 GPUs/node: VLM Qwen3-VL-32B + text Qwen3-32B + orchestrator. `NEED_LLM` auto-triggers on
`p4_top3` (`run_pool_rl_robo.sh:38-43`). `PushT-Start-v0` reposition env auto-builds.

### Smoke pass criteria (logs: `slurm_logs/pool_rl_<job>_1.out`, `logs/PushT-v1_run990_<job>.log`)
`[bootstrap] REUSING shared init … skipping rebuild` (NOT a rebuild) · `KAG loaded (N chars)`
(the suite PushT JSON render) · VLM 3 frames/episode · analysis + prescription + aggregator
fired · `capped to 1` · on empty → `re-prescribe attempt k/5` · on infeasible →
`INFEASIBLE … re-prescribing` · ≤1 demo collected · **high-loss frames saved** (the bug that
the smoke's tiny data once masked — confirm real frames reach the VLM) · no `vk::DeviceLost` /
proxy 5xx / import errors.

---

## 11. config_astar_p4top3.yaml — the single source of truth (verified)

`methods: ["p4_top3"]` (`:32`) · `environments: ["PushT-v1"]` (`:34`) · `initial_demos: 20`
(`:15`) · `budget: 100` (`:16`) · `target_sr: 1.0` (`:18`) · `max_rounds: 100` (`:19`) ·
`heldout_n: 100` (`:21`) · `nd_retrain: 1` (`:24`) · `eval_num_envs: 10` (`:28`, render-safe;
100 % 10 == 0) · `rollout_episodes: 60` (`:29`) · `max_episodes_per_arm: 5000` (`:30`).
`p4:` block (`:36-50`): `high_loss_percentile: 95` · `phase_b_workers: 8` ·
`batch_multiplier: 8` (p4's per-step diffusion-loss N_b) · `represcribe_attempts: 5` ·
`infeasible_attempts: 5` · `prompts_dir: null` (`:50`).

> ⚠️ The launcher header comment in `run_pool_rl_robo.sh:19-20` describes an OLD protocol
> (initial=50, target_sr=0.90). It is a stale comment; the active values come from this config.

---

## 12. Diff-DAgger paper faithfulness (PushT, arXiv 2410.14868 — verified)

Our diff_dagger is the paper's own algorithm with matching hyperparameters; it is NOT
undertrained. Pushing(1-Expert) Table IV: N_i=20, N_f=100, N_d=4, α=0.99, K=1, N_b=512. Ours:
`initial_demos=20`, `budget=100`, α=0.99, patience K=1, native fork query rule
(`get_action(dagger=True)`, CDF(loss) > α-quantile). We are **STATE-based** → compare ONLY to
the paper's STATE column: Diff-DAgger Pushing(1-Expert) **State = 0.96 @ 100 demos** (image
0.87/0.94 is a different setting). Our diff_dagger reaches **1.0 at ~46–75 demos** —
consistent with / better than the paper, so any comparison is against a faithful, well-trained
Diff-DAgger. We differ only in `nd_retrain=1` (vs paper N_d=4 — we retrain MORE often, benign),
applied to BOTH arms. The PushT diff_dagger seeds (98949–98953) used `batch_multiplier=8`
(N_b=128, self-consistent but ~4× noisier than the paper's N_b=512).

**Caveats to disclose:** (a) `total_expert_calls` is method-specific (the p4 arm's
expert-solve-as-feasibility-test vs diff_dagger's implicit loss-query) — secondary,
non-comparable axis; comparable budget is demonstrations added. (b) Any
`stopped_reason=max_episodes` seed is a plateau backstop (right-censored), not a clean budget
exhaustion. (c) PushT uses the fork's PPO expert + PushT-v2/PushT-Start-v0 remap.

---

## 13. Key files (current anchors)

**Fork (engine, edited):** `diffdagger/main_pipeline/pipeline.py` (budget loop: empty-retry
`:1154-1182`, infeasibility loop `:1227-1268`), `…/main_pipeline/config.py:189-190`
(BudgetConfig caps). Reuses `sim_bridge` (collect_prescribed_demo / `evaluate_heldout` /
`train_policy`), `infeasible_feedback_block`, `kag_loader`. Phase B will touch
`main_analysis/config.py`, `…/stage2_prescriptive.py`, `…/stage3_prescriptive.py`.

**Suite (`pool_rl_robo/`):**
- `orchestrator/_common.py` — bootstrap reuse `:373-389`, resolve_knobs caps `:66-68`,
  `ExpertCallCounter` `:147` (instantiated `:360`).
- `p4/pipeline.py` — `run_p4_arm` (frames `:163`, caps `:166-167`, prompt-override no-op
  `:78/:169`); `_PROMPT_KEY_TO_ATTR` `:66-75`.
- `p4/kag.py` — `kag_text_path` `:85-91` (per-env render; stale docstring `:9-12`).
- `p4/kag/PushT-v1.json` — the injected KAG (bounds at `:17-18`).
- `config_astar_p4top3.yaml` — the run config (§11).
- `run_pool_rl_robo.sh` — SLURM driver (P4_REUSE export `:75`, METHODS decode `:37`,
  NEED_LLM gate `:38-43`).
- `util/high_loss_image_saver.py` (in the fork) — the peak-frame fix `:158/:183/:207-221/:244-252`.
- `aggregation/{aggregate_astar,coverage_heatmap,render_env_topdown,reconstruct_tee_starts}.py`.

**Design doc:** `/weka/s226137394/DmNfull/P4_ARCHITECTURE.md` (the maze-project P4 spec — the
conceptual source of truth for the VLM→reason→prescribe→KAG chain).

---

## 14. Cluster / environment essentials

- **Interpreter:** bare `python`/`pip` hit the `maze` env (bashrc PATH gotcha). Always use
  `/home/s226137394/.conda/envs/diffdagger/bin/python`; from repo root
  `/weka/s226137394/DmNfull` set `PYTHONPATH=/weka/s226137394/DmNfull`. Never bare-`pip` into a
  cloned env. (vLLM servers run under `vllm_embed`; the proxy under `maze`.)
- **3 GPUs/node:** GPU0 Qwen3-VL-32B (vision), GPU1 Qwen3-32B (text), GPU2 orchestrator
  (ManiSkill GPU sim + diffusion). p4_top3 needs the LLM → 3 GPUs.
- **SLURM:** partition `gpu-large`, `qos=batch-long` (10-day max for sbatch). ALWAYS
  `--nodes=1 --gpus-per-node=N` (never `--gpus=N` — it can split across nodes).
  `--constraint=gpu-h100` (h200 nodes hit SAPIEN Vulkan device-lost on render).
- **Pipeline uses the Responses API** (`client.responses.create`, POST `/v1/responses`), not
  chat completions — any OpenAI-compatible proxy must route `/v1/responses`.
- `envs/env_setup.bootstrap_fork_path()` puts the fork on `sys.path` and binds fork subpkgs in
  `sys.modules` to dodge `DmNfull/model` shadowing. PushT-v1 → fork `PushT-v2` +
  `PushT-Start-v0` reposition env + PPO expert.

---

## 15. History (PushT) — so you don't repeat killed runs

- **Reference diff_dagger:** jobs **98949–98953** = PushT seeds 1–5, all COMPLETED; all reach
  1.0; q90 = 19/10/22/17/11 (mean 15.8). These are the bootstraps p4_top3 reuses.
- **p4_select** (the OTHER study): beat diff_dagger 1.8× on demos-to-100% (31.0 vs 55.2), won
  5/5; aggregate `results/aggregate/astar/astar_PushT-v1_summary.json`. **Not** p4_top3.
- **p4_top3 live:** jobs 99591/99593/99594/99595/99596 (seeds 1–5), the CURRENT run (§5).
  Smoke 99400 passed but masked the high-loss-frame bug; 1-seed 99587 exposed it on real data;
  fixed in `high_loss_image_saver.py`, re-launched as 99591 and validated (60 frames saved),
  then fired the full set.
- **Killed (do NOT use their data):** 98921–98925 (episode-cap + aggregation confounds),
  98930–98934 (reset-state-demo confound). `run_0/900/901` = older pre-audit single-seed
  artifacts (run_901's p4 may have run LLM-OFF — don't trust it).

## 16. NEVER DO
- Never edit the fork bootstrap (`_bootstrap_shared_init`) or put bootstrap-reuse in the fork —
  reuse is suite-side (`_common.py`).
- Never rebuild the bootstrap for a p4_top3 seed — always reuse via `P4_REUSE_INIT_CKPT`.
- Never bare-`pip` into a cloned conda env (corrupts `maze`).
- Never `--gpus=N` (splits across nodes); use `--nodes=1 --gpus-per-node=N`.
- Never set `prompts_dir` to a real path for the comparison run (breaks zero-diff parity).
- Never conflate p4_top3 (this study, lagging on SR) with p4_select (separate study, won).
