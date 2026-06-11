# claude_p4_top3_vs_diff-dagger_context.md — single-file context for the P4-top3 vs Diff-DAgger experiment

Read this top-to-bottom to resume the **P4-top3 (LLM-prescription) vs Diff-DAgger**
equal-to-equal comparison on PushT-v1. For the broader suite/harness (envs, conda, SLURM,
the fairness invariants of the *p4_select* study) see `claude_context.md` in this folder.

---

## ▶ STATE (fill in / update each session)

```
RUN TARGET:   PushT-v1, p4_top3 vs the EXISTING diff_dagger seeds 1-5 (run_1..run_5)
STAGE:        smoke submitted (job 99400, RUN_ID=990) → then 1 seed → then 5 seeds
LAST:         all behavioral fork+suite changes implemented & static-verified; smoke queued
NEXT:         (a) confirm smoke; (b) Phase B prompt-config plumbing; (c) 1 seed; (d) 5 seeds
```

## What this experiment is
Compare two **P4-LLM** variants of demonstration acquisition against the SAME Diff-DAgger
runs (jobs 98949–98953 = seeds 1–5, in `results/PushT-v1/run_<seed>/diff_dagger/`):
- **Diff-DAgger** (baseline): native diffusion-loss CDF query; corrects failures *in place*.
- **p4_top3** (this study): VLM → reasoning **analysis** → reasoning **prescription** →
  plain LLM → **prescribe a NEW object pose**, the expert solves it → ONE demo. The LLM
  *generates* new scenarios (vs `p4_select`, which only *picks* which existing failure to
  correct — see `claude_context.md`).

Headline metric = demonstrations to reach a held-out success threshold (frozen
`evaluate_heldout`, seed 7777, 100 eps — IDENTICAL to diff_dagger, so the curves are
comparable). p4_top3 ALSO wins coverage (it scatters the tee across the table via
prescription) — the thing `p4_select` could not show.

## CRITICAL architecture fact
The live p4_top3 engine is the **Diff-DAgger fork** `main_pipeline`
(`LLMGuidedDAggerPipeline.run_budget_cycle` in `/weka/s226137394/diff-dagger/diffdagger/
main_pipeline/pipeline.py`). The suite `pool_rl_robo/p4/{vlm,analysis,prescription,
config_generator}.py` are **dead reference code** (never imported). **All pipeline changes
go in the fork** (the user owns it; supervisor reviewed it). Suite-side files only:
`p4/pipeline.py` (wraps the fork), `p4/kag.py`, `orchestrator/_common.py`.

## Target flow (per round, sequential mode → 1 config / 1 demo)
VLM (3 frames: start / highest-loss / end of each of the **top-3** highest-loss failures)
→ reasoning **analysis** pass (KAG) → reasoning **prescription** pass (KAG) → plain LLM
emits ONE compressed config → **infeasibility check**: load config, run PPO expert; the
expert-solve IS both the feasibility test and the demo. If the expert can't solve it →
re-prescribe (analysis+prescription+"infeasible, redo" chain) up to a cap. Empty
prescription → re-prescribe (never skip the round) up to a cap. KAG (per-env bounds)
grounds analysis + prescription + aggregator.

## Changes implemented (this session) — all static-verified
1. **Exact-bootstrap reuse** (`orchestrator/_common.py`, ~line 338): if env var
   `P4_REUSE_INIT_CKPT` points at an existing `init_ckpt.pth`, load it + read `init_sr`
   from sibling `init_meta.json`, SKIP the rebuild (rebuild is GPU-nondeterministic →
   wouldn't match diff_dagger). `run_pool_rl_robo.sh` echoes/exports it. **Do NOT edit the
   fork bootstrap.**
2. **KAG injection fix** (`p4/kag.py::kag_text_path`): removed the PushT special-case that
   returned the fork's generic `kag_document.txt`; every env now renders its own
   `p4/kag/<env>.json` (with the object spawn box + arm/TCP bounds). This is why the
   supervisor saw "KAG not available" — the bounds weren't reaching the LLM.
3. **Frames=1** (`p4/pipeline.py`): `pcfg.analyzer.frames.top_k_high_loss = 1` → exactly
   start + single highest-loss + end per episode. ("top-3" = *episodes*/round, a different
   knob, unchanged at `max_failures_per_round=3`.)
4. **Empty-retry, not skip** (fork `pipeline.py` budget loop, ~1154): wrap
   `_analyze_and_prescribe` in a retry loop with a "you MUST prescribe" addendum, cap
   `BudgetConfig.represcribe_attempts` (default 5), then fall through to the 0-demo path.
5. **Infeasibility loop** (fork `pipeline.py`, ~1197): wrap `_collect_prescribed_demos`;
   if `n_ok==0` (expert can't solve), bank the pose into `infeasible_history`, re-prescribe
   with `infeasible_feedback_block` + "redo" addendum, cap `infeasible_attempts` (default 5).
   Failed/infeasible/empty attempts do NOT consume budget (`budget_used += n_ok`) → retries
   are budget-free → preserves "budget = one successful demo" parity with diff_dagger.
6. **Caps** added to `BudgetConfig` (`config.py`) + surfaced via `config_astar_p4top3.yaml`
   `p4.{represcribe_attempts,infeasible_attempts}` + `resolve_knobs`.
7. **All 4 KAGs** with spawn+arm bounds: PushT-v1 (already had them), authored
   PickCube-v1.json + PlugCharger-v1.json, added bounds to StackCube-v1.json. All
   parse+render. Bounds: PushT tee x[-0.20,0.20] y[-0.25,0.05] z=0.021; PickCube cube
   xy[-0.1,0.1] z=0.02, goal xy[-0.1,0.1] z[0.02,0.32]; PlugCharger charger x[-0.10,-0.026]
   y[-0.20,0.20] z=0.012, receptacle x[0.01,0.10] y[-0.10,0.10] z=0.10; StackCube cubes
   x[-0.10,0.10] y[-0.20,0.20] z=0.02.

## REMAINING — Phase B (do AFTER the smoke validates; don't edit the fork mid-run)
**Per-env configurable prompts** (the supervisor's "prompts in config" ask):
- Fork: add `*_override` fields to `AnalyzerConfig` (`main_analysis/config.py`, mirroring the
  existing `prompt_addendum_*`); in `stage2_prescriptive.py` (run_per_episode_reasoning /
  run_episode_prescription / run_cross_episode_reasoning) and `stage3_prescriptive.py`
  (run_prescriptive_stage3) use `sys = override or _DEFAULT`; thread overrides from
  `pipeline.py::_analyze_and_prescribe` (4 call sites) + the addendum assembly (~1138-1152).
- Suite: `p4/pipeline.py::_apply_prompt_overrides` is ALREADY written (loads
  `p4/prompts/<env>.yaml`, fallback `default.yaml`, sets the `*_override` attrs). Author the
  YAMLs: `p4/prompts/{default,PushT-v1,StackCube-v1,PickCube-v1,PlugCharger-v1}.yaml`. Ship
  `PushT-v1.yaml` = **verbatim** copies of the fork's current system prompts + addenda so it
  is zero-diff for the comparison run.
- The RUN uses `config_astar_p4top3.yaml: p4.prompts_dir: null` (→ fork defaults, guaranteed
  zero-diff / equal-to-equal). Set `prompts_dir: p4/prompts` to enable the per-env feature.

## RUN RECIPE (equal-to-equal)
Launcher pattern (each seed reuses ITS diff_dagger bootstrap → byte-identical start):
```bash
cd .../pool_rl_robo
# SMOKE (1 round, seed 1, throwaway RUN_ID=990):
ENV=PushT-v1 METHODS=p4_top3 SEED=1 RUN_ID=990 SMOKE=1 \
  P4_REUSE_INIT_CKPT="$(pwd)/results/PushT-v1/run_1/shared_baselines/init_ckpt.pth" \
  CONFIG="$(pwd)/config_astar_p4top3.yaml" \
  sbatch --array=1 --gpus-per-node=3 --constraint=gpu-h100 --time=02:00:00 \
         --job-name=prr_p4t3_smoke run_pool_rl_robo.sh
# ONE seed (real), then loop 1..5 for the full set:
for S in 1 2 3 4 5; do
  ENV=PushT-v1 METHODS=p4_top3 SEED=$S RUN_ID=$S \
    P4_REUSE_INIT_CKPT="$(pwd)/results/PushT-v1/run_${S}/shared_baselines/init_ckpt.pth" \
    CONFIG="$(pwd)/config_astar_p4top3.yaml" \
    sbatch --array=1 --gpus-per-node=3 --constraint=gpu-h100 --time=10-00:00:00 \
           --job-name="prr_p4t3_s${S}" run_pool_rl_robo.sh
done   # ← run seed 1 ALONE first, inspect, THEN the rest
```
Results land in `results/PushT-v1/run_<seed>/p4_top3/results/learning_curve.json` (alongside
the existing `diff_dagger/`). 3 GPUs (VLM + text-LLM + orchestrator); NEED_LLM auto-triggers
on `p4_top3`; PushT-Start-v0 reposition env auto-built.

## SMOKE PASS CRITERIA (logs: slurm_logs/pool_rl_<job>_1.out, logs/PushT-v1_run990_<job>.log)
`[bootstrap] REUSING shared init … skipping rebuild` (NOT a rebuild) · `KAG loaded (N chars)`
(= the suite PushT JSON render) · VLM 3 frames/episode · analysis + prescription + aggregator
fired · `capped to 1` · on empty → `re-prescribe attempt k/5` (not a round advance) · on
infeasible → `INFEASIBLE config (attempt k/5) — re-prescribing` · ≤1 demo collected · no
`vk::DeviceLost` / proxy 5xx / import errors.

## Equal-to-equal guardrails (must hold)
- Reuse the EXACT per-seed bootstrap (P4_REUSE_INIT_CKPT), never rebuild.
- `heldout_seed_base=7777`, `heldout_n=100`, `nd_retrain=1`, retrain-from-scratch — same as
  diff_dagger. `target_sr=1.0`, `budget=100`, `initial_demos=20`. (all in config_astar_p4top3.yaml)
- Budget unit = one *successful* demo; empty/infeasible/failed attempts are budget-free.
- `prompts_dir: null` for the comparison run (fork-default prompts = zero-diff).

## Expectation / caveat
Past p4_top3 reached only ~**0.82–0.83 held-out** on PushT (run_0: 0.82@62 demos; run_900:
0.83@37) vs diff_dagger's **1.0** — because prescribing OOD configs doesn't help the natural
held-out distribution and wasted rounds on empty/infeasible prescriptions. The changes
(KAG bounds → in-bounds prescriptions; infeasibility loop → only solvable configs become
demos; empty-retry → no wasted rounds) AIM to close that gap. The **1-seed run reveals the
real number before committing 5 seeds.** If it still underperforms on SR, the win to report
is COVERAGE (prescription explores more of the table) + the honest SR trade-off.

## Aggregation / analysis
`aggregation/aggregate_astar.py` currently compares `["p4_select","diff_dagger"]`. For
p4_top3 vs diff_dagger, point it at `p4_top3` (add it to METHODS or run a p4_top3-vs-diff
variant) once results land. Coverage figures: `aggregation/coverage_heatmap.py`
(+ `render_env_topdown.py`) work for any method dir.

## Key files
- Fork (engine, edited): `diffdagger/main_pipeline/pipeline.py` (budget loop: empty-retry
  + infeasibility loop), `…/config.py` (BudgetConfig caps). Phase B will touch
  `main_analysis/config.py` (AnalyzerConfig override fields), `…/stage2_prescriptive.py`,
  `…/stage3_prescriptive.py`. **Reuses** `sim_bridge.collect_prescribed_demo`,
  `infeasible_feedback_block`, `kag_loader`.
- Suite: `orchestrator/_common.py` (bootstrap reuse + resolve_knobs caps), `p4/pipeline.py`
  (frames, caps, `_apply_prompt_overrides`), `p4/kag.py` (injection fix), `p4/kag/*.json`
  (4 KAGs), `config_astar_p4top3.yaml`, `run_pool_rl_robo.sh`.
- Design doc: `/weka/s226137394/DmNfull/P4_ARCHITECTURE.md` (original maze-project P4 spec —
  the conceptual source of truth for the VLM→reason→prescribe→KAG chain).

## History
- Diff-DAgger PushT (jobs 98949-953, seeds 1-5): all reached 1.0; p4_select beat it 1.78×
  on demos (see `claude_context.md`). p4_top3 is the second comparison.
- A subagent once edited the fork `comparison_harness.py` (a P4_REUSE guard) — REVERTED;
  bootstrap reuse now lives suite-side in `_common.py`.
