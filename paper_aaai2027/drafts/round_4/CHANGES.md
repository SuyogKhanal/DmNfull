# Round 4 changes (vs drafts/round_3)

All edits were targeted string replacements in draft/paper.tex. Recompiled: 9 pages
total including references, 0 overfull boxes, no undefined references/citations.

## Fixes by issue id

- **R1-r4-1** (flagging attributed to the loss in intro/conclusion): intro now reads
  "Rollouts of the current policy are collected, and the learner's own per-step
  training loss locates each failed episode's peak-loss step"; conclusion now reads
  "locate each failure's peak-loss step with the learner's own loss, root-cause the
  failures with KAG-grounded foundation models". Episode selection is task-outcome
  driven everywhere, matching the method's F_r definition.
- **R1-r4-2** ("two arms, both always in play" vs Wipe/GridWorld): scoped to
  "both in play wherever the randomized quantity is an object pose (Wipe and
  GridWorld, below, are the exceptions)"; hybrid framing kept.
- **R1-r4-3 / R2-r2-1 (contradiction part)**: the "concentrated on the failure-prone
  episodes" clause is deleted. New metric text: the before evaluation is the
  pre-retrain rollout set whose failures the prescription targeted; the after
  evaluation is the following block's rollouts; per-round values run above held-out
  increments because the before set is where the targeted failures were just
  observed; fresh consecutive draws mean the sum is not the held-out rise.
- **R2-r2-1** (carried major, remainder): PARTIAL. Pipeline code inspected this
  round shows the per-round logged quantity is the WHOLE-POOL success rate
  (correction_sr = n_success/len(pool) in pool_x_selector selection and P4 loops),
  so the reviewer's failure-conditioned reading cannot be written in without
  contradicting the inspectable logging; the scatter's per-point construction lives
  in the workbook-producing analysis stage outside the paper sources. Definition
  kept at loop-log strength; granularity residue folded into the figure/archive pass.
- **R2-r2-4** (carried major, Table 1 vs 100-episode protocol): FIXED by scoping.
  Protocol: "Held-out monitoring (the learning curves) uses a fixed set ... 100
  episodes per robot task (frozen seeds), evaluated at every retraining checkpoint.
  Table 1 instead pools the evaluation pipeline's records at the budget, whose
  per-cell counts vary (caption)." Caption adds "not per-seed means and not one
  final fixed-set evaluation per seed". The 100-episode monitoring claim is
  config-verified (pool_rl_robo/config.yaml heldout_n: 100, seed base 7777) and no
  longer purports to generate Table 1.
- **R2-r4-1** (ambiguous robot before-evaluation): FIXED with the suggested clause:
  before = the rollout episodes since the previous retraining (all under the one
  unchanged policy); after = the following block's rollouts; block members share the
  checkpoint's Delta-SR.
- **R2-r4-2** (zero-failure re-roll vs budget): FIXED/DECLINED split. Added
  "re-rolls are bounded by the run's episode cap (on GridWorld, a cap on consecutive
  failure-free pools)"; protocol restated as "collects expert demonstrations under
  the same hard budget of 20, one per round" (the accounting the pipeline enforces).
  Per-cell consumption tallies are not in the results source and were not invented.
- **R2-r2-7** (prompts/fallback/cost): DECLINED again, evidence unchanged;
  camera-ready obligation stands.
- **R2-r3-2, R3-r1-6, R3-r2-4** (figure regeneration items): DEFERRED again to the
  dedicated figure pass (plot sources absent from paper sources).
- **R3-r4-1** (which/how sentence x4): intro instance reworded to "These two
  decisions, the choice of failure mode and the placement of each demonstration,
  set what a fixed budget buys." Abstract/contribution/conclusion instances kept.

## Page-budget offsets (to hold 9 pages, 0 overfull)

- Removed one single-use background citation: DART (laskey2017dart) and its
  Related Work clause (no other claim relied on it).
- Tightened, without content loss: the Q3 blind-confidence sentence ("after
  collection, retraining, and new rollouts"), the Figure 5 pointer sentence
  (dropped a clause duplicated by the caption), one Limitations parenthetical
  (the reasoning-pipeline stage list), and the Limitations "bound the damage
  without eliminating it" phrasing.
