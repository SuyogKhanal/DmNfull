# 09 — Reproducibility, results structure, aggregation, statistics

## Seeds + byte-identical bootstrap (the fairness backbone)
- **5 seeds** per cell (fix the list, e.g. `[1,2,3,4,5]`). The seed controls training RNG only.
- **Byte-identical shared bootstrap** per `(task, modality)`: collect the initial demos ONCE
  (`seed_base=0`, deterministic), pickle to `results/shared_bootstrap/<task>_<modality>_ni<N>.pkl`,
  and have **every arm and every seed of that cell load the exact same file**. This removes the
  small collection nondeterminism (global-RNG robot-init noise) that otherwise differs between
  arms. (The `diff-dagger-ur5` repo already implements `--bootstrap-dir` + a `make-bootstrap`
  mode — port it.)
- Diffusion training is non-bitwise-deterministic (cudnn.benchmark); the shared bootstrap makes
  the *data* identical, which is the fairness requirement. Seed weight-init/noise for repeatability.

## Two correctness fixes to CARRY OVER (they bit us last time)
1. **Stop rule = budget of *successful* demos**, NOT `sr >= target`. An `sr>=target` early-stop
   ends the loop at round 0 (round-0 SR ≈ the calibrated ~50%). Stop only at `n_successful_demos
   >= budget`. Identical rule for DISTIL and every baseline/ablation.
2. **SR-vs-demos curve alignment:** record the eval against the **dataset size the eval was
   computed on** (pre-collect), not the post-collect count. Otherwise arms that add different
   #demos/round get shifted differently on the x-axis and the comparison is biased.

## Results tree (deterministic, one job → one leaf)
```
results/<task>/<modality>/<ablation>/seed<s>/
   result.json        # the curve + metadata (schema below)
   run.log
   telemetry/round_*.jsonl   # per-round: clusters, descriptors, decision, confidence, infeasible
   prompts/           # the EXACT prompts used (golden rule 3)
   kag/               # the EXACT KAG doc used
   config.yaml        # the full resolved config (every flag)
```
`result.json.history` per round: `n_demos_at_eval, eval_success, mode(SELECT/BRIDGE),
confidence, n_screen_failures, k_star, cluster_label_target, n_infeasible_attempts, sec, tokens`.
Plus `final_success, n_demos, budget, seed, ablation`.

**Never** commit `telemetry/`, checkpoints (`*.pt`), or frames to git (bloat — a past run hit
548 GB). Gitignore them; keep `result.json`/`run.log`/`config.yaml` light.

## Aggregation (`aggregate.py`) → the `DISTIL_ablation_preview.xlsx` shape
Parse every `result.json` into a master table keyed by `(task, modality, ablation, seed)` with:
SR-vs-demos curve, **demos-to-target-SR**, final SR (mean ± std over seeds), and per-ablation
deltas vs full-DISTIL. Emit the xlsx (use `DISTIL_ablation_preview.xlsx` as the column template)
+ a markdown summary. Aggregation must be re-runnable and idempotent, reading `RUN_STATE.md`
(the job ledger) to know which cells exist.

## Tier-4 diagnostics — auto-compute from the logs (cheap, highest ROI; put in appendix)
1. **Cluster-label agreement** — per-cluster purity of LLM root-cause labels vs the geometric
   partition, **for ALL tasks** (not just the pretty Push-T panel). Low purity anywhere breaks
   the "semantically meaningful modes" caption.
2. **k\* distribution** across rounds × tasks. If silhouette almost always picks 3, the adaptive
   k-selection (Eq 8) is theater — concede or show the spread.
3. **Targeted-vs-bridge usage split, per task.** If bridge <10%, it isn't a real contribution.
4. **Failure-count-per-round distribution.** Clustering is skipped when N≤3; if late rounds are
   mostly N≤3, clustering+memory only act early — know that number.
5. **Fallback rate per task** and **feasibility-rejection rate** (from the infeasibility loop).
6. **Compute/wall-clock table:** sec/round and **tokens/round** for DISTIL vs baselines (the LLM
   client must log tokens — OpenRouter returns usage).
7. **Confidence diagnostic:** the per-round LLM confidence vs realized ΔSR (calibration).

## Statistics (Tier 5 — the biggest rebuttal risk)
- **Sign test** over the headline cells: an N/N clean ranking sweep under a coin-flip null gives
  `p=(1/2)^N` (10/10 → ≈0.001). Compute it; put it in the text. Stronger: **Wilcoxon signed-rank**
  over the paired per-cell means.
- **Add-seeds readiness:** be able to extend 5→10 seeds on **Push-T + Wipe** (the most likely
  single rebuttal demand). Design the runner so adding seeds is one arg.
- **Q3 pseudoreplication:** compute the honest **per-checkpoint** correlation (~25 points =
  5 seeds × 5 retrainings), or a mixed-effects model — don't report only the inflated pooled r.
- **Held-out ΔSR:** report ΔSR on the frozen held-out set, not only on the targeted-before set.

## One-command reproduce (the acceptance test)
Any cell must reproduce from: `python -m distil.run --task <t> --modality <m> --ablation <a>
--seed <s> --budget 20 --bootstrap-dir results/shared_bootstrap` — reading OpenRouter creds from
`.env`, needing 1 GPU, writing the leaf above. If that isn't true, it isn't done.
