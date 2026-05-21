# P4 vs Baseline — figure talking points (n=9, run_02 excluded)

One-line thesis: **P4 reaches the same (or slightly better) final quality as
the top-K-by-loss baseline using markedly fewer demonstrations, and it is
useful at every intermediate budget while the baseline is useful only after
spending its entire budget.**

Honest framing rule: lead with sample-efficiency; the final-accuracy edge is
small (present it as "at least as good"); n=9 so call borderline results
"trends", not "proven".

---

## A. The fair-comparison set (shows we measured this correctly)

### 1. baseline_nb_vs_swatch_p4_n9.png  — learning curves
- **What it is:** mean heldout success rate vs extra demos; P4 (purple) vs
  baseline (orange), 9 runs, target 0.90.
- **Say this:** "P4 climbs steadily from the first demo. The baseline is flat
  at 0.47 until demo ~14 then jumps — because top-K spends its whole budget in
  one shot. The two only meet at the very end."
- **If pushed:** the flat orange segment is *not* the baseline 'learning
  nothing' — it is logged only at start and end, so it is honestly comparable
  to P4 only at the final budget. That is exactly why the next figures matter.

### 2. baseline_endpoints_vs_p4_n9.png  — baseline drawn honestly
- **What it is:** P4 as a real curve; baseline as its initial level (0.47) plus
  its single end measurement (0.87 ± 0.03).
- **Say this:** "Drawn without the misleading connecting line: the baseline
  gives us two points, P4 gives a full trajectory. Same destination, but P4
  shows its work the whole way."

### 3. final_sr_at_budget_n9.png  — the only fair single comparison
- **What it is:** final heldout SR at the full budget K=15; per-run dots +
  mean ± error. Baseline 0.867, P4 0.903.
- **Say this:** "At equal budget P4 is at least as accurate as the baseline —
  here slightly higher (0.90 vs 0.87). Accuracy is a wash-to-slight-win; the
  real story is how few demos P4 needs to get there."
- **If pushed:** error bars overlap visually; treat final accuracy as parity
  with a mild edge, not the headline.

---

## B. The core efficiency story (this is "why P4 is better")

### 4. demos_to_target_n9.png  — demos to hit SR ≥ 0.90
- **What it is:** extra demos each run needed to first reach target; reached vs
  never-reached split.
- **Say this:** "P4 reached the 0.90 target in 6 of 9 runs averaging 12.2
  demos. The baseline reached it in only 2 of 9, averaging 14.5. P4 hits the
  bar more often **and** sooner."

### 5. time_to_target_survival_n9.png  — when runs cross target
- **What it is:** fraction of runs at/above target vs demos.
- **Say this:** "P4 starts clearing the target at demo 10 and ~⅔ of runs are
  there by 15. The baseline produces nothing until demo 14. P4 dominates this
  curve everywhere — it is never behind."

### 6. demos_to_match_baseline_n9.png  — demos to match baseline's *final* quality
- **What it is:** P4 curve vs baseline's full-budget quality line (0.867);
  per-run crossings; P4 median.
- **Say this:** "8 of 9 P4 runs reach the baseline's *full-budget* quality
  with a median of 12 demos — about 20% fewer demonstrations for the same
  result the baseline needs its entire budget to achieve."

### 7. paired_demos_to_target_n9.png  — same-condition paired comparison
- **What it is:** runs paired by identical correction pool & init; lines slope
  baseline→P4. Mean 15.7 → 13.4; mean diff −2.2, 95% CI [−3.6, −0.9].
- **Say this:** "Controlling for everything (same pool, same start), P4 needs
  ~2.2 fewer demos per run to hit target, and the 95% CI excludes zero — this
  is a real paired effect, not noise."

### 8. anytime_aulc_n9.png  — anytime performance
- **What it is:** area under the mean curve; AULC bar Baseline 0.498 vs
  P4 0.720.
- **Say this:** "If you have to stop at *any* point before the full budget,
  P4's expected success is far higher (0.72 vs 0.50). P4 is the better choice
  under an uncertain or limited labeling budget."
- **If pushed (important):** frame as an *anytime / early-stopping* advantage,
  not raw superiority — the baseline is low here partly because single-shot
  top-K yields no usable model until the full spend.

### 9. research_marginal_gain_n9.png  — per-demo value
- **What it is:** ΔSR from each k-th P4 demo vs the baseline's amortised
  ΔSR/demo (0.026); right panel = cumulative fraction of total gain.
- **Say this:** "P4's early demonstrations are individually far more
  informative than the baseline's average demo, and P4 banks 80% of the total
  achievable gain by demo 8. The LLM is prescribing high-value demonstrations
  early — that is the mechanism behind the efficiency."

### 10. research_budget_vs_threshold_n9.png  — dominance across all targets
- **What it is:** expected demos to reach SR ≥ θ, swept θ = 0.5→0.95.
- **Say this:** "This is the strongest single figure: at *every* quality
  target, P4 needs fewer (never more) demonstrations than the baseline. The
  advantage is not target-specific — it is a general dominance."

---

## C. Statistical rigor (pre-empts "is this just luck?")

### 11. research_bootstrap_diffs_n9.png  — uncertainty of the gap
- **What it is:** bootstrap distribution of the P4−baseline difference;
  demos-to-target CI [0.89, 3.56], AULC CI [0.20, 0.24]; both entirely > 0.
- **Say this:** "Resampling the runs, the entire distribution of the advantage
  is on P4's side for both efficiency metrics — zero is never plausible."

### 12. research_forest_stats_n9.png  — all metrics, one view
- **What it is:** paired Cohen's d ± 95% CI per metric, with permutation
  p-values. Final SR p=0.020, AULC p=0.005, Demos→target p=0.032, Reached-by-
  K=12 p=0.122 (n.s.).
- **Say this:** "Across metrics P4 is favoured and significant on final SR,
  AULC, and demos-to-target. The only non-significant one (reached-by-K=12) is
  an underpowered binary at n=9 — and even it points the right way."
- **If pushed:** AULC's huge effect size has a very wide CI (single-shot
  baseline inflates it) — quote AULC as a *direction/anytime* result and lean
  on demos-to-target + dominance for the rigorous claim. Main limitation is
  n=9; more seeds would tighten everything.

---

## Suggested 4-figure flow for the talk
1. **#3 final_sr_at_budget** — "accuracy: P4 is at least as good."
2. **#10 budget_vs_threshold** — "but P4 needs fewer demos at *every* target."
3. **#7 paired_demos_to_target** — "controlled, paired: −2.2 demos, CI excludes 0."
4. **#12 forest_stats** — "and it holds up statistically; caveat is n=9."

Verbal summary: *"At equal budget the methods are comparable in accuracy. The
contribution of P4 is sample efficiency: it reaches the target — and the
baseline's own full-budget quality — with significantly fewer demonstrations,
dominates the baseline at every intermediate budget and every quality
threshold, and the effect survives paired, bootstrap, and permutation tests.
The main limitation is n=9 runs."*
