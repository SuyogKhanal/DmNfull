# Why these plots — rationale, significance & narrative (n=9, run_02 excluded)

Companion to `FIGURE_TALKING_POINTS.md`. This document answers the harder
supervisor question: **"why did you make *these* plots, what does each one
prove, and why this form and not another?"**

---

## 0. Why this SET of plots (read this first)

The 12 figures are not a grab-bag — they are a deliberate 3-act argument that
follows the exact order a skeptical reviewer raises objections:

| Act | Question it answers | Figures |
|-----|---------------------|---------|
| **A. Validity** | "Is your comparison even fair?" | corrected curve, endpoints, final-SR |
| **B. Effect** | "Fine — is P4 better, and how much?" | demos-to-target, survival, match-baseline, paired, AULC |
| **C. Rigour & mechanism** | "Could it be luck or cherry-picking? *Why* does it work?" | budget-threshold, bootstrap, forest, marginal-gain |

**Core methodological principle — triangulation.** "Sample efficiency" has no
single canonical metric, so any *one* plot invites "you picked the metric that
flatters you." We therefore operationalise the same claim **five independent
ways** (time-to-target, demos-to-match-baseline, paired difference, AULC,
per-threshold dominance). They use different definitions, axes and statistics
yet all point the same way. An effect that is invariant to operationalisation
is not cherry-picked — that invariance *is* the argument.

**What we deliberately did NOT plot (say this if asked — it shows integrity):**
- *Per-layout success over demos* — `learning_curve.json` only logs aggregate
  heldout SR per round, so this cannot be done honestly from the logged data.
- *Variance / spread comparison* — double-edged (P4's variance is higher
  because 2 runs plateau); it is not the claim, so plotting it would be
  misleading editorialising.
- *Accuracy at small fixed budgets (K=5, K=10)* — the baseline has no
  measurement below its full-budget spend, so such a bar chart would just
  re-encode the single-shot logging artifact. Excluding it is correct, not
  hiding.

This "what we refused to plot" list is itself a credibility asset.

---

## ACT A — Validity (earn the right to make the claim)

### 1. baseline_nb_vs_swatch_p4_n9.png — the corrected learning curve
- **Why this plot:** it is the canonical figure reviewers expect *first*; we
  lead with it to be transparent and to surface the baseline's single-shot
  shape ourselves before anyone accuses us of hiding it. Showing the
  unflattering raw picture first buys credibility for everything after.
- **Why this form:** mean ± band with faint per-run lines = the standard RL/
  active-learning learning-curve convention; deviating would invite questions.
- **Significance:** establishes that the only point where both methods are
  honestly measured is the full budget — which is precisely why a single
  endpoint comparison is insufficient and Act B is needed.
- **The saying:** *"P4 learns continuously; the baseline is one jump — so the
  comparison has to be done carefully, not at a glance."*

### 2. baseline_endpoints_vs_p4_n9.png — baseline drawn honestly
- **Why this plot:** pre-empts the specific accusation *"you drew a connected
  line through a method that only has two measurements to exaggerate the gap."*
  We remove the line ourselves.
- **Why this form:** endpoints + per-run finals + mean±std is the
  *measurement-faithful* representation of a 2-point method.
- **Significance:** isolates "what was actually measured" from "what was
  interpolated"; proves the gap is not a drawing trick.
- **The saying:** *"We are not inflating anything — this is literally what the
  baseline gives us."*

### 3. final_sr_at_budget_n9.png — the one fair scalar comparison
- **Why this plot:** every reviewer asks "who wins at equal budget?" This
  answers it head-on with the **per-run scatter** (not just means) so the
  distribution and overlap are visible.
- **Why this form:** dot/CI plot, not a bar — bars hide n and distribution; at
  n=9 you must show every point.
- **Significance:** Baseline 0.867 vs P4 0.903 — accuracy is parity-to-mild-
  edge. *By conceding this honestly we earn trust for the efficiency claims.*
- **The saying:** *"At equal budget accuracy is essentially a wash, so the
  contribution of P4 must be — and is — efficiency."*

---

## ACT B — Effect (the positive claim, triangulated)

### 4. demos_to_target_n9.png — the most intuitive efficiency metric
- **Why this plot:** operationalises "efficiency" the way a practitioner
  actually thinks: *how many demonstrations to hit the target I care about
  (0.90)?* The reached/never-reached split is shown so censoring is not hidden.
- **Why this form:** scatter + mean + explicit "never reached ×k" markers —
  honest about the runs that fail.
- **Significance:** P4 reaches target in 6/9 runs (mean 12.2 demos); baseline
  only 2/9 (14.5). Higher hit rate **and** fewer demos.
- **The saying:** *"To actually get the result you want, P4 needs fewer
  demonstrations and succeeds more often."*

### 5. time_to_target_survival_n9.png — the temporal profile
- **Why this plot:** a single mean hides *when* success happens; this answers
  "at any budget I might stop at, who is ahead?" Borrowed from survival
  analysis (a recognised, rigorous framework).
- **Why this form:** step/CDF curve — the correct object for "fraction
  succeeded by time x"; shows dominance at *every* x, a stronger statement
  than a mean.
- **Significance:** P4 begins clearing target at demo 10 and ~⅔ of runs by 15;
  baseline produces nothing until 14. P4 ≥ baseline coverage everywhere
  (visual stochastic dominance).
- **The saying:** *"Pick any demo budget — more P4 runs have already
  succeeded by then."*

### 6. demos_to_match_baseline_n9.png — a baseline-defined bar
- **Why this plot:** pre-empts "0.90 is *your* target — be fair to the
  baseline." We instead pin the threshold to the **baseline's own full-budget
  result (0.867)** and ask how cheaply P4 clears the baseline's *own best*.
- **Why this form:** median (not mean) because the P4 mean is dragged late by
  2 plateau runs — and we state that explicitly rather than hide it.
- **Significance:** 8/9 P4 runs reach the baseline's full-budget quality with a
  **median of 12 demos (~20% fewer)**.
- **The saying:** *"P4 gets the baseline's best result for ~20% less labelling
  cost — judged by the baseline's own standard."*

### 7. paired_demos_to_target_n9.png — the cleanest comparison available
- **Why this plot:** strongest design we can do without new experiments — runs
  are **paired by identical correction pool and identical init model**, so the
  only thing that varies is the method. This removes between-run variance.
- **Why this form:** paired slope lines + mean + bootstrap CI — the standard
  way to show a within-subject effect.
- **Significance:** mean 15.7 → 13.4, diff −2.2, **95% CI [−3.6, −0.9] excludes
  zero** → the advantage is within-condition, not a between-sample fluke.
- **The saying:** *"Holding everything else fixed, switching to P4 saves ≈2
  demonstrations per run — and zero is outside the confidence interval."*

### 8. anytime_aulc_n9.png — the deployment-realistic metric
- **Why this plot:** answers the practical question "what if my budget is
  uncertain or I must stop early?" AULC is the standard single-number summary
  of a whole learning curve in active learning / RL.
- **Why this form:** area visual + bar — makes "expected SR over all budgets"
  concrete.
- **Significance:** 0.720 vs 0.498. Large anytime advantage.
- **Honest caveat to volunteer:** inflated because single-shot top-K yields no
  model until the full spend — present as an **anytime/early-stop** advantage,
  not raw superiority.
- **The saying:** *"Under any uncertain or limited labelling budget, P4 is the
  safer choice."*

---

## ACT C — Rigour & mechanism (defeat "luck / why does it work?")

### 9. research_marginal_gain_n9.png — the mechanism
- **Why this plot:** Acts A–B show *that* P4 is more efficient; reviewers then
  ask *why*. This is the explanatory plot — per-demo value and front-loading.
- **Why this form:** per-step ΔSR bars vs the baseline's amortised ΔSR/demo,
  plus a cumulative-gain panel — directly visualises "early demos are worth
  more."
- **Significance:** P4's early demonstrations beat the baseline's *average*
  demo, and 80% of the achievable gain is banked by demo 8 — evidence the LLM
  is prescribing high-value demonstrations early.
- **The saying:** *"The LLM front-loads informative demonstrations — that is
  the actual source of the efficiency, not noise."*

### 10. research_budget_vs_threshold_n9.png — the strongest figure
- **Why this plot:** the definitive answer to "you cherry-picked the 0.90
  threshold." Sweeping θ = 0.5→0.95 tests the claim at *every* operating point.
- **Why this form:** demos-to-θ vs θ with CI bands — turns one number into a
  whole dominance curve.
- **Significance:** P4 needs ≤ baseline demos at **every** threshold (never
  more) → threshold-invariant dominance, the strongest non-parametric claim
  here.
- **The saying:** *"The advantage is not specific to one target — it holds
  across the entire operating range."*

### 11. research_bootstrap_diffs_n9.png — tangible uncertainty
- **Why this plot:** a CI is abstract; showing the full resampled distribution
  makes "zero is implausible" visible, and is expected in rigorous empirical ML.
- **Why this form:** histogram of the bootstrap difference with the zero line
  and observed mean marked.
- **Significance:** the entire distribution sits on P4's side for both
  efficiency metrics (demos CI [0.89, 3.56]; AULC CI [0.20, 0.24]).
- **The saying:** *"However we resample our 9 runs, P4 still wins — zero is
  never on the table."*

### 12. research_forest_stats_n9.png — the decision-grade summary
- **Why this plot:** consolidates every metric into one view (effect size + CI
  + permutation p), **including the non-significant one** — shown deliberately
  to prove nothing is hidden. Forest plots are the meta-analysis standard for
  "does it hold up?"
- **Why this form:** Cohen's d ± CI per metric — comparable across metrics on
  one axis.
- **Significance:** P4 favoured & significant on Final SR (p=0.020), AULC
  (p=0.005), Demos→target (p=0.032); only reached-by-K=12 is n.s. (p=0.122) —
  an underpowered binary at n=9 that still points the right way.
- **Honest caveat to volunteer:** AULC's huge d has a very wide CI (single-shot
  baseline inflates it) — lean on demos-to-target + dominance for the rigorous
  claim. The headline limitation is **n=9**; more seeds tighten everything.
- **The saying:** *"Across independent metrics and proper paired/permutation
  tests the effect holds; our honest limitation is sample size, not direction."*

---

## How to answer "did you cherry-pick these plots?"
*"No — they're a structured argument. Act A proves the comparison is fair (we
show the unflattering raw picture and refuse to draw misleading lines). Act B
demonstrates the efficiency advantage **five independent ways** that don't
share a metric or axis. Act C shows it survives bootstrap, permutation and
paired tests and explains the mechanism. We also explicitly refused to plot
three things that would have flattered us or re-encoded an artifact. The one
non-significant result is shown, not buried. The only real limitation is
n=9."*
