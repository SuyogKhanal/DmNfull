# A13 — Memory-constant sensitivity: statistical analysis

**Source of truth:** `ablations_results/DISTIL_ablation_results.xlsx`, sheets `A13`, `A1_Memory_Off`, `S1_SignTest`.
All values below are computed from the workbook with `scipy` 1.17.1 (`friedmanchisquare`, `wilcoxon`, `binomtest`, `ttest_rel`, `rankdata`). Holm–Bonferroni is implemented manually with monotonicity enforcement, since `statsmodels` is not installed in this environment. No number in this report is hand-entered.

**Design.** Each memory constant of the recency-discounted Gaussian memory is swept one at a time with the other two held at their paper values (γ=0.6, σ=0.06, λ=1.0). Each swept value is measured on the same 10 settings (5 tasks × 2 observation modalities), treated as matched blocks. Ranks are assigned per row, rank 1 = highest final success rate.

**Tests.** (1) Friedman across the swept values over the 10 matched blocks. (2) Post-hoc two-sided Wilcoxon signed-rank of the paper's value against each other value. (3) Holm–Bonferroni within each constant's family of four comparisons only, never across constants. (4) Average rank per value across the 10 settings.

**Power note that governs the reading of every table below.** A two-sided Wilcoxon signed-rank test on `n` non-zero pairs has a hard lower bound on the p-value it can ever return: n=10 → 0.0020, n=9 → 0.0039, n=7 → 0.0156, n=5 → 0.0625, n=4 → 0.1250. Where the sweep produces identical success rates across two values, Wilcoxon discards the tied pair and the effective `n` falls. Several comparisons below return *exactly* their floor, meaning every non-tied setting favoured the paper value and the test was already as significant as it is arithmetically able to be.

---

## Cross-check: A13 λ=0 must reproduce A1_Memory_Off

λ=0 disables the memory term, so the λ=0 column of A13 and the ablated mean of A1_Memory_Off describe the same experiment and must agree.

| Setting | A13 λ=0 | A1_Memory_Off | Difference |
|---|---|---|---|
| GridWorld 5x5 state | 89.4 | 89.4 | 0.000 |
| GridWorld 5x5 image | 89.0 | 89.0 | 0.000 |
| Push-T state | 95.7 | 95.7 | 0.000 |
| Push-T image | 93.1 | 93.1 | 0.000 |
| Lift state | 99.9 | 99.9 | 0.000 |
| Lift image | 99.9 | 99.9 | 0.000 |
| Wipe state | 94.4 | 94.4 | 0.000 |
| Wipe image | 94.1 | 94.1 | 0.000 |
| Door state | 98.2 | 98.2 | 0.000 |
| Door image | 98.0 | 98.0 | 0.000 |

**PASS — no data-integrity flag.** All 10 settings agree to machine precision. The two sheets are mutually consistent.

---

## γ — recency discount (paper value 0.6)

**Friedman:** χ²(4) = 33.134, p = 1.12 × 10⁻⁶.

| value | avg_rank | mean SR (%) | wilcoxon_p_vs_chosen | holm_corrected_p |
|---|---|---|---|---|
| 0.3 | 4.00 | 94.64 | 0.0039 | 0.0078 |
| 0.5 | 2.60 | 95.51 | 0.0039 | 0.0078 |
| **0.6 (paper)** | **1.10** | **95.79** | — | — |
| 0.7 | 2.55 | 95.55 | 0.0020 | 0.0078 |
| 0.9 | 4.75 | 94.55 | 0.0020 | 0.0078 |

**Verdict:** γ = 0.6 is significantly best. It holds the best average rank (1.10 of 5) and beats every other swept value, both immediate neighbours included, at Holm-corrected p < 0.01.

γ=0.6 is top-ranked in 9 of the 10 settings. The response is a clean inverted-U: performance falls away at 0.3 and at 0.9, while the neighbouring values 0.5 and 0.7 are almost interchangeable with each other (mean 95.51 against 95.55, average rank 2.60 against 2.55) and both sit below 0.6. That pattern is the signature of a genuine interior optimum rather than a monotone trend truncated by the edge of the sweep.

---

## σ — kernel width (paper value 0.06)

**Friedman:** χ²(4) = 28.841, p = 8.42 × 10⁻⁶.

| value | avg_rank | mean SR (%) | wilcoxon_p_vs_chosen | holm_corrected_p | effective n |
|---|---|---|---|---|---|
| 0.02 | 3.65 | 95.16 | 0.0156 | 0.0469 | 7 |
| 0.04 | 2.15 | 95.62 | 0.1250 | 0.1250 | 4 |
| **0.06 (paper)** | **1.75** | **95.79** | — | — | — |
| 0.1 | 2.75 | 95.48 | 0.0625 | 0.1250 | 5 |
| 0.2 | 4.70 | 94.79 | 0.0039 | 0.0156 | 9 |

**Verdict:** σ = 0.06 is directionally best but not statistically distinguishable from its neighbours σ = 0.04 and σ = 0.1 after Holm correction. It sits on a plateau, and the failure to separate is a power artefact of the degenerate kernel rather than evidence that σ does not matter where it acts.

The naive reading of that flag is wrong, and the detail matters. Against σ=0.04 only 4 of 10 settings produce a non-zero difference, and against σ=0.1 only 5 of 10 do. In both comparisons *every* non-tied setting favours σ=0.06 (4/4 and 5/5). The returned p-values, 0.1250 and 0.0625, are precisely the minimum a two-sided Wilcoxon test can return at n=4 and n=5. Significance at the 0.05 level is arithmetically unreachable in these two comparisons however large the effect is. The evidence is perfectly one-directional and the test has run out of resolution.

The ties are not noise. They are the degenerate-kernel pathology recorded on the A13 sheet. Separating the settings where the kernel can act from those where it cannot:

| group | settings | σ=0.02 | σ=0.04 | σ=0.06 | σ=0.1 | σ=0.2 | per-setting spread |
|---|---|---|---|---|---|---|---|
| kernel live | Push-T, Wipe (both modalities) | 93.70 | 94.78 | **95.20** | 94.45 | 92.88 | 2.2 – 2.6 |
| kernel degenerate or ceiling-masked | GridWorld, Door, Lift (both modalities) | 96.13 | 96.18 | **96.18** | 96.17 | 96.07 | 0.1 – 0.2 |

On the four settings where the kernel discriminates, σ moves the success rate by 2.2 to 2.6 points and σ=0.06 is the best value (Friedman on those four blocks alone: χ² = 16.000, p = 0.0030). On the other six the sweep is flat to within 0.2 points, and those six flat settings are exactly the tied pairs that strip the post-hoc test of its power. Three separate causes produce the flatness. On GridWorld the centroids are in grid-cell units, so the kernel collapses to an identical-centroid check at every swept σ. On Door the reset range is ±0.013 m, so typical centroid separations leave the kernel saturated even at σ=0.02. On Lift the framework is at 100.0 ± 0.0 and the ceiling masks whatever the kernel does. **Lift is uninformative for every ablation, and no null result on it is evidence about any mechanism.**

The honest conclusion is the one the workbook already reaches. σ=0.06 is mis-scaled for the narrow-reset settings. A single global σ is not a virtue of the framework, it is a limitation of the present instantiation, and a per-task σ defined as a fraction of each task's reset range would let the memory act in all ten settings rather than four.

---

## λ — memory penalty weight (paper value 1.0)

**Friedman:** χ²(4) = 34.358, p = 6.29 × 10⁻⁷.

| value | avg_rank | mean SR (%) | wilcoxon_p_vs_chosen | holm_corrected_p |
|---|---|---|---|---|
| 0.0 | 2.90 | 95.17 | 0.0020 | 0.0078 |
| 0.5 | 3.90 | 94.67 | 0.0039 | 0.0078 |
| **1.0 (paper)** | **1.10** | **95.79** | — | — |
| 2.0 | 2.30 | 95.36 | 0.0039 | 0.0078 |
| 4.0 | 4.80 | 94.48 | 0.0020 | 0.0078 |

**Verdict:** λ = 1.0 is significantly best. It holds the best average rank (1.10 of 5) and beats every other swept value, λ = 0 included, at Holm-corrected p < 0.01.

Two features deserve comment. λ=1.0 separates from λ=0 at the floor p-value of the test, which is the strongest statement the 10 matched blocks can support, so the memory term contributes to the final success rate and is not decorative. The response is also not monotone: λ=0.5 (average rank 3.90, mean 94.67) ranks *below* λ=0 (average rank 2.90, mean 95.17), so a half-weight penalty is worse than no penalty at all. A weak penalty deflects the allocation away from the dominant failure mode without applying enough pressure to rotate onto a different one, which spreads the budget across regions without covering any of them. Only at λ=1.0 does the penalty commit hard enough to complete the rotation. The non-monotonicity is a finding, and it argues that λ cannot be tuned down toward zero safely.

---

## S1 — DISEIL against the strongest baseline across the 10 settings

DISEIL attains the higher mean in **10 of 10** settings. Mean margin **+3.71 points** (range +0.4 on Lift image to +6.4 on Door image).

| n | test | statistic | one-sided p | two-sided p |
|---|---|---|---|---|
| 10 settings | Sign test | 10/10 wins | 0.00098 | 0.00195 |
| 10 settings | Wilcoxon signed-rank | W = 0.0 | 0.00098 | 0.00195 |
| 5 task means | Sign test | 5/5 wins | 0.0312 | 0.0625 |
| 5 task means | Paired t-test | t(4) = 4.150 | 0.0071 | 0.0143 |
| 5 task means | Wilcoxon signed-rank | W = 0.0 | — | 0.0625 |

Per-task margins: GridWorld +2.80, Push-T +5.15, Lift +0.60, Wipe +5.20, Door +4.80.

Both n=10 tests return W = 0 and land on the floor p-value for 10 pairs, so 0.00195 two-sided is the most significant result the design can produce. The n=10 figure should not be led with. The ten settings are 5 tasks × 2 observation modalities, and the two modalities of a task share the expert, the reward structure and the reset distribution, so they are correlated by construction and the effective sample size is closer to 5 than to 10. Collapsing to the five task means, the sweep holds at 5/5, the sign test can no longer reject two-sided (p = 0.0625 is its floor at n=5), and the paired t-test on the same five means does reject (p = 0.0143 two-sided). The claim that survives a hostile reviewer is the conservative one, and it remains a claim: the ranking is consistent across every task and every modality, and the margin is large wherever there is headroom to show one.

**Verdict:** DISEIL is ahead of the strongest baseline in every setting. The aggregate advantage is significant under the conservative task-level analysis (paired t, p = 0.014 two-sided), and the setting-level tests reach the floor of what their design permits (p = 0.002 two-sided).

---

## Paste-ready paragraphs

### γ

We swept the recency discount γ over {0.3, 0.5, 0.6, 0.7, 0.9} with the remaining memory constants held at their reference values, and evaluated every value on the same ten settings, treated as matched blocks. A Friedman test rejects the null of equal performance across the sweep (χ²(4) = 33.13, p = 1.1 × 10⁻⁶). The value used throughout the paper, γ = 0.6, attains the best average rank (1.10 of 5) and the highest mean success rate (95.8%), and post-hoc two-sided Wilcoxon signed-rank tests with Holm–Bonferroni correction inside the γ family separate it from all four alternatives (corrected p < 0.01 in every comparison). The profile is an inverted-U with an interior peak: the neighbouring values γ = 0.5 and γ = 0.7 are indistinguishable from one another (95.5% against 95.6%) and both fall below γ = 0.6, while the extremes lose about one point of success rate. A discount that is too aggressive forgets which regions were recently corrected and re-spends the budget on them, and one that is too slow keeps suppressing regions long after the policy has stopped failing there.

### σ

The kernel width σ was swept over {0.02, 0.04, 0.06, 0.1, 0.2} on the same ten matched settings. The Friedman test rejects equality across the sweep (χ²(4) = 28.84, p = 8.4 × 10⁻⁶), and the reference value σ = 0.06 attains the best average rank (1.75 of 5). Holm-corrected Wilcoxon tests separate it from the extremes σ = 0.02 and σ = 0.2, but not from its immediate neighbours σ = 0.04 and σ = 0.1, so σ = 0.06 is directionally best and statistically indistinguishable from the values on either side of it. The reason is instructive, and we state it rather than presenting the plateau as stability. Only four of the ten settings, Push-T and Wipe under both observation modalities, vary at all under the sweep, and on those four the choice of σ moves the final success rate by 2.2 to 2.6 points with σ = 0.06 best. On the remaining six the sweep is flat to within 0.2 points, and those tied settings remove so much power from the paired test that a corrected p below 0.05 against the neighbouring values is arithmetically unreachable, even though every untied setting favours σ = 0.06. The flatness has three distinct causes. On GridWorld the cluster centroids are expressed in grid-cell units, so the Gaussian collapses to an identical-centroid test at every σ we swept. On Door the reset range is ±0.013 m, and typical centroid separations leave the kernel saturated even at σ = 0.02. On Lift the framework already sits at 100.0 ± 0.0, so the ceiling hides whatever the kernel does, and no null result on Lift is evidence about any mechanism. A single global σ is therefore mis-scaled for the narrow-reset settings. We report this as a limitation of the present instantiation: defining σ per task, as a fraction of that task's reset range, would let the memory term act in all ten settings instead of four.

### λ

The memory penalty weight λ was swept over {0.0, 0.5, 1.0, 2.0, 4.0}, where λ = 0 removes the memory term entirely and reproduces the memory-off ablation exactly on all ten settings. A Friedman test rejects equality across the sweep (χ²(4) = 34.36, p = 6.3 × 10⁻⁷). The reference value λ = 1.0 attains the best average rank (1.10 of 5) and separates from every alternative under Holm-corrected Wilcoxon tests (corrected p < 0.01), λ = 0 included, which establishes that the memory term contributes to the final success rate rather than decorating the objective. The response to λ is not monotone. A half-weight penalty (λ = 0.5, mean 94.7%) performs worse than no penalty at all (λ = 0, mean 95.2%), and only at λ = 1.0 does performance recover and exceed both. A weak penalty deflects the allocation away from the dominant failure mode without pushing hard enough to rotate onto a different one, which spreads the budget without covering any region properly, whereas a heavy penalty (λ = 4.0, mean 94.5%) forces rotation away from failure modes that still merit attention. The interior optimum is a trade-off between revisiting and rotating, and λ cannot be tuned toward zero without cost.

### S1

The framework attains the best mean success rate in all ten task and modality settings, with a mean margin of 3.7 points over the strongest baseline in each setting. Treating the ten settings as paired observations, both a sign test and a Wilcoxon signed-rank test reject a coin-flip ranking at p = 0.002 (two-sided), the smallest p-value attainable with ten pairs. The ten settings are not ten independent experiments. They are five tasks under two observation modalities, and the two modalities of a task share the expert, the reward structure and the reset distribution. Collapsing to the five task means to absorb that correlation, the sweep remains consistent at five wins from five, a paired t-test over the task means rejects at p = 0.014 (two-sided, t(4) = 4.15), and the sign test reaches its own floor at p = 0.031 one-sided. We report the conservative task-level result as the headline.

---

## Data-integrity summary

| check | result |
|---|---|
| A13 λ=0 column reproduces A1_Memory_Off | **PASS**, exact on all 10 settings |
| A13 sweep blocks parsed | 3 sweeps × 5 values × 10 settings |
| Paper values present in every sweep | γ=0.6, σ=0.06, λ=1.0 all present |
| Holm correction scope | within each constant's family of 4 comparisons, never across constants |
| Fabricated values | none; every figure recomputed from the workbook |

**Artefacts:** `build/stats_results.csv` (one row per constant-value pair), `build/_stats_raw.json` (full matrices and per-comparison detail).
