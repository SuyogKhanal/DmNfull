# Generated figures for the CoC report

Fifteen figures, written to `paper_aaai2027/COC_REPORT/figures_generated/` as **PDF** (vector, for LaTeX) and **PNG** (preview). Each is produced by `figures_generated/make_figures.py`, which parses the workbook at run time. No number in any figure is hand-entered.

**Source of truth.** `ablations_results/DISTIL_ablation_results.xlsx` (24 sheets). The workbook's method column carries the old internal name; it is relabelled DISEIL on every axis, legend and annotation. The rendered PDFs were scanned after generation and contain no occurrence of the dead name, nor of any internal code identifier.

**Conventions applied in every figure.**

- Δ-SR is the change in the policy's success rate on the round-level rollout evaluation.
- Margin retained = (ablated − best baseline) / (full DISEIL − best baseline), from the sheets' own helper-derived columns.
- Baselines are labelled as the DAgger family.
- The three primary settings (GridWorld image, Push-T state, Door image) are drawn in the strong hue, bold-faced and marked `*`.
- **Lift is greyed and carries an explicit ceiling note wherever it appears** (100.0 ± 0.0: no headroom, no seed variance), so that no reader can over-read a null on Lift as evidence about any mechanism.
- Seed counts: 9 (GridWorld), 5 (robot tasks). Budget B = 20, D = 1 demonstration per round.

**Palette.** Okabe-Ito colourblind-safe qualitative set. Adjacent-pair separation was computed, not eyeballed, under deuteranomaly, protanomaly and tritanomaly (CAM02-UCS ΔE, severity 100): worst pair 13.1, above the ≥ 12 target. Purple and orange are never used together (that pair falls to 11.0 under tritanomaly). Ordered arms (A9, A14, A15, D2) use a single-hue sequential ramp rather than categorical hues, because a ladder encodes magnitude, not identity. Fonts are ≥ 7 pt for footnotes and ≥ 8.5 pt for all axis and tick labels, at the printed width. No figure carries an internal title: the caption carries it.

**Sheet-layout notes that govern the parsing.**

- Knockout sheets (A1, A3, A4, A5, A6, A7, A8) share a layout: header at Excel row 8, data at rows 9–18; column C = full DISEIL, D = best baseline, E = the `mean ± std` display string, F = Δ, G = margin retained, last column = the numeric helper mean.
- **The A6 display strings (column E) are stale**: they disagree with the sheet's own Δ and margin-retained columns for eight of ten settings. Every A6 number plotted here is taken from the helper column (J), which is what the sheet's own arithmetic uses. No A6 figure uses a standard deviation, so the stale strings are never touched.
- **A2, column E is Stagger**; column D is Diff-DAgger. (Reading D as the random-allocation arm silently substitutes Diff-DAgger and inverts the finding.)

---

## F1 — Allocation ladder

**Files** `F1_allocation_ladder.pdf` / `.png` · **Covers** A2, A8, A3, GT_SR · **Format** grouped bar, three panels, best baseline drawn as a horizontal rule

**What it shows.** The single most important ablation figure. In one image: uniform-random allocation sits *below* the best baseline; the deterministic fallback rule alone recovers about a third of the margin; removing the clustering drops the system back to the baseline (and *below* it on Door image); the full system sits on top. It establishes that DISEIL is an allocation framework and that the allocation is what carries it.

**Data series and provenance.**

| Series | Source |
|---|---|
| Random allocation, robot settings | `A2_RandomAlloc_Robots` col **E** (Stagger), rows 9–16 |
| Random allocation, GridWorld | `GT_SR` col **G** (Stagger), rows 5–6 — GridWorld has no A2 row, its random-allocation control is the Stagger column of Table 1 |
| Fallback only (A8) | `A8_Fallback_Only` helper col I + s.d. from col E, rows 9–18 |
| Clustering off (A3) | `A3_Clustering_Off` helper col J + s.d. from col E, rows 9–18 |
| Full DISEIL | `GT_SR` col I, rows 5–14 |
| Best baseline + its s.d. | `GT_SR` col J (mean), with the s.d. taken from whichever baseline column attains it |

Plotted values: GridWorld image 86.6 / 87.8 / 87.4 / **87.1** / 89.6; Push-T state 82.3 / 92.5 / 92.0 / **90.7** / 96.1; Door image 84.0 / 94.8 / 92.4 / **92.8** / 99.2 (bold = best baseline rule).

**Caption.** *The allocation ladder, on the three primary settings. Each panel compares uniform-random allocation (A2), the deterministic nearest-untried fallback rule alone (A8), clustering removed in favour of greedy worst-loss selection (A3), and full DISEIL, against the strongest DAgger-family baseline for that setting (dashed rule). Bars are the mean final success rate over seeds (9 for GridWorld, 5 for the robot tasks) with one standard deviation. Randomly chosen expert labour falls below every gated baseline; structured-but-unreasoning allocation recovers roughly a third of the margin; removing the failure-mode structure returns the system to baseline performance, and on Door image drops it beneath the baseline. Budget B = 20, one demonstration per round.*

---

## F2 — Gain without allocation

**Files** `F2_gain_without_allocation.pdf` / `.png` · **Covers** A3, GT_InfoGain · **Format** scatter with paired arrows, plus a slope panel

**What it shows.** The dissociation that licenses the central claim. Each arrow runs from full DISEIL to the clustering-off system for one setting. The arrows point right-and-down: per-demonstration information gain does not fall (it rises slightly) while the success rate collapses. A bar chart would show the success-rate drop and conceal the half of the result that matters. Two measures of different scale are shown on two panels rather than on a twin axis.

**Data series and provenance.**

| Series | Source |
|---|---|
| Information gain, full | `GT_InfoGain` col I, rows 5–14 |
| Information gain, ablated | `A3_Clustering_Off` col **H**, rows 9–18 |
| Δ-SR against full DISEIL | `A3_Clustering_Off` col F, rows 9–18 |

Annotated statistics (excluding Lift): mean ΔIG = +0.06, Wilcoxon p = 0.23; mean Δ-SR = −4.01, Wilcoxon p = 0.002.

**Caption.** *Information gain is necessary and not sufficient. Each arrow joins full DISEIL (open marker) to the clustering-off ablation (filled marker) for one setting; the horizontal axis is per-demonstration information gain, defined as the policy's per-step loss on a newly acquired demonstration measured before retraining on it, and the vertical axis is the change in success rate against full DISEIL. Removing the failure-mode structure leaves information gain statistically unchanged (mean +0.06, Wilcoxon p = 0.23, Lift excluded) while success falls by 4.01 points (p = 0.002). The right-hand panel plots the paired gain values alone. Greedy worst-loss selection collects individually informative and jointly redundant demonstrations: gain has no term for redundancy between demonstrations, and allocation supplies it. Lift is at the ceiling and uninformative.*

---

## F3 — Knockout summary

**Files** `F3_knockout_summary.pdf` / `.png` · **Covers** A1, A3, A4, A5, A6, A7, A8 × 10 settings · **Format** heatmap, rows ordered by mean damage

**What it shows.** Seven knockouts against ten settings is a matrix, so it is drawn as one. Each cell gives the margin retained and, beneath it, the Δ in success-rate points. The ordering of the rows is the finding: clustering is load-bearing, the cluster memory is the *weakest* of the seven knockouts.

**Data series and provenance.** Margin retained from column G, and Δ from column F, of each of `A1_Memory_Off`, `A3_Clustering_Off`, `A4_LLM_vs_Heuristic`, `A5_VLM_Off`, `A6_KAG_Off`, `A7_Bridging_Off_`, `A8_Fallback_Only`, rows 9–18. (These columns are computed by the sheets from their helper means, so A6 is consistent here despite its stale display strings.)

**Deviation from the dossier's suggested encoding, stated deliberately.** The dossier proposes a diverging scale centred at 50 %. Margin retained has no polarity at 50 % — it is a magnitude with a critical zero — so a diverging ramp would place a hue at a meaningless midpoint. The figure instead uses a single-hue sequential ramp for 0–100 % and paints the cells that fall *below* the baseline in the alert hue, which is what the dossier wanted the diverging scale to achieve. Lift columns are hatched and marked `n/a`.

**Caption.** *Every knockout, on every setting. Cells give the percentage of the margin over the strongest DAgger-family baseline that survives the ablation, with the change in success rate (points) beneath. Rows are ordered by mean damage over the eight settings with headroom. Clustering carries the method (11.3 % retained on average); the deterministic fallback recovers about a third; knowledge-graph grounding is worth roughly half the margin; and the cluster memory, which the framework's own narrative presents as a headline mechanism, is the least damaging knockout of the seven. A13 explains why, and the explanation is a mis-scaled kernel width, not a virtue. The single cell in the alert hue is A3 on Door image, where the ablated system falls below its baseline. Lift is at the 100.0 ± 0.0 ceiling: no ablation is informative there, and its columns are struck out rather than reported.*

---

## F4 — Reasoning and vision are small

**Files** `F4_reasoning_and_vision_small.pdf` / `.png` · **Covers** A4, A5 · **Format** paired dot plot with a seed-noise band

**What it shows.** The honest format for a small effect. Every point is negative (the effect is real and consistently signed) and every point lies inside the ± 1 seed-standard-deviation band of its own full run (the effect is small). A4 and A5 are plotted on one axis because the two knockouts are indistinguishable in magnitude and separate figures would imply a distinction the data do not support.

**Data series and provenance.** Δ from `A4_LLM_vs_Heuristic` col F and `A5_VLM_Off` col F, rows 9–18. The noise band is the full-DISEIL seed s.d. from `GT_SR` col I, rows 5–14.

**Caption.** *Removing the reasoning model (A4, replaced by a fixed heuristic on the same geometric clusters) and removing the vision-language model (A5, geometric descriptor and root-cause labels only) each cost about one success-rate point: mean −1.08 and −1.01 over the eight settings with headroom. The shaded band is one seed standard deviation of the corresponding full DISEIL run. The direction is consistent across all ten settings and every gap falls inside the seed noise. Both components act downstream of the decisive step: clustering is geometric and uses no foundation-model output, so by the time either model is called, the question of which region of the failure distribution receives this round's demonstration has already been settled. The framework degrades gracefully to a heuristic that still beats every baseline.*

---

## F5 — Grounding and feasibility

**Files** `F5_grounding_and_feasibility.pdf` / `.png` · **Covers** A6 · **Format** scatter, fallback rate against Δ-SR

**What it shows.** The claim is causal and two-variable — grounding is removed, feasibility failures rise, success falls — so the mechanism is plotted against the outcome. A grouped bar of success rates would reduce A6 to "another knockout worth a couple of points" and hide the mechanism entirely.

**Data series and provenance.** Fallback rate from `A6_KAG_Off` col **H**, rows 9–18. Δ-SR from col F. Means from the helper column **J** (the display strings in column E are stale and are not used).

**Known gap, marked on the figure.** The full-DISEIL fallback rate per setting is not in the workbook, so the vertical reference line this figure wants cannot be drawn. The figure says so rather than inventing one.

**Caption.** *Removing the knowledge-augmented graph from the prompts raises the fallback rate to between 22 % and 35 % of rounds and costs 2.44 success-rate points on average, retaining 44.2 % of the margin. Equation 10 is a feasibility-verification loop: the reasoning model proposes a prescription, constraints are retrieved from the graph, the prescription is checked against them, and a violation is returned to the model for revision. Without the graph there is nothing to check against, so the model proposes placements outside the reachable set or the spawn range, the expert cannot solve the prescribed episode within the step limit, and the round falls back. Removing the reasoning itself costs one point (A4); removing the constraints the reasoning is checked against costs two and a half. The relation is loose because a fallback round is not a wasted round: the fallback rule alone still retains 31 % of the margin (A8), which is why A6 costs 2.4 points and not 6. Lift is at the ceiling and uninformative.*

---

## F6 — Bridging

**Files** `F6_bridging.pdf` / `.png` · **Covers** A7, D3 · **Format** scatter with the fitted line, plus a 100 %-stacked bar of the split

**What it shows.** An honest negative. Bridging is used in 18–30 % of accepted prescriptions, and the correlation between how often it is used and how much its removal costs is flat. Showing the flat cloud is more persuasive than omitting the analysis.

**Data series and provenance.** Bridge and targeted shares from `D3_Bridge_Split` cols C and D, rows 9–18. Δ from `A7_Bridging_Off_` col F, rows 9–18. Pearson r and p are computed in the script from those two columns, excluding Lift (r = 0.23, p = 0.58).

**Workbook contradiction, printed on the figure.** The A7 and D3 prose asserts bridging is inapplicable on GridWorld and Wipe and should be exactly 0 % there. The D3 data record 24–30 % on those settings, and A7 records a −2.9-point effect on Wipe image. The data are the source of truth; the figure states that the prose is stale and must be corrected before this ablation is written up.

**Caption.** *Bridging placement, which asks the expert for a configuration positioned between the target cluster and a region the policy already handles, is selected in 18 % to 30 % of accepted prescriptions (mean 24.4 %) and its removal costs 1.24 points on average. The correlation between the bridged share and the damage done by disabling bridging is flat (r = 0.23, p = 0.58, Lift excluded): bridging is not more valuable on the settings where it is used more often. What matters is which rounds use it. Bridging pays when the target cluster lies far outside anything the current policy solves, so that a targeted demonstration would be a large distributional jump and a bridged one is a step the policy can absorb; Wipe image, the setting with the weakest policy, is where that condition bites hardest (−2.9). The right panel gives the targeted/bridged split per setting.*

---

## F7 — Descriptor dimensionality

**Files** `F7_descriptor_dimensionality.pdf` / `.png` · **Covers** A10 · **Format** line plot, ten thin lines plus a heavy mean, marker at 6-D

**What it shows.** The inverted U, which is the argument, and which only a line plot shows. The descriptor is scored on a criterion that has nothing to do with success rate, which is the answer to the "you fished for the feature set" objection. This sheet supersedes the old Equation 7 image branch: clustering is geometric for every run, state and image alike, and the descriptor is the same 6-D vector in both. The frozen-embedding-plus-PCA branch does not appear anywhere in these figures.

**Data series and provenance.** `A10`, header at Excel row 10, variants at rows 11–17 (2-D, 4-D, 5-D, 6-D, 8-D, 10-D, 12-D), columns B–K (the ten settings). The heavy line is the mean over the ten columns, computed in the script: 0.354, 0.485, 0.532, **0.567**, 0.524, 0.469, 0.402.

**Caption.** *Mean silhouette of the failure clusters against the dimensionality of the geometric descriptor φ, one thin line per setting and the heavy line the mean over all ten. The 6-D descriptor is the argmax in ten settings out of ten (Friedman χ²(6) = 59.52, p = 5.6 × 10⁻¹¹; Wilcoxon 6-D against 5-D and against 8-D, p = 0.002 in both). Below 6-D the descriptor discards information that separates failure modes, and the largest single step in the sweep is the addition of orientation. Above 6-D nothing is removed: the loss is geometric, as pairwise distances concentrate and the agglomerative merge order becomes arbitrary. The failure sets being clustered are small (42 in round 1, falling to 2 by round 20), and distance concentration bites hardest at small sample sizes, so the descriptor is small because the failure sets are small. Silhouette scores geometric separation only and is independent of success rate; D1 is the complementary check on whether a well-separated cluster is a semantically clean one.*

---

## F8 — Budget sweep

**Files** `F8_budget_sweep.pdf` / `.png` · **Covers** A11 · **Format** line plot of the margin against B, plus an absolute-success-rate panel

**What it shows.** The margin decays monotonically in B in every one of the ten settings, and the second panel shows *why*: the baseline catches up, DISEIL does not degrade. The figure also carries the retraction. The workbook proposed the headline "DISEIL at B = 10 matches the best baseline at B = 20"; the data refute it in seven of ten settings, and the figure says so.

**Data series and provenance.** `A11`, rows 9–18. Cols C/D/E = best baseline / DISEIL / margin at B = 10; F/G/H at B = 20; I/J/K at B = 40. The mean line is over the eight non-Lift settings, computed in the script: +10.35, +4.49, +2.67.

**Caption.** *Margin over the strongest DAgger-family baseline as a function of the budget B, one line per setting, with the mean over the eight settings that have headroom (dashed). The margin is monotonically decreasing in B in all ten settings and roughly doubles when the budget is halved: +10.35 points at B = 10, +4.49 at B = 20, +2.67 at B = 40. Allocation buys the rate of coverage of the failure distribution, not its asymptote, so the advantage decays as demonstrations stop being scarce. The right panel shows the mechanism on Push-T state: the margin shrinks because the baseline catches up, not because DISEIL degrades. The framework operates under any fixed budget; B = 20 is the validated instance. The claim that DISEIL at B = 10 matches the best baseline at B = 20 is false in seven of the ten settings and is retracted.*

---

## F9 — Demonstrations per round

**Files** `F9_demos_per_round.pdf` / `.png` · **Covers** A12 · **Format** slope chart, ten lines

**What it shows.** Every line falls and no two lines cross. A slope chart is the only format that makes both facts visible at once; thirty grouped bars would obscure the monotonicity.

**Data series and provenance.** `A12`, rows 9–18, cols D/E/F = D = 1 / 2 / 3.

**Caption.** *Final success rate against the number of demonstrations prescribed per round, with the total expert labour held fixed at B = 20, so that D = 1 gives twenty rounds and D = 3 gives about seven. D = 1 is best in ten settings out of ten and the decline is monotone in every one (Friedman χ²(2) = 19.54, p = 5.7 × 10⁻⁵; Wilcoxon D = 1 against D = 3, p = 0.002). The number of rounds is the number of times the system re-analyses a freshly retrained policy: with D > 1, the later demonstrations of a round are prescribed against a policy that no longer exists by the time they are used, and the allocation is made stale by its own execution. The cost of stale allocation is largest exactly where allocation is most valuable (Door image, Wipe image) and smallest where it is least (GridWorld), which is a second and independent confirmation of the allocation thesis. D = 1 is the validated instance; the framework is stated over general D.*

---

## F10 — Memory constants

**Files** `F10_memory_constants.pdf` / `.png` · **Covers** A13, `build/stats_results.csv` · **Format** three-panel line plot, reference value marked

**What it shows.** The shape of each response, and the σ problem stated as the limitation it is. γ has an interior peak on every line. λ has a peak with a dip at 0.5, so a half-weight penalty is worse than no penalty at all. σ has a peak on Push-T and Wipe and six flat lines everywhere else, and those six ties are what strip the post-hoc test of its power.

**Data series and provenance.** `A13`: γ block at Excel rows 12–21 (values 0.3, 0.5, 0.6, 0.7, 0.9); σ block at rows 26–35 (0.02, 0.04, 0.06, 0.1, 0.2); λ block at rows 40–49 (0.0, 0.5, 1.0, 2.0, 4.0); cols C–G in each. Friedman statistics and the Holm-corrected p-values annotated on the panels are read from `build/stats_results.csv`, not retyped.

**Note on the y-limits.** The axis must span GridWorld (≈ 88.6) to Lift (100.0). A tighter limit silently clips both GridWorld lines — including GridWorld image, a primary setting — off the plot.

**Companion table.** The dossier is right that this is one of only two places where a table is warranted, because the exact Holm-corrected p-values are load-bearing for the σ claim. The table lives in `build/stats_results.csv` (columns: value, avg_rank, wilcoxon_p, holm_p, friedman_χ², friedman_p) and should be set beneath this figure.

**Caption.** *Sensitivity of the three memory constants, each swept alone with the others held at their reference values (dashed). The recency discount γ and the penalty weight λ are significantly best at their reference values, separating from every swept alternative under Holm-corrected Wilcoxon tests (corrected p < 0.01), and λ = 1.0 separates from λ = 0, which establishes that the memory term contributes rather than decorating the objective. The response to λ is not monotone: a half-weight penalty (λ = 0.5) is worse than no penalty at all, because it deflects the allocation away from the dominant failure mode without pushing hard enough to rotate onto a different one. The kernel width σ = 0.06 is directionally best but is not statistically distinguishable from its neighbours σ = 0.04 and σ = 0.1 (Holm p = 0.125, marked ns). The reason is visible in the panel: σ moves the success rate by 2.2 to 2.6 points on Push-T and Wipe and by 0.1 to 0.2 points on the six remaining settings, where the kernel is degenerate (GridWorld centroids are in grid-cell units; the Door reset range is ± 0.013 m, leaving the kernel saturated even at σ = 0.02) or masked by the Lift ceiling. A single global σ is mis-scaled for the narrow-reset tasks. A per-task σ, defined as a fraction of each task's reset range, would let the memory act in all ten settings instead of four. This is a limitation of the present instantiation, reported as one. The λ = 0 column reproduces the memory-off ablation exactly in all ten settings, which is an independent consistency check on the two runs.*

---

## F11 — Context set, cluster count, cited episodes

**Files** `F11_context_and_selection.pdf` / `.png` · **Covers** A9, A14, A15 · **Format** three grouped-bar panels, arms ordered by effect

**What it shows.** Three step-level ablations with the same answer — each is worth roughly half a point — shown together precisely so that none of them is oversold in isolation.

**Data series and provenance.**

| Panel | Source |
|---|---|
| A9 context-set composition | `A9`, header Excel row 8, arms rows 9–13, cols B–K |
| A14 cluster-count selection | `A14`, header row 8, arms rows 9–13, cols B–K |
| A15 number of cited episodes | `A15`, header row 8, arms rows 9–15, cols C–L (col B is n) |

Only the three primary settings are drawn. Lift is omitted, being at the ceiling.

**Caption.** *Three ablations of the machinery inside individual steps, on the primary settings, with arms ordered by effect. Left: the context set S is built from a forced target representative, the worst-peak-loss seed, and a farthest-point-sampling diversity fill. Dropping the forced representative hurts most, because without it the reasoning model may never see an example of the mode it has been instructed to fix; dropping the diversity fill costs less; dropping the loss seed costs least. Random selection of three episodes from the cluster is worse than any single-rule removal, so the three rules are complementary rather than redundant, and the whole spread is 2.25 points. Middle: silhouette-based selection of the cluster count, which is standard practice and cited as such, beats the best fixed alternative (k = 3) by 0.34 points. The adaptivity is real — D2 shows every k from 2 to 6 selected in at least 15 % of rounds — but its value is small, because the success-rate surface is flat between k = 3 and k = 4. Right: citing three episodes chosen by plain peak-loss rank costs 0.40 points against the three-rule construction, n = 5 buys nothing over n = 3 (which justifies the cap), and citing every failure in the target cluster is worse than citing three, because early rounds carry about forty failures and the target mode is buried. Top-1 is confounded by construction and was not run: bridging requires at least two cited failures.*

---

## F12 — Distribution of the selected cluster count

**Files** `F12_cluster_count_distribution.pdf` / `.png` · **Covers** D2 · **Format** normalised stacked bar, k = 2…6 plus a distinct skipped segment

**What it shows.** Composition is the message, and the totals differ between GridWorld (180 rounds) and the robot settings (100), so the bars are normalised. The skipped segment matters as much as the k segments: between 15 % and 34 % of all rounds never cluster at all.

**Data series and provenance.** `D2`, rows 9–18. Col C = clustered rounds, cols D–H = counts for k = 2…6, col I = rounds skipped because N ≤ 3. Percentages are normalised in the script by (clustered + skipped), which reproduces 180 for each GridWorld setting and 100 for each robot setting — a consistency check on the stated seed counts that passes exactly.

**Caption.** *How many failure modes the method actually discovers, per round. Bars give the share of all rounds selecting each cluster count k, with the rounds that skip clustering entirely (fewer than four remaining failures, so each failure becomes its own cluster) shown hatched. Pooled over the 896 clustered rounds, k = 3 is the mode at 25.1 % and k = 4 a close second at 23.8 %, and every value from 2 to 6 is selected in at least 15 % of rounds. The number of discovered modes varies by round, most often three or four; three is the mode of a broad distribution, not a property of the tasks. The hatched segment is a seam in the method and is reported rather than hidden: in the late rounds, when failures have become rare, the clustering machinery is inactive and the budget is allocated by the fallback rule, which recovers only 31 % of the margin on its own. Total rounds equal seeds × B (180 for GridWorld at 9 seeds, 100 for the robot tasks at 5 seeds), which confirms the stated seed counts.*

---

## F13 — Failures over the budget

**Files** `F13_failures_over_budget.pdf` / `.png` · **Covers** D4 · **Format** line plot, the N ≤ 3 region shaded

**What it shows.** The clustering and the memory do their work in the first two thirds of the budget and are idle at the end. This bounds the method's own mechanism and is the empirical companion to F12's hatched segment.

**Data series and provenance.** `D4_FailureCount`, rows 9–28. Col A = round, col B = mean failure count, col C = the sheet's own N ≤ 3 flag (rounds 18, 19, 20), which drives the ringed markers.

**Limitation, stated on the figure.** Only Push-T image is instrumented. The same curve should be run on GridWorld image and Door image so that each primary setting has one, and the shape may well differ where the initial success rate is higher.

**Caption.** *Mean number of recorded failures per round over the budget, on Push-T image, averaged over five seeds. The count halves by round 8 and falls by an order of magnitude by round 17. Below four remaining failures the clustering sweep is skipped and each failure becomes its own cluster (shaded), which happens in rounds 18 to 20 here and in 15 % to 34 % of rounds across the ten settings (D2). Clustering forty failures into three or four modes is a meaningful operation; clustering five failures is barely one. The descriptor, the clustering and the memory therefore act in the early and middle rounds, and the last rounds effectively run the fallback rule. Read with the budget sweep, where the margin doubles as the budget is halved, this says that the framework front-loads the value of a small budget. It also suggests an extension that is not tested here and is not claimed: stopping the reasoning stack once the failure count drops below the clustering threshold would save most of the per-round cost at no measured loss.*

---

## F14 — Aggregate significance

**Files** `F14_aggregate_significance.pdf` / `.png` · **Covers** S1, GT_SR · **Format** forest plot, ten rows plus two pooled diamonds

**What it shows.** The individual overlaps *and* the systematic direction, which is exactly the argument. A table would give ten numbers and lose the pattern. The figure leads with the conservative reading rather than the flattering one.

**Data series and provenance.** Paired margins from `S1_SignTest` col E, rows 9–18 (identical to `GT_SR` col I minus col J, verified). The horizontal bars are the standard error of the paired difference, computed in the script as √(s.d.²_DISEIL + s.d.²_baseline) / √n_seeds, with n = 9 for GridWorld and 5 for the robot tasks; the standard deviations come from `GT_SR`. The two diamonds are the pooled mean over the ten settings and the mean over the five task means.

**Caption.** *DISEIL attains the higher mean in all ten task-and-modality settings, with a mean margin of 3.71 points over the strongest DAgger-family baseline in each. Treating the settings as paired observations, a sign test and a Wilcoxon signed-rank test both reject a coin-flip ranking at two-sided p = 0.002, which is the smallest p-value attainable with ten pairs. The ten settings are not ten independent experiments: they are five tasks under two observation modalities, and the two modalities of a task share the expert, the reward structure and the reset distribution, so the effective sample size is nearer five than ten. Collapsing to the five task means, the sweep holds at five out of five and a paired t-test rejects at p = 0.014 (t(4) = 4.15). The conservative task-level result is the one to lead with. Horizontal bars are the standard error of the paired difference and are not themselves the test. Lift, at the 100.0 ± 0.0 ceiling, contributes the two smallest margins for the trivial reason that there is no headroom.*

---

## F15 — Cluster purity

**Files** `F15_cluster_purity.pdf` / `.png` · **Covers** D1 · **Format** scatter, silhouette against purity, marker shape by modality

**What it shows.** The absence of a relationship, which only a scatter can show. Geometric separation and semantic purity are uncorrelated across settings, so A10 and D1 are genuinely independent checks rather than two views of one quantity. A bar chart of purity alone would hide the independence result entirely.

**Data series and provenance.** `D1_Cluster_Purity`, rows 9–18. Col C = mean purity, col D = mean distinct root causes per cluster, col E = mean silhouette. Pearson r and p are computed in the script from cols C and E (r = 0.18, p = 0.62).

**Caption.** *Semantic purity of the geometric clusters against their geometric separation, one point per setting, marker shape by observation modality. Purity is the fraction of a cluster's failures sharing the dominant root-cause label; it ranges from 0.78 on Wipe image to 0.93 on Lift state, with a mean of 0.877. The two quantities are uncorrelated (r = 0.18, p = 0.62): a well-separated cluster is not automatically a semantically clean one, and the descriptor is therefore checked twice, once on separation (A10) and once on meaning (D1). The limit of a geometric descriptor is visible in the low-purity corner. The descriptor separates failures by where and how they occur, and recovers root cause only to the extent that configuration determines cause; on Wipe, the same end-effector position can correspond to insufficient contact force, to a missed patch, or to premature termination, and geometry cannot tell them apart. The claim that the discovered modes are semantically meaningful must be qualified with this number and must not be illustrated from the Push-T panel alone. Purity is measured against the reasoning model's own root-cause labels, so it records agreement between two components of the same system, not agreement with ground truth; there is no human-labelled root-cause set, and that circularity is a genuine limitation.*

---

## Not drawable

**D5 (compute).** The sheet has row labels for five settings and no numbers, with the instruction "Run 1 job per task to compute and fill in these matrix." The per-round cost of the reasoning stack is a stated limitation and a reviewer will ask for it. No figure and no number until those jobs run; no placeholder is used.

## Data gaps that these figures expose

1. The full-DISEIL fallback rate per setting is missing, so F5 cannot carry its reference line.
2. D4 is instrumented on Push-T image only, so F13 covers one primary setting instead of three.
3. The A6 display strings must be regenerated from the helper column before any A6 number is printed as text; the figures already bypass them.
4. The D3/A7 bridging contradiction must be resolved by inspecting the prescription logs before F6 is written up.
