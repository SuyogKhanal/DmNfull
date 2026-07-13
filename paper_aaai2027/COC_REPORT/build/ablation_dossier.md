# Ablation dossier for DISEIL

Source of truth: the ablation workbook in `paper_aaai2027/COC_REPORT/ablations_results/` (one `.xlsx`, 24 sheets). Every number below is read from that workbook with pandas. Nothing is invented. Where the workbook is internally inconsistent, the inconsistency is recorded here rather than smoothed over.

Naming: the workbook's method column carries the old internal name. Throughout this dossier that column is DISEIL. The code identifiers used inside the fork are not used in prose.

## 0. Scope, conventions, and what the workbook actually contains

Ten settings: five tasks (GridWorld 5x5, Push-T, Lift, Wipe, Door) under two observation modalities (state, image). A setting is one task under one modality. Seed counts are 9 for GridWorld and 5 for the robot tasks; the workbook's own round accounting confirms this (see D2 below, where clustered rounds plus skipped rounds equal 180 for each GridWorld setting and 100 for each robot setting, that is seeds x B with B = 20).

Three primary settings are analysed in depth here, on the author's instruction: GridWorld image, Push-T state, Door image. The other seven settings are summarised and held for supplementary material.

Delta-SR, used throughout: the change in the policy's success rate on the round-level rollout evaluation.

Two conventions used in all knockout tables:

- `Delta` = ablated mean minus full DISEIL mean, in success-rate points.
- `Margin retained` = (ablated mean - best baseline mean) / (full DISEIL mean - best baseline mean), expressed as a percentage. It answers the question a reviewer actually asks: after removing this component, how much of the advantage over the strongest competing method survives? A value near 100% means the component is decorative. A value near 0% means the component carries the result. A negative value means the ablated system has fallen below the best baseline.

Reference anchors (GT_SR, GT_InfoGain sheets):

| Setting | Best baseline (DAgger family) | DISEIL | Margin | DISEIL info gain | Best-baseline info gain |
|---|---|---|---|---|---|
| GridWorld 5x5 state | 86.8 (ThriftyDAgger) | 89.9 +/- 1.3 | +3.1 | 3.55 | 2.95 |
| GridWorld 5x5 image | 87.1 (ThriftyDAgger) | 89.6 +/- 1.8 | +2.5 | 3.21 | 2.53 |
| Push-T state | 90.7 (Diff-DAgger) | 96.1 +/- 4.5 | +5.4 | 2.81 | 2.36 |
| Push-T image | 89.0 (Diff-DAgger) | 93.9 +/- 4.9 | +4.9 | 2.82 | 2.16 |
| Lift state | 99.2 | 100.0 +/- 0.0 | +0.8 | 2.64 | 2.23 |
| Lift image | 99.6 | 100.0 +/- 0.0 | +0.4 | 2.93 | 2.18 |
| Wipe state | 90.8 (EnsembleDAgger) | 95.5 +/- 6.0 | +4.7 | 2.91 | 2.38 |
| Wipe image | 89.6 (Diff-DAgger) | 95.3 +/- 3.2 | +5.7 | 3.62 | 2.96 |
| Door state | 95.2 (Diff-DAgger) | 98.4 +/- 4.2 | +3.2 | 3.43 | 3.10 |
| Door image | 92.8 (ThriftyDAgger) | 99.2 +/- 3.4 | +6.4 | 3.00 | 2.46 |

### The Lift caveat, stated once and applied everywhere

Lift is at 100.0 +/- 0.0 under full DISEIL in both modalities. There is no headroom and no seed variance. A null result on Lift therefore carries no information about any mechanism, in this dossier or anywhere else. Every aggregate reported below is given with Lift excluded, and where a Lift number is quoted it is quoted only to show that it is uninformative. This is not a hedge invented after the fact: the workbook's own A13 sheet reaches the same conclusion independently, by showing that the memory kernel does come alive on Lift at sigma = 0.02 and yet nothing is visible in the success rate because of the ceiling.

### Three internal inconsistencies in the workbook that must be resolved before publication

1. **A6 display column is stale.** On the A6 sheet the human-readable `mean +/- std` strings disagree with the numeric helper column for eight of ten settings, and the sheet's own `Delta` and `Margin retained` columns are computed from the helper, not from the strings. Example: GridWorld state shows the string 88.8 +/- 1.3 but Delta = -1.4 against a full value of 89.9, which implies 88.5, the helper value. This dossier uses the helper column throughout, because that is what the sheet's own arithmetic uses. The strings must be regenerated before any of A6 is printed.
2. **D3 contradicts the A7 and D3 framing text.** The A7 header says bridging is disabled "on the tasks where bridging operates (Push-T, Lift, Door)" and the D3 hypothesis says bridging should be "exactly 0% on Wipe and GridWorld". The D3 data say otherwise: 30% and 24% bridging on GridWorld, 28% and 27% on Wipe. A7 also shows a measurable effect on those settings (-1.3 on GridWorld image, -2.9 on Wipe image), which is only coherent if bridging did operate there. Either the data are right and the prose is stale, or the runs did something the prose did not intend. The data are the source of truth, so this dossier reports bridging as active on all five tasks, and flags the prose as needing correction.
3. **D5 (compute) is empty.** The sheet has row labels for five settings and no numbers, plus the note "Run 1 job per task to compute and fill in these matrix." The per-round cost of the reasoning stack is a stated limitation of the method and a reviewer will ask for it. It cannot be reported until those jobs run.

One cross-check passes cleanly and should be stated in the report as evidence of internal consistency: the lambda = 0 column of A13 reproduces A1 exactly, in all ten settings (89.4, 89.0, 95.7, 93.1, 99.9, 99.9, 94.4, 94.1, 98.2, 98.0). Setting the memory penalty weight to zero is the same experiment as switching the memory off, and the two independent runs agree to the decimal.

### Summary of every knockout, ranked by damage

Mean over the eight non-Lift settings:

| Knockout | Mean Delta (pts) | Mean margin retained | Reading |
|---|---|---|---|
| A3 clustering off | -4.01 | 11.3% | The load-bearing component |
| A8 fallback only | -3.14 | 31.1% | Structure without reasoning gets a third of the way |
| A6 KAG off | -2.44 | 44.2% | Grounding is worth about half the margin |
| A7 bridging off | -1.24 | 71.7% | Real but modest |
| A4 LLM vs heuristic | -1.08 | 76.7% | Small. Discussed honestly below |
| A5 VLM off | -1.01 | 78.0% | Small. Discussed honestly below |
| A1 memory off | -0.75 | 83.3% | Smallest of all, and this is a problem for the paper's own framing |

Two things follow immediately, and both are uncomfortable. Clustering is the component that carries DISEIL. The cluster memory, which the paper currently presents as a headline equation and as half of the second contribution, is the *weakest* of the seven knockouts. A13 explains why, and the explanation is a scaling bug, not a virtue. Both points are developed below.

---

## 1. A1 — cluster memory off (lambda = 0)

**Motivation.** The memory term is the mechanism that stops the budget from being absorbed by whichever failure mode happens to be dominant in round 1. It applies a recency-discounted Gaussian penalty to clusters whose centroids sit near recently corrected regions, so that near-dominant clusters get rotated over the budget instead of one cluster being corrected twenty times. If that mechanism does nothing, the paper's second contribution is decorative.

**Setup.** lambda is set to 0, disabling the penalty. Everything else is unchanged: same descriptor, same clustering, same prescription pipeline, same budget B, same D = 1. All ten settings.

**Findings.**

| Setting | Full | Ablated | Delta | Margin retained |
|---|---|---|---|---|
| GridWorld image (primary) | 89.6 | 89.0 +/- 1.8 | -0.6 | 76.0% |
| Push-T state (primary) | 96.1 | 95.7 +/- 4.8 | -0.4 | 92.6% |
| Door image (primary) | 99.2 | 98.0 +/- 3.7 | -1.2 | 81.2% |
| GridWorld state | 89.9 | 89.4 +/- 1.6 | -0.5 | 83.9% |
| Push-T image | 93.9 | 93.1 +/- 5.5 | -0.8 | 83.7% |
| Wipe state | 95.5 | 94.4 +/- 6.0 | -1.1 | 76.6% |
| Wipe image | 95.3 | 94.1 +/- 3.6 | -1.2 | 78.9% |
| Door state | 98.4 | 98.2 +/- 4.4 | -0.2 | 93.7% |
| Lift state / image | 100.0 | 99.9 / 99.9 | -0.1 / -0.1 | uninformative |

Mean Delta over the eight non-Lift settings: -0.75 points. Mean margin retained: 83.3%. Every individual Delta is inside one standard deviation of the corresponding full-DISEIL mean.

**Why the result is what it is.** A1 was expected to be the most damaging knockout and it is the least damaging. A13 supplies the mechanism. The Gaussian kernel is applied with a single width sigma = 0.06 on every task, but the tasks do not share a spatial scale. Door has a reset range of about +/- 0.013 m, so typical centroid separations are on the order of 0.01 m and the kernel evaluates to exp(-0.125) ~ 0.88 for a pair of centroids that ought to be treated as distinct: the penalty is applied almost uniformly to every cluster, which is arithmetically close to applying no penalty at all. GridWorld centroids live in grid-cell units, where even sigma = 0.20 gives exp(-1/(2 x 0.04)) ~ 4e-6 between adjacent cells, so the kernel degenerates into an identical-centroid indicator and the only live parameter is the recency discount gamma. On Push-T and Wipe, whose reset distributions are wide relative to sigma = 0.06, the kernel does discriminate, and those are exactly the settings where A13's sigma sweep shows real movement (spread 2.2 to 2.6 points, against 0.1 to 0.2 points on GridWorld, Door and Lift). So the memory mechanism is not weak. The memory mechanism is *mostly switched off by a mis-scaled constant* on three of the five tasks, and the A1 knockout therefore measures the removal of something that was already close to inert on most of the grid.

**Implications.** The honest claim the workbook supports is narrow: with a single global sigma, cluster memory contributes about 0.75 points on average and about 17% of the margin over the best baseline, and its contribution is concentrated where the reset distribution is wide. The claim the paper currently implies, that the memory equation is a principal driver, is not supported. The fix is not a rhetorical one. A per-task sigma set as a fraction of that task's reset range would make the kernel discriminate everywhere, and that is a concrete change to the method, not a caveat to be buried.

**Limitations.** A1 cannot separate "memory does little" from "memory was disabled by a bad constant" on its own. Only the joint reading of A1 and A13's sigma block can, and that joint reading is the honest one.

**Influence on the final framework.** The memory term stays in the framework, because on the two tasks where the kernel is correctly scaled it is worth roughly 1 point and is the only component that produces rotation across near-dominant clusters. It is re-specified with a per-task sigma expressed as a fraction of the reset range, and the paper reports the global-sigma result as the limitation it is.

**Presentation.** Grouped bar chart with three panels (the primary settings), each panel showing best baseline / A1 / full DISEIL, with seed standard-deviation error bars. A table is the wrong choice here because the point is visual and comparative: the A1 bar sits close to the full bar and far above the baseline bar. Do not use a table; the exact decimals do not carry the argument.

*Data series (bar chart):*
```
labels        = ["Best baseline", "Memory off (A1)", "DISEIL"]
GridWorld_img = [87.1, 89.0, 89.6]   err = [1.9, 1.8, 1.8]
PushT_state   = [90.7, 95.7, 96.1]   err = [4.5, 4.8, 4.5]
Door_image    = [92.8, 98.0, 99.2]   err = [2.7, 3.7, 3.4]
```

---

## 2. A2 — uniform-random allocation on the robot tasks

**Motivation.** Stagger, the uniform-random allocation control, is reported in Table 1 for GridWorld only. That leaves the most damaging question unanswered on eight of the ten settings: does a randomly chosen recorded failure per round already match DISEIL? If it does, all of the clustering, memory and reasoning machinery is unnecessary. A2 fills that gap.

**Setup.** Each round corrects one uniformly chosen recorded failure. No descriptor, no clustering, no memory, no reasoning model. Same expert, same budget, same D = 1. Run on the eight robot settings.

**Findings.**

| Setting | Best gate baseline | Diff-DAgger | Stagger (random alloc.) | DISEIL | DISEIL - Stagger |
|---|---|---|---|---|---|
| Push-T state (primary) | 90.7 | 90.7 +/- 4.5 | 82.3 +/- 4.2 | 96.1 | +13.8 |
| Push-T image | 89.0 | 89.0 +/- 4.8 | 81.8 +/- 5.9 | 93.9 | +12.1 |
| Lift state | 99.2 | 99.2 | 97.9 +/- 4.4 | 100.0 | +2.1 |
| Lift image | 99.6 | 99.6 | 96.2 +/- 5.4 | 100.0 | +3.8 |
| Wipe state | 90.8 | 90.4 | 83.9 +/- 5.1 | 95.5 | +11.6 |
| Wipe image | 89.6 | 89.6 | 83.2 +/- 3.5 | 95.3 | +12.1 |
| Door state | 95.2 | 95.2 | 87.1 +/- 4.7 | 98.4 | +11.3 |
| Door image (primary) | 92.8 | 89.2 | 84.0 +/- 6.3 | 99.2 | +15.2 |

Mean Stagger over the eight robot settings: 87.05. Mean gap to DISEIL: 10.25 points.

**Why random allocation fails.** Random replay is not a weak version of DISEIL. It is a different sampling distribution. Sampling failures uniformly reproduces the *frequency* of failure modes in the current policy's failure set, which means the dominant mode is corrected in proportion to how often it occurs, and rare-but-persistent modes are almost never touched inside a budget of twenty. The whole premise of allocation is that the value of a demonstration is not proportional to how often the corresponding failure occurs. A2 is the direct measurement of that premise, and the premise holds by a wide margin.

The finer structure is more interesting than the headline. Stagger does not merely sit "among the baselines". It sits *below all four* uncertainty-gated baselines on Wipe state, Door state, Door image and both Lift settings, and below three of four on Push-T state. Only on Push-T image and Wipe image does it land mid-pack. That is a stronger result than the sheet's own hypothesis anticipated, and it says something the paper should say plainly: an uncertainty gate, however crude, is doing real work in choosing *when* to query, and replacing the gate with random replay costs more than replacing the reasoning stack with a heuristic (compare A4, -1.08). Randomly chosen expert labour is worse than gated expert labour, which is worse than allocated expert labour.

**Implications.** A2 closes the "your win is just failure replay" objection. It also gives the correct control against which A8 (deterministic fallback) should be read: A8 recovers 31% of the margin, A2 recovers none of it and goes backwards.

**Limitations.** Stagger on the robot tasks is our reimplementation, not the authors'; it inherits our expert and our reset distributions. It should be described as a random-allocation control, not as a faithful reproduction of the published system.

**Influence on the final framework.** A2 is the empirical justification for the framework being an *allocation* framework at all. It is the control that belongs next to the main table, not in supplementary.

**Presentation.** Horizontal grouped bar, eight robot settings, four bars per setting (best gate baseline, Diff-DAgger, Stagger, DISEIL). Horizontal because the setting labels are long and there are eight of them. This is one of the few places where all eight robot settings should be shown rather than the three primaries, because the point of A2 is coverage.

*Data series:*
```
settings   = ["Push-T state","Push-T image","Lift state","Lift image","Wipe state","Wipe image","Door state","Door image"]
best_gate  = [90.7, 89.0, 99.2, 99.6, 90.8, 89.6, 95.2, 92.8]
diff_dagger= [90.7, 89.0, 99.2, 99.6, 90.4, 89.6, 95.2, 89.2]
stagger    = [82.3, 81.8, 97.9, 96.2, 83.9, 83.2, 87.1, 84.0]
stagger_sd = [ 4.2,  5.9,  4.4,  5.4,  5.1,  3.5,  4.7,  6.3]
diseil     = [96.1, 93.9,100.0,100.0, 95.5, 95.3, 98.4, 99.2]
```

---

## 3. A3 — clustering off (greedy worst-loss)

This is the load-bearing ablation of the whole study.

**Motivation.** The central claim of DISEIL is that per-step loss identifies *what is informative* but not *what to collect next*, and that the missing step is allocation across failure modes. A3 is the experiment that separates those two things. It keeps the loss signal and removes the mode structure: every round targets the single highest-peak-loss failure, with no clusters, so the memory has nothing to rotate over.

**Setup.** The clustering step is removed. The round corrects the raw highest-loss failure. Descriptor, memory and prescription are all bypassed or made vacuous. All ten settings. Per-demonstration information gain (pre-retrain policy loss on each newly acquired demonstration) is recorded alongside success rate.

**Findings.**

| Setting | Info gain, full | Info gain, ablated | Success, full | Success, ablated | Delta SR | Margin retained | Best baseline |
|---|---|---|---|---|---|---|---|
| GridWorld image (primary) | 3.21 | 3.23 | 89.6 | 87.4 +/- 2.0 | -2.2 | 12.0% | 87.1 |
| Push-T state (primary) | 2.81 | 2.97 | 96.1 | 92.0 +/- 4.7 | -4.1 | 24.1% | 90.7 |
| Door image (primary) | 3.00 | 2.87 | 99.2 | 92.4 +/- 3.9 | -6.8 | **-6.2%** | 92.8 |
| GridWorld state | 3.55 | 3.46 | 89.9 | 86.9 +/- 1.4 | -3.0 | 3.2% | 86.8 |
| Push-T image | 2.82 | 3.00 | 93.9 | 89.3 +/- 5.9 | -4.6 | 6.1% | 89.0 |
| Wipe state | 2.91 | 3.07 | 95.5 | 91.7 +/- 7.1 | -3.8 | 19.1% | 90.8 |
| Wipe image | 3.62 | 3.59 | 95.3 | 90.2 +/- 3.8 | -5.1 | 10.5% | 89.6 |
| Door state | 3.43 | 3.63 | 98.4 | 95.9 +/- 4.9 | -2.5 | 21.9% | 95.2 |
| Lift state | 2.64 | 2.76 | 100.0 | 99.1 +/- 0.4 | -0.9 | uninformative | 99.2 |
| Lift image | 2.93 | 2.88 | 100.0 | 99.5 +/- 0.6 | -0.5 | uninformative | 99.6 |

The two columns move in opposite directions, and that is the entire point.

Information gain is statistically unchanged. Mean change over the eight non-Lift settings is **+0.06** (it goes *up*, not down), and a Wilcoxon signed-rank test over the ten paired means gives p = 0.23. The greedy worst-loss policy is, if anything, slightly better at collecting high-loss demonstrations than DISEIL is, which is unsurprising: greedy worst-loss maximises exactly that quantity by construction, whereas DISEIL sacrifices some of it to spread the budget across modes.

Success rate collapses. Mean Delta over the eight non-Lift settings is **-4.01 points**, Wilcoxon p = 0.002, and mean margin retained is **11.3%**. On Door image the ablated system falls *below* the best baseline (92.4 against 92.8), and on Lift state and Lift image it also nominally falls below, though the Lift numbers are uninformative. Across the ten settings the ablated system beats its best baseline in only seven, by a mean of 0.48 points, which is within noise.

**Why greedy worst-loss fails while still collecting high-loss data.** Peak loss is a property of a single failure trajectory. It is not a property of the *set* of failures the policy is still producing. Under greedy worst-loss, the highest-loss failure in round r is very likely to belong to the same failure mode as the highest-loss failure in round r-1, because the modes that produce the largest loss spikes are the ones the policy has learned least about, and one demonstration does not close a mode. The budget is therefore spent repeatedly inside a single region of the state space. Each of those demonstrations is genuinely informative in isolation, which is why the information-gain column stays high, and each is largely *redundant with the demonstration collected in the previous round*, which is why the success rate does not move. Information gain measured per demonstration has no term for redundancy across demonstrations. Allocation is precisely the term that supplies it.

This licenses a claim that is stronger than "clustering helps": per-demonstration information gain is not a sufficient objective for demonstration selection under a fixed budget, and a method that maximises it without allocating across failure modes converges to the performance of the baselines it was supposed to beat. That is the argument for the whole framework, and A3 is the only experiment in the study that makes it directly.

**Implications.** Every claim in the paper that rests on information gain must be stated as "high gain is necessary and not sufficient", with A3 as the citation. The information-gain table (GT_InfoGain) must not be presented as evidence of the method's advantage on its own, because A3 shows an ablated system with equal or higher gain and a 4-point-lower success rate.

**Limitations.** A3 removes the descriptor and the memory along with the clustering, because they have nothing to operate on once modes are gone. It is a knockout of the allocation *stack*, not of the clustering step in isolation. A cleaner variant would keep the descriptor and replace agglomerative clustering with a random partition of the failures into k groups; that would separate "grouping by geometry" from "grouping at all". It is not in the workbook and should be flagged as future work rather than claimed.

**Influence on the final framework.** A3 is why the clustering step, presented in the method as a generic partition step C instantiated here with agglomerative clustering, is the non-negotiable core. It is also why the framework's contribution is stated as allocation and not as uncertainty estimation.

**Presentation.** A two-axis scatter, one point per setting, with information gain on the x-axis and Delta-SR against full DISEIL on the y-axis, plus a paired arrow from the full-DISEIL point to the A3 point for each setting. Nothing else shows the dissociation as directly: the arrows point right-and-down, which is the finding. A grouped bar would show the success-rate drop and hide the gain, which is the half of the result that matters. Add a second small panel with the paired information-gain values (full vs ablated) as a slope chart, to make the "no change" visually explicit.

*Data series (scatter, primary settings emphasised):*
```
setting            IG_full  IG_abl  SR_full  SR_abl  dSR
GridWorld state      3.55    3.46     89.9    86.9   -3.0
GridWorld image*     3.21    3.23     89.6    87.4   -2.2
Push-T state*        2.81    2.97     96.1    92.0   -4.1
Push-T image         2.82    3.00     93.9    89.3   -4.6
Lift state           2.64    2.76    100.0    99.1   -0.9   (grey, uninformative)
Lift image           2.93    2.88    100.0    99.5   -0.5   (grey, uninformative)
Wipe state           2.91    3.07     95.5    91.7   -3.8
Wipe image           3.62    3.59     95.3    90.2   -5.1
Door state           3.43    3.63     98.4    95.9   -2.5
Door image*          3.00    2.87     99.2    92.4   -6.8
* = primary setting
Annotations: mean dIG = +0.06 (Wilcoxon p = 0.23); mean dSR = -4.01 (Wilcoxon p = 0.002), both excluding Lift.
```

---

## 4. A4 — reasoning model against a fixed heuristic

**Motivation.** The prescription step asks a reasoning model to decide, given the target cluster and the cited failures, what correction to demand and whether to place it as a targeted in-place correction or a bridging placement. That model is the most expensive component in the system and the one a reviewer will attack first. A4 replaces it with a deterministic rule ("always target the dominant cluster representative") operating on the same geometric clusters, so the comparison isolates the reasoning, not the clustering.

**Setup.** Same descriptor, same clustering, same memory, same budget. The reasoning model's output is replaced by the heuristic. All ten settings.

**Findings.**

| Setting | Full | Heuristic | Delta | Margin retained |
|---|---|---|---|---|
| GridWorld image (primary) | 89.6 | 89.1 +/- 1.8 | -0.5 | 80.0% |
| Push-T state (primary) | 96.1 | 94.2 +/- 5.2 | -1.9 | 64.8% |
| Door image (primary) | 99.2 | 97.6 +/- 4.2 | -1.6 | 75.0% |
| GridWorld state | 89.9 | 89.0 +/- 1.5 | -0.9 | 71.0% |
| Push-T image | 93.9 | 93.1 +/- 6.0 | -0.8 | 83.7% |
| Wipe state | 95.5 | 94.3 +/- 6.7 | -1.2 | 74.5% |
| Wipe image | 95.3 | 94.0 +/- 3.6 | -1.3 | 77.2% |
| Door state | 98.4 | 98.0 +/- 4.9 | -0.4 | 87.5% |
| Lift state / image | 100.0 | 99.9 / 99.8 | -0.1 / -0.2 | uninformative |

Mean Delta over the eight non-Lift settings: **-1.08 points**. Mean margin retained: **76.7%**. Every single Delta is smaller than the seed standard deviation of the corresponding full-DISEIL run.

**What this licenses and what it does not.** It licenses this: the reasoning model contributes about one success-rate point on average, roughly a quarter of the margin over the best baseline, consistently in the same direction on all ten settings. It does not license claiming that the reasoning model is the source of DISEIL's advantage. Three quarters of the margin survives its removal. A reader who deletes the reasoning model and keeps the geometric clustering, the memory and the deterministic heuristic still beats every baseline on every setting.

The reason for the small gap is structural and should be said out loud rather than discovered by a reviewer. Clustering is geometric. It uses no output from any foundation model. By the time the reasoning model is called, the hard decision, which region of the failure distribution gets this round's demonstration, has already been made by the descriptor and the memory. The reasoning model is choosing the *form* of the correction inside a region that has already been selected. A component that acts downstream of the decisive step cannot be expected to produce a large effect, and the measurement agrees.

The direction of the effect is nevertheless consistent, and one pattern is worth reporting: the gap is largest on Push-T state (-1.9) and Door image (-1.6), the two settings with the highest bridging share among the primaries (28% and 21%), and smallest on Door state (-0.4) and the GridWorld settings. The heuristic cannot bridge; it can only target the dominant representative. So a plausible reading, which A7 supports, is that most of what the reasoning model buys is the decision to bridge. That reading is testable and is not yet tested: the experiment would be a heuristic that bridges by a fixed rule.

**Implications.** The paper must not claim that language-model reasoning is what makes DISEIL work, and the abstract must not imply it. The defensible claim is that the framework is a demonstration-allocation framework in which the reasoning model supplies the prescription, and that the prescription step is worth about a point. If the cost of the reasoning model is a concern for deployment, A4 is also the answer: the system degrades gracefully to a heuristic that still beats the baselines.

**Limitations.** The heuristic is a strong one. "Always target the dominant cluster representative" is itself an allocation rule that inherits the memory's rotation, since the dominant cluster changes as the memory penalises recently corrected regions. A weaker heuristic would produce a larger gap and a more flattering number, and choosing one would be dishonest.

**Influence on the final framework.** The prescription step stays, and it is described as one instantiation of a generic step that could be filled by a heuristic, by a learned policy, or by a reasoning model. The paper's contribution is relocated to where the evidence puts it, in the allocation, and the reasoning model is presented as the component that turns an allocation into an executable prescription and as the component that makes bridging possible.

**Presentation.** A single dot plot of Delta with error bars, all ten settings on one axis, ordered by Delta, with a shaded band showing +/- 1 seed standard deviation of the full run. This makes the honest point visible at a glance: every dot is negative, every dot is inside the noise band. A table would invite the reader to hunt for the one setting where the gap looks big. A bar chart of raw success rates would make a 1-point gap invisible and would look like special pleading. The dot plot with a noise band is the format that shows both that the effect is real and that it is small.

*Data series (paired dot plot, shared with A5):*
```
setting            A4_delta  A5_delta  seed_sd_full
GridWorld state      -0.9      -0.7        1.3
GridWorld image*     -0.5      -0.6        1.8
Push-T state*        -1.9      -2.0        4.5
Push-T image         -0.8      -0.8        4.9
Lift state           -0.1      -0.2        0.0   (grey)
Lift image           -0.2      -0.3        0.0   (grey)
Wipe state           -1.2      -0.8        6.0
Wipe image           -1.3      -1.4        3.2
Door state           -0.4      -0.4        4.2
Door image*          -1.6      -1.4        3.4
mean excl. Lift      -1.08     -1.01
```

---

## 5. A5 — vision-language model off

**Motivation.** The pipeline passes three frames of each cited failure (start, the peak-loss step, end) to a vision-language model, whose description of what went wrong is then given to the reasoning model. A5 removes those frames and gives the reasoning model only the geometric descriptor and the root-cause labels. It is a distinct knockout from A4: A4 removes the decision, A5 removes the perception that informs it.

**Setup.** The vision-language model is removed from the pipeline. Clustering is unaffected, because clustering is geometric in every run. All ten settings.

**Findings.** Mean Delta over the eight non-Lift settings: **-1.01 points**. Mean margin retained: **78.0%**. Primaries: GridWorld image -0.6 (76.0% retained), Push-T state -2.0 (63.0%), Door image -1.4 (78.1%). Full series in the A4 data block above.

**What this licenses and what it does not.** The visual channel is worth about one point, in the same direction on all ten settings, and every individual gap is inside the seed noise. It is not dead weight, and it is not the reason the method works. Removing it costs almost exactly what removing the reasoning model's decision costs (-1.01 against -1.08), and the two knockouts are close enough that the study cannot rank them.

The mechanism is the same one that explains A4. The visual frames feed the root-cause labelling, which feeds the prescription. They do not feed the clustering, which is where the allocation decision is made. In the earlier version of the method, image-modality runs clustered in a frozen visual-embedding space, and the visual channel would then have been on the critical path. A10 supersedes that design: clustering is geometric for state and image runs alike, and the descriptor is the same 6-D vector in both. Having removed the visual channel from the clustering, we should not be surprised that removing it from the prompt costs one point.

The largest gap is on Push-T state (-2.0), which is a state-modality setting, and that is worth a sentence in the paper because it is counterintuitive. The visual frames are not there to compensate for a missing state vector. They are there to let the model see *why* the T ended up where it did, and Push-T is the task where the same terminal geometry can be reached by several different failure processes (pushing on the wrong face, losing contact, over-rotating). On Door, where the geometry of a failure largely determines its cause, the visual channel adds less.

**Implications.** The V in the pipeline stays, described as informing root-cause quality, and its measured value is reported as approximately one point. It is not to be described as a core component.

**Limitations.** A5 does not test whether a *better* visual model would help more, and it does not test the counterfactual that matters most for the image-modality settings: clustering in a visual space. The workbook, through A10, argues that geometric clustering is the right choice, and A5 is consistent with that, but neither experiment measures a visual-clustering arm directly. That arm was run in an earlier version of the study, produced the frozen-embedding-plus-PCA branch, and is now superseded. It should not be resurrected as evidence.

**Influence on the final framework.** The vision-language model remains the root-cause labeller and does not enter the clustering. The architecture figure must show it feeding the prescription branch only.

**Presentation.** Shares the A4 dot plot. Plot A4 and A5 as two series on the same axis, one marker each per setting, so the reader sees that the two knockouts are indistinguishable in magnitude. Presenting them in separate figures would suggest a distinction the data do not support.

---

## 6. A6 — knowledge graph grounding off

**Motivation.** Equation 10 is the feasibility verification mechanism. The reasoning model proposes a prescription, the constraints are retrieved from the knowledge-augmented graph (workspace bounds, reachability, object and spawn ranges, controller limits, stored as structured key-value knowledge), the prescription is checked against them, and if a constraint is violated the violation is returned to the model, which revises the prescription until it is feasible. If the graph is removed, the model is proposing placements with no knowledge of where the robot can reach, so the proportion of prescriptions that cannot be executed should rise, and the round should more often fall back to the deterministic rule (the expert fails to solve the prescribed episode within the step limit).

Note that feasibility verification against the graph is one of *two* checks in the architecture. The other is the policy-solvability loop: the prescription is rolled out under the current policy, and if the current policy already solves it, the prescription is uninformative and is revised. A6 knocks out the first check only. The second is not ablated in the workbook, and that is a gap in the study.

**Setup.** The rendered knowledge graph (embodiment, geometry, workspace bounds, failure taxonomy, per-mode rules) is dropped from both the vision-language and the reasoning prompts. Fallback rate is recorded alongside success rate.

**Findings.** (Numeric helper column; the display strings on this sheet are stale, see section 0.)

| Setting | Full | Ablated | Delta | Margin retained | Fallback rate |
|---|---|---|---|---|---|
| GridWorld image (primary) | 89.6 | 88.1 | -1.5 | 40.0% | 27.1% |
| Push-T state (primary) | 96.1 | 93.4 | -2.7 | 50.0% | 27.0% |
| Door image (primary) | 99.2 | 96.3 | -2.9 | 54.7% | 34.8% |
| GridWorld state | 89.9 | 88.5 | -1.4 | 54.8% | 29.0% |
| Push-T image | 93.9 | 90.0 | -3.9 | 20.4% | 29.0% |
| Wipe state | 95.5 | 93.8 | -1.7 | 63.8% | 34.2% |
| Wipe image | 95.3 | 92.5 | -2.8 | 50.9% | 27.4% |
| Door state | 98.4 | 95.8 | -2.6 | 18.7% | 23.5% |
| Lift state / image | 100.0 | 99.7 / 99.7 | -0.3 / -0.3 | uninformative | 31.3% / 22.0% |

Mean Delta over the eight non-Lift settings: **-2.44 points**, and mean margin retained **44.2%**. A6 is the third most damaging knockout, and it is more than twice the size of A4 and A5.

**Why grounding matters more than the reasoning model itself.** This is the most informative comparison in the study after A3. Removing the *reasoning* costs one point. Removing the *constraints the reasoning is checked against* costs two and a half. The reasoning model is not the bottleneck; the model's ignorance of the environment is. Without workspace bounds in the prompt, the model proposes placements outside the reachable set or outside the spawn range, the feasibility check has nothing to check against, and the prescription reaches the expert as an episode the expert cannot solve inside the step limit. That round then produces no usable demonstration and the deterministic fallback consumes a unit of a budget of twenty. A fallback rate of 23% to 35% means roughly five to seven of twenty rounds are spent on a fallback rather than on a prescribed correction, which is a direct loss of a quarter to a third of the budget.

The fallback rate is the mechanism, and it should be plotted against the success-rate drop rather than reported as a separate number. The relationship is not clean (Push-T image loses 3.9 points at a 29% fallback rate; Wipe state loses 1.7 points at 34.2%), which is expected: a fallback is not worthless, it is the nearest untried recorded failure, and A8 shows that a system built entirely on that rule still retains 31% of the margin. So a wasted round costs the difference between a prescribed correction and a fallback correction, not the whole round.

**Implications.** The graph earns its authoring cost, and the "bounds the damage" claim is supported: the graph does not make the reasoning model smarter, it stops the reasoning model from producing prescriptions the environment cannot instantiate. That is the honest framing, and it is a better story than a vaguer claim about grounding, because it comes with a mechanism (fallback rate) and a number.

**Limitations.** The per-task graph is authored by hand. The study does not measure how much of its content is load-bearing, so we cannot say whether the workspace bounds alone would recover most of the 2.44 points, or whether the failure taxonomy and per-mode rules matter too. That decomposition is the obvious follow-up. The policy-solvability check is not ablated at all.

**Influence on the final framework.** Feasibility verification against the graph is retained and is presented as the mechanism of Equation 10, described as the retrieve-check-feed-back-revise loop rather than as a scoring function. The fallback rate becomes a diagnostic reported for the method itself, alongside the ablation.

**Presentation.** Scatter with the fallback rate on the x-axis and Delta-SR on the y-axis, one point per setting, Lift points greyed. Overlay the full-DISEIL fallback rate as a vertical reference line if it is available (it is not currently in the workbook and should be added). This is the right format because the claim is causal and two-variable: grounding is removed, feasibility failures rise, success falls. A grouped bar of success rates would hide the mechanism entirely and reduce A6 to "another knockout that costs a couple of points".

*Data series:*
```
setting            fallback_%   dSR
GridWorld state       29.0      -1.4
GridWorld image*      27.1      -1.5
Push-T state*         27.0      -2.7
Push-T image          29.0      -3.9
Lift state            31.3      -0.3   (grey)
Lift image            22.0      -0.3   (grey)
Wipe state            34.2      -1.7
Wipe image            27.4      -2.8
Door state            23.5      -2.6
Door image*           34.8      -2.9
MISSING: full-DISEIL fallback rate per setting (needed for the reference line).
```

---

## 7. A7 — bridging off (targeted-only)

**Motivation.** The prescription step chooses between a targeted in-place correction, which asks the expert to demonstrate the failure episode as configured, and a bridging placement, which asks for a configuration positioned between the target cluster and a region the policy already handles. Bridging is the mechanism by which a prescription can be made *easier* than the failure it addresses, and it is the only part of the prescription that changes the environment configuration rather than just selecting an episode. A7 disables it.

**Setup.** Every prescription becomes a targeted in-place correction. All ten settings.

**Findings.**

| Setting | Full | Ablated | Delta | Margin retained | Bridge share (D3) |
|---|---|---|---|---|---|
| GridWorld image (primary) | 89.6 | 88.3 +/- 2.0 | -1.3 | 48.0% | 24% |
| Push-T state (primary) | 96.1 | 95.0 +/- 5.2 | -1.1 | 79.6% | 28% |
| Door image (primary) | 99.2 | 97.8 +/- 4.8 | -1.4 | 78.1% | 21% |
| GridWorld state | 89.9 | 89.0 +/- 1.8 | -0.9 | 71.0% | 30% |
| Push-T image | 93.9 | 92.7 +/- 5.1 | -1.2 | 75.5% | 19% |
| Wipe state | 95.5 | 94.8 +/- 5.3 | -0.7 | 85.1% | 28% |
| Wipe image | 95.3 | 92.4 +/- 3.7 | -2.9 | 49.1% | 27% |
| Door state | 98.4 | 98.0 +/- 4.5 | -0.4 | 87.5% | 30% |
| Lift state / image | 100.0 | 99.9 / 99.8 | -0.1 / -0.2 | uninformative | 18% / 19% |

Mean Delta over the eight non-Lift settings: **-1.24 points**. Mean margin retained: **71.7%**.

**Reading.** Bridging is chosen in 19% to 30% of accepted prescriptions across all ten settings (mean 24.4%), and turning it off costs about 1.2 points. That is a real effect and a modest one, and the size is what D3 predicts: a component used in a quarter of rounds cannot produce a large aggregate effect unless the rounds where it is used are the decisive ones.

The correlation between bridge share and A7 damage is essentially zero (Pearson r = 0.24, p = 0.58, excluding Lift). Bridging is *not* more valuable on the settings where it is used more often. The largest damage is on Wipe image (-2.9) at a 27% bridge share, and the smallest is on Door state (-0.4) at a 30% bridge share. So the value of bridging depends on *which* rounds it is used in, not on how many. That is a coherent story: bridging matters when the target cluster is far outside anything the policy currently solves, so that a targeted demonstration would be a large distributional jump and a bridged one is a step the policy can actually absorb. Wipe image, the setting with the weakest policy among the primaries and the largest baseline spread (SafeDAgger 69.6, ThriftyDAgger 69.2), is exactly where that condition should bite.

**This is where the workbook contradicts itself.** The A7 and D3 prose asserts that bridging is inapplicable on GridWorld (a corridor) and Wipe (path-randomised), and that it should be exactly 0% there. The D3 data record 24% to 30% bridging on those settings, and A7 records a -2.9-point effect on Wipe image. The data are the source of truth. The prose is stale and must be corrected. The mechanism should be re-derived for the path-randomised and discrete cases and stated, or the runs re-inspected. This must be resolved before the ablation is written up, because a reviewer who reads the method's claim that bridging requires pose randomisation and then sees a 27% bridge share on a path-randomised task will conclude that the implementation does not match the paper.

**Implications.** Bridging survives as a component and is described as a placement rule that is selected roughly a quarter of the time and is worth about a point. It is not a headline.

**Limitations.** Bridging is entangled with the reasoning model in A4: the heuristic cannot bridge, so part of A4's -1.08 is A7's -1.24 measured through a different knockout. The two ablations are not independent and the paper must not add their effects.

**Influence on the final framework.** Bridging stays inside the prescription step, and the method section presents targeted and bridged placement as two options the prescription step chooses between, with the selection frequency reported as a diagnostic.

**Presentation.** Scatter of bridge share (x) against A7 Delta (y), one point per setting, with the near-zero regression line drawn and labelled with its r and p. The figure's job is to make an *honest negative* visible: the more you bridge, the more you lose by not bridging, is a claim the data do not support, and showing the flat cloud is more persuasive than omitting the analysis. Pair it with a small stacked bar of the targeted/bridge split per setting (that is the D3 figure, section 17).

*Data series:* the bridge-share and Delta columns of the table above.

---

## 8. A8 — deterministic fallback only

**Motivation.** The fallback rule, used when the reasoning model fails to produce a feasible prescription after five attempts, is to take the nearest untried recorded failure. A8 promotes that rule to the whole method: no descriptor, no clustering, no memory, no reasoning model, just nearest-untried-failure every round. It is the structured-but-unreasoning control, and it sits between random allocation (A2) and full DISEIL.

**Setup.** Every round takes the nearest untried recorded failure. All ten settings.

**Findings.**

| Setting | Full | Ablated | Delta | Margin retained |
|---|---|---|---|---|
| GridWorld image (primary) | 89.6 | 87.8 +/- 2.1 | -1.8 | 28.0% |
| Push-T state (primary) | 96.1 | 92.5 +/- 5.4 | -3.6 | 33.3% |
| Door image (primary) | 99.2 | 94.8 +/- 4.1 | -4.4 | 31.2% |
| GridWorld state | 89.9 | 87.9 +/- 1.5 | -2.0 | 35.5% |
| Push-T image | 93.9 | 90.2 +/- 5.0 | -3.7 | 24.5% |
| Wipe state | 95.5 | 92.2 +/- 6.8 | -3.3 | 29.8% |
| Wipe image | 95.3 | 90.7 +/- 3.8 | -4.6 | 19.3% |
| Door state | 98.4 | 96.7 +/- 5.2 | -1.7 | 46.9% |
| Lift state / image | 100.0 | 99.7 / 99.9 | -0.3 / -0.1 | uninformative |

Mean Delta over the eight non-Lift settings: **-3.14 points**. Mean margin retained: **31.1%**. The margin-retained figures are remarkably consistent, clustering between 19% and 47%.

**Reading.** The three-way comparison is the point, and it should be presented as a ladder:

- Random allocation (A2, robot settings): mean 87.05, *below* the best baseline on every robot setting, and below all four uncertainty-gated baselines on five of eight.
- Nearest-untried fallback (A8): retains 31% of the margin. Above every baseline on every setting, comfortably.
- Full DISEIL: 100%.

Nearest-untried is a spatial heuristic. It implicitly spreads the budget, because a failure adjacent to one already corrected is less likely to be selected than a distant one, and that is a crude form of the same coverage pressure that clustering plus memory supply deliberately. It gets a third of the way. It cannot get further because "nearest untried" has no notion of *mode*: it will happily walk along a chain of adjacent failures that all belong to the same failure mode, and it has no mechanism for deciding that a mode has been addressed.

This is the strongest available answer to the "the deterministic scaffolding is carrying the method" objection. The scaffolding carries 31%. The clustering carries the rest (A3 retains 11%, which is the same statement viewed from the other side).

**Implications.** The fallback rule is a good fallback, which is why it is the fallback. A6's fallback rates (23% to 35% when grounding is removed) are therefore less catastrophic than they look, and that should be said, because it explains why A6 costs 2.44 points and not 6.

**Limitations.** A8's "nearest" is defined in the same 6-D descriptor space that DISEIL uses, so A8 inherits the descriptor. It is not a descriptor-free control. A2 is.

**Influence on the final framework.** The fallback stays, unchanged. A8 also sets the honest floor for what the reasoning stack must beat, and every claim about the reasoning stack in the paper is calibrated against A8 rather than against the baselines.

**Presentation.** One grouped bar chart carrying A2, A8, A3 and full DISEIL against the best baseline, on the three primary settings. This is the "allocation ladder" figure, and it is the single most important ablation figure in the report. Four ablation arms plus baseline plus full, three panels. It shows in one image that random allocation is below the baseline, that structured-but-unreasoning allocation is a third of the way, that removing modes drops the method to the baseline, and that the full system is on top.

*Data series (allocation ladder, 3 panels):*
```
arms          = ["Random alloc. (A2)","Fallback only (A8)","Clustering off (A3)","Best baseline","DISEIL"]
GridWorld_img = [86.6, 87.8, 87.4, 87.1, 89.6]   # A2 for GridWorld = Stagger from Table 1 (86.6 +/- 2.3)
PushT_state   = [82.3, 92.5, 92.0, 90.7, 96.1]
Door_image    = [84.0, 94.8, 92.4, 92.8, 99.2]
err_GW        = [ 2.3,  2.1,  2.0,  1.9,  1.8]
err_PT        = [ 4.2,  5.4,  4.7,  4.5,  4.5]
err_Door      = [ 6.3,  4.1,  3.9,  2.7,  3.4]
Note: on Door image, A3 (92.4) falls BELOW the best baseline (92.8). Draw the baseline as a horizontal
rule per panel so this is visible.
```

---

## 9. A9 — context-set composition

**Motivation.** The context set S given to the reasoning model contains three cited failure episodes chosen by three different rules: the forced representative of the target cluster, the worst-peak-loss seed, and a farthest-point-sampling fill for diversity. Each rule costs a line of method text and a citation. A9 removes each in turn, replaces the farthest-point fill with a random fill, and adds a floor control of three random episodes from the cluster. The target cluster is fixed by the memory in every arm, so only the *composition* of S varies.

**Setup.** Five arms, all ten settings, kappa = 3 throughout.

**Findings** (success rate, mean over seeds):

| Arm | GridWorld image* | Push-T state* | Door image* | Mean (excl. Lift) | Delta vs full (excl. Lift) | Wilcoxon p |
|---|---|---|---|---|---|---|
| Full S (rep + worst-loss + FPS) | 89.6 +/- 1.8 | 96.1 +/- 4.5 | 99.2 +/- 3.4 | 94.74 | — | — |
| minus worst-loss seed | 89.0 +/- 2.2 | 95.0 +/- 4.8 | 98.2 +/- 3.9 | 94.00 | -0.74 | 0.002 |
| minus farthest-point fill (random fill) | 88.9 +/- 2.2 | 94.4 +/- 5.3 | 97.1 +/- 3.8 | 93.59 | -1.15 | 0.002 |
| minus forced representative | 88.9 +/- 2.2 | 93.5 +/- 4.7 | 96.0 +/- 3.8 | 92.81 | -1.93 | 0.002 |
| Random 3 from cluster | 88.4 +/- 2.2 | 92.9 +/- 5.4 | 95.9 +/- 4.2 | 92.49 | -2.25 | 0.002 |

Friedman across the five arms over the ten settings: chi-square = 31.54, p = 2.4e-06.

**Reading.** The ordering is exactly the one the mechanism predicts, and the ordering is the result, not the individual numbers. Dropping the forced representative hurts most (-1.93), because without it the reasoning model may never see an example of the mode it has been instructed to fix: the memory selects the cluster, and if S is filled by loss rank and diversity alone, the cited episodes can all come from the cluster's periphery. Dropping the diversity fill costs -1.15, because a context set of three episodes that are all near the loss peak describes the mode narrowly and the model prescribes a narrow correction. Dropping the worst-loss seed costs least (-0.74), which makes sense: the representative and the diversity fill between them already span the mode, and the loss peak adds sharpness rather than coverage.

Random 3 (-2.25) is worse than any single-rule removal, which confirms that the three rules are complementary rather than redundant. The construction is not ceremony. The workbook's own "if the gap is ~0 this is ceremony, cut it and drop two citations" test is failed in the direction that keeps the construction.

But note the magnitudes. The whole spread from Full S to Random 3 is 2.25 points, which is smaller than A3 and comparable to A6. The context-set construction is worth roughly as much as the knowledge graph and considerably less than the clustering.

**Implications.** Keep all three rules. Report the ordering, which is interpretable, rather than the individual gaps, which are within seed noise on any single setting.

**Limitations.** kappa is fixed at 3 throughout A9. The interaction between kappa and the composition rules is tested separately in A15 and only along the peak-loss axis.

**Influence on the final framework.** The context set retains its three-rule construction, and the method text justifies each rule by the mechanism A9 exposes, with the forced representative given first because it is the one that matters most.

**Presentation.** Grouped bar, three primary settings, five arms per setting, ordered by mean effect so the monotone ladder is visible. A slope chart from Full S down to Random 3 would also work and would emphasise the ordering, but the bar chart is easier to read against the error bars, which matter here because the gaps are of the same order as the seed spread. Do not use a table: the exact decimals are not the point and would invite over-reading of individual cells.

*Data series:*
```
arms = ["Full S","- worst-loss seed","- FPS (random fill)","- forced rep","Random 3"]
GridWorld_img = [89.6, 89.0, 88.9, 88.9, 88.4]  err = [1.8, 2.2, 2.2, 2.2, 2.2]
PushT_state   = [96.1, 95.0, 94.4, 93.5, 92.9]  err = [4.5, 4.8, 5.3, 4.7, 5.4]
Door_image    = [99.2, 98.2, 97.1, 96.0, 95.9]  err = [3.4, 3.9, 3.8, 3.8, 4.2]
```

---

## 10. A10 — descriptor dimensionality, scored by silhouette

**Motivation.** The descriptor is designed, not learned, and that is stated as a limitation. A reviewer will suspect the feature set was fitted to the result. A10 answers by scoring the descriptor on the thing it is *for*, which is separating failure modes, and by scoring it with a criterion that has nothing to do with success rate: the mean silhouette of the resulting clusters. If the descriptor were over-specified, the silhouette curve would be flat and the choice of 6-D would be arbitrary.

This sheet also supersedes the paper's Equation 7 image branch. Clustering is geometric for every run, state and image alike. The descriptor is 6-D in both: for the robot tasks [p_x, p_y, sin(theta), cos(theta), rho, delta], where rho is task progress and delta is contact distance; for GridWorld [agent cell (2), signed offset to goal (2), progress, Manhattan distance]. The frozen visual-embedding branch with PCA reduction is out of date and must be deleted from the method, along with the N <= 2 PCA edge case and the changing-PCA-dimension workaround in the cluster-memory paragraph. The two columns for one task under the two modalities differ only because an image policy fails in different places, not because the features differ.

**Setup.** Features are removed from and added to the 6-D descriptor. Mean silhouette is measured over all clustering rounds. All ten settings.

**Findings** (mean silhouette):

| Descriptor | GridWorld image* | Push-T state* | Door image* | Mean over 10 settings |
|---|---|---|---|---|
| 2-D, position only | 0.37 | 0.40 | 0.35 | 0.354 |
| 4-D, + orientation | 0.49 | 0.54 | 0.49 | 0.485 |
| 5-D, + task progress | 0.54 | 0.60 | 0.53 | 0.532 |
| **6-D, full descriptor** | **0.58** | **0.64** | **0.56** | **0.567** |
| 8-D, + end-effector velocity | 0.54 | 0.59 | 0.52 | 0.524 |
| 10-D, + gripper state, z-height | 0.48 | 0.52 | 0.47 | 0.469 |
| 12-D, + joint-angle summary | 0.42 | 0.46 | 0.39 | 0.402 |

The 6-D descriptor has the highest silhouette in **all ten settings without exception**. Friedman across the seven dimensionalities: chi-square = 59.52, p = 5.6e-11. Wilcoxon 6-D against 5-D: p = 0.002. Wilcoxon 6-D against 8-D: p = 0.002. The curve is a clean inverted U with a single interior maximum.

**Why both arms of the U are what they are.** Below 6-D the descriptor discards information that distinguishes modes. The step from 2-D to 4-D is the largest in the whole sweep (+0.131 mean silhouette), and it is the step that adds orientation. Position alone cannot separate a Push-T failure in which the T is in the right place with the wrong angle from one in which it is in the right place with the right angle and the pusher lost contact, so those failures collapse into one cluster and the silhouette of that cluster is poor. Adding progress (+0.047) and contact distance (+0.035) each buy less, which is consistent with orientation being the dominant discriminator on these tasks.

Above 6-D the loss is not an information loss. Every added feature carries *some* signal. The loss is geometric: as dimensionality rises, the ratio of the nearest to the farthest pairwise distance approaches one, so all failures look equidistant, the agglomerative merge order becomes arbitrary, and the silhouette falls even though no information was removed. End-effector velocity (8-D) is a weakly informative pair that adds two dimensions of mostly noise to a distance computation over a few dozen points, and by 12-D the joint-angle summary has added six more. The number of failures being clustered is small (D4: 42 in round 1, falling to 2 by round 20), and distance concentration bites hard at small sample sizes. The right way to say this in the paper is that the descriptor is small because the failure sets are small, and that the two are linked by the geometry of distance concentration.

**Implications.** The descriptor is not over-specified, and the choice of 6-D is defensible on a criterion that was not tuned to the reported success rates. This is the answer to the "you fished for the feature set" objection, and it should be given in the method section, not buried in an appendix.

**Limitations.** Silhouette is a criterion of geometric separation, not of semantic correctness. A descriptor could produce beautifully separated clusters that do not correspond to distinct root causes. D1 is the check that closes that hole (mean purity 0.877), and A10 and D1 must be read together. Also, the descriptor remains designed by hand: A10 shows that *this* family of descriptors peaks at 6-D, and says nothing about whether a learned descriptor would do better.

**Influence on the final framework.** The descriptor is fixed at 6-D and is stated as a generic geometric descriptor whose dimensionality is chosen by silhouette. Equation 7's image branch is deleted.

**Presentation.** Line plot, x-axis = descriptor dimensionality, y-axis = mean silhouette, one line per setting (ten thin lines) plus a heavy line for the mean, with a vertical marker at 6-D. The inverted U is the argument and only a line plot shows it. A table would require the reader to find the maximum of each row by eye. Do not fold this into a success-rate figure: the point of A10 is that it scores the descriptor on a criterion independent of success rate, and mixing the two axes would destroy that.

*Data series:*
```
dims = [2, 4, 5, 6, 8, 10, 12]
GridWorld_state = [0.38, 0.51, 0.57, 0.61, 0.56, 0.50, 0.44]
GridWorld_image = [0.37, 0.49, 0.54, 0.58, 0.54, 0.48, 0.42]
PushT_state     = [0.40, 0.54, 0.60, 0.64, 0.59, 0.52, 0.46]
PushT_image     = [0.37, 0.52, 0.56, 0.61, 0.57, 0.50, 0.42]
Lift_state      = [0.32, 0.44, 0.48, 0.52, 0.48, 0.43, 0.37]
Lift_image      = [0.31, 0.42, 0.46, 0.49, 0.46, 0.40, 0.34]
Wipe_state      = [0.33, 0.48, 0.52, 0.55, 0.50, 0.46, 0.39]
Wipe_image      = [0.34, 0.46, 0.51, 0.53, 0.49, 0.44, 0.37]
Door_state      = [0.37, 0.50, 0.55, 0.58, 0.53, 0.49, 0.42]
Door_image      = [0.35, 0.49, 0.53, 0.56, 0.52, 0.47, 0.39]
mean            = [0.354, 0.485, 0.532, 0.567, 0.524, 0.469, 0.402]
Annotate: 6-D is the argmax in 10/10 settings; Friedman chi2 = 59.5, p = 5.6e-11.
```

---

## 11. A11 — budget sweep

**Motivation.** The framework is claimed to work under any fixed budget B, with B = 20 as the validated instance. A11 is the experiment that supports or refutes that claim. It also tests the hypothesis that allocation matters most when demonstrations are scarce, which is the mechanistic reason the method should exist at all.

**Setup.** B is set to 10, 20 and 40. Every method gets the same B. B = 20 reproduces Table 1.

**Findings** (best baseline / DISEIL / margin):

| Setting | B = 10 | B = 20 | B = 40 |
|---|---|---|---|
| GridWorld image* | 79.1 / 85.1 / **+6.0** | 87.1 / 89.6 / +2.5 | 92.5 / 94.0 / +1.5 |
| Push-T state* | 75.5 / 87.9 / **+12.4** | 90.7 / 96.1 / +5.4 | 94.5 / 97.7 / +3.2 |
| Door image* | 78.4 / 92.9 / **+14.5** | 92.8 / 99.2 / +6.4 | 95.7 / 99.5 / +3.8 |
| GridWorld state | 78.0 / 85.4 / +7.4 | 86.8 / 89.9 / +3.1 | 92.2 / 94.1 / +1.9 |
| Push-T image | 75.1 / 86.1 / +11.0 | 89.0 / 93.9 / +4.9 | 93.6 / 96.5 / +2.9 |
| Wipe state | 75.0 / 86.0 / +11.0 | 90.8 / 95.5 / +4.7 | 94.6 / 97.4 / +2.8 |
| Wipe image | 72.5 / 85.5 / +13.0 | 89.6 / 95.3 / +5.7 | 93.9 / 97.3 / +3.4 |
| Door state | 84.5 / 92.0 / +7.5 | 95.2 / 98.4 / +3.2 | 97.2 / 99.1 / +1.9 |
| Lift state | 96.8 / 100.0 / +3.2 | 99.2 / 100.0 / +0.8 | 100.0 / 100.0 / 0.0 |
| Lift image | 97.6 / 100.0 / +2.4 | 99.6 / 100.0 / +0.4 | 100.0 / 100.0 / 0.0 |

Mean margin over the eight non-Lift settings: **+10.35 at B = 10, +4.49 at B = 20, +2.67 at B = 40**. The margin is monotonically decreasing in B in every one of the ten settings.

**The headline the workbook hoped for does not hold, and this must be said.** The A11 sheet proposes the claim "DISEIL at B = 10 matches the best baseline at B = 20, the same policy for half the expert labour". The data refute it. DISEIL at B = 10 is *below* the best baseline at B = 20 in seven of the ten settings (GridWorld state -1.4, GridWorld image -2.0, Push-T state -2.8, Push-T image -2.9, Wipe state -4.8, Wipe image -4.1, Door state -3.2), matches it on Door image (+0.1) and exceeds it only on the two Lift settings, which are at ceiling and uninformative. The halved-labour claim is false and must not appear in the paper. Someone will check it.

**What does hold.** The margin widens as the budget shrinks, in every setting, by roughly a factor of two from B = 20 to B = 10 and again from B = 40 to B = 20. That is the claim the data support, and it is the claim that matters for the framework: allocation is most valuable when there is least to allocate. At B = 40 the margin is 2.67 points on average and Lift has closed entirely, which is the expected saturation. The method is a sample-efficiency method, and it stops paying when samples stop being scarce.

**Why the margin shrinks with B.** With a large budget, even a poorly allocated stream of demonstrations eventually covers the failure distribution, because coverage is a coupon-collector problem and the collector wins if it draws long enough. Allocation buys the *rate* of coverage, not the asymptote. This is also why the effect is largest on the settings with the widest failure distributions (Push-T, Wipe, Door image) and smallest on Lift, where there is essentially one mode and it is covered immediately.

**Implications.** The budget-agnostic framing is supported, with B = 20 as an instance and with the honest statement that the advantage decays as B grows. The correct headline is that the margin roughly doubles when the budget is halved. The half-the-labour headline is retracted.

**Limitations.** Only three budget values, and only the best baseline is reported per budget rather than the full baseline set, so we cannot say whether the *ranking* of baselines is stable across B.

**Influence on the final framework.** B appears as a symbol in the method and in the algorithm, whose loop header reads "for r = 1 to B". The value 20 appears only in the experimental setup. A11 is the evidence for that separation.

**Presentation.** Line plot with B on a log-spaced x-axis (10, 20, 40) and margin over the best baseline on the y-axis, one line per setting, Lift greyed, plus a heavy mean line over the eight non-Lift settings. The decay is the finding and a line plot shows it. Add a second panel plotting absolute success rate for DISEIL and the best baseline against B, so that the saturation of both curves is visible and the reader can see that the margin shrinks because the baseline catches up, not because DISEIL degrades.

*Data series:*
```
B = [10, 20, 40]
margins:
  GridWorld state [ 7.4, 3.1, 1.9]   GridWorld image* [ 6.0, 2.5, 1.5]
  Push-T state*   [12.4, 5.4, 3.2]   Push-T image     [11.0, 4.9, 2.9]
  Lift state      [ 3.2, 0.8, 0.0]   Lift image       [ 2.4, 0.4, 0.0]   (grey)
  Wipe state      [11.0, 4.7, 2.8]   Wipe image       [13.0, 5.7, 3.4]
  Door state      [ 7.5, 3.2, 1.9]   Door image*      [14.5, 6.4, 3.8]
  mean excl Lift  [10.35, 4.49, 2.67]
panel 2 (Push-T state): DISEIL [87.9, 96.1, 97.7]; best baseline [75.5, 90.7, 94.5]
```

---

## 12. A12 — demonstrations per round

**Motivation.** The framework prescribes D = 1 demonstration per round. That commitment is flagged in the paper's own limitations. If D = 3 matched D = 1, the choice would be arbitrary and the method should be generalised over D or the choice justified. A12 justifies it.

**Setup.** D is set to 1, 2 and 3 with the total budget held at B = 20, so D = 1 gives 20 rounds, D = 2 gives 10 rounds and D = 3 gives about 7. Total expert labour is identical in all three arms. All ten settings.

**Findings.**

| Setting | D = 1 | D = 2 | D = 3 | Delta (D3 - D1) |
|---|---|---|---|---|
| GridWorld image* | 89.6 +/- 1.8 | 89.2 +/- 1.9 | 88.9 +/- 2.1 | -0.7 |
| Push-T state* | 96.1 +/- 4.5 | 95.3 +/- 4.9 | 94.2 +/- 5.5 | -1.9 |
| Door image* | 99.2 +/- 3.4 | 97.8 +/- 4.2 | 97.1 +/- 3.6 | -2.1 |
| GridWorld state | 89.9 +/- 1.3 | 89.2 +/- 1.6 | 89.0 +/- 1.4 | -0.9 |
| Push-T image | 93.9 +/- 4.9 | 93.0 +/- 4.9 | 92.3 +/- 5.7 | -1.6 |
| Wipe state | 95.5 +/- 6.0 | 94.8 +/- 6.1 | 93.9 +/- 6.9 | -1.6 |
| Wipe image | 95.3 +/- 3.2 | 94.2 +/- 3.9 | 93.1 +/- 3.5 | -2.2 |
| Door state | 98.4 +/- 4.2 | 97.6 +/- 5.1 | 97.1 +/- 4.6 | -1.3 |
| Lift state | 100.0 | 99.9 | 99.5 | -0.5 (uninformative) |
| Lift image | 100.0 | 99.9 | 99.9 | -0.1 (uninformative) |

D = 1 is best in all ten settings. Mean over the eight non-Lift settings: 94.74 (D = 1), 93.89 (D = 2), 93.20 (D = 3). The decline is monotone in every setting. Friedman: chi-square = 19.54, p = 5.7e-05. Wilcoxon D = 1 against D = 3: p = 0.002. Wilcoxon D = 1 against D = 2: p = 0.002.

**Why one demonstration per round wins at fixed labour.** The number of *rounds* is the number of times the system gets to re-analyse a freshly retrained policy. With D = 3, the second and third demonstrations of a round are prescribed against a policy that no longer exists by the time they are used, because the first demonstration has already changed it. The allocation is made stale by its own execution. This is the same reason the effect is largest exactly where allocation matters most: Door image (-2.1) and Wipe image (-2.2) are the two settings with the largest DISEIL margin over baseline (+6.4, +5.7), and GridWorld (-0.7, -0.9) is where the margin is smallest (+2.5, +3.1). The cost of stale allocation is proportional to the value of allocation. That correspondence is the mechanism, and it is worth stating because it turns A12 from a hyperparameter choice into a second, independent confirmation of the allocation thesis.

**Implications.** D = 1 is the validated instance and the framework is stated over general D. The design is justified rather than arbitrary. The one-demonstration-per-round structure follows the precedent set by Stagger, and A12 shows the precedent is right for this framework as well.

**Limitations.** Retraining after every demonstration is what makes D = 1 affordable in wall-clock terms here; on a task where retraining is expensive, larger D would be forced and A12 quantifies what that costs (about 1.5 points at D = 3). That is the practical reading and it should be given, because it is the question a practitioner will ask.

**Influence on the final framework.** D appears as a symbol in the method. The algorithm's loop collects D demonstrations per round. The setup states D = 1.

**Presentation.** Slope chart, ten lines (one per setting) from D = 1 to D = 3, Lift greyed. Every line goes down and no line crosses another, and a slope chart is the only format that makes both facts visible at once. A grouped bar with 30 bars would obscure the monotonicity. If a single number is wanted in the text, give the Friedman result and the mean decline.

*Data series:*
```
D = [1, 2, 3]
GridWorld state [89.9, 89.2, 89.0]   GridWorld image* [89.6, 89.2, 88.9]
Push-T state*   [96.1, 95.3, 94.2]   Push-T image     [93.9, 93.0, 92.3]
Lift state      [100.0, 99.9, 99.5]  Lift image       [100.0, 99.9, 99.9]  (grey)
Wipe state      [95.5, 94.8, 93.9]   Wipe image       [95.3, 94.2, 93.1]
Door state      [98.4, 97.6, 97.1]   Door image*      [99.2, 97.8, 97.1]
Annotate: Friedman chi2 = 19.54, p = 5.7e-05; D=1 best in 10/10.
```

---

## 13. A13 — memory constants

**Motivation.** The memory term has three constants: gamma (recency discount, paper value 0.6), sigma (kernel width, 0.06) and lambda (penalty weight, 1.0). The paper currently presents "the same constants on every task" as a virtue. A13 tests each constant on every setting and shows that the framing is wrong.

**Setup.** Each constant is swept alone with the others held at their paper values. Five values each. All ten settings. Statistical analysis follows the plan recorded on the sheet: Friedman across the sweep with the ten settings as matched blocks, post-hoc Wilcoxon of the chosen value against each other value, and Holm-Bonferroni within each constant's family. Results are in `build/stats_results.csv`.

**Findings.**

*gamma (0.3, 0.5, 0.6, 0.7, 0.9).* Friedman chi-square = 33.13, p = 1.1e-06. Average ranks: 0.6 -> 1.10 (best), 0.7 -> 2.55, 0.5 -> 2.60, 0.3 -> 4.00, 0.9 -> 4.75. All Holm-corrected p-values against the chosen 0.6 are 0.0078. Spread across the sweep is 1.0 to 1.9 points on every setting including GridWorld and Door. **gamma is live everywhere and 0.6 is significantly best.**

*sigma (0.02, 0.04, 0.06, 0.1, 0.2).* Friedman chi-square = 28.84, p = 8.4e-06. Average ranks: 0.06 -> 1.75 (best), 0.04 -> 2.15, 0.1 -> 2.75, 0.02 -> 3.65, 0.2 -> 4.70. Holm-corrected p against 0.02 is 0.047 and against 0.2 is 0.016, both significant; against 0.04 and 0.1 it is 0.125, **not** significant. The correct statement is that sigma = 0.06 is directionally best but not statistically distinguishable from its neighbours 0.04 and 0.1. Spread by setting: Push-T 2.3 and 2.2, Wipe 2.2 and 2.6, against 0.1 to 0.2 on GridWorld, Door and Lift.

*lambda (0.0, 0.5, 1.0, 2.0, 4.0).* Friedman chi-square = 34.36, p = 6.3e-07. Average ranks: 1.0 -> 1.10 (best), 2.0 -> 2.30, 0.0 -> 2.90, 0.5 -> 3.90, 4.0 -> 4.80. All Holm-corrected p-values against 1.0 are 0.0078. **lambda = 1.0 is significantly best.** Note the non-monotone ordering: lambda = 0 (no memory) ranks *better* than lambda = 0.5. A half-strength penalty is worse than no penalty at all, which is a real and slightly awkward finding, and the natural reading is that a weak penalty perturbs the cluster ranking without being strong enough to force rotation, so it produces neither the clean greedy behaviour of lambda = 0 nor the rotation of lambda = 1.

**The sigma finding, stated as the limitation it is.** sigma is inert on GridWorld, Door and Lift, for three different reasons, and only one of them is benign.

- GridWorld: centroids are in grid-cell units. Even sigma = 0.20 gives exp(-1/(2 x 0.04)) ~ 4e-6 for adjacent cells, so the kernel is an identical-centroid indicator at every swept sigma. Genuinely inert, and the recency discount gamma is the only live parameter of the memory on this task.
- Door: the reset range is +/- 0.013 m, so typical centroid separations are about 0.01 m. Even sigma = 0.02 gives exp(-0.125) ~ 0.88, still saturated. Inert across the swept range. It would take sigma ~ 0.005 to make the kernel discriminate.
- Lift: the kernel *does* come alive at sigma = 0.02 (reset +/- 0.03 m gives exp(-0.78) ~ 0.46, real discrimination), but DISEIL is at 100.0 +/- 0.0 and nothing is visible. Flat by ceiling, not by degeneracy.

The Lift observation is the one that matters, because it proves the point. If sigma = 0.02 makes the kernel discriminate on a task with a +/- 0.03 m reset range, then sigma = 0.06 is mis-scaled for every task whose reset range is narrower than that, which is Door and (in its own units) GridWorld. The memory is not a constant that happens to transfer. It is a constant that happens to be *in range* on Push-T and Wipe and out of range elsewhere. A per-task sigma expressed as a fraction of that task's reset range would make the memory function on all five tasks. This is a limitation of the current instantiation and it must be reported as one. It also explains A1 completely.

**Cross-check.** The lambda = 0 column reproduces A1 exactly on all ten settings. The two experiments were run independently and agree to the decimal. Report this.

**Implications.** Two of the three constants are significantly best at their paper values. The third is directionally best, is not distinguishable from its neighbours, and is mis-scaled on three of five tasks. The paper reports all three honestly and re-specifies sigma per task.

**Limitations.** The sweep is one constant at a time. Interactions between gamma and sigma are not measured, and the natural interaction (a narrow kernel with a slow discount behaves differently from a wide kernel with a fast one) is exactly the one a per-task sigma would change.

**Influence on the final framework.** sigma becomes a per-task quantity defined as a fraction of the reset range. gamma and lambda stay at 0.6 and 1.0 and are reported with their Friedman and Holm statistics.

**Presentation.** Three-panel line plot, one panel per constant, x-axis = constant value, y-axis = success rate, one line per setting, Lift greyed, with the paper value marked by a vertical rule. The panels immediately show the shape of the story: gamma has a peak in every panel line, lambda has a peak with a dip at 0.5, and sigma has a peak on Push-T and Wipe and three flat lines. Beneath the figure, a small table is warranted here (and only here) because the exact Holm-corrected p-values are load-bearing for the "directionally best but not statistically distinguishable" statement about sigma, and that phrase must be backed by the number.

*Data series (lines, per setting; paper value in bold position):*
```
gamma  = [0.3, 0.5, 0.6, 0.7, 0.9]
  GW_st [89.1, 89.8, 89.9, 89.8, 88.8]   GW_im* [88.6, 89.5, 89.6, 89.5, 88.7]
  PT_st*[94.6, 95.8, 96.1, 95.7, 94.5]   PT_im  [92.3, 93.5, 93.9, 93.8, 92.3]
  Lf_st [100.0, 99.9, 100.0, 99.8, 99.5] Lf_im  [99.0, 100.0, 100.0, 99.8, 99.6]  (grey)
  Wp_st [94.2, 95.2, 95.5, 95.2, 94.0]   Wp_im  [93.7, 94.7, 95.3, 94.9, 93.5]
  Dr_st [97.5, 98.1, 98.4, 98.2, 97.3]   Dr_im* [97.4, 98.6, 99.2, 98.8, 97.3]
sigma  = [0.02, 0.04, 0.06, 0.1, 0.2]
  GW_st [89.9, 89.9, 89.9, 89.9, 89.8]   GW_im* [89.5, 89.6, 89.6, 89.6, 89.6]
  PT_st*[94.5, 95.8, 96.1, 95.4, 93.8]   PT_im  [92.4, 93.4, 93.9, 93.2, 91.7]
  Lf_st [99.9, 100.0, 100.0, 100.0, 99.9] Lf_im [100.0, 100.0, 100.0, 100.0, 99.9] (grey)
  Wp_st [94.2, 95.2, 95.5, 94.8, 93.3]   Wp_im  [93.7, 94.7, 95.3, 94.4, 92.7]
  Dr_st [98.4, 98.4, 98.4, 98.3, 98.2]   Dr_im* [99.1, 99.2, 99.2, 99.2, 99.0]
lambda = [0.0, 0.5, 1.0, 2.0, 4.0]
  GW_st [89.4, 89.0, 89.9, 89.5, 88.9]   GW_im* [89.0, 88.8, 89.6, 89.3, 88.8]
  PT_st*[95.7, 94.4, 96.1, 95.6, 94.3]   PT_im  [93.1, 92.4, 93.9, 93.4, 92.0]
  Lf_st [99.9, 99.9, 100.0, 99.9, 99.7]  Lf_im  [99.9, 100.0, 100.0, 100.0, 99.8] (grey)
  Wp_st [94.4, 93.9, 95.5, 94.9, 93.9]   Wp_im  [94.1, 93.2, 95.3, 94.5, 93.3]
  Dr_st [98.2, 97.7, 98.4, 98.1, 97.1]   Dr_im* [98.0, 97.4, 99.2, 98.4, 97.0]

Companion table (exact numbers matter here):
constant | value | avg rank | Holm p vs chosen | Friedman chi2 | Friedman p
gamma  | 0.3 | 4.00 | 0.0078 | 33.13 | 1.1e-06
gamma  | 0.5 | 2.60 | 0.0078 |       |
gamma  | 0.6 | 1.10 |   —    |       |
gamma  | 0.7 | 2.55 | 0.0078 |       |
gamma  | 0.9 | 4.75 | 0.0078 |       |
sigma  | 0.02| 3.65 | 0.047  | 28.84 | 8.4e-06
sigma  | 0.04| 2.15 | 0.125 (ns) |   |
sigma  | 0.06| 1.75 |   —    |       |
sigma  | 0.1 | 2.75 | 0.125 (ns) |   |
sigma  | 0.2 | 4.70 | 0.016  |       |
lambda | 0.0 | 2.90 | 0.0078 | 34.36 | 6.3e-07
lambda | 0.5 | 3.90 | 0.0078 |       |
lambda | 1.0 | 1.10 |   —    |       |
lambda | 2.0 | 2.30 | 0.0078 |       |
lambda | 4.0 | 4.80 | 0.0078 |       |
```

---

## 14. A14 — cluster-count selection

**Motivation.** The cluster count k is chosen per round by maximum mean silhouette, with k_max = max(2, min(6, N-1)). That is standard practice and is cited as such. A14 asks whether the adaptive choice earns its place against a constant.

**Setup.** Silhouette selection is replaced by a fixed k in {2, 3, 4, 5}. All ten settings.

**Findings.**

| Arm | GridWorld image* | Push-T state* | Door image* | Mean (excl. Lift) | Delta vs silhouette |
|---|---|---|---|---|---|
| Silhouette (adaptive) | 89.6 +/- 1.8 | 96.1 +/- 4.5 | 99.2 +/- 3.4 | 94.74 | — |
| fixed k = 3 | 89.5 +/- 2.2 | 96.0 +/- 5.3 | 98.7 +/- 4.1 | 94.40 | **-0.34** |
| fixed k = 4 | 89.5 +/- 2.1 | 95.5 +/- 4.5 | 98.4 +/- 4.0 | 94.10 | -0.64 |
| fixed k = 2 | 88.6 +/- 1.9 | 94.6 +/- 4.5 | 97.5 +/- 4.0 | 93.47 | -1.26 |
| fixed k = 5 | 88.6 +/- 2.2 | 94.0 +/- 5.1 | 96.9 +/- 3.8 | 93.20 | -1.54 |

Friedman: chi-square = 30.63, p = 3.6e-06. Wilcoxon silhouette against fixed k = 3: p = 0.002, but the effect is **0.34 points**.

**The honest reading.** Silhouette selection is significantly better than fixed k = 3 in the paired sense (it wins in every setting, which is what a Wilcoxon over ten paired means detects), and the size of the win is a third of a point, which is a tenth of the seed standard deviation on most settings. The workbook's own hypothesis was that silhouette should *match* the best fixed k rather than beat it, and that is essentially what happened. Fixed k = 3 recovers 91% of the margin that silhouette selection achieves over fixed k = 5.

Cross-read with D2. Silhouette does not simply always pick 3. Pooled over all settings and rounds, k = 2 is chosen 17.5% of the time, k = 3 25.1%, k = 4 23.8%, k = 5 18.6% and k = 6 15.0%. The distribution is broad with a mode at 3, which means the adaptivity is real. But A14 shows that the *value* of that adaptivity is small, because the success-rate surface is flat between k = 3 and k = 4. Both facts are true and both should be reported: the method really does discover different numbers of modes in different rounds, and it would lose only a third of a point by not bothering.

**Why k = 2 and k = 5 are both worse.** Too few clusters merges distinct failure modes, so the memory penalises a merged cluster and suppresses correction of a mode that was never addressed. Too many clusters splits a single mode across several clusters, so the memory fails to recognise that a mode has been corrected and rotation is diluted across fragments of the same region. The flat optimum between 3 and 4 corresponds to the number of modes these tasks actually have, and D2's spread over 2 to 6 is the round-to-round variation in how many modes the current policy still exhibits.

**Implications.** Silhouette-based k selection is standard practice and is presented as such, with a citation, in the background rather than as a contribution. A14 justifies keeping it and honestly reports that a fixed k = 3 would cost a third of a point.

**Limitations.** A14 fixes k across all rounds within a run. It does not test a schedule (large k early when there are many failures, small k late), which D4 suggests could match the adaptive choice at lower cost.

**Influence on the final framework.** The silhouette criterion stays and is explicitly flagged as standard practice with a citation, which is what the supervisor asked for. It is not claimed as novelty.

**Presentation.** Grouped bar, three primary settings, five arms. Alternatively fold A14 and A15 into one two-panel figure, since both answer "does the machinery in this step earn its cost" and both have the same honest answer of "yes, by about half a point". Do not use a table, because the differences are within noise and a table would invite over-reading.

*Data series:*
```
arms = ["Silhouette","k=2","k=3","k=4","k=5"]
GridWorld_img = [89.6, 88.6, 89.5, 89.5, 88.6]  err = [1.8, 1.9, 2.2, 2.1, 2.2]
PushT_state   = [96.1, 94.6, 96.0, 95.5, 94.0]  err = [4.5, 4.5, 5.3, 4.5, 5.1]
Door_image    = [99.2, 97.5, 98.7, 98.4, 96.9]  err = [3.4, 4.0, 4.1, 4.0, 3.8]
```

---

## 15. A15 — number of cited episodes and the selection rule

**Motivation.** A9 varied the *composition* of the context set at kappa = 3. A15 varies the *number* of cited episodes jointly with the selection rule, and it contains the one comparison that could have deleted a component of the method: top-3-by-peak-loss against the full three-rule construction. If they tied, the construction would be over-engineering and should be cut.

**Setup.** Instead of the three-rule set, n episodes are cited from the same target cluster, selected by plain peak-loss rank (top-n), by the current rule, or at random. The target cluster is fixed by the memory in every arm. All ten settings.

**Findings.**

| Arm | n | GridWorld image* | Push-T state* | Door image* | Mean (excl. Lift) | Delta vs full S |
|---|---|---|---|---|---|---|
| Full S (rep + worst-loss + FPS) | 3 | 89.6 +/- 1.8 | 96.1 +/- 4.5 | 99.2 +/- 3.4 | 94.74 | — |
| Top-1 by peak loss | 1 | not computed | not computed | not computed | — | — |
| Top-2 by peak loss | 2 | 89.0 +/- 2.2 | 94.5 +/- 5.4 | 97.7 +/- 3.6 | 93.76 | -0.97 |
| Top-3 by peak loss | 3 | 89.5 +/- 2.1 | 95.7 +/- 4.8 | 98.4 +/- 3.6 | 94.34 | **-0.40** |
| Top-5 by peak loss | 5 | 89.1 +/- 2.1 | 95.4 +/- 5.6 | 98.3 +/- 3.8 | 94.27 | -0.46 |
| All failures in target cluster | N | 88.8 +/- 2.2 | 93.9 +/- 4.9 | 97.1 +/- 4.2 | 93.34 | -1.40 |
| Random 3 from target cluster | 3 | 88.7 +/- 2.2 | 93.9 +/- 4.5 | 96.7 +/- 3.6 | 93.14 | -1.60 |

Friedman: chi-square = 39.31, p = 2.1e-07.

Top-1 was not computed, and the sheet gives the reason: n = 1 makes bridging impossible, because bridging needs at least two cited failures to define a placement between a failing region and a solved one. The arm is confounded by construction and should be dropped from the paper rather than run.

**Reading.** The key comparison, top-3-by-loss against full S, gives -0.40 points. That is a real gap in the paired sense (full S wins in every setting) and it is a small one. The three-rule construction earns about four tenths of a point over the simplest possible selection rule at the same n. The workbook's own framing was that a null here would be "a gift", permitting the construction to be cut. It is not a null, but it is close enough that the construction must be described modestly, and the two citations that support it (farthest-point sampling for the diversity fill) are justified by the mechanism rather than by the effect size.

The rest of the curve is clean and interpretable. n = 2 costs a point relative to n = 3. n = 5 gives nothing over n = 3 (-0.46 against -0.40, a difference of six hundredths of a point), which is the direct justification of the kappa = 3 cap. Citing *all* failures in the target cluster is worse than citing three (-1.40), and that is the prompt-bloat effect: D4 shows early rounds carry about 42 failures, and a context set of forty episodes buries the target mode in detail the model cannot weigh. Random 3 is the floor (-1.60), and it is worse than any principled selection at any n, which confirms that selection matters even when the target cluster is already fixed.

**Implications.** kappa = 3 is justified against both a smaller and a larger set. The selection rule matters, and the specific three-rule construction is worth about half a point over a loss-ranked top-3. Report the effect size and do not oversell.

**Limitations.** A15 and A9 both hold the target cluster fixed by the memory, so neither can say anything about the interaction between context-set construction and cluster selection. Top-1 remains unmeasured and unmeasurable in the current design.

**Influence on the final framework.** kappa = 3 stays. The three-rule construction stays, described accurately as buying a modest improvement over a loss-ranked top-3, with the mechanism (forced representative guarantees the model sees the mode it is instructed to fix) given as the reason.

**Presentation.** Line plot with n on the x-axis for the top-n family (2, 3, 5, and N as a right-hand category), one line per primary setting, with full S and random 3 drawn as horizontal reference rules. The plateau from n = 3 to n = 5 and the fall at n = N is the shape of the argument. Alternatively a grouped bar sharing a figure with A14. Do not table it.

*Data series:*
```
n_axis = [2, 3, 5, "all"]
top_n:  GridWorld_img [89.0, 89.5, 89.1, 88.8]
        PushT_state   [94.5, 95.7, 95.4, 93.9]
        Door_image    [97.7, 98.4, 98.3, 97.1]
reference lines (Full S):   GW 89.6 | PT 96.1 | Door 99.2
reference lines (Random 3): GW 88.7 | PT 93.9 | Door 96.7
Note: Top-1 omitted (confounded: bridging requires >= 2 cited failures).
```

---

## 16. D1 — cluster purity

**Motivation.** A10 shows the geometric descriptor produces well-separated clusters. Separation is not meaning. D1 asks whether a geometric cluster corresponds to a single root cause, by measuring the fraction of a cluster's failures that share the reasoning model's dominant root-cause label. This is the check that stops the "coherent, well-separated modes" claim from being a statement about geometry alone.

**Setup.** For each setting, mean purity across clusters and the mean number of distinct root causes per cluster, alongside the mean silhouette.

**Findings.**

| Setting | Mean purity | Mean root causes per cluster | Mean silhouette |
|---|---|---|---|
| Lift state | 0.93 | 1.31 | 0.52 |
| Lift image | 0.92 | 1.43 | 0.49 |
| GridWorld state | 0.91 | 1.38 | 0.61 |
| Push-T state* | 0.91 | 1.35 | 0.64 |
| Push-T image | 0.90 | 1.30 | 0.61 |
| GridWorld image* | 0.89 | 1.62 | 0.58 |
| Door state | 0.86 | 1.71 | 0.58 |
| Door image* | 0.84 | 1.86 | 0.56 |
| Wipe state | 0.83 | 1.78 | 0.55 |
| Wipe image | 0.78 | 1.91 | 0.53 |

Mean purity 0.877, range 0.78 to 0.93. State settings average 0.888, image settings 0.866.

Two correlations, both informative. Purity against the number of root causes per cluster: r = -0.92, p = 0.0002, which is close to arithmetic and confirms the two columns measure the same thing from opposite sides. Purity against silhouette: **r = 0.18, p = 0.62**. Geometric separation and semantic purity are essentially uncorrelated across settings.

**Why the near-zero purity/silhouette correlation is the interesting result.** It says that a well-separated cluster is not automatically a semantically clean one, and it therefore says that A10 and D1 are genuinely independent checks rather than two views of the same quantity. Push-T state has the best silhouette (0.64) and a purity of 0.91. Lift state has one of the worst silhouettes (0.52) and the best purity (0.93), because Lift has essentially one failure mode and any partition of it is pure by default. Wipe image has a middling silhouette (0.53) and the worst purity (0.78), because Wipe failures at a given geometric location genuinely have several causes: the same end-effector position can correspond to insufficient contact force, to a missed patch of the surface, or to premature termination, and geometry cannot tell them apart.

That is the honest limit of a geometric descriptor, and it should be stated: the descriptor separates failures by *where and how* they occur, and it separates root causes only to the extent that root cause is determined by configuration. On tasks where it is not, purity falls to 0.78 and the clusters mix causes. The method still works on Wipe (margin +4.7 and +5.7), which suggests that a cluster mixing two causes at the same location is still a useful unit of allocation, because a demonstration at that location addresses both. But the "semantically meaningful modes" claim must be qualified with the number, and it must not be illustrated only with the Push-T panel.

**Implications.** Report the range (0.78 to 0.93), not the maximum. Qualify the mode-coherence claim with Wipe.

**Limitations.** Purity is measured against the reasoning model's own root-cause labels, so it measures agreement between two components of the same system, not agreement with ground truth. There is no human-labelled root-cause set. This is a genuine circularity and it must be admitted.

**Influence on the final framework.** The clustering step stays geometric, and the paper states its limit: geometry recovers root cause only where configuration determines cause. This is also the strongest argument for the vision-language model's continued presence, since the visual channel is what supplies root cause where geometry cannot.

**Presentation.** Scatter, silhouette on the x-axis and purity on the y-axis, one point per setting, marker shape by modality, labelled points, with the near-flat regression line drawn and its r annotated. The finding is the absence of a relationship and only a scatter can show an absence. A bar chart of purity alone would hide the independence result entirely.

*Data series:* the table above, all three columns.

---

## 17. D2 — distribution of the selected cluster count

**Motivation.** If silhouette almost always picks k = 3, the adaptivity is theatre and A14 is already decided. D2 measures the distribution directly.

**Setup.** For each setting, the number of rounds selecting each k in {2..6}, and the number of rounds where clustering was skipped because N <= 3 (each failure becomes its own cluster).

**Findings.**

| Setting | Clustered rounds | k=2 | k=3 | k=4 | k=5 | k=6 | Skipped (N<=3) | Total rounds |
|---|---|---|---|---|---|---|---|---|
| GridWorld state | 149 | 27 | 35 | 38 | 26 | 23 | 31 (17%) | 180 |
| GridWorld image* | 143 | 19 | 39 | 35 | 27 | 23 | 37 (21%) | 180 |
| Push-T state* | 85 | 13 | 23 | 26 | 15 | 8 | 15 (15%) | 100 |
| Push-T image | 71 | 13 | 26 | 25 | 5 | 2 | 29 (29%) | 100 |
| Lift state | 69 | 21 | 11 | 16 | 2 | 19 | 31 (31%) | 100 |
| Lift image | 77 | 13 | 14 | 17 | 22 | 11 | 23 (23%) | 100 |
| Wipe state | 81 | 12 | 19 | 22 | 16 | 12 | 19 (19%) | 100 |
| Wipe image | 75 | 13 | 21 | 13 | 16 | 12 | 25 (25%) | 100 |
| Door state | 66 | 12 | 18 | 10 | 14 | 12 | 34 (34%) | 100 |
| Door image* | 80 | 14 | 19 | 11 | 24 | 12 | 20 (20%) | 100 |
| **Pooled** | **896** | **157 (17.5%)** | **225 (25.1%)** | **213 (23.8%)** | **167 (18.6%)** | **134 (15.0%)** | 264 | 1160 |

The total-rounds column is a consistency check and it passes: 180 = 9 seeds x 20 rounds for each GridWorld setting, 100 = 5 seeds x 20 rounds for each robot setting. The seed counts stated in the setup are confirmed by the round accounting.

**Reading.** k = 3 is the mode at 25.1% of clustered rounds, and k = 4 is a close second at 23.8%. Every value from 2 to 6 is selected in at least 15% of rounds. The adaptivity is real, and the figure that shows k = 3 in the paper is showing the mode of a broad distribution, not a discovery. That must be said plainly, because the current framing invites the reader to think 3 is *the* number of failure modes, and it is not: it is the most common number of modes the policy still exhibits when clustering runs.

The skipped column matters as much as the k column. Between 15% and 34% of all rounds never cluster at all, because fewer than four failures remain. Combined with D4, this says the clustering machinery is active in early and middle rounds and inactive in late ones. The allocation story is a story about the first two thirds of the budget. That is a seam in the method and it should be reported rather than left for a reviewer to find. It is also a design opportunity: if clustering is inactive in the last third of the budget, the last third of the budget is being allocated by the fallback rule, which A8 shows recovers only 31% of the margin.

**Implications.** The claim is "the number of discovered modes varies by round, most often three or four", not "there are three modes".

**Limitations.** D2 records the selected k, not whether the selected k was correct. There is no ground-truth mode count.

**Influence on the final framework.** Silhouette selection is retained, and the paper reports the distribution rather than a single number. The N <= 3 path is documented as a real and frequent branch, not an edge case.

**Presentation.** Stacked bar, one bar per setting, segments for k = 2..6 plus a distinct segment for the skipped rounds, normalised to total rounds so the skipped fraction is visible. Stacked because the composition is the message and the totals differ between GridWorld (180) and the robot settings (100). Do not use a heatmap, because the skipped category is not on the same ordinal scale as k.

*Data series:* the table above, as counts, normalised per row by the total-rounds column.

---

## 18. D3 — bridge and targeted split

**Findings.** Bridging share of accepted prescriptions: GridWorld state 30%, GridWorld image 24%, Push-T state 28%, Push-T image 19%, Lift state 18%, Lift image 19%, Wipe state 28%, Wipe image 27%, Door state 30%, Door image 21%. Mean 24.4%, range 18% to 30%.

**Reading and the contradiction.** The measured range matches the workbook's expectation for the pose-randomised tasks. It contradicts the workbook's expectation of exactly 0% on Wipe and GridWorld, where 24% to 30% of prescriptions are recorded as bridged. See section 0, item 2. This is the single most urgent thing to resolve in the whole workbook, because it is a mismatch between what the method section claims the mechanism requires and what the implementation did. Resolve it by inspecting the prescription logs for a GridWorld run before writing the bridging paragraph.

**Influence on the final framework.** D3 bounds what A7 can show, and it is the reason A7's effect is modest. Report the split so that the reader can see that a component used a quarter of the time produced a 1.2-point effect, which is proportionate.

**Presentation.** Horizontal stacked bar (targeted / bridge), one bar per setting, 100% width. Small, and placed adjacent to the A7 scatter so the two are read together.

---

## 19. D4 — failures per round

**Setup.** Mean failure count per round over the budget, Push-T image, averaged over 5 seeds. Only one setting is instrumented.

**Findings.** N by round r = 1..20: 42, 38, 35, 31, 29, 26, 24, 21, 18, 16, 14, 11, 9, 7, 6, 5, 4, 3, 3, 2. The count falls by half by round 8 and by an order of magnitude by round 17. Three rounds (18, 19, 20) have N <= 3 and therefore skip the clustering sweep. Five rounds have N <= 5.

**Reading.** The decline is the intended behaviour of the system and it also bounds the system's own mechanism. Clustering forty failures into three or four modes is a meaningful operation. Clustering five failures into three modes is barely one. So the descriptor, the clustering and the memory do their real work in the first twelve to fifteen rounds, and the last five rounds are effectively running the fallback rule on a handful of remaining failures. D2 confirms this across all settings, where 15% to 34% of rounds skip clustering.

This has a design consequence worth stating. The budget's marginal value is highest early (A11: the margin doubles when B is halved) and the method's machinery is most active early. Those two facts are consistent, and together they suggest that the honest description of DISEIL is a method that front-loads the value of a small budget. It also suggests the obvious extension, which is to stop the reasoning stack once N drops below the clustering threshold and spend the remaining rounds on the fallback, saving most of the per-round reasoning cost at no measured loss. The workbook does not test that, so it is future work and not a claim.

**Limitations.** One setting only. D4 should be run on GridWorld image and Door image before the report is submitted, because the three primary settings should each have a failure-count curve and the shape may differ where the initial success rate is higher (Door image starts far above Push-T image).

**Presentation.** Line plot, round on the x-axis, mean failure count on the y-axis, with the N <= 3 region shaded and the three skipped rounds marked. Overlay the round-level success rate on a secondary axis if it is available, because the two curves are mirror images and the pairing makes the point in one figure. Currently the success trajectory per round is not in the workbook.

*Data series:*
```
round = [1..20]
N     = [42,38,35,31,29,26,24,21,18,16,14,11,9,7,6,5,4,3,3,2]   # Push-T image, 5-seed mean
shade y < 3.5; mark rounds 18,19,20 as "clustering skipped"
MISSING: same curve for GridWorld image and Door image.
```

---

## 20. D5 — compute

**Status: not computed.** The sheet lists five settings (Door state, Push-T image, Wipe image, Door image, GridWorld image) and contains no numbers, with the instruction "Run 1 job per task to compute and fill in these matrix."

The per-round cost of the reasoning stack is already listed as a limitation of the method, and a reviewer will ask for the number. It cannot be reported until those five jobs run. Required columns: baseline seconds per round, DISEIL seconds per round, vision-language tokens per round, reasoning tokens per round, overhead multiplier, and the graph's token contribution. Nothing about compute can be written until this sheet is filled, and no placeholder number should be used.

**Presentation, once the data exist.** A table, because exact numbers matter for a cost claim and because a reader will want to divide the token count by their own price per token. This is one of only two places in the whole dossier where a table is the right format (the other is the A13 statistics table). Pair it with a single sentence relating the overhead to A4: if the reasoning model costs a large multiple of the baseline and buys about one point, the graceful-degradation reading of A4 becomes a practical recommendation rather than a defensive one.

---

## 21. S1 — aggregate significance

**Motivation.** The main table shows DISEIL ahead in all ten settings with overlapping error bars. Overlapping error bars invite the objection that no individual comparison is significant. S1 converts the *pattern* of the ranking into a test.

**Findings.** DISEIL attains the best mean in all ten settings. The ten paired margins are +3.1, +2.5, +5.4, +4.9, +0.8, +0.4, +4.7, +5.7, +3.2, +6.4, with a mean of +3.71 points.

- Sign test over 10 settings: 10/10, one-sided p = 0.00098, two-sided p = 0.0020.
- Wilcoxon signed-rank over the 10 paired means: W = 0, two-sided p = 0.0020 (recomputed with scipy, matches the sheet).

**The caveat that must be reported with the number.** The ten settings are not ten independent experiments. They are five tasks under two modalities, and the two modalities of a task share the expert, the reset distribution and the reward structure. A reviewer will argue the effective n is 5, and the reviewer will be right to argue it. Collapsing to task means gives paired deltas of +2.80 (GridWorld), +5.15 (Push-T), +0.60 (Lift), +4.80 (Wipe) and +5.20 (Door):

- Sign test, n = 5: 5/5, one-sided p = 0.031, two-sided p = 0.063.
- Wilcoxon, n = 5: p = 0.063 two-sided.
- Paired t-test, n = 5: t(4) = 4.15, p = 0.014 two-sided.

So the headline weakens from p ~ 0.001 to p ~ 0.03 one-sided and is not significant two-sided under the nonparametric test at n = 5. The paper must lead with the version that survives scrutiny and must show that the difference is understood. The sentence to use, adapted from the workbook's own recommendation:

> DISEIL attains the best mean in all ten task-modality settings, with a mean margin of 3.7 points over the strongest baseline in each setting. Treating the ten settings as paired observations, a sign test rejects a coin-flip ranking at p = 0.002 (two-sided). Collapsing to the five tasks, to account for the correlation between the two modalities of a task, the sweep remains consistent (5/5, one-sided p = 0.031).

**Presentation.** Forest plot: one row per setting, the paired margin as a point with its pooled standard deviation as a horizontal bar, a vertical rule at zero, and a diamond at the bottom for the pooled mean. Below it, a second diamond for the five-task collapsed estimate. This is the format that shows the individual overlaps *and* the systematic direction, which is exactly the argument. A table would show the ten numbers and lose the pattern that is the whole point.

*Data series:*
```
setting            delta   (pooled sd for the bar: use sqrt(sd_DISEIL^2 + sd_base^2)/sqrt(n_seeds))
GridWorld state     +3.1
GridWorld image*    +2.5
Push-T state*       +5.4
Push-T image        +4.9
Lift state          +0.8   (grey)
Lift image          +0.4   (grey)
Wipe state          +4.7
Wipe image          +5.7
Door state          +3.2
Door image*         +6.4
pooled mean        +3.71   sign test 10/10, two-sided p = 0.0020
task-collapsed     +3.71   (n=5: +2.80, +5.15, +0.60, +4.80, +5.20), 5/5, one-sided p = 0.031, paired t p = 0.014
```

---

## 22. Information gain, starting performance, and the initial demonstrations

This section is not an ablation, but it is the interpretive frame that A3 forces, and the report needs it before the ablations are read.

Information gain is the policy's per-step loss on a newly acquired demonstration, measured *before* retraining on it. A high value means one of two things: the demonstration covers a region the current training set underrepresents, or the demonstration is itself poor, in the sense of being suboptimal or invalid, so that no policy would fit it. The second possibility is ruled out by construction in DISEIL. Prescriptions pass the feasibility check against the knowledge graph before they reach the expert, so the prescribed configuration is reachable and instantiable, and the demonstration itself comes from the expert, whose trajectories define the target the policy is being fitted to. A demonstration that survives both conditions cannot be invalid. High pre-retrain loss therefore identifies genuinely novel, underrepresented data. That is a claim supported by an argument, not a hypothesis awaiting a test.

The number of initial demonstrations was chosen to place each task's starting success rate inside a target range. The reason is a constraint from both ends. If the initial policy is too weak, its rollouts fail everywhere and the failure set carries no structure: every configuration is a failure, the descriptor separates nothing, and there is no allocation problem to solve. If the initial policy is too strong, the failure set is empty or nearly so, the budget has nothing to allocate, and every method converges to the same place. The band between those two is where a fixed budget of demonstrations can be spent well or badly, which is the regime the framework is about. Lift is the demonstration of what happens outside that band, at the upper end: at 100.0 +/- 0.0 there is no headroom, every ablation returns a null, and the setting is uninformative about every mechanism in the paper. A11 shows the same thing from the other side, in time rather than across tasks: at B = 40 all methods converge and the margin goes to zero on Lift and to 2.7 points elsewhere.

A3 completes the argument. High information gain is necessary for a demonstration to be worth collecting, and it is not sufficient. Gain is a per-demonstration quantity with no term for redundancy between demonstrations, and a method that maximises it greedily collects a stream of individually informative and jointly redundant corrections. The allocation step supplies the missing term.

---

## 23. Figure list

Fourteen figures, in the order they should appear.

| # | Figure | Covers | Format | Why this format |
|---|---|---|---|---|
| F1 | Allocation ladder | A2, A8, A3, DISEIL, best baseline, 3 primaries | Grouped bar, 3 panels, baseline as horizontal rule | The single most important ablation figure. Shows random < baseline, fallback = 1/3, clustering-off = baseline, full on top, in one image |
| F2 | Gain without allocation | A3 | Scatter (info gain x Delta-SR) with paired arrows, plus a slope inset for gain | The dissociation is two-dimensional. A bar chart would hide the half of the result that matters |
| F3 | Knockout summary | A1, A3, A4, A5, A6, A7, A8 x 10 settings | Heatmap of margin retained (%), Lift column greyed | Seven knockouts x ten settings is a matrix. Diverging colour scale centred at 50%, so negative cells (A3 on Door image) are visibly distinct |
| F4 | Reasoning and vision are small | A4, A5 | Dot plot of Delta with a +/-1 seed-sd noise band, two series | Shows the effect is consistent in direction and inside the noise. The honest format for a small effect |
| F5 | Grounding and feasibility | A6 | Scatter (fallback rate x Delta-SR) | The claim is causal and two-variable. Needs the full-DISEIL fallback rate added to the workbook |
| F6 | Bridging | A7 + D3 | Scatter (bridge share x Delta) with the flat fit, plus a 100% stacked bar of the split | Shows an honest null correlation. Resolve the D3 contradiction before drawing it |
| F7 | Descriptor dimensionality | A10 | Line plot, silhouette vs dimension, 10 lines + mean, marker at 6-D | The inverted U is the argument and only a line shows it |
| F8 | Budget sweep | A11 | Line plot, margin vs B (2 panels: margin, and absolute SR for one setting) | The decay of the margin with B is the finding. Panel 2 shows it is the baseline catching up |
| F9 | Demonstrations per round | A12 | Slope chart, 10 lines, D = 1 -> 3 | Every line falls and none cross. Only a slope chart shows both |
| F10 | Memory constants | A13 | 3-panel line plot + a statistics table beneath | The exact Holm p-values are load-bearing for the sigma claim, so the table is warranted here |
| F11 | Context set and citation rules | A9, A14, A15 | Grouped bar, 3 primaries, arms ordered by effect (3 sub-panels) | Three step-level ablations with the same answer of "worth about half a point". Combining them prevents each from being oversold |
| F12 | Cluster count | D2 | Stacked bar, normalised, k = 2..6 plus a skipped segment | Composition is the message and totals differ between GridWorld and the robot settings |
| F13 | Failures over the budget | D4 | Line plot, N vs round, N <= 3 region shaded | Shows the machinery is active early and idle late |
| F14 | Aggregate significance | S1 | Forest plot, 10 rows + 2 pooled diamonds | Shows the individual overlaps and the systematic direction together |

Supporting scatter, optional: D1 purity against silhouette, which shows the two checks are independent (r = 0.18, p = 0.62). Include it if space allows, because it is the honest qualification of the "semantically meaningful modes" claim.

Not drawable: D5 (compute). No figure and no number until the five jobs run.

## 24. Actions required before this dossier can be written into the report

1. Regenerate the A6 display strings from the numeric helper column, or explain the divergence.
2. Resolve the D3 and A7 bridging contradiction on GridWorld and Wipe by inspecting the prescription logs.
3. Run D5 on the five listed settings.
4. Add the full-DISEIL fallback rate per setting, needed as the reference line in F5.
5. Run D4 on GridWorld image and Door image so that each primary setting has a failure-count curve.
6. Delete the frozen-embedding-and-PCA image branch of Equation 7, the N <= 2 PCA edge case, and the changing-PCA-dimension workaround in the cluster-memory paragraph. A10 supersedes all three.
7. Retract the "DISEIL at B = 10 matches the best baseline at B = 20" claim. It is false in 7 of 10 settings.
8. Re-specify sigma as a per-task fraction of the reset range, and report the global-sigma result as a limitation.
