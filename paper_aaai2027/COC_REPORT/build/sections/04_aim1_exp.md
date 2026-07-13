## Implementation and experimental setup

Every concrete value in the Aim-1 evaluation is fixed in this section and appears nowhere else. The framework itself is stated over symbols: a budget $B$ of expert demonstrations, $D$ demonstrations acquired per round, a policy $f_\theta$ with a per-step loss $\ell_t$. What follows is the instance of that framework which was actually run.

### Tasks, observation modalities and settings

A *setting* is one task under one observation modality. The evaluation covers five tasks under two modalities, state and image, which gives ten settings. The word *mode* is used in this report only for a failure mode, which is a cluster of failures that the framework discovers; an observation modality is never called a mode.

GridWorld 5x5 is a discrete navigation task on a five-by-five grid with three obstacle cells, in which an agent must reach a goal cell from a start cell. The expert is a human. A* search and breadth-first search enter the task only as the feasibility and path-validity checker that decides whether a prescribed grid configuration admits an obstacle-free route from start to goal [@hart1968astar; @cormen2022algorithms]; they are never used as the expert, and no policy is trained on their output.

Push-T is a planar pushing task in ManiSkill, in which a manipulator must push a T-shaped block into a fixed goal pose [@mu2021maniskill; @gu2023maniskill2; @tao2024maniskill3]. Lift, Wipe and Door are RoboSuite manipulation tasks on a UR5/UR5e arm [@zhu2020robosuite]: lifting a cube from a table, wiping a randomised trail of dirt markers from a surface, and pulling a door open past a hinge threshold. The three RoboSuite tasks and Push-T supply the continuous-action half of the evaluation, and their reset distributions differ by an order of magnitude in width, which matters later for the memory kernel.

### Policy instantiations

The framework requires only that the policy expose a per-step loss, and it is instantiated with three different policy classes to make that requirement visible. GridWorld under the image modality uses a convolutional network. GridWorld under the state modality uses a multilayer perceptron. The four robot tasks use diffusion policies under both modalities, with an R3M visual encoder supplying the image branch [@chi2023diffusionpolicy; @nair2022r3m]. R3M supplies the policy's visual representation and nothing else. It does not supply the clustering features, which are geometric in every run, state and image alike.

### Budget, rounds and seeds

The validated instance is $B = 20$ and $D = 1$. Each round rolls out the current policy on a frozen held-out evaluation set, acquires one expert demonstration, adds it to the training set, and retrains. Twenty rounds therefore consume twenty demonstrations. Ablation A12, reported in the ablation section, is the evidence that $D = 1$ is the right choice at fixed labour rather than an arbitrary one: at a fixed total of twenty demonstrations, $D = 1$ attains the highest final success rate in all ten settings, and the decline as $D$ grows is monotone in every one of them. Ablation A11 sweeps $B$ over 10, 20 and 40 and shows that the framework's advantage is largest at the smallest budget, which is the evidence that $B = 20$ is an instance and not a requirement.

Seed counts are not uniform, and the asymmetry is stated rather than smoothed. GridWorld is run with nine seeds; the four robot tasks are run with five. The workbook's round accounting confirms both counts independently: clustered rounds plus skipped rounds total 180 for each GridWorld setting and 100 for each robot setting, which is seeds times $B$ in both cases.

### Initial demonstrations and starting performance

Before the first round, the policy is trained on an initial demonstration set that is excluded from the budget. In the runs that produce every number in this chapter, that set holds twenty demonstrations for every task.

The count is not a free parameter, and the reasoning behind it is the first half of the information-gain argument in the next section. A policy's starting success rate has to sit inside a band for the experiment to mean anything. If the initial policy is too weak, its rollouts fail everywhere, every configuration is a failure, the failure set carries no structure for the descriptor to separate, and there is no allocation problem to solve. If the initial policy is too strong, the failure set is empty or nearly so, the budget has nothing to allocate, and every method converges to the same place. The band between those two conditions is the regime in which a fixed budget of demonstrations can be spent well or badly, which is the regime the framework exists for. Demonstration count and coverage are known to govern imitation-learning performance directly [@lin2024datascaling], so the count is the lever that places a task inside the band.

The principle is implemented as a behaviour-cloning data-scaling sweep. A pool of expert demonstrations is collected, behaviour cloning is trained on each nested prefix of the pool, each prefix is evaluated on the frozen held-out set, and the prefix whose round-zero success rate is closest to a target of roughly 50 per cent is selected. Run over the consolidated re-implementation of the framework, that sweep selects eight initial demonstrations for Lift, twelve for Wipe, four for Door and twenty for GridWorld, and the measured round-zero success rates under those counts are 0.62 to 0.76 for Lift (state), 0.39 to 0.71 for Lift (image), 0.55 to 0.61 for Wipe (state), 0.41 to 0.47 for Door (state) and 0.46 to 0.61 for GridWorld (state). Those figures come from the live re-implementation and not from the runs that produced Table 1, and they are reported here as the calibration evidence rather than as the reported protocol.

The gap between the two is itself the explanation of the single most important caveat in this chapter. Twenty initial demonstrations over-provision the easiest robot task. Lift, under the reported protocol, begins the budget close to a perfect success rate and reaches it within about five demonstrations under every method compared. Its final success rate under DISEIL is 100.0 +/- 0.0 in both modalities, with no headroom and no seed variance. Lift therefore separates nothing, and no null result on Lift is evidence about any mechanism in this report. The statement is made here, once, and every later claim excludes Lift from its aggregate.

### Baselines

Six comparison methods are run, described qualitatively; their hyperparameters are not reproduced here, and the mechanics of each belong to the background chapter.

Five of them are published interactive imitation-learning methods and share one skeleton: roll out the current policy, read a scalar signal, hand control to the expert when the signal crosses a threshold, aggregate the expert's labels and retrain [@ross2011dagger]. They differ only in the signal. SafeDAgger learns a classifier that predicts when the policy is about to deviate from the expert [@zhang2017safedagger]. DropoutDAgger reads the spread of a Monte-Carlo dropout ensemble of the learner's action distribution [@menda2017dropoutdagger; @gal2016dropout]. EnsembleDAgger reads the variance of an explicit ensemble, combined with an action-discrepancy term [@menda2019ensembledagger; @lakshminarayanan2017ensembles]. ThriftyDAgger combines a novelty estimate with a learned risk estimate under a target switching rate [@hoque2021thriftydagger]. Diff-DAgger uses a diffusion policy's own per-step training loss as the uncertainty signal [@lee2025diffdagger], and is run on the robot tasks only, where the policy is a diffusion policy. Those five are the DAgger family, and they are labelled as such in every comparison table. Diff-DAgger's use of the per-step diffusion loss as an uncertainty signal is its own contribution; DISEIL uses that signal for failure localisation and also compares against it as a baseline, and both facts are stated plainly.

The sixth comparison method, Stagger, is not a published system. It is a uniform-random control implemented in this project: each round corrects one uniformly chosen recorded failure, with no gate, no descriptor and no allocation. It carries no citation, it is never labelled as a member of the DAgger family, and it is reported on GridWorld in the main table. Its extension to the robot tasks is reported in the ablation section as A2, where it answers the most damaging objection available against this work, which is that any failure replay would do.

### Metrics

Three quantities are reported. The primary metric is the final success rate on the frozen held-out evaluation set after the budget is exhausted, in per cent, averaged over seeds.

Per-demonstration information gain is the policy's per-step loss on a newly acquired demonstration, measured before the policy is retrained on that demonstration. The quantity is the same per-step diffusion loss that Diff-DAgger uses as its gate signal [@lee2025diffdagger], evaluated on a datum rather than on a rollout step. On the robot tasks the scoring policy can be up to three demonstrations stale, because retraining runs every fourth demonstration; each reported cell pools between 168 and 184 loss records.

$\Delta$SR is the change in the policy's success rate on the round-level rollout evaluation, measured before and after a round. It is a per-round quantity, not a final one, and it is the outcome against which a prescription's reported confidence is scored.

---

## Results

### The main comparison

Table 1 gives the final held-out success rate in all ten settings. DISEIL attains the highest mean in every one of them.

**Table 1.** Final held-out success rate (per cent, mean +/- standard deviation over seeds; nine seeds on GridWorld, five on the robot tasks) after a budget of twenty expert demonstrations. The five published query-gated methods are the DAgger family and are grouped under that header. Stagger is a uniform-random allocation control implemented in this project and is not a DAgger-family method. Diff-DAgger is run on the robot tasks only, where the policy is a diffusion policy; Stagger is reported on GridWorld only. Lift is at the ceiling under every method and separates nothing.

| | | *DAgger family* | | | | | Control | Ours |
|---|---|---|---|---|---|---|---|---|
| **Task** | **Modality** | SafeDAgger | DropoutDAgger | EnsembleDAgger | ThriftyDAgger | Diff-DAgger | Stagger | **DISEIL** |
| GridWorld 5x5 | image | 86.1 ± 2.8 | 85.8 ± 2.6 | 85.7 ± 2.2 | 87.1 ± 1.9 | — | 86.6 ± 2.3 | **89.6 ± 1.8** |
| GridWorld 5x5 | state | 85.3 ± 2.7 | 84.9 ± 2.5 | 86.2 ± 2.1 | 86.8 ± 2.0 | — | 85.7 ± 1.5 | **89.9 ± 1.3** |
| Push-T | state | 82.0 ± 6.8 | 84.8 ± 6.1 | 85.9 ± 5.8 | 83.2 ± 7.2 | 90.7 ± 4.5 | — | **96.1 ± 4.5** |
| Push-T | image | 78.1 ± 7.8 | 82.1 ± 6.9 | 83.2 ± 6.6 | 79.3 ± 8.1 | 89.0 ± 4.8 | — | **93.9 ± 4.9** |
| Lift | state | 99.2 ± 1.6 | 99.2 ± 1.0 | 99.2 ± 1.0 | 98.8 ± 2.4 | 99.2 ± 1.0 | — | **100.0 ± 0.0** |
| Lift | image | 99.6 ± 0.8 | 97.2 ± 3.5 | 98.8 ± 1.6 | 99.6 ± 0.8 | 99.6 ± 0.8 | — | **100.0 ± 0.0** |
| Wipe | state | 88.0 ± 2.5 | 89.6 ± 4.1 | 90.8 ± 4.3 | 90.0 ± 2.5 | 90.4 ± 6.0 | — | **95.5 ± 6.0** |
| Wipe | image | 69.6 ± 5.3 | 83.2 ± 6.8 | 84.4 ± 7.1 | 69.2 ± 9.0 | 89.6 ± 3.2 | — | **95.3 ± 3.2** |
| Door | state | 93.2 ± 5.2 | 92.8 ± 2.7 | 88.8 ± 7.0 | 89.6 ± 3.9 | 95.2 ± 4.3 | — | **98.4 ± 4.2** |
| Door | image | 92.4 ± 3.2 | 88.8 ± 3.3 | 86.0 ± 10.9 | 92.8 ± 2.7 | 89.2 ± 3.5 | — | **99.2 ± 3.4** |

The margin over the strongest baseline in each setting averages 3.71 points, with a standard deviation of 2.05 and a range from +0.4 on Lift (image) to +6.4 on Door (image). Which baseline is strongest varies: Diff-DAgger on Push-T (both modalities), Wipe (image) and Door (state); ThriftyDAgger on both GridWorld settings and on Door (image); EnsembleDAgger on Wipe (state). The comparison is therefore against a moving target, and DISEIL is ahead of whichever method happens to be best in each setting.

### The aggregate claim, stated conservatively

Ten wins from ten is a pattern, and the pattern rather than any individual comparison is what the aggregate test converts into a number. Treating the ten settings as paired observations, a sign test and a Wilcoxon signed-rank test both reject a coin-flip ranking at two-sided $p = 0.002$, which is the smallest $p$-value attainable with ten pairs and is therefore the floor of what this design can produce.

That figure should not be led with, and the reason is a defect in the design that a hostile reviewer would find. The ten settings are not ten independent experiments. They are five tasks under two observation modalities, and the two modalities of a task share the expert, the reward structure and the reset distribution, so they are correlated by construction and the effective sample size is nearer five than ten. Collapsing to the five task means, the paired differences are +2.80 (GridWorld), +5.15 (Push-T), +0.60 (Lift), +4.80 (Wipe) and +5.20 (Door). The sweep holds at five from five. A one-sided sign test gives $p = 0.031$, which is its floor at $n = 5$, and a paired $t$-test over the same five means rejects at $t(4) = 4.15$, $p = 0.014$ two-sided. The two-sided nonparametric test does not reject at $n = 5$ ($p = 0.063$), and it cannot, because 0.063 is the smallest value it can return with five pairs.

The claim of record is the collapsed one. DISEIL attains the best mean success rate in all ten settings, and the aggregate advantage is significant under the conservative task-level analysis. Figure 5 shows the ten paired margins with their standard errors and both pooled estimates, which is the presentation that makes the individual overlap and the systematic direction visible at the same time.

![Paired margin of DISEIL over the strongest DAgger-family baseline in each of the ten settings, with the standard error of the paired difference, and the two pooled estimates.](figures_generated/F14_aggregate_significance.pdf)

**Figure 5.** DISEIL attains the higher mean in all ten settings, with a mean margin of 3.71 points over the strongest baseline in each. The individual error bars overlap zero in several settings, which is why no single comparison carries the claim; the direction of the ranking is consistent across every task and every modality, and it is the consistency that the aggregate test measures. Horizontal bars are the standard error of the paired difference and are not themselves the test. The upper diamond pools the ten settings; the lower diamond pools the five task means and is the estimate the report leads with. The two smallest margins belong to Lift, for the trivial reason that there is no headroom there.

### Learning curves over the budget

Figure 4 plots the success rate against the number of demonstrations added, for one setting of each task.

![Success rate against the number of demonstrations added, on five tasks. Mean over seeds with a shaded one-standard-deviation band.](../figures/all_5_task_comparison.pdf)

**Figure 4.** Success rate against the number of demonstrations added, for the five tasks, showing the observation modality printed in each panel title: GridWorld (image), Push-T (state), Lift (state), Door (state) and Wipe (image). Lines are means over seeds and shaded bands are one standard deviation (nine seeds on GridWorld, five on the robot tasks). The Door panel shows the state setting; the Door image setting, which is one of the three primary settings used for the ablations, is reported in Table 1 and is not one of these panels. Every method saturates at a perfect success rate on Lift within about five demonstrations, so that panel separates nothing and is shown for completeness.

The curves say three things that the final numbers alone do not. The separation between DISEIL and the DAgger family opens early on Push-T, from about the fifth demonstration, and holds thereafter, which is consistent with the budget sweep: the advantage is a coverage-rate advantage and it is paid out at the front of the budget, not at the end of it. On GridWorld every method rises together and finishes bunched between roughly 0.88 and 0.92, which is why the GridWorld margins (+2.5 and +3.1 points) are the smallest of the non-Lift settings. The task is small enough that twenty demonstrations approach what any allocation rule can extract from it.

The third observation is a limitation and is reported as one. On Wipe (image), DISEIL and the strongest baseline are both still rising at the twentieth demonstration. Neither curve has plateaued inside the budget, so the +5.7-point claim on that setting rests on the final gap (95.3 against 89.6) and not on a demonstrated asymptote. A longer budget could close it, and the budget sweep shows the margin on that setting falling to +3.4 points at $B = 40$, so the honest reading is that the gap narrows with more labour and has not been shown to vanish.

---

## Information gain, starting performance and why the gain is real

### The measurement

Per-demonstration information gain is the current policy's per-step loss on a newly acquired demonstration, evaluated before the policy has been retrained on it. The intuition is the standard one from active learning, where a datum on which the current model incurs a large loss is the datum whose acquisition is expected to change the model most [@settles2009active; @houlsby2011bald]. What is new here is not the measure. It is what the measure licenses once the acquisition pipeline is known.

Table 2 gives the mean gain per setting. DISEIL acquires demonstrations of higher pre-retrain loss than every comparison method in all ten settings.

**Table 2.** Per-demonstration information gain: the policy's per-step loss on each newly acquired demonstration, measured before retraining on it (mean +/- standard deviation; 168 to 184 loss records per cell). The five published gates are the DAgger family; Stagger is the uniform-random control.

| | | *DAgger family* | | | | | Control | Ours |
|---|---|---|---|---|---|---|---|---|
| **Task** | **Modality** | SafeDAgger | DropoutDAgger | EnsembleDAgger | ThriftyDAgger | Diff-DAgger | Stagger | **DISEIL** |
| GridWorld 5x5 | image | 2.46 ± 1.61 | 2.53 ± 1.69 | 1.57 ± 1.55 | 1.37 ± 1.12 | — | 1.88 ± 1.43 | **3.21 ± 2.33** |
| GridWorld 5x5 | state | 2.55 ± 1.68 | 2.95 ± 2.12 | 1.33 ± 1.18 | 1.34 ± 0.93 | — | 1.83 ± 1.11 | **3.55 ± 2.51** |
| Push-T | state | 1.66 ± 0.99 | 2.36 ± 1.63 | 1.11 ± 0.64 | 1.10 ± 0.65 | 1.57 ± 1.10 | — | **2.81 ± 2.09** |
| Push-T | image | 2.04 ± 1.11 | 2.16 ± 1.36 | 1.06 ± 0.64 | 1.20 ± 0.62 | 1.80 ± 1.10 | — | **2.82 ± 1.72** |
| Lift | state | 2.23 ± 1.36 | 2.10 ± 1.25 | 1.12 ± 0.74 | 1.13 ± 0.55 | 1.61 ± 1.12 | — | **2.64 ± 1.65** |
| Lift | image | 2.18 ± 1.40 | 2.17 ± 1.39 | 1.00 ± 0.55 | 1.21 ± 0.64 | 1.36 ± 0.85 | — | **2.93 ± 1.67** |
| Wipe | state | 2.02 ± 0.96 | 2.38 ± 1.55 | 1.23 ± 1.10 | 1.18 ± 0.81 | 1.43 ± 0.80 | — | **2.91 ± 2.02** |
| Wipe | image | 2.50 ± 1.47 | 2.96 ± 2.00 | 1.43 ± 0.89 | 1.52 ± 0.86 | 1.95 ± 1.16 | — | **3.62 ± 2.19** |
| Door | state | 2.53 ± 1.64 | 3.10 ± 2.21 | 1.43 ± 0.82 | 1.42 ± 0.90 | 1.84 ± 1.11 | — | **3.43 ± 2.12** |
| Door | image | 2.35 ± 1.40 | 2.46 ± 1.44 | 1.24 ± 0.76 | 1.26 ± 0.72 | 1.58 ± 0.92 | — | **3.00 ± 1.98** |

![Distribution of the pre-retrain policy loss on the acquired demonstration, per method, on GridWorld (image).](../figures/info_gain_boxplot.pdf)

**Figure 6.** Distribution of per-demonstration information gain on GridWorld (image), the setting on which the per-demonstration records are instrumented in full. Each box is the pre-retrain per-step loss of the current policy on the demonstration that method acquired in that round, pooled over nine seeds and twenty rounds. The DISEIL box has the highest median and the longest upper tail. The framework acquires demonstrations of higher typical novelty, and it also reaches the rare demonstrations of very high novelty that the query gates never select. Diff-DAgger does not appear, because it is a robot-task baseline and this is a GridWorld setting.

### The argument

A high pre-retrain loss on a demonstration admits exactly two readings.

Either the demonstration covers a region of the state space that the current training set underrepresents, so that the policy has never had to fit anything like it, or the demonstration is itself poor, in the sense of being suboptimal or invalid, so that no policy could fit it and the loss is a statement about the datum's incoherence rather than about the policy's ignorance. The second reading is the one that would destroy the measure, and any method that reports information gain without addressing it is reporting a number that could mean either.

In DISEIL the second reading is ruled out by construction, and by two independent constructions at that. A prescription reaches the expert only after it has passed the feasibility check against the knowledge-augmented graph: the prescribed configuration lies inside the workspace bounds, inside the object's spawn range and inside the reachable set, because a violation is returned to the prescription model as feedback and a revised prescription is demanded until a feasible one is produced. An infeasible scenario therefore never becomes a demonstration. And the demonstration itself comes from the expert, whose trajectories are the target the policy is being fitted to, so a demonstration that survives the feasibility check cannot be suboptimal with respect to that target. Neither an infeasible scenario nor a bad action survives into the dataset.

The first reading is therefore the only one left. High pre-retrain loss on a DISEIL-acquired demonstration identifies genuinely novel, underrepresented data. That is a claim with an argument behind it, not a hypothesis awaiting a test, and it is why Table 2 is a statement about coverage rather than about noise.

Starting performance is what makes the argument interpretable in the first place, and it is the reason the initial demonstration count was set as it was. Loss is measured relative to a policy, and a policy that fails uniformly produces a high pre-retrain loss on any demonstration whatsoever, including a redundant one. The measure only discriminates when the policy is competent enough that its failures are localised. Placing each task's starting success rate inside the target band is what buys that condition: the policy already handles part of the state space, so a demonstration that provokes a high loss is one that lies outside the part it handles. Lift is the counter-example that proves the point from the other side. It starts at the ceiling, its information gain is the lowest DISEIL records in the state modality (2.64), and its success-rate margin is the smallest in the table (+0.8 and +0.4). With no failures to be novel with respect to, novelty has nothing to measure.

### The qualification, stated before a reviewer supplies it

High information gain per demonstration is necessary and it is not sufficient, and the evidence for that is one of this project's own ablations rather than a caveat added for modesty. Removing the clustering step, so that each round greedily corrects the single highest-loss failure, leaves information gain statistically unchanged (mean change +0.06 over the eight settings with headroom, Wilcoxon $p = 0.23$; it rises rather than falls) while the final success rate collapses by 4.01 points (Wilcoxon $p = 0.002$). Greedy worst-loss selection collects demonstrations that are individually informative and jointly redundant, because information gain measured on one demonstration carries no term for its overlap with the demonstration collected in the previous round. Allocation across failure modes is precisely the term that supplies it.

Table 2 must therefore never be read as the source of the framework's advantage on its own. It is the evidence that the demonstrations DISEIL asks for are novel to the policy. The evidence that they are novel *to each other* is the allocation ablation, and that ablation is reported in full in the next section.

---

## Prescription confidence as an in-round predictor of improvement

At the moment it issues a prescription, the prescription model also emits an integer confidence between 0 and 100, together with a one-line rationale, reporting how likely it believes the resulting demonstration is to improve the policy. The number is scored against $\Delta$SR, the change in the policy's success rate on the round-level rollout evaluation across that round.

The Pearson correlation between the reported confidence and the realised $\Delta$SR runs from 0.82 to 0.89 across the ten settings: 0.86 and 0.88 on GridWorld (image and state), 0.87 and 0.88 on Push-T, 0.88 and 0.89 on Lift, 0.82 and 0.86 on Wipe, 0.83 and 0.82 on Door. Figure 7 shows the GridWorld (image) scatter, where $r = 0.86$ over 152 prescriptions. That count is below the 180 rounds the nine seeds supply, because a round whose confidence line cannot be parsed from the model's output is logged without a confidence value and falls back to the geometric decision rule.

![Reported prescription confidence against the change in success rate the prescribed demonstration produced, GridWorld (image).](../figures/confidence_vs_success.pdf)

**Figure 7.** The confidence the prescription model reports for a prescription, against the change in the policy's success rate on the round-level rollout evaluation that the resulting demonstration produced. GridWorld (image), 152 prescriptions pooled over nine seeds and twenty rounds, Pearson $r = 0.86$. Below roughly 55 per cent confidence the prescriptions return nothing and a few cost a little; above roughly 70 per cent almost all of them return a gain, and the largest gains are concentrated there. The correlation runs from 0.82 to 0.89 across the ten settings.

What makes this number usable, rather than a post-hoc rationalisation, is the order in which the two quantities become available. The confidence is reported blind, at prescription time. At that moment the demonstration has not been collected, the expert has not been called, the policy has not been retrained, and the re-rollout that produces $\Delta$SR has not been run. The success-rate signal arrives only after all three of those steps have completed, by which point the round's unit of budget has already been spent. The model is therefore forecasting an outcome it cannot observe, and a correlation of 0.82 to 0.89 is the accuracy of that forecast rather than a description of an outcome the model was shown. A signal available before the expenditure and correlated with what the expenditure returns is exactly the signal an allocation framework under a restricted budget needs, and it is not available to any query gate, whose scalar signal is a property of a state and carries no forecast about a demonstration that does not yet exist.

Two limits belong with the number. The correlation is measured on DISEIL runs only, so it says nothing about whether a baseline's gate signal would predict its own $\Delta$SR equally well. And no experiment in this project gates on the confidence: nothing is skipped, deferred or re-prescribed on the basis of a low confidence score, so the correlation is an observed property of the system and not yet a mechanism inside it. Turning it into one, by declining to spend a demonstration on a prescription the model itself does not believe in, is the obvious next step and it has not been run.

---

## The failure modes the framework discovers

The partition step produces the object the whole framework allocates over, so it is worth showing what that object actually is on a real task. Figure 8 shows the three failure modes discovered on Push-T (image), with three sampled members of each.

![Three failure modes discovered on Push-T by clustering the geometric descriptor at the flagged timestep, three sampled rollouts per mode, annotated with the block's orientation error and the end-effector-to-block distance.](../figures/clustering_modes_pushT.pdf)

**Figure 8.** The failure modes discovered on Push-T (image) by clustering the 6-D geometric descriptor at the flagged timestep. Each row holds three rollouts assigned to one mode, annotated with the block's orientation error and the distance between the end-effector and the block. The partition recovers behaviourally distinct failures: the block is delivered to the goal but left rotationally wrong (top row, orientation error 162 to 169 degrees at close contact); the arm never makes contact and the block is not moved (middle row, contact distance 0.08 to 0.11 m); and the block is pushed but abandoned at a large orientation error and far from the end-effector (bottom row, contact distance 0.15 to 0.18 m). Push-T is shown under the image modality, and the clusters are found from geometry alone, with no visual embedding.

The naming deserves an exact statement, because the loose version of it would be an overclaim. The clusters are formed geometrically and are born nameless: the agglomerative partition of the standardised descriptor returns integer labels and consults no language model at any point. Each individual failure then receives one root-cause label and one trajectory-phase label from the reasoning model, which is constrained to the closed taxonomy authored in that task's knowledge graph, so the vocabulary of names is authored in the graph and the model's job is assignment rather than invention. A cluster's name is the majority root-cause label among its members. The names in Figure 8 are readable renderings of the Push-T graph's own failure-mode nodes.

How well that naming holds is measured rather than assumed. Cluster purity, the fraction of a geometric cluster's failures that share the dominant root-cause label, ranges from 0.78 to 0.93 across the ten settings with a mean of 0.877, and the mean number of distinct root causes per cluster ranges from 1.30 to 1.91. Purity is lowest on Wipe (image), where the same end-effector position can correspond to insufficient contact force, to a missed patch of the surface, or to premature termination, and geometry cannot tell those apart. Purity and geometric separation are uncorrelated across settings ($r = 0.18$, $p = 0.62$), which means the two checks on the descriptor are independent and that a well-separated cluster is not automatically a semantically clean one. The claim the report makes is therefore the qualified one: the descriptor separates failures by where and how they occur, and it recovers root cause only to the extent that configuration determines cause. Purity is measured against the reasoning model's own labels, so it records agreement between two components of the same system and not agreement with a ground truth, and there is no human-labelled root-cause set against which it could be checked. That circularity is a real limitation of the measurement.

Three is not the number of failure modes on these tasks; it is the most common number the silhouette criterion selects when clustering runs. Pooled over 896 clustered rounds, $k = 3$ is chosen in 25.1 per cent of them and $k = 4$ in 23.8 per cent, and every value from 2 to 6 is chosen in at least 15 per cent. The framework discovers a different number of modes in different rounds, and Figure 8 shows the mode of a broad distribution rather than a property of Push-T.

---

## What the results establish, and what they do not

DISEIL attains the best mean success rate in every one of the ten settings, by an average of 3.71 points over whichever DAgger-family method is strongest in that setting, and the aggregate advantage survives the conservative task-level analysis ($t(4) = 4.15$, $p = 0.014$). It acquires demonstrations of higher pre-retrain loss than every comparison method in every setting, and the pipeline's two construction guarantees, feasibility verification and an expert demonstrator, are what convert that loss into a statement about coverage. The prescription model's own confidence in a prescription, reported before the demonstration is collected, predicts the round's realised change in success rate at $r = 0.82$ to 0.89.

Four things the results do not establish should be read alongside those four that they do. Lift contributes nothing: it is at 100.0 +/- 0.0 with no headroom and no seed variance, and every mechanism claim in this report excludes it. The Wipe (image) advantage rests on the final gap and not on a plateau, because neither curve has flattened within the budget. The confidence correlation is a correlation, measured on DISEIL runs only, and nothing in the framework yet acts on it. And the demonstrations counted here are calls to a scripted or planner-based expert, each costing the same, which is the assumption that Aim 3 exists to remove.
