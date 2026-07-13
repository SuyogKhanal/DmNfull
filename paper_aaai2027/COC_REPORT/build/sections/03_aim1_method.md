## Aim 1 — Demonstration distillation for sample-efficient imitation learning

### Motivation and problem statement

Every ingredient of an imitation-learning pipeline has become cheap except one. Compute is bought, simulators are free, and policy architectures are downloaded and fine-tuned. The expert demonstration is the input whose price does not fall: a person teleoperates a trajectory, or a scripted oracle is written and run, one episode at a time, and the number of episodes available is set by something other than the researcher's patience. Policy performance, meanwhile, scales with the number and the coverage of the demonstrations it is trained on [@lin2024datascaling]. A practitioner who holds a fixed allowance of demonstrations therefore faces a question that the scaling relationship does not answer: given that the allowance cannot grow, what should each demonstration in it contain?

The whole of Aim 1 is an answer to that question. Write $B$ for the number of demonstrations the expert will supply beyond an initial set, and $D$ for the number acquired in one round of interaction, so that the loop runs for $B/D$ rounds. Neither symbol carries a value in this section. The framework is defined for any fixed, restricted budget and any per-round acquisition count, and the values at which it was validated appear once, in the experimental setup. What the framework maximises, under that fixed budget, is the information content of each demonstration: informally, how much of the policy's remaining error a demonstration can remove, and formally, the policy's per-step loss on the demonstration measured before it is trained on it (the argument that this quantity means what it appears to mean is made in the results, and it depends on the initial demonstration count, which is also set there).

The policy is any function $f_\theta$ that maps an observation to an action and exposes a per-step loss $\ell_t$ at a state-action pair. The requirement stops there. A multilayer perceptron on a discrete grid, a convolutional network on grid images and a diffusion policy on a manipulator [@chi2023diffusionpolicy] all satisfy it, and all three are used in the experiments. The framework is a way of spending a demonstration budget, and the only thing it asks of the learner is a loss it can read, which is what allows one loop to run on a discrete grid policy and on a continuous manipulation policy without modification.

### The gap in the DAgger family

Interactive imitation learning already exists to solve the failure that offline cloning cannot: a policy trained on expert states compounds its own errors once its own actions determine where it goes, and the correction is to label the states the learner actually visits [@ross2010reductions; @ross2011dagger]. Every method in the family that follows shares one skeleton. Roll out the current policy, decide whether to hand control to the expert, aggregate the expert's labels into the dataset, retrain. The members differ in exactly one component, the scalar signal that opens the gate. SafeDAgger trains a classifier to predict when the policy will deviate from the expert [@zhang2017safedagger]. DropoutDAgger reads the spread of a Monte-Carlo dropout ensemble [@menda2017dropoutdagger] and EnsembleDAgger reads the variance of an explicit ensemble [@menda2019ensembledagger], both importing standard deep uncertainty estimators [@gal2016dropout; @lakshminarayanan2017ensembles]. ThriftyDAgger combines novelty with a learned risk estimate under a target switching rate [@hoque2021thriftydagger]. Diff-DAgger reads a diffusion policy's own per-step training loss and hands over when it crosses a quantile of the training-loss distribution [@lee2025diffdagger]. A survey of the field organises the whole literature around who decides and when [@celemin2022iil].

Each of these answers the question of *when* to ask for help. Three properties follow from answering only that question, and they are the opening this work uses.

A per-state gate cannot see a batch. It fires on one state of one rollout. Presented with the twenty rollouts a policy failed this round, it has no representation in which two of those failures are the same mistake and a third is a different one, so it cannot tell a redundant correction from a novel one, and it cannot spend a scarce budget on the failure that is most worth removing.

A per-state gate has no memory across rounds. The signal is recomputed from scratch on the current policy at the current state. A failure that persists across rounds will keep tripping the gate, and nothing in the gate records that the same region was corrected twice already, so a single stubborn failure can absorb a large share of the budget while other failures go untouched.

A per-state gate inherits the state that tripped it. The corrective demonstration begins wherever the policy happened to be when the signal crossed threshold. That state is often already corrupted, and the expert then spends the demonstration recovering from a bad configuration rather than teaching the behaviour that would have avoided it.

So *when* is one decision of three, and the other two are unclaimed: *which* failure to correct, and *where* the corrective demonstration begins. The adjacent literature does not claim them either. Active learning selects which point to label, but from a pool of points that already exist [@settles2009active; @houlsby2011bald]. Coreset and diversity methods cover a representation space, again by selecting from what has been collected [@sener2018coreset; @ash2020badge]. Demonstration curation retrieves sub-trajectories from an existing corpus [@memmel2025strap], and dataset distillation compresses a dataset that has already been gathered into a smaller synthetic one [@cazenavette2022mtt]. None of these prescribes a demonstration that does not exist yet and then has an expert produce it, which is the operation this framework performs and the reason it is called demonstration distillation.

### Problem formulation

The interactive loop skeleton is shared by every method compared in this chapter, and it is stated in the background rather than re-derived here [@ross2011dagger]. What follows fixes the notation and isolates the one component that differs between methods.

The policy $f_\theta$ is trained on an initial demonstration set $\mathcal{D}_0$ by behaviour cloning. Rounds are indexed by $r$. At the start of round $r$ the policy is rolled out on a fresh pool of episodes drawn from the task's reset distribution, and the episodes it fails are collected into the round's failure set. The only requirement placed on the policy is that it expose a per-step loss at its own executed action,

$$\ell^{(i)}_t \;=\; \mathcal{L}\big(f_\theta,\; s^{(i)}_t,\; a^{(i)}_t\big), \tag{1}$$

where $i$ indexes an episode, $s_t$ is the observation at step $t$ and $a_t$ is the action the policy itself executed. For a diffusion policy $\mathcal{L}$ is the denoising loss, which is what Diff-DAgger uses as its gate signal [@lee2025diffdagger]; for a discrete policy it is the negative log-likelihood of the executed action. The failure set of the round is

$$\mathcal{F}_r \;=\; \big\{\, f_i \;=\; (\tau_i,\; \ell^{(i)}_{1:T_i}) \;:\; \tau_i \text{ is a failed rollout of } f_\theta \,\big\}, \qquad N \;=\; |\mathcal{F}_r|, \tag{2}$$

with $\tau_i$ the trajectory and $T_i$ its length. Success is measured on a frozen held-out evaluation set that no method sees during acquisition, and the same set is used for every method.

Each round ends by acquiring $D$ demonstrations from the expert and aggregating them,

$$\mathcal{D}_r \;=\; \mathcal{D}_{r-1} \,\cup\, \{d_{r,1},\dots,d_{r,D}\}, \qquad \theta_r \;=\; \arg\min_\theta\ \mathbb{E}_{(s,a)\sim\mathcal{D}_r}\big[\mathcal{L}_{\mathrm{BC}}\big], \tag{3}$$

and the loop stops when the budget is exhausted, that is when $|\mathcal{D}_r| - |\mathcal{D}_0| = B$. Retraining is from scratch, at a per-task cadence, and follows standard practice.

Every method in this chapter instantiates the same three-part acquisition rule. Writing $A$ for the rule, a round's acquisition is fully specified by

$$A \;=\; \big(\underbrace{t^\star}_{\text{when}},\; \underbrace{C_{\mathrm{tgt}}}_{\text{which}},\; \underbrace{\xi}_{\text{where}}\big), \tag{4}$$

where $t^\star$ is the step at which the expert takes over, $C_{\mathrm{tgt}}$ is the failure mode the round is spent on, and $\xi$ is the specification of the state from which the corrective demonstration begins. The DAgger family fixes the second and third components trivially: $C_{\mathrm{tgt}}$ is whichever rollout tripped the gate first, and $\xi$ is the state the rollout was already in. DISEIL computes all three. The rest of the method is a description of how the second and third are computed, and nothing else in the loop is changed, which is what makes the comparison a controlled one.

### The DISEIL framework

The framework has four stages, run once per round: perceive the round's failures, partition them into failure modes, prioritise one mode against a memory of what has already been corrected, and prescribe the one demonstration the expert is asked to supply. Standard machinery is used inside three of the four stages and is flagged as standard where it appears. The novelty is the pairing of a cross-round memory over failure modes with a within-round prescription that is verified against an explicit model of what the environment permits.

![Figure 1. The demonstration-distillation loop. A policy trained on a small demonstration set fails repeatedly on the same configurations. A language model reads those failures and prescribes the configuration of the next demonstration to collect. The expert supplies one demonstration at that configuration, the demonstration is added to the training set, and the policy is retrained. Each round of the loop spends a single unit of the demonstration budget.](../figures/Teaser_Diagram.pdf)

#### Perceive

The first act of a round is to say where each failure went wrong, and to say it in two languages at once: a geometric one, which the partition will use, and a natural one, which the prescription will use. The two descriptions are computed from the same step of the same episode, and they never mix. Clustering is geometric for every run, under both observation modalities. There is no visual-embedding branch anywhere in the framework, and the earlier version of the method, which clustered image runs in a frozen visual-representation space reduced by principal components, has been retired on the evidence of ablation A10, reported with the other design-choice studies below. A pre-trained visual representation is still used, but only as the encoder of the image-modality *policy* [@nair2022r3m], and it supplies no feature to the partition.

**The flagged step.** Each failure is anchored at a single step $t^\star_i$, defined as the first step at which the per-step loss crosses an out-of-distribution threshold $\eta$ and stays across it for $K$ consecutive steps,

$$t^\star_i \;=\; \min\Big\{\, t \;:\; \ell^{(i)}_u > \eta \ \ \text{for all } u \in [\,t,\, t+K\,] \,\Big\}, \qquad \text{with } t^\star_i \leftarrow \arg\max_t \ell^{(i)}_t \ \text{if no crossing occurs}. \tag{5}$$

The threshold is a quantile of the training-loss distribution, recalibrated at every retrain, which is the Diff-DAgger construction used unchanged [@lee2025diffdagger]. The choice of the *first* crossing rather than the loss peak is a deliberate departure from the obvious definition, and it was made for a practical reason recorded during implementation: in a failing episode the peak arrives late (on Door, at a median of 0.91 of the episode length), so an expert who takes over at the peak inherits a badly corrupted state and has almost no episode left in which to correct it. The first crossing is early, the state is less corrupted, and the expert has budget to work with.

**The geometric descriptor.** Each failure is reduced to a six-dimensional vector $\phi_i \in \mathbb{R}^6$ computed from the privileged simulator state at $t^\star_i$. For the manipulation tasks the canonical form is

$$\phi_i \;=\; \big[\, p_{x},\; p_{y},\; \sin\theta,\; \cos\theta,\; \rho_i,\; \delta_i \,\big], \qquad \rho_i = \frac{t^\star_i}{T_i}, \qquad \delta_i = \big\|\, p^{\mathrm{tcp}}_i - p^{\mathrm{obj}}_i \,\big\|_2, \tag{6}$$

where $(p_x, p_y)$ is the planar position of the task-relevant object, $\theta$ its yaw (entered through its sine and cosine so that the wrap at $\pm\pi$ does not create a false distance), $\rho_i$ the fraction of the episode completed before the failure was flagged, and $\delta_i$ the distance between the end-effector and the object at the flagged step. For the discrete grid task the same six slots are filled with the agent's cell, its signed offset to the goal, its progress, and the Manhattan distance remaining. Where a task randomises no yaw, the two orientation slots carry the task's own state variables in their place. The instantiation actually used for each task is given below.

| Task | The six components of $\phi$ |
|---|---|
| GridWorld | agent cell $(2)$, signed offset to goal $(2)$, progress, Manhattan distance to goal |
| Push-T | block planar position $(2)$, $\sin\theta$, $\cos\theta$, progress, end-effector-to-block distance |
| Lift | cube planar position $(2)$, progress, gripper-to-cube planar distance, gripper height above cube, grasp indicator |
| Door | door-frame position $(2)$, frame yaw, normalised hinge angle, end-effector-to-handle distance, progress |
| Wipe | remaining-dirt centroid $(2)$, proportion wiped, end-effector-to-centroid distance, fraction of markers remaining, progress |

The width of the descriptor is not a free choice made to fit the results. Ablation A10 scores descriptors of two to twelve dimensions on the mean silhouette of the clusters they produce, a criterion with no dependence on success rate, and finds an inverted U with a single interior maximum at six dimensions in all ten settings. The descriptor is small because the failure sets are small, and the two facts are connected by distance concentration: adding weakly informative dimensions to a distance computation over a few dozen points pushes all pairwise distances toward each other and makes the merge order of the clustering arbitrary.

**The perceptual and causal description.** In parallel with the descriptor, three rendered frames of the failing episode, at its start, at $t^\star_i$ and at its end, are passed to a vision-language model [@bai2025qwen3vl], which returns a short spatial account of what went wrong. A second, text-only reasoning model then converts that account into a root cause and a trajectory phase, drawn from a closed taxonomy that is stored in the task's knowledge-augmented graph rather than invented by the model. The literature supports this division of labour precisely. Vision-language models are competent at naming a cause when they are given structured evidence [@liu2023reflect; @duan2025aha] and unreliable at metric and spatial reasoning from pixels alone [@chen2024spatialvlm; @fu2024blink]. The framework therefore asks them for the cause, and computes the geometry itself.

#### Partition

The round's failures are partitioned into failure modes by a generic clustering step $\mathcal{C}$ applied to the standardised descriptors,

$$\tilde{X}_i \;=\; \frac{\phi_i - \mu}{\sigma_\phi}, \qquad \{C_1,\dots,C_{k^\star}\} \;=\; \mathcal{C}\big(\tilde{X},\, k^\star\big), \qquad k^\star \;=\; \arg\max_{k \in [2,\,k_{\max}]} \operatorname{sil}(k), \tag{7}$$

with $k_{\max} = \max(2, \min(6, N-1))$. The step is generic by design: agglomerative clustering is the instantiation used here [@ward1963hierarchical], k-means or any other partition method would serve [@lloyd1982kmeans], and the number of modes is selected by the silhouette criterion, which is standard and is used unmodified [@rousseeuw1987silhouette; @pedregosa2011sklearn]. The framework claims the *presence* of a partition step, not its implementation.

Each mode carries three quantities that the later stages consume: its centroid in the raw pose coordinates, its mean peak loss $\bar{L}_C$, and a representative $\mathrm{rep}(C)$, defined as the member nearest the cluster mean in the standardised feature space. The dominant mode $C^\star$ is the one with the most members, ties broken by mean peak loss.

Two honest seams belong here rather than in a later section. When fewer than four failures remain, the silhouette sweep is skipped and each failure becomes its own singleton, so in the late rounds of a budget the partition is inactive and the round is allocated by the fallback rule described below. The frequency of that event, and its consequences, are measured (diagnostics D2 and D4). And the modes the partition discovers are geometric, so they recover cause only to the extent that configuration determines cause; the measured agreement between a geometric mode and a root cause is 0.78 to 0.93 (diagnostic D1), and the framework's claim about semantic modes is qualified by that number wherever it is made.

![Figure 2. The three failure modes discovered on Push-T by clustering the geometric descriptor at the flagged step. Each row shows three rollouts assigned to one mode, annotated with the block's orientation error and the distance between the end-effector and the block. The partition recovers behaviourally distinct failures: the block is delivered to the goal but rotationally wrong, the arm never makes contact, and the block is moved but left badly rotated and abandoned. The modes are found from geometry alone, under both observation modalities, with no visual embedding.](../figures/clustering_modes_pushT.pdf)

#### Prioritise

Two decisions are made in this stage, and they are the pair the framework owns. The first chooses which mode this round's demonstration is spent on. The second chooses which failures are shown to the prescription model as evidence.

**The cluster memory.** Left to itself, a rule that always targets the largest or the highest-loss mode will target the same mode round after round, because one demonstration rarely removes a mode outright. The framework keeps a memory $\mathrm{Mem}$ of the centroids of every mode it has already corrected, tagged with the round in which the correction happened, and penalises a candidate mode in proportion to how recently and how closely it has been corrected,

$$P_{\mathrm{mem}}(c) \;=\; \sum_{(r_i,\, c_i)\, \in\, \mathrm{Mem}} \gamma^{\,\max(0,\; r - r_i)} \, \exp\!\left(-\,\frac{\|c - c_i\|_2^2}{2\sigma_{\mathrm{mem}}^2}\right), \tag{8a}$$

$$C_{\mathrm{tgt}} \;=\; \arg\max_{C \,:\, |C| \,\ge\, |C^\star| - 1} \Big(\, \bar{L}_C \;-\; \lambda\, P_{\mathrm{mem}}\big(c_C\big) \,\Big). \tag{8b}$$

The recency discount $\gamma$ lets a mode become eligible again once the policy has moved on, the kernel width $\sigma_{\mathrm{mem}}$ sets how far the penalty reaches in the workspace, and the weight $\lambda$ sets how hard it pushes; values are given in the setup. The constraint $|C| \ge |C^\star| - 1$ keeps the target within one member of the dominant mode, so the memory rotates the target across the failure distribution without ever letting it wander onto a mode that barely exists. Setting $\lambda = 0$ removes the memory entirely and recovers a plain highest-loss rule, which is the memory-off ablation.

One property of this term must be stated where the term is defined, and not buried in the ablations. The kernel width $\sigma_{\mathrm{mem}}$ is a single global constant, and the tasks do not share a spatial scale. On Door, whose reset range is on the order of a centimetre, and on GridWorld, whose centroids are in grid-cell units, the kernel is degenerate at every width swept in ablation A13: it either saturates near one or collapses to an identical-centroid indicator, and in both cases the penalty is applied almost uniformly and is arithmetically close to no penalty at all. The memory is therefore active on four of the ten settings and inert on the rest. A per-task $\sigma_{\mathrm{mem}}$, expressed as a fraction of that task's own reset range, is the identified fix, and it has not been run. The global constant is a limitation of this instantiation, and it is reported as one.

Nothing in the literature does quite this. Coverage-driven selection over a representation space [@sener2018coreset], batch acquisition that mixes uncertainty with diversity [@ash2020badge] and the reweighting of intervention data [@mandlekar2020iwr; @liu2023sirius] are the nearest relatives, and none of them is a cross-round memory over discovered failure modes.

**The context set.** The prescription model is not shown every failure in the target mode. It is shown a small set $S$ of cited failures, capped at $\kappa$ members and built by three rules,

$$S_0 \;=\; \big\{\mathrm{rep}(C_{\mathrm{tgt}})\big\} \cup \big\{\arg\max_i \mathrm{peak}_i\big\}, \qquad S \;\leftarrow\; S \cup \Big\{ \arg\max_{i \,\notin\, S} \ \min_{j \in S} \ \big\| \tilde{X}_i - \tilde{X}_j \big\|_2 \Big\} \ \ \text{until } |S| = \kappa. \tag{9}$$

The representative of the target mode is forced into the set, because without it the model can be asked to fix a mode of which it has seen no example. The worst-loss failure is seeded next. The remaining slots are filled by farthest-point selection, which is standard [@eldar1997fps] and is used here to make the cited failures span the mode rather than crowd its loss peak. Ablation A9 removes each rule in turn and finds the ordering the mechanism predicts, with the forced representative the most damaging to remove.

#### Prescribe

The prescription model [@yang2025qwen3] receives the target mode's anchor geometry, the cited failures in $S$ with their root-cause labels, and the rendered constraints of the task, and returns one demonstration request together with an integer confidence score and a one-line rationale for it. The request takes one of two forms.

A **targeted correction** names one cited failure. That exact episode is re-instantiated, and the expert takes over at the flagged step and completes it. The mode is tight, or one failure clearly stands for it, and the demonstration is the correction of that failure on-policy.

A **bridging placement** names two or three cited failures and asks for a new configuration positioned between them, from which the expert demonstrates a complete episode. Bridging is the only part of the framework that changes the environment's configuration rather than selecting an episode, and it is what allows a prescription to be *easier* than any failure it addresses: when a mode lies far outside anything the current policy can solve, a targeted correction is a large distributional jump, and a bridged one is a step the policy can absorb. Bridging is selected in 18% to 30% of accepted prescriptions across the ten settings, so it is exercised rather than decorative.

Which of the two arms exists is a property of the task, and the framework reads that property from the knowledge store rather than hard-coding it. Wipe randomises a path of dirt markers rather than the pose of a single object, so there is no object pose to place in a middle ground; the task's graph declares the task targeted-only and the second arm is removed from the prompt.

**Feasibility verification against the knowledge-augmented graph.** A prescription is a request for a configuration of the world, and a language model asked for a configuration will sometimes ask for one the world cannot produce: an object outside the reachable set, a pose outside the spawn range, a grid layout with no path from start to goal. The knowledge-augmented graph (KAG) is the store that makes such a request checkable. It holds explicit environmental constraints as structured key-value knowledge, not as prose: workspace bounds, object and spawn ranges, reachability, controller limits, the success predicate, and the task's failure-mode and phase vocabulary. It is not a document store to be retrieved from in the manner of retrieval-augmented generation [@lewis2020rag; @edge2024graphrag]; it is closer to the explicit, queryable environment and action knowledge of a robot knowledge base [@tenorth2013knowrob], and it is queried during verification.

Verification is a loop, and it is the mechanism of Equation 10. The prescription model proposes; the constraints are retrieved from the graph; a map $g$ turns the proposal into a concrete reset specification $\xi$; the specification is checked against the retrieved constraints; and if a constraint is violated the violation is returned to the model as feedback, which proposes again:

$$
\begin{aligned}
\mathrm{cmd}^{(j)} &= \mathrm{LLM}\big(\, A,\ S,\ \mathcal{K},\ \text{violation}(\xi^{(j-1)}) \,\big), \qquad \xi^{(j)} \;=\; g\big(\mathrm{cmd}^{(j)}\big), \\[2pt]
V(\xi) &= \mathbf{1}\big[\, \xi \in \mathcal{W}_{\mathcal{K}} \,\big] \;\wedge\; \mathbf{1}\big[\, \mathrm{reachable}_{\mathcal{K}}(\xi) \,\big] \;\wedge\; \mathbf{1}\big[\, \mathrm{valid\text{-}path}_{\mathcal{K}}(\xi) \,\big], \\[2pt]
\xi^\star &= \xi^{(j)} \ \ \text{for the first } j \le J_{\max} \text{ with } V\big(\xi^{(j)}\big) = 1, \qquad \text{else } \xi^\star = \text{nearest untried failure},
\end{aligned}
\tag{10}
$$

where $\mathcal{K}$ is the task's graph, $\mathcal{W}_{\mathcal{K}}$ its workspace bounds, and the conjuncts of $V$ are the constraints the graph actually stores for that task. On the manipulation tasks the reachability and workspace conjuncts are box constraints on the object pose, padded from a measurement of the simulator's own reset sampler, so a prescribed configuration can never leave the task's native reset distribution. On the grid task the constraint is not a box at all but a path-validity predicate: the prescribed layout must place start, goal and obstacles on distinct in-grid cells and must admit an obstacle-free path from start to goal, and that predicate is decided by breadth-first search [@cormen2022algorithms], with A\* available for the same purpose [@hart1968astar]. Neither search is ever the expert. The GridWorld expert is a human, and the search is the checker that decides whether the human can be asked for a demonstration at all.

A failed attempt consumes no budget, because the budget counts demonstrations collected, not prescriptions proposed. After $J_{\max}$ attempts the round falls back to the deterministic rule of taking the nearest untried recorded failure, which is a correction the environment is guaranteed to be able to instantiate. The propose-verify-revise pattern is not new in itself: a language model's proposal has been checked by an external planner and the planner's verdict returned as feedback [@liu2023llmp], and the check has been iterated to convergence [@chen2024autotamp]. What the framework adds is the object being verified, which is a request for a training demonstration rather than a plan to be executed.

**Policy solvability.** Feasibility asks whether the environment can instantiate the prescribed configuration. A second and separate question is whether the configuration is worth an expert's time at all. A prescription that the current policy can already solve carries no information: the expert would demonstrate a behaviour the policy has, and a unit of a restricted budget would be spent for nothing. The architecture therefore contains a second check, drawn as its own loop. The prescribed configuration $P = \xi^\star$ is rolled out under the current policy, and

$$\mathrm{SR}_{f_\theta}(P) \;\ge\; \tau_{\mathrm{solve}} \quad \Longrightarrow \quad \text{revise } P, \tag{11}$$

so that a solvable prescription is returned to the prescription model rather than to the expert. The nearest intellectual relatives are the reverse-curriculum and reset-state literatures, which choose start states by what the learner can and cannot yet do [@florensa2017reversecurriculum; @eysenbach2018leavenotrace], and they are cited as neighbours rather than as precedents for this check.

Two things must be said plainly about Equation 11. The two checks are distinct mechanisms and must not be run together in the reader's mind: the first rejects a configuration the world cannot produce, the second rejects a configuration the policy does not need. And the solvability check is a design element of the framework as drawn in the architecture; it is not exercised in the experiments reported in this chapter, it is not ablated in the ablation programme, and no number anywhere in this report is attributable to it. Implementing and ablating it is outstanding work, and the honest position is to describe it as an element of the architecture and to claim nothing for it.

#### Naming the discovered failure modes

A partition returns integers. A method that reports "the policy fails in mode 2" has told the reader nothing, and the framework's prescriptions are only legible because its modes carry names. The naming is a three-step pipeline, and the precise version of it is more defensible than the loose one.

Modes are born nameless. The partition of Equation 7 runs on the standardised geometric descriptors and uses no output of any language model, so a mode at this point is an integer index over a set of failures.

Each failure, separately, is assigned a root cause and a trajectory phase by the reasoning model, and the model may only choose from the enumerated categories stored as failure-mode and phase nodes in that task's knowledge-augmented graph. The prompt says so explicitly. The vocabulary of names is authored in the graph, and the model's job is assignment rather than invention.

A mode's name is then the majority root cause among its members. The fraction of a mode's failures that share its dominant label is its purity, and purity is measured rather than assumed: it runs from 0.78 to 0.93 across the ten settings, with a mean of 0.877 (diagnostic D1). It is lowest on Wipe, where the same end-effector position can correspond to insufficient contact force, to a missed patch, or to a premature stop, and where geometry consequently cannot separate causes that a human would call different. The names that reach Figure 2 are readable renderings of the Push-T graph's own failure-mode labels, and they are not coined by the model.

#### Algorithm

```
Input:  initial demonstration set D0; policy class f_theta; expert pi*;
        budget B; demonstrations per round D; knowledge graph K;
        context-set cap kappa; memory constants (gamma, sigma_mem, lambda);
        re-prescription limit J_max.
Output: the trained policy f_theta.

 1: train f_theta on D0 by behaviour cloning
 2: Mem <- empty
 3: for r = 1 to B do
 4:     roll out f_theta on a fresh pool of episodes; record the per-step loss (Eq. 1)
 5:     F_r <- the failed episodes of the pool                                  (Eq. 2)
 6:     for each failure i in F_r do
 7:         t*_i <- first sustained threshold crossing of the loss, else its peak (Eq. 5)
 8:         phi_i <- the 6-D geometric descriptor at t*_i                        (Eq. 6)
 9:         VLM describes the start, t*_i and end frames of failure i
10:         reasoning LLM assigns a root cause and a phase from the taxonomy in K
11:     end for
12:     standardise the descriptors; k* <- argmax silhouette; cluster into modes  (Eq. 7)
13:     name each mode by the majority root cause of its members
14:     C_tgt <- the mode maximising mean peak loss minus the memory penalty      (Eq. 8)
15:     S <- forced representative of C_tgt, plus the worst-loss failure,
16:          plus a farthest-point fill, up to kappa members                      (Eq. 9)
17:     for j = 1 to J_max do
18:         cmd <- prescription LLM(anchor of C_tgt, S, K, previous violation)
19:         xi <- g(cmd)                        # targeted correction or bridging placement
20:         retrieve the constraints of K; if V(xi) = 1 then break                (Eq. 10)
21:         violation <- the constraint that xi breaks
22:     end for
23:     if no feasible xi was produced then xi <- nearest untried failure in F_r
24:     if the current policy already solves xi then revise xi                    (Eq. 11)
25:     collect D demonstrations from pi* at xi
26:     D_r <- D_{r-1} + those demonstrations; append the centroid of C_tgt to Mem
27:     retrain f_theta from scratch on D_r at the per-task cadence               (Eq. 3)
28: end for
29: return f_theta
```

The loop header is symbolic. A budget of any size runs the same algorithm, and the value used in the experiments appears in the setup. Lines 4 to 11 are the perceive stage, line 12 the partition, lines 14 to 16 the prioritise stage, and lines 17 to 25 the prescribe stage. Line 23 is the fallback, and its cost when it fires often is measured in the knowledge-graph ablation. Line 24 is the solvability check, which is drawn in the architecture and not exercised in the reported runs.

### Architecture

![Figure 3. The DISEIL framework. A policy is trained on an initial expert demonstration set and rolled out on held-out episodes. A query gate flags the step t* of greatest policy uncertainty. Two descriptions of that step are formed: a geometric descriptor, which the cluster engine partitions into k failure modes against a memory of previously corrected modes, and a vision-and-language description of the start, t* and end frames, which a reasoning model turns into a root-cause account grounded in geometry facts retrieved from the knowledge-augmented graph. A prescription model proposes the configuration P of the next demonstration. The proposal is screened before any expert time is spent: if the current policy can already solve P, the prescription carries no information and P is revised. The expert then demonstrates the surviving configuration, the demonstration is added to the dataset, and the policy is retrained for the next round.](<../figures/Architectural Diagram.pdf>)

The figure is read left to right, and each block is described below by what it consumes, what it emits, and what breaks without it. The last of those three is answered by an ablation, and the ablation is named.

**Initial expert demonstrations and Train Policy.** The loop opens on a database of seed demonstrations and a behaviour-cloning fit. The size of that set is not incidental. It is chosen to place the policy's starting success rate inside a target band, because a policy that fails everywhere produces a failure set with no structure to partition, and a policy that fails nowhere produces no failure set at all. The reasoning behind the band, and the sweep that sets the count, are given in the setup and in the information-gain argument.

**Policy Rollout.** The trained policy is rolled out on held-out episodes. The block emits the round's failure set. Without it there is nothing to reason over, and every method in the comparison contains it.

**Flag Uncertainty at t\*.** The orange gate in the figure is the query gate, and it is the component the DAgger family consists of. It reads the per-step loss and returns the step at which the policy first becomes unreliable. The framework uses it as the DAgger family uses it, and then keeps going.

**The perception branch.** Three rendered frames of each failure, at the start, at $t^\star$ and at the end, feed the vision-language model by the dotted arrows, whose output feeds the reasoning model, which performs the root-cause analysis. The knowledge-augmented graph feeds the reasoning model by a dashed arrow, supplying the geometry facts and the failure vocabulary. Removing the vision-language model costs about one success-rate point, and removing the graph costs about two and a half, chiefly by raising the rate at which rounds fall back.

**The geometric branch.** A long arrow labelled with the geometric descriptor at $t^\star$ runs across the top of the figure into a scatter panel of three coloured point clouds and then into the cluster engine, which emits $k$ failure modes. The cluster memory feeds the engine by a dashed arrow. The two branches leave the same gate and do not meet until the prescription model, which is the visual statement of the fact that the partition uses no output of any foundation model. Removing the cluster engine is the single most damaging knockout in the ablation programme.

**Prescription LLM.** The reasoning model and the cluster engine both feed the prescription model, which prescribes the configuration $P$ of the next demonstration. It emits a targeted correction or a bridging placement, with a confidence score.

**Policy Rollout on P, and the return arrow.** The prescribed configuration is rolled out under the current policy, and the orange dashed arrow labelled "Solvable ⇒ Revise P" returns to the prescription model. That arrow is the solvability check of Equation 11. The figure draws this loop and does not draw a second return arrow for the feasibility check of Equation 10: in the current drawing the knowledge-augmented graph appears only as a one-way input to the reasoning model. The feasibility loop is a mechanism of the implemented framework and the solvability loop is not, so the drawing understates one and depicts the other, and the two are described separately in the prescribe stage above for exactly that reason.

**Expert Demo, Add demo, Update Policy.** The surviving configuration goes to the expert, whose single demonstration is added to the dataset, on which the policy is retrained, and the teal arrow returns to the rollout for the next round. One unit of the budget has been spent. Everything the framework has done in the round bears on the choice of what the policy is trained on, and nothing in it touches how the policy is trained, which is what makes the comparison against the DAgger family a controlled one.

### Representative prompts

The three model calls of a round are reproduced below in the form the implementation writes to disk. Every run stores its exact prompts, so these are records rather than reconstructions.

The perception call takes the system instruction and the three frames.

```
SYSTEM:  You are analysing a robot manipulation failure from rendered frames. Be
         concrete and spatial; describe what you actually see, not generic advice.

USER:    You are analysing a robot manipulation failure. The attached frames are, in
         order: start, high_loss, end (the peak-loss frame is the policy's most-uncertain
         step, t*=146).
         Task: A UR5e robot with a wiping pad (no fingers) must WIPE a trail of dirt
         markers off the table by pressing down and sweeping along the dirt path until
         all markers are cleared (coverage).
         Describe what went wrong. Focus on: where in the trajectory the failure occurs,
         the robot/gripper configuration at peak loss, and what object or contact state
         caused it. ~120 words, concrete and spatial.
         [start frame] [high_loss frame] [end frame]
```

The model's reply on that instance, from a Wipe run, reads: *"The failure occurs during the sweep at the beginning of the trajectory, around step t\*=146. At peak loss, the robot's wiping pad is positioned at the start of the dirt trail, but the pad is tilted and not fully in contact with the table surface. The gripper is misaligned, causing the pad to lift slightly and only partially engage the dirt. This lack of stable contact prevents effective wiping, leaving the initial segment of the trail untouched."* One defect in the prompt template is visible in the quotation and is recorded here rather than left for a reader to find: the template names the anchor the "peak-loss frame", while the frame actually passed is the first-crossing step of Equation 5. The wording is stale and the frame is correct.

The reasoning call is constrained to the graph's vocabulary and to strict JSON.

```
SYSTEM:  You are a robot-manipulation failure analyst. Classify the root cause and
         trajectory phase using ONLY the provided categories and the KAG facts.
         Output strict JSON, no prose, no code fences.

USER:    TASK: {task_description}
         {kag_text}
         VLM FAILURE DESCRIPTION (the only visual evidence): {vlm_report}
         Identify the root cause category and the trajectory phase where the failure
         occurred.
         root_cause in [grasp_failure, approach_failure, placement_error,
                        contact_instability, pose_mismatch, timeout]
         phase      in [pre_grasp, grasp, transport, placement, insertion]
         Output ONLY this JSON:
         {"root_cause": "<one of the categories>", "phase": "<one of the phases>",
          "rationale": "<one sentence grounded in the VLM description and a KAG fact>"}
```

Its reply on the same failure is the grounding claim in miniature, because the rationale cites both the visual evidence and a named node of the graph:

```json
{"root_cause": "contact_instability", "phase": "transport",
 "rationale": "The wiping pad tilts and lifts during the sweep (transport phase) as
 described in the VLM evidence, causing intermittent contact with the table surface and
 aligning with the KAG-reasoned 'contact_instability' failure due to loss of downward
 pressure/alignment."}
```

The prescription call states the budget rule to the model, offers the two arms, and requires a confidence line.

```
SYSTEM:  You are a demonstration coach for an interactive imitation-learning loop. Each
         round you spend ONE expert demonstration to fix the dominant failure mode. You
         decide HOW to spend it, grounded in the KAG facts and the per-failure analyses.
         Reason briefly, then end with EXACTLY two lines: (1) a decision line in the exact
         required format, and (2) a confidence line 'CONFIDENCE: <integer 0-100> - <one-line
         rationale>' reporting how confident you are that this demonstration will improve
         the policy.

USER:    (A) SELECT ep<ID> - one recorded failure represents the whole mode. That exact
             scene is re-run and the expert corrects it on-policy from the divergence point
             t*. Use when the cluster is TIGHT or one failure clearly dominates.
         (B) BRIDGE ep<ID>,ep<ID> - no single failure covers the mode. Prescribe ONE new
             object placement in the MIDDLE GROUND between 2-3 cited failures (e.g. failures
             at (1,1) and (5,5) -> a demo near (3,3)); the expert demonstrates from there.
             Use when the members are geometrically SPREAD but share a root cause.
```

The cited failures are handed to the model one line each, and the model answers in two lines. A Wipe round and a GridWorld round, taken verbatim from the logs, show the same machinery on a manipulator and on a discrete grid:

```
DOMINANT FAILURE CLUSTER (members with their VLM+analysis findings):
  - ep3000001: object_xy=(0.221,-0.027) progress=74/500  peak_loss=0.0434
               root_cause=contact_instability phase=transport
  - ep3000003: object_xy=(0.165,-0.068) progress=146/500 peak_loss=0.0181
               root_cause=contact_instability phase=transport
  - ep3000006: object_xy=(0.123,-0.157) progress=234/500 peak_loss=0.0186
               root_cause=pose_mismatch         phase=placement
===== RESPONSE =====
SELECT ep3000001
CONFIDENCE: 85 - Ep3000001's contact_instability root cause (higher peak_loss) at early
progress best represents the dominant failure mode, ensuring the expert demonstration
directly addresses unstable wiping pressure causing missed coverage.
```

```
DOMINANT FAILURE CLUSTER (members with their VLM+analysis findings):
  - ep3000004: object_xy=(1.000,2.000) progress=3/60 peak_loss=0.4643
               root_cause=hit_fire phase=corridor
  - ep3000013: object_xy=(0.000,1.000) progress=0/60 peak_loss=0.7154
               root_cause=timeout  phase=junction
  - ep3000016: object_xy=(1.000,2.000) progress=1/60 peak_loss=0.0126
               root_cause=hit_fire phase=approach
===== RESPONSE =====
SELECT ep3000004
CONFIDENCE: 75 - The cluster has two hit_fire failures at the same start cell (1,2),
allowing the expert to demonstrate safe corridor navigation from a common fire-adjacent
starting position.
```

The Wipe example also shows the imperfection of geometric naming from the inside: two of the three cited members carry the label `contact_instability` and the third carries `pose_mismatch`, so this mode has a purity of two thirds, and the reported mean of 0.877 is an average over instances like it. The confidence line is not decoration. Its correlation with the improvement the prescribed demonstration actually produces is reported in the results.

### Representative environmental constraints

The knowledge-augmented graph of a task is a JSON document with a fixed schema: metadata (domain, robot, controller, action dimension), typed nodes with key-value properties, relations between them, and a block of reasoning implications, one per failure mode plus a workspace constraint and a non-emptiness rule. A renderer turns the document into the text block that is injected into the reasoning and prescription prompts. The graph is authored once per task, and its constraints are measurements of the environment rather than opinions about it.

Push-T stores its bounds as typed workspace nodes and its controller as a node in its own right.

```json
{"id":"ws_tee","type":"Workspace","label":"Reliable tee init range",
 "properties":{"x":[-0.20,0.20],"y":[-0.25,0.05],"z":0.021}},
{"id":"ws_tcp","type":"Workspace","label":"Reliable tcp range",
 "properties":{"x":[-0.35,0.35],"y":[-0.35,0.35],"z":[0.02,0.08]}},
{"id":"ctrl","type":"Controller","label":"pd_joint_pos / rel_joint_pos",
 "properties":{"policy_action":"7 joint deltas (rel_joint_pos)",
               "expert_action":"PPO -> joint_delta_pos (same 7-joint space)"}},
{"id":"goal","type":"Goal","label":"Fixed goal T-pose",
 "properties":{"goal_offset":[-0.156,-0.1],"goal_z_rot_rad":1.5708,
               "fixed_per_episode":true}}
```

The predicate that Equation 10 checks is stored as an implication and is written in the imperative, because it is addressed to the model as much as to the checker:

> `"workspace_constraint": "Every prescribed config MUST keep tee_xyz within x[-0.20,0.20] y[-0.25,0.05] z=0.021 and tcp_xyz within x[-0.35,0.35] y[-0.35,0.35] z[0.02,0.08]; out-of-range poses are dropped (the PPO expert is unreliable there) and waste the round."`

> `"non_emptiness": "A failure is present, so the prescription MUST be a concrete, fully-specified config (non-empty tee_xyz, tee_zrot, tcp_xyz). Never emit an empty prescription - that collects zero demos and wastes the round."`

Door's constraint is tighter by an order of magnitude, and the numbers are a padded empirical measurement of the simulator's own reset sampler, so a prescribed configuration cannot leave the task's native reset distribution:

```json
{"id": "ws_door", "type": "Workspace", "label": "Reliable door-frame range",
 "properties": {"x": [-0.135, -0.108], "y": [-0.366, -0.340], "z": 1.10,
                "yaw_rad": [-1.82, -1.57]}},
{"id": "succ", "type": "SuccessCondition", "label": "Door open",
 "properties": {"metric": "hinge_qpos > 0.3 rad", "info_key": "success"}}
```

GridWorld shows that the store is not merely a bounding box. For a discrete task the environmental constraint is a reachability predicate, and the graph says so:

> `"workspace_constraint": "Every prescribed layout MUST keep start, goal, and the 3 fires as DISTINCT in-grid cells in [0..4]^2, with start != goal, Manhattan(start,goal) >= 4, and a fire-free BFS path from start to goal (fires never block all routes). Out-of-grid or unsolvable layouts are rejected and waste the round."`

Wipe shows the store doing something a bounding box cannot do at all. Its graph carries an implication that removes an entire arm of the prescription from the task:

> `"select_only": "Wipe randomizes a whole marker PATH, not a single object pose, so BRIDGE is infeasible - always choose SELECT of the most representative failed episode."`

The planner reads that implication structurally and the prompt omits the bridging option. The graph therefore does two jobs: it constrains where a demonstration may be placed, and it determines which prescription arms exist for a task at all. Both jobs are done by knowledge that is written down and checkable, which is the sense in which the framework's model of the environment is explicit rather than implicit in a network's weights.
