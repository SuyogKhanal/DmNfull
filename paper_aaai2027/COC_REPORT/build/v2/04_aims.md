# 4. Aims and approaches

The programme has three aims. Aim 1 raises the value of a demonstration within a round of interaction, and it is the aim on which work has been completed; its implementation, experiments and ablations are reported in Section 5. Aim 2 raises the value of a demonstration across the dataset the loop is building. Aim 3 raises it across tasks, embodiments and teachers, and prices it against the human time it consumes. Each aim is stated here as a problem, a formulation and a method. Each is the correction to the limitation the previous aim's own evaluation exposes.

## 4.1 Aim 1. Demonstration distillation under a fixed budget

### 4.1.1 Motivation and problem statement

A practitioner who holds a fixed allowance of demonstrations faces a question the scaling relationship of Section 1.1 does not answer. Policy performance rises with the number and the coverage of the demonstrations a policy is trained on [51], and the allowance cannot grow, so the only open question is what each demonstration in it should contain. That question is RQ1, stated in Section 3.2: under a fixed budget, does choosing which failure mode to correct and where the corrective demonstration begins yield a higher final success rate than choosing only when to intervene?

Write $B$ for the number of demonstrations the expert will supply beyond an initial set, and $D$ for the number acquired in one round of interaction, so that the loop runs for $B/D$ rounds. Neither symbol carries a value in this section. The framework is defined for any fixed budget and any per-round acquisition count, and the values at which it was validated appear once, in the experimental setup of Section 5.1.2. What the framework maximises under that budget is the information content of each demonstration: informally, how much of the policy's remaining error the demonstration can remove, and formally, the policy's per-step loss on the demonstration measured before it is trained on it. Section 5.1.5 is the argument that this quantity means what it appears to mean.

The gap it addresses is the first of the three stated in Section 3.1, and its two halves are the decisions the DAgger family leaves fixed by default: which failure to correct, and where the corrective demonstration begins. The adjacent selection literatures do not claim them either, because every one of them chooses from data that has already been collected, which is the distinction Table 2 draws. This framework prescribes a demonstration that does not exist yet and then has an expert produce it, and that operation is what the word distillation names in the title of the Aim-1 paper.

The policy is any function $f_\theta$ that maps an observation to an action and exposes a per-step loss $\ell_t$ at a state-action pair. The requirement stops there. A multilayer perceptron on a discrete grid, a convolutional network on grid images and a diffusion policy on a manipulator [17] all satisfy it, and all three are used in the experiments. The framework is a way of spending a demonstration budget, and the only thing it asks of the learner is a loss it can read.

### 4.1.2 Problem formulation

The interactive loop skeleton is shared by every method compared in this report, it is stated in Section 2.2, and it is not re-derived here [79]. What follows fixes the notation and isolates the one component that differs between methods.

The policy $f_\theta$ is trained on an initial demonstration set $\mathcal{D}_0$ by behaviour cloning. Rounds are indexed by $r$. At the start of round $r$ the policy is rolled out on a fresh pool of episodes drawn from the task's reset distribution, and the episodes it fails are collected into the round's failure set. The only requirement placed on the policy is that it expose a per-step loss at its own executed action,

$$\ell^{(i)}_t \;=\; \mathcal{L}\big(f_\theta,\; s^{(i)}_t,\; a^{(i)}_t\big), \tag{1}$$

where $i$ indexes an episode, $s_t$ is the observation at step $t$ and $a_t$ is the action the policy itself executed. For a diffusion policy $\mathcal{L}$ is the denoising loss, which is what Diff-DAgger uses as its gate signal [47]; for a discrete policy it is the negative log-likelihood of the executed action. The failure set of the round is

$$\mathcal{F}_r \;=\; \big\{\, f_i \;=\; (\tau_i,\; \ell^{(i)}_{1:T_i}) \;:\; \tau_i \text{ is a failed rollout of } f_\theta \,\big\}, \qquad N \;=\; |\mathcal{F}_r|, \tag{2}$$

with $\tau_i$ the trajectory and $T_i$ its length. Success is measured on a frozen held-out evaluation set that no method sees during acquisition, and the same set is used for every method.

Each round ends by acquiring $D$ demonstrations from the expert and aggregating them,

$$\mathcal{D}_r \;=\; \mathcal{D}_{r-1} \,\cup\, \{d_{r,1},\dots,d_{r,D}\}, \qquad \theta_r \;=\; \arg\min_\theta\ \mathbb{E}_{(s,a)\sim\mathcal{D}_r}\big[\mathcal{L}_{\mathrm{BC}}\big], \tag{3}$$

and the loop stops when the budget is exhausted, that is when $|\mathcal{D}_r| - |\mathcal{D}_0| = B$. Retraining is from scratch, at a per-task cadence, and follows standard practice.

Every method compared in this report instantiates the same three-part acquisition rule. Writing $A$ for the rule, a round's acquisition is fully specified by

$$A \;=\; \big(\underbrace{t^\star}_{\text{when}},\; \underbrace{C_{\mathrm{tgt}}}_{\text{which}},\; \underbrace{\xi}_{\text{where}}\big), \tag{4}$$

where $t^\star$ is the step at which the expert takes over, $C_{\mathrm{tgt}}$ is the failure mode the round is spent on, and $\xi$ is the specification of the state from which the corrective demonstration begins. The DAgger family fixes the second and third components trivially: $C_{\mathrm{tgt}}$ is whichever rollout tripped the gate first, and $\xi$ is the state the rollout was already in. DISEIL computes all three. The methodology below is a description of how the second and third are computed, and nothing else in the loop is changed, which is what makes the comparison a controlled one.

### 4.1.3 Methodology

DISEIL runs four stages once per round: perceive the round's failures, partition them into failure modes, prioritise one of those modes, and prescribe the $D$ demonstrations the expert is asked to supply. $B$ and $D$ are symbols throughout this section and throughout Algorithm 1. Standard machinery is used inside three of the four stages and is flagged as standard where it appears. The contribution is the pairing of a partition of the round's failures into modes with a prescription that is verified against an explicit model of what the environment permits before an expert is asked to satisfy it. This section states the framework; the implementation and the evidence are in Section 5.

![](../figures/Architectural_Diagram.pdf)

**Figure 2.** The DISEIL framework. A policy is trained on an initial demonstration set and rolled out; a query gate flags the step $t^\star$; a geometric branch and a perception branch describe that step; a prescription model proposes the configuration $P$ of the next demonstration; the expert demonstrates it and the policy is retrained.

Figure 2 draws the framework as implemented, and the four stages below are the four bands of it. The reading of the figure is given at the end of this section, once the stages it draws have been stated.

**Perceive.** The first act of a round is to say where each failure went wrong, in two languages at once: a geometric one, which the partition consumes, and a natural one, which the prescription consumes. The two descriptions are computed from the same step of the same episode and they never mix.

Each failure is anchored at a single step $t^\star_i$, defined as the first step at which the per-step loss crosses an out-of-distribution threshold $\eta$ and stays across it for $K$ consecutive steps,

$$t^\star_i \;=\; \min\Big\{\, t \;:\; \ell^{(i)}_u > \eta \ \ \text{for all } u \in [\,t,\, t+K\,] \,\Big\}, \qquad \text{with } t^\star_i \leftarrow \arg\max_t \ell^{(i)}_t \ \text{if no crossing occurs}. \tag{5}$$

The threshold is a quantile of the training-loss distribution, recalibrated at every retrain, which is the Diff-DAgger construction used unchanged [47]. Taking the first crossing rather than the loss peak is a deliberate departure from the obvious definition. In a failing episode the peak arrives late, so an expert who takes over at the peak inherits a badly corrupted state and has almost no episode left in which to correct it. The first crossing is early, the state is less corrupted, and the expert has budget to work with.

Each failure is then reduced to a six-dimensional vector $\phi_i \in \mathbb{R}^6$ computed from the privileged simulator state at $t^\star_i$. For the manipulation tasks the canonical form is

$$\phi_i \;=\; \big[\, p_{x},\; p_{y},\; \sin\theta,\; \cos\theta,\; \rho_i,\; \delta_i \,\big], \qquad \rho_i = \frac{t^\star_i}{T_i}, \qquad \delta_i = \big\|\, p^{\mathrm{tcp}}_i - p^{\mathrm{obj}}_i \,\big\|_2, \tag{6}$$

where $(p_x, p_y)$ is the planar position of the task-relevant object, $\theta$ its yaw, entered through its sine and cosine so that the wrap at $\pm\pi$ does not create a false distance, $\rho_i$ the fraction of the episode completed before the failure was flagged, and $\delta_i$ the distance between the end-effector and the object at the flagged step. Where a task randomises no yaw, the two orientation slots carry the task's own state variables in their place. Table 4 gives the instantiation used for each task. The descriptor is geometric under both observation modalities, and clustering is geometric for every run: no output of any foundation model enters the partition.

**Table 4.** The six-dimensional geometric descriptor, per task. The descriptor is computed from the privileged simulator state at the flagged step and is the same under both observation modalities.

| Task | The six components of $\phi$ |
|---|---|
| GridWorld | agent cell (2), signed offset to goal (2), progress, Manhattan distance to goal |
| Push-T | block planar position (2), $\sin\theta$, $\cos\theta$, progress, end-effector-to-block distance |
| Lift | cube planar position (2), progress, gripper-to-cube distance, gripper height, grasp indicator |
| Door | door-frame position (2), frame yaw, normalised hinge angle, end-effector-to-handle distance, progress |
| Wipe | remaining-dirt centroid (2), proportion wiped, end-effector-to-centroid distance, markers remaining, progress |

The descriptor is small because the failure sets are small, and the two facts are connected by distance concentration: adding weakly informative dimensions to a distance computation over a few dozen points pushes all pairwise distances toward each other and makes the merge order of the clustering arbitrary. The width is a design choice and it is swept in Section 5.1.7.3.

In parallel with the descriptor, three rendered frames of the failing episode, at its start, at $t^\star_i$ and at its end, are passed to a vision-language model [4], which returns a short spatial account of what went wrong. A text-only reasoning model then converts that account into a root cause and a trajectory phase, drawn from a closed taxonomy stored in the task's knowledge-augmented graph rather than invented by the model. The literature supports this division of labour. Vision-language models are competent at naming a cause when they are given structured evidence [22, 55] and unreliable at metric and spatial reasoning from pixels alone [15, 28]. The framework therefore asks them for the cause and computes the geometry itself.

**Partition.** The round's failures are partitioned into failure modes by a generic clustering step $\mathcal{C}$ applied to the standardised descriptors,

$$\tilde{X}_i \;=\; \frac{\phi_i - \mu}{\sigma_\phi}, \qquad \{C_1,\dots,C_{k^\star}\} \;=\; \mathcal{C}\big(\tilde{X},\, k^\star\big), \qquad k^\star \;=\; \arg\max_{k \in [2,\,k_{\max}]} \operatorname{sil}(k), \tag{7}$$

with $k_{\max} = \max(2, \min(6, N-1))$. The step is generic by design: agglomerative clustering is the instantiation used here [93], k-means or any other partition method would serve [56], and the cluster count is selected by the silhouette criterion, which is standard and is used unmodified [72, 81]. The framework claims the presence of a partition step, not its implementation.

Each mode carries three quantities that the later stages consume: its centroid in the raw pose coordinates, its mean peak loss $\bar{L}_C$, and a representative $\mathrm{rep}(C)$, defined as the member nearest the cluster mean in the standardised feature space. The dominant mode $C^\star$ is the one with the most members, ties broken by mean peak loss. When fewer than four failures remain, the silhouette sweep is skipped and each failure becomes its own singleton, so in the late rounds of a budget the partition is inactive and the round is allocated by the fallback rule stated below. The modes are geometric, so they recover cause only to the extent that configuration determines cause, and the framework's claim about semantic modes is qualified by the purity measured in Section 5.1.7.4 wherever it is made.

A partition returns integers, and a method that reports a failure in mode 2 has said nothing. A mode's name is therefore the majority root cause among its members, taken from the per-failure labels the reasoning model assigned. The model may only choose from the categories enumerated in that task's knowledge-augmented graph, so the vocabulary of names is authored in the graph and the model's job is assignment rather than invention.

**Prioritise.** This stage makes the pair of decisions the framework owns: which mode the round's demonstrations are spent on, and which failures are shown to the prescription model as evidence.

The round is spent on the mode of highest mean peak loss among the modes that are near-dominant, which is to say within one member of the largest,

$$C_{\mathrm{tgt}} \;=\; \arg\max_{C \,:\, |C| \,\ge\, |C^\star| - 1} \ \bar{L}_C. \tag{8}$$

The size constraint $|C| \ge |C^\star| - 1$ keeps the target inside the bulk of the round's failures, so a mode that barely exists cannot capture the round's budget on the strength of one badly failed episode. Coverage-driven selection over a representation space [83], batch acquisition that mixes uncertainty with diversity [3] and the reweighting of intervention data [54, 61] are the nearest relatives of this decision, and none of them chooses among failure modes discovered inside the round it is allocating.

A rule of this shape can return the same mode round after round, because one demonstration rarely removes a mode outright. The framework therefore carries a cluster memory, and the memory is a configurable, task-dependent component rather than a part of the core rule. When it is switched on, it holds the centroids of the modes already corrected, tagged with the round in which the correction happened, and subtracts from each candidate's score a recency-discounted Gaussian penalty on the distance to those centroids, under a discount $\gamma$, a kernel width $\sigma$ and a weight $\lambda$. Setting $\lambda = 0$ switches it off and returns Equation 8 exactly. It becomes active on a task whose failures form recurring clusters, where it rotates the budget away from modes that have already been corrected, and in an environment without that recurrence it costs negligible overhead and changes performance very little. The evidence says both halves of that plainly. Switching the memory off costs 0.6, 0.4 and 1.2 points on the three ablation settings of Section 5.1.7, the smallest of the seven knockouts; and the kernel is inert in most rounds whatever its constants are, because the candidate set of near-dominant modes is a singleton in 56 to 84 per cent of rounds on the settings with enough telemetry to measure it, and the dominant mode is then returned regardless of the penalty. The kernel width is a single global constant in this instantiation and the tasks do not share a spatial scale; Section 5.1.9 records the consequence and the identified fix. The memory is switched on in every run reported in Section 5, and it is a component of the instantiation and not part of the framework's contribution, which is the pairing of the partition with a feasibility-verified prescription.

The prescription model is not shown every failure in the target mode. It is shown a small set $S$ of cited failures, capped at $\kappa$ members and built by three rules,

$$S_0 \;=\; \big\{\mathrm{rep}(C_{\mathrm{tgt}})\big\} \cup \big\{\arg\max_i \mathrm{peak}_i\big\}, \qquad S \;\leftarrow\; S \cup \Big\{ \arg\max_{i \,\notin\, S} \ \min_{j \in S} \ \big\| \tilde{X}_i - \tilde{X}_j \big\|_2 \Big\} \ \ \text{until } |S| = \kappa. \tag{9}$$

The representative of the target mode is forced into the set, because without it the model can be asked to fix a mode of which it has seen no example. The worst-loss failure is seeded next. The remaining slots are filled by farthest-point selection, which is standard [24] and is used here so that the cited failures span the mode rather than crowd its loss peak.

**Prescribe.** The prescription model [96] receives the target mode's anchor geometry, the cited failures in $S$ with their root-cause labels, and the rendered constraints of the task, and returns the round's request for $D$ demonstrations together with an integer confidence score and a one-line rationale. Each requested demonstration takes one of two forms. A targeted correction names one cited failure; that exact episode is re-instantiated, and the expert takes over at the flagged step and completes it. A bridging placement names two or three cited failures and asks for a new configuration positioned between them, from which the expert demonstrates a complete episode. Bridging changes the environment's configuration instead of selecting a recorded episode, and it is what allows a prescription to be easier than any failure it addresses: when a mode lies far outside anything the current policy can solve, a targeted correction is a large distributional jump and a bridged one is a step the policy can absorb. Which of the two arms exists is a property of the task, and the framework reads that property from the knowledge store instead of hard-coding it. Wipe randomises a path of dirt markers rather than the pose of a single object, so there is no object pose to place in a middle ground, and the task's graph declares the task targeted-only.

A prescription is a request for a configuration of the world, and a language model asked for a configuration will sometimes ask for one the world cannot produce: an object outside the reachable set, a pose outside the spawn range, a grid layout with no path from start to goal. The knowledge-augmented graph is the store that makes such a request checkable. It holds explicit environmental constraints as structured key-value knowledge rather than as prose: workspace bounds, object and spawn ranges, reachability, controller limits, the success predicate, and the task's failure-mode and phase vocabulary. It is not a document store to be retrieved from in the manner of retrieval-augmented generation [23, 48]; it is closer to the explicit, queryable environment and action knowledge of a robot knowledge base [91].

Verification is a loop. The prescription model proposes, the constraints are retrieved from the graph, a map $g$ turns the proposal into a concrete reset specification $\xi$, the specification is checked against the retrieved constraints, and a violation is returned to the model as feedback so that it can propose again:

$$
\begin{aligned}
\mathrm{cmd}^{(j)} &= \mathrm{LLM}\big(\, A,\ S,\ \mathcal{K},\ \text{violation}(\xi^{(j-1)}) \,\big), \qquad \xi^{(j)} \;=\; g\big(\mathrm{cmd}^{(j)}\big), \\[2pt]
V(\xi) &= \mathbf{1}\big[\, \xi \in \mathcal{W}_{\mathcal{K}} \,\big] \;\wedge\; \mathbf{1}\big[\, \mathrm{reachable}_{\mathcal{K}}(\xi) \,\big] \;\wedge\; \mathbf{1}\big[\, \mathrm{valid\text{-}path}_{\mathcal{K}}(\xi) \,\big], \\[2pt]
\xi^\star &= \xi^{(j)} \ \ \text{for the first } j \le J_{\max} \text{ with } V\big(\xi^{(j)}\big) = 1, \qquad \text{else } \xi^\star = \text{nearest untried failure},
\end{aligned}
\tag{10}
$$

where $\mathcal{K}$ is the task's graph, $\mathcal{W}_{\mathcal{K}}$ its workspace bounds, and the conjuncts of $V$ are the constraints the graph stores for that task. On the manipulation tasks the reachability and workspace conjuncts are box constraints on the object pose, padded from a measurement of the simulator's own reset sampler, so a prescribed configuration can never leave the task's native reset distribution. On the grid task the constraint is a path-validity predicate rather than a box: the prescribed layout must place start, goal and obstacles on distinct in-grid cells and must admit an obstacle-free path from start to goal, and that predicate is decided by breadth-first search [20]. The search is never the expert. A failed attempt consumes no budget, because the budget counts demonstrations collected and not prescriptions proposed, and after $J_{\max}$ attempts the round falls back to the deterministic rule of taking the nearest untried recorded failure, which is a correction the environment is guaranteed to be able to instantiate. The propose-verify-revise pattern is not new in itself [16, 53]. What the framework adds is the object being verified, which is a request for a training demonstration rather than a plan to be executed.

Feasibility asks whether the environment can instantiate the prescribed configuration. A second and separate question is whether the configuration is worth an expert's time at all. A prescription the current policy can already solve carries no information, and a unit of a restricted budget would be spent for nothing. The framework therefore contains a second screen. The prescribed configuration $P = \xi^\star$ is rolled out under the current policy, and

$$\mathrm{SR}_{f_\theta}(P) \;\ge\; \tau_{\mathrm{solve}} \quad \Longrightarrow \quad \text{revise } P, \tag{11}$$

so that a solvable prescription is returned to the prescription model rather than to the expert. The nearest relatives are the reverse-curriculum and reset-state literatures, which choose start states by what the learner can and cannot yet do [25, 27]. The two screens are distinct mechanisms: the first rejects a configuration the world cannot produce, the second rejects a configuration the policy does not need. The solvability screen is a design element of the framework and no more than that. It is not exercised in the runs reported in Section 5, and no number in this report is attributable to it.

Algorithm 1 states the round. The loop header is symbolic, and a budget of any size runs the same algorithm.

\begin{algorithm}[t]
\caption{DISEIL}
\label{alg:diseil}
\begin{algorithmic}[1]
\Require initial demonstration set $\mathcal{D}_0$; policy $f_\theta$; expert $\pi^\star$; budget $B$; demonstrations per round $D$; knowledge-augmented graph $\mathcal{K}$; context-set cap $\kappa$; re-prescription limit $J_{\max}$
\Ensure the trained policy $f_\theta$
\State train $f_\theta$ on $\mathcal{D}_0$ by behaviour cloning
\For{$r = 1$ to $B/D$}
    \State $\mathcal{F}_r \gets$ the failed rollouts of $f_\theta$ on a fresh pool of episodes \Comment{Eq. 2}
    \ForAll{failures $f_i \in \mathcal{F}_r$}
        \State $t^\star_i \gets$ the flagged step of $f_i$ \Comment{Eq. 5}
        \State $\phi_i \gets$ the geometric descriptor of $f_i$ at $t^\star_i$ \Comment{Eq. 6}
        \State assign $f_i$ a root cause from the frames at $t^\star_i$ and the taxonomy in $\mathcal{K}$
    \EndFor
    \State $\{C_1,\dots,C_{k^\star}\} \gets$ partition $\{\phi_i\}$ into failure modes \Comment{Eq. 7}
    \State name each mode by the majority root cause of its members
    \State $C_{\mathrm{tgt}} \gets$ the near-dominant mode of highest mean peak loss \Comment{Eq. 8}
    \State $S \gets$ the cited failures of $C_{\mathrm{tgt}}$, at most $\kappa$ of them \Comment{Eq. 9}
    \Repeat
        \State $\xi \gets$ a configuration prescribed from $C_{\mathrm{tgt}}$, $S$, $\mathcal{K}$ and the last violation
        \State $v \gets$ the constraint of $\mathcal{K}$ that $\xi$ violates, if any
    \Until{$V(\xi) = 1$ or $J_{\max}$ attempts are spent} \Comment{Eq. 10}
    \If{$V(\xi) = 0$}
        \State $\xi \gets$ the nearest untried failure in $\mathcal{F}_r$
    \EndIf
    \If{$f_\theta$ already solves $\xi$}
        \State revise $\xi$ \Comment{Eq. 11}
    \EndIf
    \State $d_{r,1},\dots,d_{r,D} \gets$ $D$ demonstrations from $\pi^\star$ at $\xi$
    \State $\mathcal{D}_r \gets \mathcal{D}_{r-1} \cup \{d_{r,1},\dots,d_{r,D}\}$ \Comment{Eq. 3}
    \State retrain $f_\theta$ from scratch on $\mathcal{D}_r$ \Comment{Eq. 3}
\EndFor
\State \Return $f_\theta$
\end{algorithmic}
\end{algorithm}

**Architecture.** Figure 2 is read from left to right. The loop opens on a set of seed demonstrations and a behaviour-cloning fit, and the trained policy is rolled out on held-out episodes to produce the round's failure set. The query gate reads the per-step loss and returns the step at which the policy first becomes unreliable, which is the component the DAgger family consists of and the point at which DISEIL keeps going. Two branches leave that gate. The geometric branch carries the descriptor at $t^\star$ into the cluster engine, which emits the round's failure modes. The perception branch carries three rendered frames into the vision-language model and its output into the reasoning model, which is fed the geometry facts and the failure vocabulary of the knowledge-augmented graph. The two branches do not meet until the prescription model, which is the visual statement of the fact that the partition consumes no output of any foundation model. The prescription model emits the configuration $P$, which is screened for solvability by a rollout under the current policy before any expert time is spent on it. The expert then demonstrates the surviving configuration, the demonstration is added to the dataset, the policy is retrained, and one unit of the budget has been spent. Everything the framework does in a round bears on the choice of what the policy is trained on, and nothing in it touches how the policy is trained, which is what makes the comparison against the DAgger family a controlled one.

Two things the drawing understates are stated here. The feasibility loop of Equation 10 is a mechanism of the implemented framework, and the figure shows the knowledge-augmented graph only as a one-way input to the reasoning model. The solvability screen of Equation 11 is drawn as a return arrow, and it is the mechanism that is not exercised in the reported runs. The cluster memory is not drawn at all: it is a configurable, task-specific component that is active only when a task exhibits recurring failure clusters, it is the least damaging of the seven knockouts and not a headline contribution of the framework, and it is therefore left out of the framework figure and carried in the text as a feature of the loop that a task enables or disables.

## 4.2 Aim 2. Reverse vision-language-action, a coverage memory

### 4.2.1 The limitation Aim 1 leaves

DISEIL's selector reasons about the failure in front of it and knows nothing about the dataset behind it. Every input the language model receives in a round is drawn from that round: the flagged step of a failed rollout, three frames around it, a six-dimensional geometric descriptor, the partition of the round's failures into modes, and the constraints the environment imposes. The model can say that the policy failed at a particular configuration and that the cause is a grasp that closes before the gripper is aligned. It cannot say that the training set already holds six demonstrations of that cause, so the failure is not a gap in what the policy has been taught but a gap in how well it has been taught. Under a fixed budget, a demonstration spent re-teaching material the dataset already covers is a demonstration lost, which inverts the objective the programme sets out to serve.

Section 5 measures that limitation rather than asserting it, and three of its findings carry into the design of Aim 2. The knockouts of Section 5.1.7.2 show that each language-model component is worth about one success-rate point, and it is worth that little because the allocation decision is taken by a geometric partition that consumes no output from any foundation model. The one piece of cross-round state DISEIL holds is the cluster memory, which records where corrections have been placed and not what the training set contains; its kernel width is mis-scaled for the narrow-reset tasks (Section 5.1.9), and a per-task width would repair the scaling but not the representation, because a correctly scaled geometric memory still answers whether a demonstration has been placed near a location and not whether the dataset already contains a behaviour. The purity diagnostic of Section 5.1.7.4 measures where those two questions come apart: geometry mixes causes exactly where it separates them poorly, and no measurement inside the system tells the two failures apart. A memory indexed on coordinates cannot represent the difference between a failure mode it has covered and one it has merely visited.

The gap is therefore that no demonstration selector holds a model of the dataset it is building. Interactive imitation learning selects by local uncertainty and consults no training set [35, 47, 66, 79]. Selection and curation reason about a dataset but choose from a pool that already exists [3, 6, 64, 83, 99]. Language is used in robot learning on the way to an action [8, 11, 43, 69] or as an intermediate en route to a motor command [7, 100], and trajectory captioning exists as a primitive scored as description [44, 89, 94, 95], but none of that work keeps a cross-episode record of what has been taught or uses the description to decide what to collect next. Aim 2 addresses RQ2, stated in Section 3.2.

### 4.2.2 Core idea and proposed method

**Core idea.** A vision-language-action model consumes an instruction and an image and emits motor commands. Aim 2 runs the mapping backwards. A captioner $C_\phi$ consumes a trajectory's visual observations together with its executed action sequence and emits language:

$$C_\phi:\ \tau = \{(o_t, a_t)\}_{t=0}^{H-1} \ \longmapsto\ (\ell_{\text{traj}},\ \ell_{\text{act}},\ \ell_{\text{fail}}). \tag{12}$$

The three outputs are at three granularities. The trajectory caption states the intent the trajectory realises. The action caption segments the trajectory into sub-skill spans and names each span. The failure caption, emitted only for a failed rollout, states the root cause and is anchored at the flagged step $t^\star$, which is the same localisation signal Aim 1 uses [47]. The executed action sequence is the input that makes the inversion something other than video captioning: two trajectories can look nearly identical in pixels and differ in what the robot did, and proprioceptive and action signals are known to improve this class of caption and segmentation [89]. The captioner primitive is borrowed [89] and is not the contribution. What the captions are used for is the contribution. Every demonstration that enters the training set is captioned and its captions are stored, so the system accumulates a persistent, language-indexed record of what the policy has been taught, and the selector reads the coverage of that record before it prescribes.

**The captioner.** The captioner is a small vision-language model with an action channel. Keyframes are sampled from the trajectory, with the first frame, the flagged step and the final frame forced into the sample, and are encoded with a frozen pre-trained visual encoder of the kind already used for the image-modality policies [57, 68, 75]. The executed actions are quantised into tokens, interleaved with the visual tokens, and projected into a language backbone. The three caption heads are selected by a query token, so one forward pass serves all three. The failure head is where the two aims meet most directly: Aim 1's closed root-cause taxonomy was a compression forced by the absence of a memory, because a stateless pipeline can compare across rounds only by identity of labels. With a memory, comparison is by embedding and the vocabulary can be open.

**Training the captioner.** Manual captioning at the required scale is not affordable, so the training signal is constructed. Privileged simulator state supplies programmatic predicates, which a grammar renders into correct but stiff captions; a larger vision-language teacher paraphrases them into fluent language with decoding constrained to the template's facts; sub-skill spans come from change-point detection on the action, velocity and contact signals; and failure captions are built contrastively against the nearest successful demonstration of the same intent. The training objective adds to the language-modelling term a span-segmentation term, an alignment term that draws a caption and its trajectory together in a shared embedding space, and a fact-consistency term that re-parses the generated caption into predicates and penalises disagreement with the simulator's own. The alignment term is what makes the memory possible, because coverage is computed in the space it builds, and the fact-consistency term is what keeps a hallucinated cause out of it.

**The coverage memory.** The memory $M = \{(e_i, \ell_i, m_i)\}$ stores, for every demonstration in the training set, its caption embeddings $e_i$, its captions $\ell_i$, and metadata $m_i$ recording the task, the round of acquisition and the sub-skill spans it contains. Storage is at the trajectory level and at the span level, because a demonstration collected for one purpose usually teaches more than one thing. Coverage of a query embedding $e_q$ is the mean similarity of the query to the memory under a kernel $k$,

$$\rho(e_q) \;=\; \frac{1}{|M|} \sum_{i=1}^{|M|} w_i \, k(e_q, e_i), \qquad w_i \;=\; 1 - \widehat{\text{SR}}(e_i), \tag{13}$$

and the competence weight $w_i$ is the part that matters. A skill can be present in the dataset and still not learned, so each stored skill is weighted by one minus the policy's measured success rate on it, and a skill that is present but taught badly still reads as a gap. A memory that conflated presence with competence would refuse to re-teach precisely the material that has been taught badly, and the data-quality literature is the reason to expect the two to come apart [6]. The coverage gap is the low-$\rho$ region in the neighbourhood of the current failure's embedding, and it is what the selector is asked to fill. Aim 1's geometric descriptor is retained as one channel of the index rather than discarded, because Section 5.1.7.3 shows it produces the best-separated failure clusters and Section 5.1.7.4 shows where it stops working, which is where language is expected to earn its place. The coverage memory subsumes Aim 1's cluster memory as its degenerate case, and the mis-scaled kernel width does not survive the change, because coverage in an embedding space is not parameterised by a metric width that must be re-tuned against each task's reset range.

**The memory-conditioned selector.** One model does the reasoning that Aim 1 splits across three calls:

$$(P, R) \;=\; \mathrm{LLM}_\theta\big(\ell_{\text{fail}},\ \varphi,\ \mathrm{Retrieve}(M, e_{\text{fail}})\big). \tag{14}$$

It consumes the failure caption, the geometric descriptor $\varphi$, and what retrieval over the memory returns for the failure's embedding: the nearest stored skills, their support counts and their competence weights. Retrieval over a structured store is standard and is used as such [23, 48]. The model emits the prescription $P$ and a rationale $R$ in language, which states why that demonstration and not another, and which the person who holds the budget can read. The claim is not that one model is better than three. The claim is that the selector's reasoning is stateful and coverage-aware, and the ablation that decides between the two readings is stated in advance in Section 4.2.4.

The outer protocol is Aim 1's, which is the point of the design: the same query gate, the same budget, the same retraining step. Both screens of Aim 1 are retained. Feasibility verification against the constraint store runs unchanged, because Section 5.1.7.2 measures what happens without it, and a prescription must still be checkable against what the environment permits before an expert is asked to satisfy it. The naming function of the knowledge-augmented graph is absorbed into the memory, because the failure-mode vocabulary becomes open language; its verification function is not absorbed. Policy solvability is the second screen and is unchanged.

### 4.2.3 Architecture

![](figures_generated/aim2_architecture.pdf)

**Figure 3.** Proposed architecture for Aim 2. The imitation loop of Aim 1 is retained along the top left; the trajectory band supplies the frames and the executed actions the inversion consumes; the dashed enclosure holds the single language-grounded selector, in which the reverse vision-language-action model emits captions, the captions enter the language skill memory, and the unified model reasons over the coverage the memory returns together with the current failure.

The figure is drawn in three bands. The imitation loop along the top left is inherited from Aim 1 without change, and the figure keeps its four blocks to make visible that the protocol is unchanged. The trajectory band supplies the two inputs to the inversion, the frames and the executed action sequence, and the presence of the second is the whole difference between this and a video captioner. The selector, drawn as a dashed enclosure, holds the contribution: the reverse vision-language-action model maps frames and actions to language, the captions enter the language skill memory, the memory returns coverage, and the unified model holds two things at once, what has been taught and what has just gone wrong. The prescribe-and-learn band closes the loop along the bottom. The drawing does not yet carry the constraint store or the geometric channel of the fused index, both of which the method retains; the revision adds two edges and changes no block.

### 4.2.4 Evaluation strategy

The protocol is Aim 1's, unchanged, so that the head-to-head is a comparison of selectors and of nothing else. Push-T is carried forward as the continuity benchmark, because the Aim-1 numbers on it are directly comparable and a regression would be visible immediately. A graded manipulation suite [62, 102] is the sample-efficiency testbed, LIBERO [52] ships language instructions and a defined skill taxonomy that supply ground truth for caption scoring at no annotation cost, and Meta-World [97] provides a named skill inventory against which a claim of complementary rather than redundant selection can be checked. The primary metrics are demonstrations-to-threshold and the area under the success-versus-demonstrations curve, reported against Aim 1 over at least five seeds with confidence intervals. Secondary metrics are the final success rate, caption grounding scored as agreement between the generated caption's predicates and the simulator's, the redundant-demonstration rate, and the faithfulness of the rationale, tested by removing the coverage gap the rationale cites and checking that the selection changes. Baselines hold the policy and the retraining loop fixed and vary only the acquisition rule: passive behaviour cloning, random selection, the DAgger-family gates carried over from Aim 1 [35, 47, 66, 79, 101], and full DISEIL.

The decisive experiment is a matched-information ablation, and it is designed to be the first results figure of the Aim-2 paper. The selection loop is frozen and only the representation the selector consumes is varied, at equal capacity.

**Table 5.** The matched-information ablation. The selection loop is frozen and only the representation the selector consumes varies, at equal capacity.

| Arm | Representation the selector consumes | What it tests |
|---|---|---|
| Generated captions | the captioner's output, indexed in language | the proposed method |
| Geometric descriptors | Aim 1's descriptor and its cluster signatures | whether language adds anything over Aim 1's representation |
| Learned trajectory embedding | a trajectory encoder trained at the same budget, no language | whether the gain is coverage in some space rather than in language |
| Scrambled captions | captions with their content permuted across trajectories | whether the selector uses caption content at all |
| Oracle captions | human-written captions | the ceiling the generated captions are measured against |

The interpretation is committed before the experiment is run. The contribution holds only if generated captions beat both the learned embedding and the scrambled placebo, and trend toward the oracle. If the learned embedding matches the captions, the contribution is interpretability and cross-task composability rather than sample efficiency, Aim 2 is written as such, and Aim 3 proceeds on an embedding index. Three further ablations isolate the components: memory on against memory off, which should show its effect in the redundant-demonstration rate before it shows it in the success rate; a language-indexed memory against a raw-embedding memory, which separates the index from the memory; and the memory bolted onto three separate calls against the single unified model, which settles whether the unification is a mechanism or an engineering convenience.

The limitation Aim 2 will leave is stated here, because it is what makes Aim 3 the next step. The coverage memory is task-local: it records what one policy has been taught about one task, and a skill shared between tasks, such as the reach-and-align that precedes both a door pull and an insertion, cannot be credited across them. The supplier is a scripted expert who is always available and identically priced. Outside simulation the budget is not a count of demonstrations, it is a person's time, and demonstrations differ by an order of magnitude in what they cost that person to produce. The selector spends but does not price.

## 4.3 Aim 3. Demonstration demand across tasks, embodiments and teachers

### 4.3.1 The limitation Aim 2 leaves

Aim 3 is future work. Nothing in this section has been implemented and no number in it is a measurement.

Aim 2 gives the selector a memory of what it has been taught, and the memory is task-local. Skills are shared across tasks in a way that record cannot express: the reach-and-align that precedes a door pull is the reach-and-align that precedes an insertion, and under a task-local memory a demonstration collected for the first cannot be credited against the shortfall of the second. Generalist policies are trained on data pooled from many tasks and many robots precisely because that sharing exists [9, 43, 69, 70, 76]. What is pooled in that literature is supply, and the demand side has no representation at all.

The second limitation is the supplier. Aims 1 and 2 address a scripted or planner-based expert that answers instantly and charges the same for every question, and under that assumption the demonstration budget is a counter. Outside simulation the assumption fails on every clause. Demonstrations are produced by people, at a cost measured in minutes, and the cost varies by an order of magnitude with what is being asked: a short push and a long contact-rich insertion are one demonstration each and are not one price each. Large-scale collection efforts are budgeted in human hours and in operator interfaces, not in trajectory counts [42, 60]. A framework whose purpose is to spend a scarce resource well is therefore measuring the wrong resource. The Aim-2 selector spends but does not price: it can say which demonstration is most informative, and it cannot say what that demonstration is worth relative to what it costs.

The components of the answer exist and the pairing does not. Cross-embodiment data pooling trains one policy on many robots and holds no ledger of what the policy is short of [8, 43, 69, 70]. Cost-sensitive active learning weighs the value of a query against its labelling cost, for a datum that already exists [84]. Sub-trajectory retrieval shows that one collected trajectory serves several tasks, and it performs that crediting at consumption time, over a corpus that is already fixed [64]. Aim 3 addresses RQ3, stated in Section 3.2.

### 4.3.2 Proposed method

Aim 3 makes demonstration demand a priced object that transfers across tasks. Four components extend the components of Aim 2 rather than replacing them.

**A cross-task skill inventory.** Aim 2's captions are aggregated into a shared skill space, annotated with the embodiment on which each instance was demonstrated. A skill is a language-indexed cluster of sub-trajectory captions, for example aligning a gripper with a vertical handle and pulling along the hinge arc. It is neither a task nor a trajectory. Coverage is measured over the inventory, so the question of whether the policy can align with a handle is answerable without reference to the task in which the handle appeared. Open-ended skill libraries built by a language model are the nearest existing object [92], and benchmark suites with a named skill taxonomy supply the ground truth against which an inventory can be scored [52, 97]. The inventory is where the language index must earn the claim Aim 2 makes for it, because Aim 1's geometric descriptor does not compose across tasks: its coordinates are defined against one task's objects and one task's reset distribution.

**A demand model with a price.** For each skill in the inventory the demand model maintains a shortfall, the distance between the policy's competence on that skill and what the task family requires, weighted by how often the skill lies on the critical path of a task the policy is currently failing. Every candidate request then carries two numbers. The first is an expected information gain. Aim 1 measures information gain after a demonstration has been collected, as the per-step loss on it before retraining, and Aim 3 predicts the same quantity before collection, from the current coverage of the requested skill and the policy's measured competence on it. Aim 1's measurements are the training data for that predictor, which is the most direct link between the three aims: one quantity, measured in Aim 1, contextualised in Aim 2 and predicted in Aim 3. Expected information gain is the standard way to price a query before it is answered [36, 84], and what is new is the object being priced. The second number is an expected human cost, in minutes of teacher time, estimated from the length and difficulty of comparable demonstrations already collected. Selection maximises expected information gain per unit of teacher time, and the budget stops being a count of demonstrations and becomes a time budget, which is what it always was outside simulation.

**A non-expert teaching interface.** A demand is rendered as a request a person can act on: a natural-language instruction, a scene specification, and the reason the demonstration is being asked for. The scene specification has already passed the feasibility check, which is the propose-verify-revise loop Aim 1 runs against the knowledge-augmented graph and Aim 2 retains for exactly this moment [16, 53, 91], so no request is issued that the environment cannot instantiate or the robot cannot reach. Aim 1's solvability screen is retained and acquires an economic reading: a request the current policy can already satisfy wastes a person's time, and a framework that prices human time cannot afford to issue one. A non-expert demonstration also breaks the second half of Aim 1's information-gain argument, because a high pre-retrain loss on a non-expert demonstration is ambiguous between novelty and incompetence, so the loop requires a quality filter and the demand model must be able to reject a satisfied request. Learning from suboptimal and preference-based human input is the starting point for that filter [12, 18, 99].

**Transfer credit.** An arriving demonstration is captioned by the Aim-2 captioner, decomposed into its sub-skill spans, and credited against the outstanding shortfall of every task in the inventory that those spans partially satisfy. A demonstration requested to close a door-opening shortfall reduces the alignment shortfall of an insertion task, and the ledger records the reduction. Crediting at collection time is a different operation from retrieval at consumption time, because it can change what is collected next. Transfer credit is the mechanism by which a fixed number of human hours buys more competence than the same hours spent one task at a time.

The loop, for the algorithm float that will accompany the Aim-3 paper: roll out the generalist policy across the task family; caption the failures; update the competence estimate for every skill in the inventory; compute each skill's shortfall; price every candidate request by predicted information gain per unit of teacher time; verify the highest-value request against the constraint store and against policy solvability; issue the request to a teacher; caption the returned demonstration; credit it against every task whose shortfall it reduces; aggregate and retrain.

### 4.3.3 Component lineage across the three aims

Table 6 is the lineage the panel should be able to check. Each row is one object followed through the programme, and no row begins in Aim 3.

**Table 6.** The lineage of each component across the three aims.

| Component | Aim 1 | Aim 2 | Aim 3 |
|:-----------------|:---------------------------|:---------------------------|:---------------------------|
| Index of failures | 6-D geometric descriptor, one task, one round | caption embedding fused with the geometric descriptor | skill inventory, shared across tasks and embodiments |
| Memory | configurable penalty over corrected cluster centroids, active where failures recur | coverage of what the dataset contains, weighted by competence | demand ledger: shortfall per skill, settled by transfer credit |
| Value of a demonstration | measured after the fact as the pre-retrain per-step loss | conditioned on what the dataset already holds | predicted before collection, divided by expected teacher time |
| Screening | feasibility against the knowledge graph, and policy solvability | both retained | both retained; solvability becomes a cost argument |
| Supplier | scripted or planner-based expert | scripted or planner-based expert | non-expert human, whose time is the budget |

### 4.3.4 Evaluation strategy

Benchmarks are multi-task suites with a defined skill taxonomy, so that coverage is measured against a ground-truth inventory instead of against the system's own captions [39, 52, 63, 97], together with cross-embodiment evaluation on pooled multi-robot data, to test whether a skill demanded on one embodiment can be satisfied on another [70]. The primary metric is teacher-time-to-threshold, the number of minutes of human demonstration time required to bring the policy family above a target success rate. Demonstrations-to-threshold is reported beside it, and the gap between the two curves is itself a result, because a framework that reduces the demonstration count while raising the cost per demonstration has achieved nothing. Three further quantities are reported: transfer credit, in tasks advanced per demonstration; the rate at which a non-expert can act on a generated request; and the calibration of the price, as predicted against realised information gain, which is the successor of Aim 1's prescription-confidence measurement in Section 5.1.6. Controls remove one component of the proposal each: per-task demand with no transfer credit, uniform demand across skills, demand without a price, and Aim-2 single-task selection, which keeps the chain of comparisons unbroken from Aim 1 through Aim 3. The DAgger-family gates remain the outer reference point [47, 79].

The human study is the evaluation the Aim-3 claim depends on and the first point in the programme at which humans enter. Non-expert participants satisfy generated requests in simulation, using a teleoperation interface of the kind established for crowdsourced demonstration collection [54, 60]. Ethics approval will be sought from the Deakin human-research ethics committee before any participant is recruited, and the arrangements are set out in Section 7.2. The study makes the programme's central claim checkable from outside it: a person who has never read the thesis is handed a request, satisfies it, and the policy improves by approximately the amount the demand model predicted. If it does not, the calibration curve says so.

Two of Aim 1's components carry all the way through, and they are the reason the programme is one programme rather than three papers. The constraint store, which Aim 1 uses to verify a prescription before an expert is called, is what makes it possible to hand a request to a person without wasting their time. The geometric descriptor, which the purity diagnostic shows is semantically blunt where configuration does not determine cause, is what keeps language from collapsing kinematically distinct skills into one entry of the inventory. The value of one demonstration is measured in Aim 1, contextualised in Aim 2, and priced in Aim 3.
