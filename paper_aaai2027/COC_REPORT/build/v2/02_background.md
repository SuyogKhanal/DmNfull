# 2. Background and literature review

This chapter sets out the material the programme uses and locates the three aims in the literature they extend. The query gates of the DAgger family are both the baselines against which Aim 1 is measured and the point at which the programme departs from prior work, so they are treated at length and qualitatively. Everything the programme uses unmodified is named here, once, with attribution, and is flagged as standard practice: the behaviour-cloning objective, the aggregate-and-retrain skeleton, the query-gate template, the silhouette criterion, farthest-point selection, A\* and breadth-first search. None of it is re-derived later. Abbreviations are expanded at first use in this chapter and used in short form thereafter.

## 2.1 Imitation learning and behaviour cloning

A task is modelled as a finite-horizon Markov decision process, or a partially observed one when the learner sees images instead of privileged simulator state,

$$\mathcal{M} = \big(\mathcal{S},\ \mathcal{A},\ P(s' \mid s, a),\ R(s,a),\ H\big),
\qquad
\pi_\theta : \mathcal{S} \to \Delta(\mathcal{A}),$$

with $\mathcal{S}$ the state space, $\mathcal{A}$ the action space, $P$ the transition kernel, $R$ the reward, $H$ the horizon, and $\pi_\theta$ the learner's policy with parameters $\theta$. An expert $\pi^\star$ supplies trajectories, and the demonstration set is the collection of state-action pairs those trajectories contain,

$$\mathcal{D} = \big\{ (s_t, a_t) \ : \ a_t = \pi^\star(s_t) \big\}.$$

Behaviour cloning fits the policy to that set by supervised learning [5, 73]:

$$\theta^\star = \arg\min_\theta \ \mathbb{E}_{(s,a) \sim \mathcal{D}} \Big[\, \mathcal{L}_{\mathrm{BC}}\big(\pi_\theta(\cdot \mid s),\, a\big) \,\Big],$$

with $\mathcal{L}_{\mathrm{BC}} = -\log \pi_\theta(a \mid s)$ where the action space is discrete and $\mathcal{L}_{\mathrm{BC}} = \lVert \pi_\theta(s) - a \rVert_2^2$ where it is continuous. The reduction is old and it works: the first system of this kind steered a road vehicle from camera input through a three-layer network trained on logged human driving [73], and the formulation was later given as a named framework [5] and surveyed as one branch of learning from demonstration [2, 71].

The reduction carries one defect, and every method in this report exists because of it. Supervised learning assumes the training and test inputs are drawn from the same distribution. In imitation learning they are not. The policy is trained on the states the expert visits, and at deployment it visits the states its own actions produce. Any error moves the learner off the expert's state distribution, the next prediction is made on an input the training set under-represents, and the error grows. The statistics literature calls the mismatch covariate shift [85]; the imitation-learning consequence is quantitative. If the cloned policy incurs supervised loss $\epsilon$ under the expert's state distribution, its cost over a horizon $H$ can grow as $O(H^2 \epsilon)$, and the quadratic term is a property of the offline reduction and not an artefact of a loose bound. Allowing the learner to be corrected on the states it actually reaches removes it, leaving a bound linear in the horizon [78, 79].

The problem was visible in the first system. A policy trained only on a good driver's centred trajectory never observes a recovery from the road edge, because a good driver never produces one. Pomerleau's remedy was to synthesise the missing data: each camera image was shifted and rotated laterally, and the steering label was corrected to the command that would return the vehicle to the centre [74]. Noise injection into the expert's control stream is the modern off-policy version of the same idea [46].

The objective above is fixed for the whole programme, and it is fixed for every method compared. Neither Aim 1 nor either of the later aims changes the loss that is minimised, the optimiser, or the policy architecture. They change which demonstrations enter $\mathcal{D}$.

## 2.2 Dataset aggregation and the interactive loop

Dataset aggregation, introduced as DAgger, corrects covariate shift by moving the labelling effort onto the learner's own state distribution [79]. In round $r$ the current policy $\pi_{\theta_r}$ is rolled out, the expert is asked for its action at the states the rollout visits, those labelled pairs are added to the dataset, and the policy is refitted to the whole aggregate:

$$\mathcal{D}_{r+1} = \mathcal{D}_r \ \cup\ \big\{ (s,\, \pi^\star(s)) \ : \ s \sim d_{\pi_{\theta_r}} \big\},
\qquad
\theta_{r+1} = \arg\min_\theta \ \mathbb{E}_{\mathcal{D}_{r+1}} \big[ \mathcal{L}_{\mathrm{BC}} \big],$$

where $d_{\pi_{\theta_r}}$ is the state distribution induced by rolling out the round-$r$ policy. The analysis casts the loop as online learning against an adversarially chosen sequence of state distributions, so that a no-regret supervised learner attains a performance bound linear in the horizon rather than quadratic [79]. Variants replace action agreement with the expert's cost-to-go as the aggregation signal, which lets the learner be told how much a mistake costs and not only that it was a mistake [80, 88].

The loop has become a field rather than a single algorithm, with a taxonomy of feedback types and of who initiates the handover [14]. Human-gated variants give the decision to the person: the human watches the rollout and takes control when they judge it necessary, which removes the need for a machine-readable uncertainty signal at the cost of continuous human attention [41]. Data collected during an intervention has a different value from data collected on-policy, and can be reweighted accordingly, or used as the substrate of a deployment-time learning system in which the robot runs, a human intervenes, and the intervention becomes training data [54, 61].

One cost is the reason the rest of this chapter exists. The aggregation step as originally stated asks the expert to label every state the learner visits. An expert who must answer at every step of every rollout is an expert whose time scales with the number of rollouts, and outside simulation that expert is a person. The interactive loop trades the compounding-error problem for an expert-effort problem, and the query-efficient variants below are the field's answer to the second problem.

Aim 1 keeps this skeleton without modification, and so does every method it is compared against. The rollout, the aggregation and the retraining are shared. The only free variable is how the round's new demonstration is chosen, and holding everything else fixed is what makes the comparison a comparison.

## 2.3 The query gates of the DAgger family

Query-efficient interactive imitation learning replaces the ask-at-every-state rule with a gate. At each visited state the method computes a scalar score, compares it against a threshold, and hands control to the expert at the first state where the comparison fires:

$$\mathrm{Query}(s_t) = \mathbf{1}\big[\ \mathrm{score}(s_t) \ \gtrless \ \tau \ \big],
\qquad
t^\star = \min \{\, t \ : \ \mathrm{Query}(s_t) = 1 \,\}.$$

The expert then takes over from $t^\star$ and completes the episode, and the expert's segment is the round's new demonstration. The published methods differ in one place only: what they put in $\mathrm{score}(\cdot)$. The template is stated explicitly because it makes the family's shared limitation legible.

SafeDAgger learns an auxiliary safety classifier that predicts, from the current observation, whether the policy's action will deviate from the expert's by more than a tolerance, and hands over when the classifier predicts a large deviation. The classifier is trained on the policy's own rollouts, so the gate is a learned model of where the policy is unsafe and not a direct measurement of it [101].

DropoutDAgger reads the spread of the policy's own action distribution under Monte-Carlo dropout. Several stochastic forward passes are drawn at the visited state, and the expert is called when the sampled actions stop concentrating near the expert's action [65]. The signal is imported from the Bayesian deep-learning literature, where dropout at inference time is interpreted as approximate posterior sampling [29].

EnsembleDAgger replaces the dropout samples with an ensemble of independently trained policies and reads their variance, which it calls doubt. The gate opens on high doubt or on a large discrepancy between the ensemble mean and the expert's action, so that the epistemic term and the safety term each have an arm [66]. Ensembles are the second canonical deep-uncertainty estimator and, like dropout, are used here as published [45].

ThriftyDAgger adds a second quantity to novelty. Alongside the ensemble doubt it trains an estimate of task risk, a value function predicting the probability that the episode will fail from the current state and action, and it opens the gate on either. Both thresholds are set as quantiles of the observed distributions, calibrated so that the method hands over at a target switching rate, which is what makes the method budget-aware [35]. A related design reduces the number of context switches by using asymmetric thresholds for handing over and handing back, so that control does not oscillate between learner and expert [34]; it is context here and not a baseline.

Diff-DAgger reads the learner's own training loss. For a diffusion policy the per-step denoising loss on a state-action pair is a usable score of how far that pair lies outside the training distribution, so the method thresholds it at a quantile of the training-loss distribution, recalibrated at each retrain, and hands over when the loss stays in the tail for a run of consecutive steps [47]. Using the diffusion loss as an uncertainty signal is Diff-DAgger's idea. Aim 1 uses that same per-step loss, both to localise a failure within a rollout and to measure the information content of an acquired demonstration, and it also compares against Diff-DAgger as a baseline. Both facts are stated plainly wherever the signal appears.

A uniform-random control completes the comparison, and it is a control on the *which* decision and not on the *when* decision. Referred to in this report as Stagger, it holds no gate, no score and no threshold. Each round it draws one of the round's recorded failures uniformly at random and has the expert correct it. It is not a published method: it is a floor implemented in this project, and it is never labelled as a member of the DAgger family. Its purpose is to establish what an uninformed allocation of the same expert effort buys, so that any margin a gated method reports can be read against a random one.

| Gate | Scalar signal the gate reads | What opens the gate |
|---|---|---|
| SafeDAgger [101] | learned safety classifier predicting policy-expert deviation | predicted deviation exceeds tolerance |
| DropoutDAgger [65] | spread of Monte-Carlo dropout action samples | too few samples agree with the expert |
| EnsembleDAgger [66] | ensemble variance (doubt) and mean action discrepancy | either term exceeds its threshold |
| ThriftyDAgger [35] | ensemble novelty and a learned task-risk estimate | either term exceeds a budget-calibrated quantile |
| Diff-DAgger [47] | the policy's own per-step denoising loss | the loss stays in the tail of the training-loss distribution |
| *Uniform-random allocation control (Stagger)* | *none* | *no gate: one recorded failure of the round, drawn uniformly at random, is corrected* |

**Table 1.** The five published query gates of the DAgger family as instances of one template: a scalar score, a threshold and a handover. The methods differ only in the score. The last row is the uniform-random allocation control implemented in this project, which has no gate and is not a member of the family.

Descriptions here are qualitative by design. The thresholds, ensemble sizes and calibration constants each published method specifies are not reproduced, and the choices made when these gates were re-implemented for the comparison are recorded with the experimental setup in the progress report.

The three properties of the template set out in Section 1.2 are properties of the form of $\mathrm{score}(\cdot)$ and not of any one choice of it. The gate maps one visited state to one scalar, so it holds no representation in which two failures are the same mistake and a third is a different one. Nothing in $\mathrm{score}(\cdot)$ carries across rounds, so a persistent failure mode can absorb the entire budget. And the gate inherits the state that tripped it, so the corrective demonstration starts wherever the score happened to cross the threshold. Each of the five published gates answers *when* to hand over, and answers it well. Neither *which* of a batch of failures to correct nor *where* the corrective demonstration should begin is answered by any of them.

## 2.4 Uncertainty estimation

The gates read their signals from the deep-uncertainty literature, and the three families of signal are worth separating because their limitations are shared. Monte-Carlo dropout treats dropout at inference as approximate posterior sampling and reads the spread of the resulting predictions [29]. Deep ensembles train several independent members and read their disagreement, which is simpler to implement and generally better calibrated [45]. Density-style signals score how far an input lies from the training distribution, and the per-step denoising loss of a diffusion policy is one such score: a state-action pair the model has not seen produces a high reconstruction error, which is what makes it usable as a query trigger [47].

All three produce a number attached to a state. A number attached to a state is enough to decide whether that state is a problem. It is not enough to decide whether that state's problem is the same problem as another state's, because two scalars can be equal for entirely different reasons and can differ while describing one underlying failure. The comparison the DAgger family cannot make is a comparison between failures, and no refinement of the scalar supplies it. What supplies it is a representation in which failures are points and not magnitudes, which is where clustering, and the standard machinery of Section 2.6, enter the programme.

## 2.5 Policy classes

Explicit regression onto expert actions is a poor fit for demonstration data whose action distribution has several peaks. Where two different actions are both correct at a state, a network trained under a squared-error loss learns their average, and the average may be correct under neither of the two actions. Energy-based and generative formulations of the policy fit such distributions instead of averaging them [26]. Denoising diffusion probabilistic models are the generative family that has proved most usable for this [33]. Applied to trajectories they give a planner [40], and applied to short action sequences conditioned on recent observations they give the visuomotor diffusion policy that is the learner for the robot tasks in this programme [17]. The conventions for training such policies on offline human-style manipulation data, including observation encoders and the treatment of action chunks, follow the empirical study that established them [62].

A diffusion policy is trained by noising the clean action target $x_0 = a$ over $K$ steps and learning to reverse the corruption,

$$x_k = \sqrt{\bar\alpha_k}\, x_0 + \sqrt{1 - \bar\alpha_k}\, \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I),$$

with the network trained to predict the noise, or an equivalent reparameterisation of it, and the resulting per-pair denoising loss written $L_{\mathrm{dif}}(s, a)$. Evaluated at the state the policy visited and the action it executed, that loss gives a per-step signal along a rollout,

$$\ell_t = L_{\mathrm{dif}}\big(s_t,\, a_t\big).$$

The quantity $\ell_t$ is the one Diff-DAgger thresholds [47], and the one Aim 1 uses to localise the step at which a rollout goes wrong.

The framework is stated over any policy $f_\theta$ that exposes a per-step loss at a visited state under the executed action. A multilayer perceptron trained with cross-entropy on a discrete grid exposes one. A convolutional network on grid images exposes one. A diffusion policy exposes one, in the form above. Nothing else about the policy enters the loop: not its architecture, not its action parameterisation, not whether the observation is a state vector or an image. The programme is therefore run on all three policy classes, and the diffusion policy is one instantiation of the requirement and not the requirement itself.

## 2.6 Standard machinery

Five routines are used unmodified. Each is named here, cited, and flagged as standard, so that the method chapter can use them without appearing to claim them.

The partition step is generic. Aim 1 groups a round's failures into failure modes with a clustering step $\mathcal{C}$, instantiated as agglomerative clustering under Ward's linkage [93]. The choice is one instantiation and not a commitment: k-means [56] or any other partition method that returns a labelling and a set of centroids would serve, and the framework is stated over $\mathcal{C}$ and not over the particular algorithm.

The number of clusters is chosen by the silhouette criterion, which scores a partition by comparing, for each point $i$, its mean distance $a(i)$ to the other members of its own cluster against its mean distance $b(i)$ to the members of the nearest other cluster,

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}},$$

and takes the mean over points [81]. The criterion is used exactly as published, and the number of clusters is the value that maximises the mean silhouette over a bounded range.

Diversity selection within a cluster is farthest-point, or k-centre, selection: given a set already chosen, add the candidate whose minimum distance to that set is largest, and repeat [24]. The greedy rule is standard and is used as such.

Path validity on the discrete grid task is checked with A\* [31] and breadth-first search [20]. Their role must be stated precisely, because it is easy to misread. They verify that a prescribed grid configuration admits a valid path from the start cell to the goal cell around the obstacles, which is a feasibility check on the prescription. They are never the expert. The GridWorld expert is a human, and the demonstrations on that task are human trajectories.

Clustering, the silhouette computation and the feature standardisation that precedes them are taken from a standard library implementation [72].

One point of ownership is settled here and not left to the method chapter. Pre-trained visual representations for manipulation supply the visual encoder for the image-modality policies. R3M is the one used here [68], and masked visual pre-training [75] and value-implicit pre-training [57] are the alternatives that were available. Their role ends at the policy. They do not supply the features that are partitioned: clustering in this programme is geometric in every run, under both observation modalities, over a low-dimensional descriptor of the robot and object configuration at the flagged step.

## 2.7 Language and vision-language models

Large language models, which are autoregressive models over text, and vision-language models, which condition the same generation on images, have been used in robotics in several distinct roles, and the distinctions matter because they determine which capability this programme depends on.

As planners, language models decompose a natural-language instruction into a sequence of steps over a fixed repertoire of learned skills. The decomposition is only useful if it is grounded in what the robot can actually do, which is why the influential version of the idea scores each candidate skill by the product of the language model's likelihood that the skill is useful and a value function's estimate that the skill will succeed from the current state [1]. As programmers, they write executable code against a perception and control interface, so that the plan is a program with loops and conditionals and not a flat list [50, 87]. Multimodal variants take sensor observations directly into the language model's embedding space, so that the plan is conditioned on what the robot sees and not on a textual scene description produced by another module [21]. As designers of objectives, they write reward functions and cost maps that a downstream optimiser consumes, which is the setting in which a language model's output is a specification and never an action [38, 58, 98].

Failure reasoning and self-correction matter more directly to this programme. A vision-language model given a summary of a robot's execution, in the form of a small number of frames and a record of what happened, can name the cause of a failure, and the naming is accurate enough to drive a recovery [55]. A model trained specifically on manipulation failures does better than a general one, which is evidence that the capability is a learnable perceptual skill and not an emergent accident [22]. Closed-loop textual feedback improves an embodied planner, because a plan that fails can be revised when the failure is described back to the model that wrote it [37], and verbal self-critique with iterative refinement is by now an established pattern in its own right [59, 86]. A language model can also be calibrated to recognise when it does not know, and to ask a human instead of guessing [77], which is the direct precedent for the request interface proposed in Aim 3.

Against those capabilities stands a well-documented weakness. Vision-language models are unreliable at metric and spatial reasoning from pixels alone. They mistake relative depth, distance and size, and they fail on spatial-relation questions that a human answers instantly [15, 28]. The failure is a property of how these models are trained and not a matter of model scale, and it is measured as such on purpose-built benchmarks.

The two findings together dictate a design commitment, and the commitment is inherited by the Aim-1 method without further argument. The model is reliable at naming a cause when it is handed structured evidence, and unreliable at recovering geometry from an image. It is therefore handed the geometry, in the form of a low-dimensional numeric descriptor and an explicit store of what the environment permits, and it is asked for the thing it is good at. The model that reads frames and the model that writes the prescription are open-weight instruction-tuned models [4, 96]. Neither ever emits a robot action. In this programme a language model is a component that reads a structured summary of the policy's failures and returns a request for a demonstration, and it is not a controller.

## 2.8 Structured environmental knowledge and constraint grounding

Retrieval-augmented generation conditions a language model's output on passages fetched from an external store, so that the model's factual claims are anchored in retrievable text and not in its parameters [48]. Graph-structured retrieval organises the store instead of treating it as a flat pile of passages, which improves queries whose answer is spread across many documents [23]. Robot knowledge bases predate both and do something different again: they hold explicit, queryable, symbolic knowledge about objects, actions and the environment, and a robot's planner asks them questions instead of reading them as text [91].

The knowledge-augmented graph used in Aim 1 belongs to the third tradition. It stores explicit environmental constraints as structured key-value entries: workspace bounds, reachability limits, the ranges within which objects may be placed and spawned, and the limits of the controller. A prescription is checked against those entries during verification, and no passage of the store is retrieved into the model's context as evidence in the manner of retrieval-augmented generation.

The pattern in which it is used is established. A language model's proposal can be handed to an external checker and the checker's verdict returned to the model as feedback: a plan can be validated by a symbolic planner [53], and the propose-verify-revise cycle can be iterated until the proposal passes [16]. Aim 1's feasibility loop is an instance of that pattern, with the knowledge-augmented graph as the checker, so the feasibility check is not the novel part of the method.

A second check in Aim 1 has no precedent in the list above and must not be conflated with the first. Before expert time is spent, the prescribed configuration is rolled out under the current policy. If the current policy already solves it, the prescription carries no information and the configuration is revised. The nearest relatives in the literature are methods that choose start states by what the learner can and cannot yet do, in particular reverse curriculum generation, which grows the start-state distribution outward from states the learner already handles [27], and reset-learning work in which an auxiliary policy returns the system to states from which learning can continue [25]. They are the intellectual neighbours of the solvability check and they do not perform it. The two checks are separate mechanisms: one asks whether the environment permits the prescribed configuration, the other asks whether the prescribed configuration would teach the policy anything.

## 2.9 Demonstration selection, curation and active learning

Active learning studies which unlabelled point to send to an oracle. An acquisition function scores each candidate and the highest-scoring one is labelled, and the survey of acquisition functions is the standard entry point [84]. Two families are relevant here. Uncertainty-style acquisition scores a candidate by how unsure the model is about it, of which expected information gain, formalised as the mutual information between the label and the model parameters, is the Bayesian version [36]. Coverage-style acquisition ignores uncertainty and instead selects a subset that covers the representation space, on the argument that a model trained on a good cover of the input distribution generalises to the rest of it; the core-set formulation makes that argument precise and solves it with a greedy k-centre rule [83]. Batch acquisition needs both, because a batch of individually uncertain points can be a batch of near-duplicates, and combining an uncertainty term with a diversity term is the standard remedy [3]. The observation that individually informative selections can be jointly redundant recurs in this report as an empirical finding about the Aim-1 framework itself.

Within imitation learning, the corresponding literature is about curating demonstration data. Sub-trajectory retrieval selects segments from an existing corpus that resemble the target task and trains on them, which turns a large heterogeneous dataset into a task-relevant one at consumption time [64]. Data quality in imitation learning has been characterised directly, and the finding that demonstrations differ in value, so that more data is not automatically better data, is the premise the whole programme rests on [6]. Diverse and partly suboptimal demonstrations can be exploited and not discarded [99], and the mixture weights over data sources in large-scale training can be optimised and not set by hand [32]. Performance itself follows a scaling relationship in the number of demonstrations and in their diversity, which is what licenses the claim that a demonstration has a measurable marginal return [51].

| Approach | What it selects over | Where the data comes from |
|---|---|---|
| Active learning acquisition [36, 84] | unlabelled points | an existing unlabelled pool |
| Core-set and diversity selection [3, 83] | points in a representation space | an existing unlabelled pool |
| Sub-trajectory retrieval [64] | segments of collected trajectories | an existing demonstration corpus |
| Data-mixture optimisation [32] | source datasets | existing datasets |
| Dataset distillation [13] | synthesised training examples | compressed from an existing dataset |
| DAgger-family gates [47, 79] | the timestep at which to hand over | the rollout the policy just produced |
| Aim 1 | the configuration of a demonstration that does not exist | the expert, after the request is issued |

**Table 2.** Selection methods by what they choose and where the data they choose from comes from. Every method above the last row selects from something that already exists. Aim 1 specifies a demonstration that has not been collected, and an expert then produces it.

The distinction in the table is the reason none of these methods is a baseline in Aim 1. An acquisition function ranks candidates in a pool. A retrieval method ranks segments in a corpus. Both presuppose that the data exists and that the only question is which of it to use. Aim 1 answers a different question: given that the policy is failing in a particular way, what demonstration should be collected next, from an expert who has not yet been asked. The demonstration does not exist until the request is made, so there is no pool to rank, and a coverage criterion over a pool cannot be evaluated.

One neighbour needs to be distinguished by name, because the title of the Aim-1 paper uses the word. Dataset distillation compresses a large training set into a small synthetic one that trains a model to comparable accuracy, by matching training trajectories or gradients [13]. The operation runs after collection and operates on data that is already held. Demonstration distillation, in the sense used in this programme, runs before collection and decides which demonstration to acquire next. The two share a word and share no mechanism.

## 2.10 Vision-language-action models

A vision-language-action model maps an image observation and a natural-language instruction to a robot action, and is trained end to end on demonstration data. The line began with a transformer trained on a large corpus of real robot episodes [10] and continued by initialising the same mapping from a vision-language model pre-trained on web data, so that semantic knowledge acquired from images and text transfers into control [11]. Open reproductions followed, trained on pooled cross-embodiment data and released with weights, which made the class of model available to laboratories that cannot collect at that scale [43, 69]. The pooling itself is a research object: a collaboration assembled demonstration data from many robots and many laboratories into a single corpus and showed that policies trained on the pool transfer across embodiments [70]. Later variants change the action decoder, replacing autoregressive token prediction with flow matching to emit continuous action chunks at high rate [8], and the generalist-agent line trains one network across tasks and embodiments with the same supervised recipe [9, 76].

Language has also been used as an intermediate representation inside the mapping. One design predicts a short language motion primitive from the instruction and the observation and then predicts the action from the primitive, which gives a hierarchy in which the middle layer is human-readable [7]. Another emits an explicit chain of embodied reasoning steps, including sub-tasks and object positions, before emitting the action [100]. In both, the language is produced on the way to an action and is consumed by the action decoder.

Every model in this section maps vision and language to an action. Aim 2 inverts the mapping.

## 2.11 Open problems

Three problems are left open by the sections above. They are named here and stated as the gap of this programme in Section 3.1, where the three research questions that answer them are also set out.

The first lies in the interactive loop. The published gates decide when to hand control to the expert, from a scalar attached to a single state, and the selection literatures decide which item to draw from a pool that has already been collected. Neither decides which of a round's failure modes should receive the next demonstration, or from which configuration that demonstration should begin.

The second lies in what a selector knows. A gate reads the current state and a failure-reasoning model reads the current episode, and neither holds a representation of the training set assembled so far, so neither can separate a genuinely novel failure from one the dataset already covers.

The third lies in what a demonstration is taken to cost. Demonstration supply has been pooled across embodiments [70], and cost-sensitive acquisition is standard where the queries are labels for data that already exists [84]. No framework holds demonstration demand: an explicit statement of which skills a policy is short of, priced against the time of the person who would have to produce them.
