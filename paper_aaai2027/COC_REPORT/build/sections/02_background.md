# Background and literature review

The chapter states the material the programme uses and does not claim, and it locates the three aims inside the literature they extend. It opens with the imitation-learning formulation and the failure of offline cloning that motivates every interactive method, then gives the dataset-aggregation loop and the family of query gates built on top of it. Those gates are both the baselines against which Aim 1 is measured and the exact point at which the programme departs from prior work, so they are treated at length and qualitatively. The chapter then covers the policy classes the framework is run on, the clustering and search routines it uses unmodified, the evidence on what language and vision-language models are and are not reliable at, structured environmental knowledge as a constraint store, demonstration selection and active learning, and vision-language-action models.

The behaviour-cloning objective, the aggregate-and-retrain skeleton, the generic query-gate template, the silhouette criterion, farthest-point selection, A\* and breadth-first search all appear here, once, with attribution, and are flagged as standard practice. They are not re-derived in the Aim-1 chapter, which is reserved for what is new. Abbreviations are expanded at first use in this chapter and are used in short form thereafter.

## Imitation learning and behaviour cloning

A task is modelled as a finite-horizon Markov decision process, or a partially observed one when the learner sees images rather than privileged simulator state,

$$\mathcal{M} = \big(\mathcal{S},\ \mathcal{A},\ P(s' \mid s, a),\ R(s,a),\ H\big),
\qquad
\pi_\theta : \mathcal{S} \to \Delta(\mathcal{A}),$$

with $\mathcal{S}$ the state space, $\mathcal{A}$ the action space, $P$ the transition kernel, $R$ the reward, $H$ the horizon, and $\pi_\theta$ the learner's policy with parameters $\theta$. An expert $\pi^\star$ supplies trajectories, and the demonstration set is the collection of state-action pairs those trajectories contain,

$$\mathcal{D} = \big\{ (s_t, a_t) \ : \ a_t = \pi^\star(s_t) \big\}.$$

Behaviour cloning fits the policy to that set by supervised learning [@pomerleau1988alvinn; @bain1995cloning]:

$$\theta^\star = \arg\min_\theta \ \mathbb{E}_{(s,a) \sim \mathcal{D}} \Big[\, \mathcal{L}_{\mathrm{BC}}\big(\pi_\theta(\cdot \mid s),\, a\big) \,\Big],$$

with $\mathcal{L}_{\mathrm{BC}} = -\log \pi_\theta(a \mid s)$ where the action space is discrete and $\mathcal{L}_{\mathrm{BC}} = \lVert \pi_\theta(s) - a \rVert_2^2$ where it is continuous. The reduction is old and it works: the first system of this kind steered a road vehicle from camera input through a three-layer network trained on logged human driving [@pomerleau1988alvinn], and the formulation was later given as a named framework [@bain1995cloning] and surveyed as one branch of learning from demonstration [@argall2009survey; @osa2018algorithmic].

The reduction carries one defect, and every method in this report exists because of it. Supervised learning assumes the training and test inputs are drawn from the same distribution. In imitation learning they are not. The policy is trained on the states the expert visits, and at deployment it visits the states its own actions produce. Any error moves the learner off the expert's state distribution, the next prediction is made on an input the training set under-represents, and the error grows. The statistics literature calls the mismatch covariate shift [@shimodaira2000covariate]; the imitation-learning consequence is quantitative. If the cloned policy incurs supervised loss $\epsilon$ under the expert's state distribution, its cost over a horizon $H$ can grow as $O(H^2 \epsilon)$, and the quadratic term is not an artefact of a loose bound but a property of the offline reduction. Allowing the learner to be corrected on the states it actually reaches removes it, leaving a bound linear in the horizon [@ross2010reductions; @ross2011dagger].

The problem was visible in the first system. A policy trained only on a good driver's centred trajectory never observes a recovery from the road edge, because a good driver never produces one. Pomerleau's remedy was to synthesise the missing data: each camera image was shifted and rotated laterally, and the steering label was corrected to the command that would return the vehicle to the centre [@pomerleau1991efficient]. Noise injection into the expert's control stream is the modern off-policy version of the same idea, in which the demonstrations are deliberately perturbed so that the expert is recorded recovering from small deviations [@laskey2017dart].

The objective above is fixed for the whole programme, and it is fixed for every method compared. Neither Aim 1 nor either of the later aims changes the loss that is minimised, the optimiser, or the policy architecture. They change which demonstrations enter $\mathcal{D}$.

## Dataset aggregation and the interactive loop

Dataset aggregation, introduced as DAgger, corrects covariate shift by moving the labelling effort onto the learner's own state distribution [@ross2011dagger]. In round $r$ the current policy $\pi_{\theta_r}$ is rolled out, the expert is asked for its action at the states the rollout visits, those labelled pairs are added to the dataset, and the policy is refitted to the whole aggregate:

$$\mathcal{D}_{r+1} = \mathcal{D}_r \ \cup\ \big\{ (s,\, \pi^\star(s)) \ : \ s \sim d_{\pi_{\theta_r}} \big\},
\qquad
\theta_{r+1} = \arg\min_\theta \ \mathbb{E}_{\mathcal{D}_{r+1}} \big[ \mathcal{L}_{\mathrm{BC}} \big],$$

where $d_{\pi_{\theta_r}}$ is the state distribution induced by rolling out the round-$r$ policy. The analysis casts the loop as online learning against an adversarially chosen sequence of state distributions, so that a no-regret supervised learner attains a performance bound linear in the horizon rather than quadratic [@ross2011dagger]. Variants replace action agreement with the expert's cost-to-go as the aggregation signal, which lets the learner be told how much a mistake costs rather than only that it was a mistake [@ross2014aggrevate; @sun2017aggrevated].

The loop has become a field rather than a single algorithm, with a taxonomy of feedback types and of who initiates the handover [@celemin2022iil]. Human-gated variants give the decision to the person: the human watches the rollout and takes control when they judge it necessary, which removes the need for a machine-readable uncertainty signal at the cost of continuous human attention [@kelly2019hgdagger]. Data collected during an intervention has a different value from data collected on-policy, and can be reweighted accordingly, or used as the substrate of a deployment-time learning system in which the robot runs, a human intervenes, and the intervention becomes training data [@mandlekar2020iwr; @liu2023sirius].

One cost is the reason the rest of this chapter exists. The aggregation step as originally stated asks the expert to label every state the learner visits. An expert who must answer at every step of every rollout is an expert whose time scales with the number of rollouts, and outside simulation that expert is a person. The interactive loop trades the compounding-error problem for an expert-effort problem, and the query-efficient variants below are the field's answer to the second problem.

Aim 1 keeps this skeleton without modification, and so does every method it is compared against. The rollout, the aggregation and the retraining are shared. The only free variable is how the round's new demonstration is chosen, and holding everything else fixed is what makes the comparison a comparison.

## The query gates of the DAgger family

Query-efficient interactive imitation learning replaces the ask-at-every-state rule with a gate. At each visited state the method computes a scalar score, compares it against a threshold, and hands control to the expert at the first state where the comparison fires:

$$\mathrm{Query}(s_t) = \mathbf{1}\big[\ \mathrm{score}(s_t) \ \gtrless \ \tau \ \big],
\qquad
t^\star = \min \{\, t \ : \ \mathrm{Query}(s_t) = 1 \,\}.$$

The expert then takes over from $t^\star$ and completes the episode, and the expert's segment is the round's new demonstration. The published methods differ in one place only: what they put in $\mathrm{score}(\cdot)$. The template is worth stating explicitly, because it makes the family's shared limitation legible, and because Aim 1's claim is that the template is answering one of three questions and leaving the other two unanswered.

SafeDAgger learns an auxiliary safety classifier that predicts, from the current observation, whether the policy's action will deviate from the expert's by more than a tolerance, and hands over when the classifier predicts a large deviation. The classifier is trained on the policy's own rollouts, so the gate is a learned model of where the policy is unsafe rather than a direct measurement of it [@zhang2017safedagger].

DropoutDAgger reads the spread of the policy's own action distribution under Monte-Carlo dropout. Several stochastic forward passes are drawn at the visited state, and the expert is called when the sampled actions stop concentrating near the expert's action [@menda2017dropoutdagger]. The signal is imported from the Bayesian deep-learning literature, where dropout at inference time is interpreted as approximate posterior sampling [@gal2016dropout].

EnsembleDAgger replaces the dropout samples with an ensemble of independently trained policies and reads their variance, which it calls doubt. The gate opens on high doubt or on a large discrepancy between the ensemble mean and the expert's action, so that the epistemic term and the safety term each have an arm [@menda2019ensembledagger]. Ensembles are the second canonical deep-uncertainty estimator and, like dropout, are borrowed rather than invented here [@lakshminarayanan2017ensembles].

ThriftyDAgger adds a second quantity to novelty. Alongside the ensemble doubt it trains an estimate of task risk, a value function predicting the probability that the episode will fail from the current state and action, and it opens the gate on either. Both thresholds are set as quantiles of the observed distributions, calibrated so that the method hands over at a target switching rate, which is what makes the method budget-aware [@hoque2021thriftydagger]. A related design reduces the number of context switches by using asymmetric thresholds for handing over and handing back, so that control does not oscillate between learner and expert [@hoque2021lazydagger]; it is context here rather than a baseline.

Diff-DAgger reads the learner's own training loss. For a diffusion policy the per-step denoising loss on a state-action pair is a usable score of how far that pair lies outside the training distribution, so the method thresholds it at a quantile of the training-loss distribution, recalibrated at each retrain, and hands over when the loss stays in the tail for a run of consecutive steps [@lee2025diffdagger]. Using the diffusion loss as an uncertainty signal is Diff-DAgger's idea. Aim 1 uses that same per-step loss, both to localise a failure within a rollout and to measure the information content of an acquired demonstration, and it also compares against Diff-DAgger as a baseline. Both facts are stated plainly wherever the signal appears.

A uniform-random control completes the comparison. Referred to in this report as Stagger, it draws the handover step uniformly at random and is not a published method: it is a floor implemented in this project, and it is never labelled as a member of the DAgger family. Its purpose is to establish what an uninformed allocation of the same expert effort buys, so that any margin a gated method reports can be read against a random one.

| Gate | Scalar signal the gate reads | What opens the gate |
|---|---|---|
| SafeDAgger [@zhang2017safedagger] | learned safety classifier predicting policy-expert deviation | predicted deviation exceeds tolerance |
| DropoutDAgger [@menda2017dropoutdagger] | spread of Monte-Carlo dropout action samples | too few samples agree with the expert |
| EnsembleDAgger [@menda2019ensembledagger] | ensemble variance (doubt) and mean action discrepancy | either term exceeds its threshold |
| ThriftyDAgger [@hoque2021thriftydagger] | ensemble novelty and a learned task-risk estimate | either term exceeds a budget-calibrated quantile |
| Diff-DAgger [@lee2025diffdagger] | the policy's own per-step denoising loss | the loss stays in the tail of the training-loss distribution |
| *Uniform-random control (Stagger)* | *none* | *a step drawn uniformly at random* |

**Table 1. The five published query gates of the DAgger family, presented as instances of one template: a scalar score, a threshold and a handover. The methods differ only in the score. The uniform-random control in the last row is not a published method and is not a member of the family; it is an internal floor implemented for this project, and it is listed separately so that it is never mistaken for one.**

Descriptions here are qualitative by design. The thresholds, ensemble sizes, sample counts and calibration constants that each published method specifies are not reproduced, and the implementation choices made when these gates were re-implemented for the comparison, including the cases where a learned auxiliary predictor is replaced by a direct oracle signal, are recorded with the experimental setup in the Aim-1 chapter rather than here.

Three properties of the template follow from its form, and they are the opening the programme works in.

The gate is per-state. It maps one visited state to one scalar. Given twenty failed rollouts in a round, the gate has no representation in which two of them are the same mistake and a third is a different one, so it cannot tell a redundant correction from a novel one.

The gate is memoryless across rounds. Nothing in $\mathrm{score}(\cdot)$ records that a particular kind of failure has already been corrected twice, so a persistent failure mode can absorb the entire budget.

The gate inherits the state that tripped it. The corrective demonstration begins wherever the policy happened to be when the score crossed the threshold, which is frequently a state so far off the expert's distribution that the expert spends the demonstration recovering rather than teaching.

Each of the five published gates answers *when* to hand over, and answers it well. Neither *which* of a batch of failures to correct nor *where* the corrective demonstration should begin is answered by any of them.

## Uncertainty estimation and the limits of a scalar

The gates read their signals from the deep-uncertainty literature, and the three families of signal are worth separating because their limitations are shared. Monte-Carlo dropout treats dropout at inference as approximate posterior sampling and reads the spread of the resulting predictions [@gal2016dropout]. Deep ensembles train several independent members and read their disagreement, which is simpler to implement and generally better calibrated [@lakshminarayanan2017ensembles]. Density-style signals score how far an input lies from the training distribution, and the per-step denoising loss of a diffusion policy is one such score: a state-action pair the model has not seen produces a high reconstruction error, which is what makes it usable as a query trigger [@lee2025diffdagger].

All three produce a number attached to a state. A number attached to a state is enough to decide whether that state is a problem. It is not enough to decide whether that state's problem is the same problem as another state's, because two scalars can be equal for entirely different reasons and can differ while describing one underlying failure. The comparison the DAgger family cannot make is a comparison *between* failures, and no refinement of the scalar supplies it. What supplies it is a representation in which failures are points rather than magnitudes, which is where clustering, and the standard machinery of the next section, enter the programme.

## Policy classes and why the framework does not depend on them

Explicit regression onto expert actions is a poor fit for demonstration data whose action distribution is multimodal. Where two different actions are both correct at a state, a network trained under a squared-error loss learns their average, which may be correct under neither mode. Energy-based and generative formulations of the policy fit such distributions instead of averaging them [@florence2021implicitbc]. Denoising diffusion probabilistic models are the generative family that has proved most usable for this [@ho2020ddpm]. Applied to trajectories they give a planner [@janner2022diffuser], and applied to short action sequences conditioned on recent observations they give the visuomotor diffusion policy that is the learner for the robot tasks in this programme [@chi2023diffusionpolicy]. The conventions for training such policies on offline human-style manipulation data, including observation encoders and the treatment of action chunks, follow the empirical study that established them [@mandlekar2021robomimic].

A diffusion policy is trained by noising the clean action target $x_0 = a$ over $K$ steps and learning to reverse the corruption,

$$x_k = \sqrt{\bar\alpha_k}\, x_0 + \sqrt{1 - \bar\alpha_k}\, \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I),$$

with the network trained to predict the noise, or an equivalent reparameterisation of it, and the resulting per-pair denoising loss written $L_{\mathrm{dif}}(s, a)$. Evaluated at the state the policy visited and the action it executed, that loss gives a per-step signal along a rollout,

$$\ell_t = L_{\mathrm{dif}}\big(s_t,\, a_t\big).$$

The quantity $\ell_t$ is the one Diff-DAgger thresholds [@lee2025diffdagger], and the one Aim 1 uses to localise the step at which a rollout goes wrong.

The framework is stated over any policy $f_\theta$ that exposes a per-step loss at a visited state under the executed action. A multilayer perceptron trained with cross-entropy on a discrete grid exposes one. A convolutional network on grid images exposes one. A diffusion policy exposes one, in the form above. Nothing else about the policy enters the loop: not its architecture, not its action parameterisation, not whether the observation is a state vector or an image. The programme is therefore run on all three policy classes, and the framework is never described as a method for diffusion policies, because the diffusion policy is one instantiation of the requirement and not the requirement itself.

## Standard machinery this work uses and does not claim

Five routines are used unmodified. Each is named here, cited, and flagged as standard, so that the Aim-1 method chapter can use them without appearing to claim them.

The partition step is generic. Aim 1 groups a round's failures into failure modes with a clustering step $\mathcal{C}$, instantiated as agglomerative clustering under Ward's linkage [@ward1963hierarchical]. The choice is one instantiation and not a commitment: k-means [@lloyd1982kmeans] or any other partition method that returns a labelling and a set of centroids would serve, and the framework is stated over $\mathcal{C}$ rather than over the particular algorithm.

The number of clusters is chosen by the silhouette criterion, which scores a partition by comparing, for each point $i$, its mean distance $a(i)$ to the other members of its own cluster against its mean distance $b(i)$ to the members of the nearest other cluster,

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}},$$

and takes the mean over points [@rousseeuw1987silhouette]. The criterion is used exactly as published, with no modification, and the number of clusters is the value that maximises the mean silhouette over a bounded range.

Diversity selection within a cluster is farthest-point, or k-centre, selection: given a set already chosen, add the candidate whose minimum distance to that set is largest, and repeat [@eldar1997fps]. The greedy rule is standard and is used as such.

Path validity on the discrete grid task is checked with A\* [@hart1968astar] and breadth-first search [@cormen2022algorithms]. Their role must be stated precisely, because it is easy to misread. They verify that a prescribed grid configuration admits a valid path from the start cell to the goal cell around the obstacles, which is a feasibility check on the prescription. They are never the expert. The GridWorld expert is a human, and the demonstrations on that task are human trajectories.

Clustering, the silhouette computation and the feature standardisation that precedes them are taken from a standard library implementation [@pedregosa2011sklearn].

One further point of ownership needs to be settled here rather than left to the method chapter, because an earlier version of this work got it wrong. Pre-trained visual representations for manipulation, of which R3M is the one used here [@nair2022r3m] and masked visual pre-training [@radosavovic2022mvp] and value-implicit pre-training [@ma2023vip] are the alternatives that were available, supply the visual encoder for the image-modality policies. They do not supply the features that are clustered. Clustering in this programme is geometric in every run, under both observation modalities, over a low-dimensional descriptor of the robot and object configuration at the flagged step. The frozen-embedding-plus-projection branch that once handled the image modality has been retired, and it does not appear anywhere in this report. The evidence that settles the question is reported with the Aim-1 ablations.

## What language and vision-language models are and are not reliable at

Large language models, which are autoregressive models over text, and vision-language models, which condition the same generation on images, have been used in robotics in several distinct roles, and the distinctions matter because they determine which capability this programme is depending on.

As planners, language models decompose a natural-language instruction into a sequence of steps over a fixed repertoire of learned skills. The decomposition is only useful if it is grounded in what the robot can actually do, which is why the influential version of the idea scores each candidate skill by the product of the language model's likelihood that the skill is useful and a value function's estimate that the skill will succeed from the current state [@ahn2022saycan]. As programmers, they write executable code against a perception and control interface, so that the plan is a program with loops and conditionals rather than a flat list [@liang2023codeaspolicies; @singh2023progprompt]. Multimodal variants take sensor observations directly into the language model's embedding space, so that the plan is conditioned on what the robot sees rather than on a textual scene description produced by another module [@driess2023palme]. As designers of objectives, they write reward functions and cost maps that a downstream optimiser consumes, which is the setting in which a language model's output is a specification and never an action [@ma2024eureka; @yu2023language2rewards; @huang2023voxposer].

Two capabilities matter more directly to this programme. The first is failure reasoning. A vision-language model given a summary of a robot's execution, in the form of a small number of frames and a record of what happened, can name the cause of a failure, and the naming is accurate enough to drive a recovery [@liu2023reflect]. A model trained specifically on manipulation failures does better than a general one, which is evidence that the capability is a learnable perceptual skill rather than an emergent accident [@duan2025aha]. The second is self-correction. Closed-loop textual feedback improves an embodied planner, because a plan that fails can be revised when the failure is described back to the model that wrote it [@huang2022innermonologue], and verbal self-critique with iterative refinement is by now an established pattern in its own right [@shinn2023reflexion; @madaan2023selfrefine]. A language model can also be calibrated to recognise when it does not know, and to ask a human instead of guessing [@ren2023knowno], which is the direct precedent for the request interface proposed in Aim 3.

Against those capabilities stands a well-documented weakness. Vision-language models are unreliable at metric and spatial reasoning from pixels alone. They mistake relative depth, distance and size, and they fail on spatial-relation questions that a human answers instantly [@chen2024spatialvlm; @fu2024blink]. The failure is not a matter of model scale; it is a property of how these models are trained, and it is measured as such on purpose-built benchmarks.

The two findings together dictate a design commitment, and the commitment is inherited by the Aim-1 method without further argument. The model is reliable at naming a cause when it is handed structured evidence, and unreliable at recovering geometry from an image. It is therefore handed the geometry, in the form of a low-dimensional numeric descriptor and an explicit store of what the environment permits, and it is asked for the thing it is good at. The model that reads frames and the model that writes the prescription are open-weight instruction-tuned models [@bai2025qwen3vl; @yang2025qwen3]. Neither ever emits a robot action. In this programme a language model is a component that reads a structured summary of the policy's failures and returns a request for a demonstration, and it is not a controller.

## Structured environmental knowledge and constraint grounding

Retrieval-augmented generation conditions a language model's output on passages fetched from an external store, so that the model's factual claims are anchored in retrievable text rather than in its parameters [@lewis2020rag]. Graph-structured retrieval organises the store rather than treating it as a flat pile of passages, which improves queries whose answer is spread across many documents [@edge2024graphrag]. Robot knowledge bases predate both and do something different again: they hold explicit, queryable, symbolic knowledge about objects, actions and the environment, and a robot's planner asks them questions rather than reading them as text [@tenorth2013knowrob].

The knowledge-augmented graph used in Aim 1 belongs to the third tradition and not to the first. It is not a document store. It is a store of explicit environmental constraints held as structured key-value knowledge: workspace bounds, reachability limits, the ranges within which objects may be placed and spawned, and the limits of the controller. It is queried during verification, and its entries are checked against, not read from.

The pattern in which it is used is established. A language model's proposal can be handed to an external checker, and the checker's verdict returned to the model as feedback: a plan can be validated by a symbolic planner [@liu2023llmp], and the propose-verify-revise cycle can be iterated until the proposal passes [@chen2024autotamp]. Aim 1's feasibility loop is an instance of that pattern. The prescription model proposes a demonstration configuration; the constraints that bear on it are retrieved from the knowledge-augmented graph; the configuration is checked against them; if a constraint is violated, the violation is returned to the model as feedback and a revised configuration is requested; the loop repeats until a feasible configuration is produced. Saying so makes the mechanism legible to a reader who has seen it before, and it makes clear that the feasibility check is not the novel part of the method.

A second check in Aim 1 has no precedent in the list above and must not be conflated with the first. Before expert time is spent, the prescribed configuration is rolled out under the current policy. If the current policy already solves it, the prescription carries no information and the configuration is revised. The nearest relatives in the literature are methods that choose start states by what the learner can and cannot yet do, in particular reverse curriculum generation, which grows the start-state distribution outward from states the learner already handles [@florensa2017reversecurriculum], and reset-learning work in which an auxiliary policy returns the system to states from which learning can continue [@eysenbach2018leavenotrace]. They are the intellectual neighbours of the solvability check and they do not perform it. The two checks are presented separately in the method chapter, and they are separate mechanisms: one asks whether the environment permits the prescribed configuration, the other asks whether the prescribed configuration would teach the policy anything.

## Demonstration selection, curation and active learning

Active learning studies which unlabelled point to send to an oracle. An acquisition function scores each candidate and the highest-scoring one is labelled, and the survey of acquisition functions is the standard entry point [@settles2009active]. Two families are relevant here. Uncertainty-style acquisition scores a candidate by how unsure the model is about it, of which expected information gain, formalised as the mutual information between the label and the model parameters, is the Bayesian version [@houlsby2011bald]. Coverage-style acquisition ignores uncertainty and instead selects a subset that covers the representation space, on the argument that a model trained on a good cover of the input distribution generalises to the rest of it; the core-set formulation makes that argument precise and solves it with a greedy k-centre rule [@sener2018coreset]. Batch acquisition needs both, because a batch of individually uncertain points can be a batch of near-duplicates, and combining an uncertainty term with a diversity term is the standard remedy [@ash2020badge]. The observation that individually informative selections can be jointly redundant recurs in this report as an empirical finding about the Aim-1 framework itself.

Within imitation learning, the corresponding literature is about curating demonstration data. Sub-trajectory retrieval selects segments from an existing corpus that resemble the target task and trains on them, which turns a large heterogeneous dataset into a task-relevant one at consumption time [@memmel2025strap]. Data quality in imitation learning has been characterised directly, and the finding that demonstrations differ in value, so that more data is not automatically better data, is the premise the whole programme rests on [@belkhale2023dataquality]. Diverse and partly suboptimal demonstrations can be exploited rather than discarded [@yue2024diversedemos], and the mixture weights over data sources in large-scale training can be optimised rather than set by hand [@hejna2024remix]. Performance itself follows a scaling relationship in the number of demonstrations and in their diversity, which is what licenses the claim that a demonstration has a measurable marginal return [@lin2024datascaling].

| Approach | What it selects over | Where the data comes from |
|---|---|---|
| Active learning acquisition [@settles2009active; @houlsby2011bald] | unlabelled points | an existing unlabelled pool |
| Core-set and diversity selection [@sener2018coreset; @ash2020badge] | points in a representation space | an existing unlabelled pool |
| Sub-trajectory retrieval [@memmel2025strap] | segments of collected trajectories | an existing demonstration corpus |
| Data-mixture optimisation [@hejna2024remix] | source datasets | existing datasets |
| Dataset distillation [@cazenavette2022mtt] | synthesised training examples | compressed from an existing dataset |
| DAgger-family gates [@ross2011dagger; @lee2025diffdagger] | the timestep at which to hand over | the rollout the policy just produced |
| Aim 1 | the configuration of a demonstration that does not exist | the expert, after the request is issued |

**Table 2. Selection methods by what they choose and where the data they choose from comes from. Every method above the rule selects from something that already exists. Aim 1 specifies a demonstration that has not been collected, and an expert then produces it, which is why the coverage and curation literature is background to this programme rather than a set of baselines for it.**

The distinction in the table is the reason none of these methods is a baseline in Aim 1. An acquisition function ranks candidates in a pool. A retrieval method ranks segments in a corpus. Both presuppose that the data exists and that the only question is which of it to use. Aim 1 answers a different question: given that the policy is failing in a particular way, what demonstration should be collected next, from an expert who has not yet been asked. The demonstration does not exist until the request is made, so there is no pool to rank, and a coverage criterion over a pool cannot be evaluated.

One neighbour needs to be distinguished by name, because the title of the Aim-1 paper uses the word. Dataset distillation compresses a large training set into a small synthetic one that trains a model to comparable accuracy, by matching training trajectories or gradients [@cazenavette2022mtt]. The operation runs after collection and operates on data that is already held. Demonstration distillation, in the sense used in this programme, runs before collection and decides which demonstration to acquire next. The two share a word and share no mechanism.

## Vision-language-action models

A vision-language-action model maps an image observation and a natural-language instruction to a robot action, and is trained end to end on demonstration data. The line began with a transformer trained on a large corpus of real robot episodes [@brohan2023rt1] and continued by initialising the same mapping from a vision-language model pre-trained on web data, so that semantic knowledge acquired from images and text transfers into control [@brohan2023rt2]. Open reproductions followed, trained on pooled cross-embodiment data and released with weights, which made the class of model available to laboratories that cannot collect at that scale [@kim2024openvla; @octo2024]. The pooling itself is a research object: a collaboration assembled demonstration data from many robots and many laboratories into a single corpus and showed that policies trained on the pool transfer across embodiments [@oneill2023openx]. Later variants change the action decoder, replacing autoregressive token prediction with flow matching to emit continuous action chunks at high rate [@black2025pi0], and the generalist-agent line trains one network across tasks and embodiments with the same supervised recipe [@reed2022gato; @bousmalis2023robocat].

Language has also been used as an intermediate representation inside the mapping. One design predicts a short language motion primitive from the instruction and the observation and then predicts the action from the primitive, which gives a hierarchy in which the middle layer is human-readable [@belkhale2024rth]. Another emits an explicit chain of embodied reasoning steps, including sub-tasks and object positions, before emitting the action [@zawalski2024ecot]. In both, the language is produced on the way to an action and is consumed by the action decoder.

Every model in this section maps vision and language to an action. Aim 2 inverts the mapping.

## Open problems and the gap this programme addresses

Three gaps follow from the sections above, and each corresponds to one aim.

The first is a gap in the interactive loop. The published query gates decide when to hand control to the expert, and they decide it from a scalar attached to a single state. Selection methods from active learning and dataset curation decide which item to take from a pool that already exists. Between the two lies a decision nobody makes: given a round's worth of failures, which failure mode should receive this round's demonstration, and from which configuration should that demonstration begin. Aim 1 makes that decision, and the report's central empirical question is whether making it raises the final success rate under a fixed budget.

The second is a gap in what the selector knows. A gate reads the current state. A failure-reasoning model reads the current episode. Neither holds any representation of the training set that has been assembled so far, so neither can tell a genuinely novel failure from one the dataset already covers several times over. Under a restricted budget, a demonstration spent re-teaching material the policy has already been shown is a demonstration lost. Aim 2 gives the selector a memory of what it has already taught, indexed in language, by inverting the vision-language-action mapping so that executed trajectories are described rather than generated.

The third is a gap in what a demonstration is taken to cost. Aims 1 and 2 count demonstrations, which is the right accounting in simulation, where a demonstration is a call to a motion planner and every call costs the same. Cross-embodiment pooling has made demonstration *supply* a shared object [@oneill2023openx], and cost-sensitive acquisition is standard where the queries are labels for data that already exists [@settles2009active], but no framework holds demonstration *demand*: an explicit statement of which skills a policy is short of, priced against the time of the person who would have to produce them, and shared across the tasks and embodiments over which those skills recur. Aim 3 constructs that object and issues the resulting requests to a person.
