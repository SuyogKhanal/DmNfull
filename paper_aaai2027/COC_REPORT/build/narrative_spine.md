# Narrative spine — the connective tissue of the Confirmation of Candidature

Student: Suyog Khanal (s226137394), A2I2, Deakin University.
Supervisors: A/Prof Santu Rana; Dr Arun Kumar Anjanapura Venkatesh.
Thesis: *Leveraging Large Language Models for Sample-Efficient Imitation Learning*.
Candidature start 13 November 2025. Confirmation of Candidature 13 August 2026. Thesis submission November 2028.

This document is the spine every other section hangs on. It states the one question the programme asks, the three stages in which it is answered, and the exact sentence by which each stage becomes the motivation for the next. Chapters 1, 3, 4.1, 4.11, 5.1, 6 and 7 of `build/coc_structure.md` draw their argument from here, and no chapter may contradict it.

## Binding conventions used throughout this document

The method is **DISEIL**. The name is derived, once and only in the executive summary, by bolding six letters of the Aim-1 paper title: **D**emonstration d**I**stillation for **S**ample-**E**fficient **I**mitation **L**earning. The derivation is not repeated in the title, in any heading, or anywhere else in the report.

A *setting* is one task under one observation modality. Five tasks and two modalities give ten settings. *Mode* is reserved for failure modes, which are the clusters the framework discovers. Modality is never called a mode.

ΔSR is the change in the policy's success rate on the round-level rollout evaluation. The definition appears at the first use of the symbol and is not restated.

The framework operates under any fixed, restricted budget B, with D demonstrations acquired per round. B = 20 and D = 1 are the validated instance and appear only in the experimental setup.

Lift is at 100.0 ± 0.0 under DISEIL in both modalities. It has no headroom and no seed variance, so a null result on Lift is evidence about nothing. Every mechanism claim in the report excludes it, and the exclusion is stated the first time Lift appears in an ablation.

---

## 1. The research vision

### 1.1 The premise, stated once

Imitation learning converts expert demonstrations into a policy. Everything else in the pipeline has become cheap. Compute is bought, simulators are free, network architectures are downloaded. The demonstration is the one input whose cost does not fall: a person or a scripted oracle must produce it, one trajectory at a time, and the number that can be produced is bounded by something other than the researcher's willingness to wait. In any realistic deployment the constraint is not "collect more data" but "you hold a budget of B demonstrations and you must spend it well".

That constraint sets the objective of the whole thesis. The quantity to maximise is not the number of demonstrations, which is fixed. It is the *information content of each demonstration*, and the programme's claim is that a large language model, given the right structured evidence, can raise it.

### 1.2 What the field already does, and the decision it leaves unclaimed

Interactive imitation learning corrects the compounding error of offline cloning by labelling the states the learner actually visits. The DAgger family and its descendants all share one skeleton: roll out the current policy, decide whether to hand control to the expert, aggregate the expert's labels, retrain. The members differ only in the scalar signal that opens the gate. SafeDAgger learns a classifier that predicts when the policy will deviate. DropoutDAgger reads the spread of a Monte-Carlo dropout ensemble. EnsembleDAgger reads ensemble variance. ThriftyDAgger combines novelty with a learned risk estimate under a target switching rate. Diff-DAgger reads a diffusion policy's own per-step training loss.

Every one of these answers the question *when* to ask for help. Three consequences follow, and they are the opening this programme works in.

A per-state gate cannot see a batch. Given twenty failed rollouts, it has no representation in which two of them are the same mistake and a third is a different one, so it cannot tell a redundant correction from a novel one.

A per-state gate has no memory across rounds. One failure mode that persists can absorb a whole budget, because nothing in the gate records that the mode was already corrected twice.

A per-state gate inherits the state that tripped it. The corrective demonstration begins wherever the policy happened to be when the signal crossed threshold, which is frequently a state so corrupted that the expert spends the demonstration recovering rather than teaching.

So *when* is one decision of three, and the other two are unclaimed: **which failure to correct**, and **where the corrective demonstration begins**.

### 1.3 The thesis statement

A language model is not a controller here, and the report should say so before a panel member assumes otherwise. It is the component that reads a structured summary of the policy's own failures together with an explicit statement of what the environment permits, and returns a request for a specific demonstration. The thesis statement is one sentence: *language models can raise the information content of each demonstration under a restricted budget, and raising it is what makes imitation learning sample-efficient.*

### 1.4 One quantity through three aims

The programme traces a single quantity, the value of one demonstration, through three levels at which it can be raised.

| | What the selector reasons over | What it decides | Level at which value is raised |
|---|---|---|---|
| Aim 1 (DISEIL) | this round's failures, partitioned into modes, plus environmental constraints | which mode to correct and where the demonstration starts | within a round |
| Aim 2 (Reverse VLA) | the same failures, plus a language index of everything already taught | whether the failure is a genuine coverage gap or a re-teaching of known material | across the dataset |
| Aim 3 (demonstration demand) | the coverage of a skill inventory shared across tasks, embodiments and teachers, priced against human time | which demonstration to buy next, from whom, and what it is worth | across tasks and teachers |

The three research questions are the three rows.

- **RQ1.** Under a fixed budget B, does choosing which failure mode to correct and where the corrective demonstration begins yield a higher final success rate than choosing only when to intervene?
- **RQ2.** Can the selector be given a model of what it has already taught, by inverting the vision-language-action mapping into language descriptions of executed trajectories, and does coverage-gap selection beat failure-local selection?
- **RQ3.** Can demonstration demand be made explicit, priced against a teacher's time, and satisfied across tasks and embodiments, so that a generalist policy asks a non-expert human for exactly the demonstrations it lacks?

---

## 2. Aim 1 — DISEIL

### 2.1 What it does

DISEIL takes the two unclaimed decisions and makes them. Each round, the current policy is rolled out; the per-step loss flags the step at which the policy first becomes unreliable; the state at that step is reduced to a 6-D geometric descriptor; the descriptors of all the round's failures are partitioned into failure modes; a cross-round memory penalises modes that have recently been corrected, which rotates the target across the failure distribution instead of letting one mode absorb the budget; a small, diverse context set of cited failures is assembled; a language model prescribes one demonstration, either as a correction of a cited failure or as a placement positioned between cited failures; the prescription is verified against a store of explicit environmental constraints and revised until it is feasible; the expert supplies the demonstration; the dataset grows by one and the policy is retrained.

The descriptor is geometric in every run, state modality and image modality alike. The report must state this once and unambiguously, because the earlier version of the method clustered image runs in a frozen visual-embedding space and that branch is retired (ablation A10). The visual channel feeds root-cause reasoning, and it does not feed the partition.

Two checks screen a prescription before expert time is spent on it, and they are distinct mechanisms that must never be conflated. The first is feasibility verification against the knowledge-augmented graph: the model proposes, constraints are retrieved (workspace bounds, reachability, object and spawn ranges, controller limits), the proposal is checked, and a violation is returned to the model as feedback so that it can revise, until a feasible prescription is produced. The second is policy solvability: the prescribed scenario is rolled out under the current policy, and if the current policy already solves it, the prescription carries no information and the scenario is revised.

### 2.2 What the evidence says

DISEIL attains the best mean success rate in all ten settings, with a mean margin of 3.7 points over the strongest baseline in each setting. The ten settings are not ten independent experiments, because the two modalities of a task share the expert, the reset distribution and the reward structure, so the claim of record is the conservative one: collapsed to five task means the sweep is 5/5, one-sided sign test p = 0.031, paired t-test t(4) = 4.15, p = 0.014.

The ablations then say something more interesting than "the method works", and the report's authority depends on repeating it without softening.

Allocation is the mechanism. Removing the partition (A3) costs 4.01 points on average and retains 11.3% of the margin over the best baseline, and on Door (image) the ablated system falls below its best baseline. In the same experiment, per-demonstration information gain does not fall; it rises very slightly (mean +0.06, Wilcoxon p = 0.23) while success rate collapses (Wilcoxon p = 0.002). Greedy worst-loss selection collects demonstrations that are individually informative and jointly redundant, because information gain measured on one demonstration has no term for its overlap with the demonstration collected in the previous round. Allocation is the term that supplies it.

The controls bracket the claim from both sides. Uniform-random replay of a recorded failure (A2) lands *below* the uncertainty-gated baselines on most robot settings, 10.25 points below DISEIL on average, so the win is not "any failure replay". The deterministic nearest-untried fallback promoted to a whole method (A8) retains 31.1% of the margin, so structured allocation without any reasoning gets a third of the way and no further, because "nearest untried" has no notion of a mode and cannot decide that a mode has been addressed.

The margin is largest where the budget is smallest: +10.35 points at B = 10, +4.49 at B = 20, +2.67 at B = 40, monotone in every setting. Allocation buys the rate of coverage, not the asymptote. (The workbook's proposed headline that DISEIL at B = 10 matches the best baseline at B = 20 is false in seven of ten settings and is retracted; the report must not carry it.)

### 2.3 The honest residual weakness, established by our own ablations

The weakness is not a hedge added for modesty. It is what A4, A5, A1 and A13 measured, and it is the exact hinge on which Aim 2 turns.

**The reasoning stack contributes less than the allocation machinery it sits on.** Replacing the prescription model with a deterministic heuristic (A4) costs 1.08 points and retains 76.7% of the margin. Removing the vision-language model (A5) costs 1.01 points and retains 78.0%. Every individual gap in both studies is smaller than the seed standard deviation of the corresponding full run. The reason is structural and the report states it rather than waiting for a reviewer to find it: the partition is geometric and uses no output from any foundation model, so by the time the language model is called, the decision that matters — which region of the failure distribution receives this round's demonstration — has already been made. The language model chooses the *form* of the correction inside a region that was selected without it. A component downstream of the decisive step cannot produce a large effect, and the measurement agrees.

**The selector is stateless and dataset-blind.** It reasons about the current round's failures. Its only memory is a recency-discounted Gaussian penalty over the centroids of clusters already corrected — a geometric memory of *where* corrections have been placed, holding no representation of *what the training set contains*. The selector can say "the policy failed here, and the cause is a grasp failure". It cannot say "and the training set already holds six demonstrations of that cause, so this is not the gap". Under a restricted budget, a demonstration spent re-teaching known material is a demonstration lost, which is the opposite of the objective.

**The one memory it does have is mis-scaled.** Switching the memory off (A1) costs 0.75 points, the smallest of the seven knockouts, and A13 explains why. The Gaussian kernel width σ is a single global constant, and the tasks do not share a spatial scale. On Door, whose reset range is about ±0.013 m, typical centroid separations of 0.01 m give a kernel value near 0.88 even at the narrowest swept σ, so the penalty is applied almost uniformly and is arithmetically close to no penalty at all. On GridWorld, whose centroids live in grid-cell units, the kernel degenerates into an identical-centroid indicator at every swept width. On Lift the kernel does come alive at σ = 0.02, and the 100.0 ± 0.0 ceiling hides it. The sweep is therefore inert on six of ten settings, and σ = 0.06 is directionally best but not statistically distinguishable from its neighbours (Holm-corrected p = 0.125), while γ = 0.6 and λ = 1.0 are each significantly best (Friedman χ²(4) = 33.13, p = 1.1e-06 and χ²(4) = 34.36, p = 6.3e-07; all Holm-corrected p = 0.0078). The fix, a per-task σ expressed as a fraction of that task's reset range, is identified and not yet run. The report presents this as a limitation, not as a virtue, and it presents the λ = 0 column of A13 reproducing A1 exactly in all ten settings as the internal-consistency check that it is.

**Geometry recovers cause only where configuration determines cause.** Cluster purity, the fraction of a geometric cluster's failures sharing the reasoning model's dominant root-cause label, runs from 0.78 to 0.93 (mean 0.877). It is lowest on Wipe, where the same end-effector position can correspond to insufficient contact force, to a missed patch, or to premature termination. Purity and silhouette are essentially uncorrelated across settings (r = 0.18, p = 0.62), so geometric separation is not semantic separation, and the descriptor's reach ends where the two come apart.

**The machinery is active early and idle late.** Failures per round fall from 42 to 2 over the budget, and between 15% and 34% of rounds never cluster at all because fewer than four failures remain. The allocation story is a story about the first two thirds of the budget; the last third runs the fallback rule.

Taken together: DISEIL's advantage is carried by an allocation mechanism over a *hand-designed geometric descriptor of the current round's failures*, and the language model's marginal contribution is small precisely because it is never told anything the descriptor does not already encode. That sentence is Aim 1's honest self-assessment and Aim 2's motivation, and it is the same sentence.

---

## 3. Bridge — how Aim 1 leads to Aim 2

Write this transition as an argument in three moves, not as an announcement.

**Move 1, from the measurement.** The ablations locate the mechanism in the allocation and not in the language model (A4: −1.08, 76.7% of the margin retained; A5: −1.01, 78.0%). Two readings are available. Either language models add little to demonstration selection, or the language model in DISEIL was given too little to reason over. The second reading is the one the evidence supports, because the model's entire input is a description of three frames and a handful of geometric coordinates from the round that has just finished. It is asked to choose a demonstration while being told nothing whatsoever about the demonstrations already collected. The component was not weak. It was blindfolded.

**Move 2, from the mechanism.** What the selector lacks is exactly what the cluster memory gestures at and fails to deliver. The memory records the *coordinates* of past corrections, in a space whose kernel width is mis-scaled on three of five tasks (A13). Even correctly scaled, it would answer "have we placed a demonstration near here?" and not "does the training set already contain this behaviour?". The two questions come apart wherever geometry does not determine cause, which D1 measures directly: purity falls to 0.78 on Wipe, where distinct causes share a location. A memory indexed on geometry cannot represent the difference between a mode it has covered and a mode it has merely visited.

**Move 3, to the proposal.** Give the selector a memory of what it has been taught, indexed in a representation that carries cause rather than coordinates. Language is the candidate, because the same model that must reason over the memory already reasons in it, because a caption composes across tasks in a way a Push-T contact descriptor does not, and because a language index yields a rationale the human supervising the budget can read. The mapping from a trajectory to that index is the inverse of the mapping vision-language-action models already learn. Hence Aim 2.

The sentence to close Chapter 4's limitations with, and to open Chapter 5 with, is: *DISEIL's selector reasons about the failure in front of it and knows nothing about the dataset behind it, and that is the limitation Aim 2 exists to remove.*

---

## 4. Aim 2 — Reverse VLA, a coverage memory of what has already been taught

Target venue CoRL 2027, full paper late May 2027.

### 4.1 The idea

A vision-language-action model maps (vision, language) to action: an instruction goes in, motor commands come out. Aim 2 inverts the mapping. A captioner takes a trajectory's visual observations *together with its executed action sequence* and emits language at three granularities: the trajectory's intent, the spans of the sub-skills it contains, and the root cause of its failure anchored at the flagged step. The action sequence is the discriminative input and the reason the inversion is not simply video captioning: two trajectories can look alike and differ in what the robot did, and the executed actions carry the skill semantics that frames alone miss.

Captions accumulate. Every demonstration that enters the dataset is captioned and stored, so the system holds a persistent, language-indexed record of what the policy has been taught. Coverage of a query is computed in that language space, and each stored skill is weighted by the policy's measured success rate on it, so that a skill which is *present in the dataset but not yet learned* still reads as a gap. Presence is not competence, and the memory must distinguish them or it will refuse to re-teach exactly the material that has been taught badly.

A single memory-conditioned selector then reasons over the current failure's caption together with what the memory returns, and prescribes the demonstration that fills a coverage gap. It also emits the rationale, in language, for why that demonstration and not another.

### 4.2 How Aim 1's components are subsumed, component by component

The report must make this table explicit, because it is what turns Aim 2 from a new idea into the next stage of one programme.

| Aim-1 component | What it did | What subsumes it in Aim 2 | Why the replacement is a generalisation and not a substitution |
|---|---|---|---|
| Vision-language model reading three frames | produced a free-text description of one failure, consumed once and discarded | the captioner's failure head | the same perceptual act, but its output is stored rather than discarded, so a description written in round 3 is still available in round 17 |
| Reasoning model assigning a root cause from a closed taxonomy | mapped that description to one of a fixed list of labels authored in the knowledge graph | the captioner's failure head, emitting open language anchored at the flagged step | the closed taxonomy was a compression forced by the absence of a memory: a fixed vocabulary is the only kind a stateless pipeline can compare across rounds. With a memory, comparison is by embedding, and the vocabulary can be open |
| Geometric descriptor and cluster engine | partitioned the round's failures into modes by position, orientation, progress and contact distance | embedding-space clustering over caption embeddings fused with the geometric descriptor | the geometric descriptor is retained as one component of the index, because A10 shows it separates modes well (silhouette peaking at 6-D in 10/10 settings) and because D1 shows language will be needed exactly where it fails (purity 0.78 on Wipe). Language is added to the index, not swapped for it |
| Cluster memory (recency-discounted Gaussian penalty over corrected centroids) | discouraged re-correcting a region recently corrected | the coverage memory | the cluster memory is the degenerate, geometry-only, single-round-scale case of a coverage memory. It answers "where have we placed demonstrations?"; the coverage memory answers "what does the dataset contain, and how well has it been learned?". The mis-scaled σ (A13) disappears with it, because coverage in an embedding space is not parameterised by a metric kernel width that must be re-tuned per task |
| Knowledge-augmented graph | supplied the constraints against which a prescription is verified, and the failure-mode vocabulary the reasoning model was allowed to use | the memory absorbs the vocabulary; **the constraint store is retained** | the graph does two jobs in Aim 1 and only one of them is subsumed. Its taxonomy is replaced by open language. Its constraints are not: a prescription must still be checkable against workspace bounds and reachability before an expert is asked to satisfy it, and A6 measures what happens without that check (−2.44 points, fallback rate rising to 23–35%). The report must not claim the graph is absorbed wholesale. The *naming* function is absorbed. The *verification* function survives into Aim 2 and becomes load-bearing again in Aim 3, where the person satisfying the request is not a scripted expert |
| Three separate model calls passing lossy text between them | perception, reasoning, prescription | one grounded, stateful selector | the claim is not "one model is better than three". The claim is that the reasoning is grounded, stateful and coverage-aware, and that this happens to be realisable in one component. The ablation that decides it is stated in advance (below) |

### 4.3 The experiment that decides whether Aim 2 is real

The question a panel will ask, and the one the CoC should ask first, is whether language does causal work or is a readable veneer on a retrieval loop that would work without it. The load-bearing experiment is a matched-information ablation: freeze the selection loop and vary only the representation the selector consumes, at equal capacity. Four arms: generated captions; Aim-1 geometric descriptors; a learned trajectory embedding at the same budget; content-scrambled captions as a placebo. Add an oracle-caption ceiling. The contribution holds only if generated language beats the learned embedding and the placebo and trends toward the oracle.

Pre-commit to the interpretation, in the CoC, before the experiment is run. If the learned embedding matches the captions, the contribution is interpretability and cross-task composability, not sample efficiency, and Aim 3 proceeds on the embedding index. Chapter 7's contingency section carries this and it does not break the arc, because Aim 3 needs an inventory that composes across tasks, and an embedding inventory composes; it simply cannot be read by a human, which costs Aim 3 its non-expert interface and forces a different one.

Primary metric: demonstrations-to-threshold and the area under the success-versus-demonstrations curve, with Aim 1 as the head-to-head. Continuity benchmark: Push-T. Breadth: a graded manipulation suite, plus a suite that ships language instructions and therefore supplies free ground truth for caption scoring and a defined skill taxonomy against which coverage can be measured.

### 4.4 The residual limitation Aim 2 will leave

State this in Chapter 5 while proposing the aim, not after. It is what makes the panel read Aim 3 as inevitable rather than appended.

The coverage memory of Aim 2 is **task-local**. It records what one policy has been taught about one task. Skills are shared across tasks in reality — the reach-and-align that precedes a door pull is the reach-and-align that precedes an insertion — and a task-local memory cannot see it. A demonstration collected for one task cannot be credited against the demand of another.

The supplier is **a scripted expert who is always available and always correct**. Aims 1 and 2 both count demonstrations, because in simulation a demonstration is a call to a motion planner and every demonstration costs the same. Outside simulation the budget is not a count. It is a person's time, and demonstrations differ enormously in what they cost that person to produce. A framework whose whole purpose is to spend a scarce resource well is measuring the wrong resource.

The selector **spends but does not price**. It knows which demonstration is most informative. It does not know what that demonstration is worth relative to what it costs, so it cannot choose between an expensive demonstration that closes a large gap and two cheap ones that close two small ones, and it cannot tell a human teacher which of several requests to satisfy first.

---

## 5. Bridge — how Aim 2 leads to Aim 3

The transition is the same shape as the first one, and the report should let the reader notice that.

Aim 1 gave the selector a partition of the failures in front of it, and its evaluation showed the selector was blind to the dataset behind it. Aim 2 gives the selector the dataset. Its evaluation will show that the dataset it is given is one task's dataset, held by one policy, filled by one inexhaustible expert. The object Aim 2 makes explicit — *what has been taught* — has a dual that Aim 2 leaves implicit: *what still needs to be taught, by whom, and at what price*. Aim 3 makes the dual explicit and closes the programme, because a demonstration's value was the quantity the thesis set out to raise, and the value of a thing is not established until it is priced against its cost.

---

## 6. Aim 3 — the proposal in full

**Recommended title:** *Ask for What You Lack: A Demonstration Demand Model for Sample-Efficient Imitation Learning*

Alternatives, if a reviewer finds "demand model" unfamiliar: *Pricing the Next Demonstration: Cross-Task Skill Demand for Sample-Efficient Imitation Learning*; *Teaching on Request: Coverage-Priced Demonstration Collection from Non-Expert Humans*.

Target venue CoRL 2028, abstract and full paper late May 2028, conference early November 2028.

### 6.1 Motivation

A generalist policy is trained on many tasks and often on several embodiments. Its demonstration budget is not held in demonstrations. It is held in the hours of the people who will produce them, and those people are not roboticists. Every part of the programme so far has assumed the opposite: an oracle that answers instantly, answers correctly, and answers any question at the same price. Under that assumption a demonstration budget is a counter, and the only question worth asking is which demonstration is most informative.

Drop the assumption and three questions appear at once, none of which Aims 1 or 2 can answer. Which *skill*, across the whole task family, is the policy short of? On which task and which embodiment should that skill be demonstrated, given that a demonstration of reach-and-align collected on one task may partly satisfy the shortage on another? And what is that demonstration worth, relative to the minutes of human time it will cost, compared against every other demonstration that could be requested instead?

Aim 3 answers all three by making demonstration *demand* an explicit, transferable, priced object, and by rendering a demand into a request that a person who has never seen the policy can satisfy.

### 6.2 The gap

Each of the four pieces below rests on something real in the literature. None of the four combinations exists.

Cross-embodiment data pooling is established: generalist policies are trained on data collected from many robots and many tasks. What is pooled is *supply*. Nobody has represented the *demand*: an explicit ledger of which skills the policy is short of, over an inventory shared across tasks and embodiments.

Cost-sensitive acquisition is standard in active learning, where the expected information gain of a query is weighed against its labelling cost. The queries priced are labels for data points that already exist. A robot demonstration does not exist until a human produces it, its cost varies by an order of magnitude with what is being asked, and no acquisition framework prices it.

Language models that recognise the limits of their own competence and ask a human for help exist, and are the direct precedent for the request interface. What they ask for is a disambiguation of the current instruction. They do not ask for a training demonstration, and they do not choose *which* demonstration to ask for on the basis of what the training set lacks.

Sub-trajectory retrieval shows that one collected trajectory can serve several downstream tasks. Retrieval is done at consumption time, from a corpus that is already fixed. Crediting a demonstration, at collection time, against the outstanding demand of every task it partially satisfies, is a different operation and it requires the demand ledger that does not yet exist.

### 6.3 Proposed approach

Four components. Each extends an Aim-2 component rather than replacing it, and the report should present them that way.

**A cross-task skill inventory.** Aim 2's captions are aggregated into a shared skill space annotated with the embodiment on which each instance was demonstrated. A skill is a language-indexed cluster of sub-trajectory captions ("align the gripper with a vertical handle and pull along the hinge arc"), not a task and not a trajectory. Coverage is measured over the inventory, so the question "does the policy know how to align with a handle?" is answerable independently of which task the handle appeared in. The inventory is where Aim 2's task-local memory becomes a programme-level object, and it is also where the caption index has to earn the claim made for it in Aim 2's matched-information ablation: geometric descriptors do not compose across tasks, and language is the interface that might.

**A demand model with a price.** For every skill in the inventory the demand model maintains a shortfall: how far the policy's competence on that skill falls below what the task family requires, weighted by how often the skill is on the critical path of a task the policy is failing. Each candidate demonstration request then carries two numbers. The first is an *expected* information gain — the Aim-1 quantity, which was measured after the fact, now predicted before the demonstration is collected, from the current coverage of the requested skill and the policy's measured competence on it. Aim 1's own measurement is the training signal for that predictor, which is the most direct way the three aims are linked by a single quantity. The second is an *expected human cost*: minutes of teacher time, estimated from the length and difficulty of comparable demonstrations already collected. Selection maximises expected information gain per unit of teacher time. The budget stops being a count of demonstrations and becomes a time budget, which is what it always was outside simulation.

**A non-expert teaching interface.** A demand is rendered as a request a person can act on: a natural-language instruction, a scene specification, and the rationale for why this demonstration is being asked for. The scene specification has already passed the constraint check — the same verification loop Aim 1 runs against the knowledge-augmented graph, retained through Aim 2 for exactly this moment — so no request is issued that the environment cannot instantiate or the robot cannot reach. The policy-solvability check is retained for the same reason and now has an economic reading: a request the current policy can already satisfy is a request that wastes a person's time, and a framework that prices human time cannot afford to issue one. The interface is the point at which the whole programme's central claim becomes checkable by someone outside it: a person who has never read the paper is handed a request, satisfies it, and the policy improves by the amount the demand model predicted.

**Transfer credit.** When a demonstration arrives it is captioned by the Aim-2 captioner, decomposed into its sub-skill spans, and credited against the outstanding demand of every task in the inventory that those spans partially satisfy. A demonstration requested for a door-opening shortfall reduces the alignment shortfall of an insertion task, and the ledger records it. Transfer credit is what makes the demonstration economy cross-task rather than per-task, and it is the mechanism by which a fixed number of human hours buys more competence than the same hours spent task by task.

The loop, stated for the algorithm float: roll out the generalist policy across the task family; caption the failures; update the competence estimate for every skill in the inventory; compute the shortfall; price each candidate request by predicted information gain per unit of teacher time; verify the highest-value request against the constraint store and against policy solvability; issue it to a teacher; caption the returned demonstration; credit it against every task whose demand it satisfies; aggregate and retrain.

### 6.4 Novelty

The demand ledger. A generalist policy that maintains an explicit, priced, cross-task statement of what it lacks does not exist. Coverage has been used as a selection criterion within a dataset; it has not been turned into a demand that can be issued, satisfied and settled.

Pricing a demonstration in human time, which converts the budget from a count into the resource that is actually scarce, and which makes the programme's central quantity — the value of one demonstration — an explicit, measurable, comparable number rather than an implicit objective.

Transfer credit at collection time, which is the operation that a retrieval-at-consumption-time framework cannot perform, because retrieval cannot influence what is collected.

The closed loop through a non-expert human. Aim 1 and Aim 2 both address a scripted oracle. Aim 3 addresses a person, which is the only version of the problem that anyone outside simulation has.

### 6.5 Evaluation strategy

Multi-task suites with a defined skill taxonomy, so that coverage is measurable against a ground-truth inventory rather than against the system's own captions. Cross-embodiment evaluation on pooled multi-robot data, to test whether a skill demanded on one embodiment can be satisfied on another.

The primary metric is economic and it is new to the programme: **teacher-time-to-threshold**, the number of minutes of human demonstration time required to bring the policy family above a target success rate. Demonstrations-to-threshold is reported alongside it, and the gap between the two curves is itself a result, because a framework that reduces the count while raising the cost per demonstration has achieved nothing.

Secondary metrics: transfer credit, measured as the number of tasks a single demonstration advances; request-satisfaction rate, the fraction of issued requests a non-expert teacher can actually fulfil; calibration of the demand model, measured as the correlation between predicted and realised information gain, which is the direct successor of Aim 1's confidence-versus-ΔSR correlation (r = 0.82 to 0.89 across the ten settings) and should be reported against it; and downstream policy gain per teacher-minute.

Controls, each removing exactly one component: per-task demand with no transfer credit; uniform demand across skills; demand without a price (maximise information gain, ignore cost); Aim-2 single-task selection, which is the head-to-head that keeps the programme's chain of comparisons unbroken from Aim 1 through Aim 3.

The human study is the load-bearing evaluation and it is the first point in the programme at which humans enter. Non-expert participants satisfy generated requests in simulation. Ethics approval will be sought from the Deakin human-research ethics committee before any participant is recruited, no personal data is collected beyond the demonstration itself, and no request is issued that has not passed the feasibility check.

### 6.6 Risks and mitigations

*Language may be too coarse an index across embodiments.* Two skills that read identically in language can be kinematically distinct on different robots, in which case coverage is mis-measured and the demand model asks for a demonstration it already holds. The mitigation is the one already flagged in Aim 2's risk list and adopted in Aim 3's inventory: fuse the caption embedding with a geometric or action embedding rather than indexing on language alone. Aim 1's descriptor survives into Aim 3 as the geometric half of that fusion, which is the reason it is retained rather than discarded in Aim 2.

*Non-expert demonstrations are suboptimal by construction.* Aim 1's information-gain argument rules out invalid demonstrations *by construction*, because prescriptions pass the feasibility check and demonstrations come from an expert. The second half of that argument fails the moment the demonstrator is a member of the public, and the report must say so plainly rather than let the argument carry over unexamined. A high pre-retrain loss on a non-expert demonstration is then genuinely ambiguous between novelty and incompetence. Aim 3 must therefore add a quality filter to the demand loop, and the demand model must be able to reject a satisfied request. Work on learning from suboptimal and preference-based human input is the starting point, and the demand ledger gives a natural test: a demonstration that reduces predicted shortfall but does not reduce realised failure is a demonstration the filter should have rejected.

*The price model may be wrong.* Estimating human time from comparable demonstrations is a prediction, and a systematically wrong one would misallocate the budget while appearing to optimise it. The mitigation is measurement: the human study yields realised times, the price model is scored against them, and the calibration curve is reported as a result rather than assumed.

*Ethics and recruitment may delay the human study.* The contingency, recorded in Chapter 7, is that the demand model is validated against scripted teachers with simulated cost models drawn from the demonstration-time distributions measured in existing large-scale human demonstration corpora. The economic claim then rests on a modelled cost rather than a measured one, which is weaker and is publishable, and the human study follows.

### 6.7 What Aim 3 completes

Aim 1 decides which failure to fix. Aim 2 knows what it has already taught. Aim 3 knows what it is worth to be taught next, and can ask a person for it. The quantity the thesis set out to raise, the value of one demonstration, is implicit in Aim 1, made observable in Aim 2, and made explicit, priced and transferable in Aim 3. That is the arc, and the closing chapter should state it in those terms and stop.

---

## 7. Coherence — what Chapter 7 must show

One thesis, three levers, and a single quantity traced through all three. The report earns the word "programme" by doing three things in Chapter 7 and by doing them concretely.

Trace the quantity. The value of one demonstration is *measured* in Aim 1 (pre-retrain per-step loss on the acquired demonstration, highest for DISEIL in all ten settings, with A3 as the warning that a high per-demonstration value does not license a claim about the *set*). It is *contextualised* in Aim 2, where the same demonstration's value depends on what the dataset already holds. It is *priced* in Aim 3, against the teacher-time it costs. Aim 1's confidence-versus-ΔSR correlation becomes Aim 3's demand-model calibration curve. Aim 1's after-the-fact information gain becomes Aim 3's predicted information gain, and Aim 1's measurements are the training data for the predictor. These are not analogies. They are the same quantity at three levels of resolution.

Show the components surviving. The geometric descriptor is retained in Aim 2 as half of a fused index and in Aim 3 as the guard against language collapsing distinct skills. The constraint store is retained in Aim 2 for feasibility verification and becomes load-bearing in Aim 3, where the person satisfying a request cannot be sent an impossible one. The policy-solvability check acquires an economic reading in Aim 3. Nothing in Aim 1 is discarded except the components its own ablations retired: the visual-embedding clustering branch (retired by A10) and the global memory kernel width (identified as mis-scaled by A13, re-specified per task).

State the contingencies without breaking the arc. If Aim 2's matched-information ablation shows that a learned embedding matches generated captions, Aim 2's contribution is reframed as interpretability and cross-task composability, and Aim 3 proceeds on the embedding inventory with a different teaching interface. If the Aim-3 human study is not approvable in time, the demand model is validated against scripted teachers with simulated cost models. Neither contingency removes a rung from the ladder.

---

## 8. Passages the writing agents must reuse verbatim in argument, if not in wording

These are the load-bearing sentences of the spine. Every one of them is traceable to the workbook, to the ablation dossier, or to a project fact. None may be softened.

1. The demonstration is the binding cost, so the objective is the information content of each demonstration under a fixed budget B.
2. The DAgger family decides *when*. Which failure to correct, and where the corrective demonstration begins, are unclaimed.
3. DISEIL attains the best mean in all ten settings, mean margin 3.7 points; the claim of record is the collapsed five-task result (5/5, one-sided p = 0.031, paired t(4) = 4.15, p = 0.014).
4. Removing the partition costs 4.01 points while information gain does not fall (mean +0.06, p = 0.23). High per-demonstration information gain is necessary and is not sufficient, because it carries no term for redundancy between demonstrations.
5. The language model and the vision-language model are each worth about one point (A4 −1.08, A5 −1.01) because the allocation decision is made by a geometric descriptor before either is called.
6. The selector is stateless and dataset-blind. It has a geometric cluster memory of *where* it has corrected, and no representation of *what* the dataset contains.
7. The memory kernel width is mis-scaled for narrow-reset tasks and is inert on six of ten settings. A per-task width is the identified fix and it has not been run.
8. Lift is uninformative for every ablation.
9. Aim 2's coverage memory subsumes Aim 1's cluster memory as its degenerate geometry-only case, and subsumes the knowledge graph's failure-mode vocabulary. It does not subsume the knowledge graph's constraints, which survive into Aim 2 and become load-bearing in Aim 3.
10. Aim 3 prices a demonstration in the resource that is actually scarce, which is a person's time, and asks that person for exactly the demonstration the policy lacks.
