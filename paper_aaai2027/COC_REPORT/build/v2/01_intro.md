# 1. Introduction

## 1.1 The demonstration budget

Imitation learning turns expert demonstrations into a policy. The earliest working system fitted a network to logged pairs of camera image and steering angle and drove a vehicle with the result [73], and the framework that generalises it fits a policy to logged expert state-action pairs [2, 5]. Model capacity and observation richness have grown since. The premise has not: somebody, at some point, produced the demonstrations.

Every other input to that pipeline has become cheap. Compute is bought, simulators run faster than real time, and architectures and pre-trained encoders are downloaded. The demonstration is the one input whose cost has not fallen, because it is produced by a person operating a robot one trajectory at a time, or by a scripted oracle written by an engineer who first had to solve the task by hand. Collecting demonstrations at scale is a logistics problem in its own right, and the projects that have done it read as such: a distributed teleoperation platform, and a year of coordinated collection across dozens of institutions [42, 60]. The number of demonstrations available to a given project is bounded by something other than the researcher's willingness to wait.

<!-- Teaser_Diagram_rot.pdf: small bottom float, held with its own caption. Placement is tuned so it lands on page 5 of the built PDF; do not move the block. -->

\begin{figure}[!b]
\centering
\includegraphics[width=0.40\textwidth]{../figures/Teaser_Diagram_clean.png}
\par\vspace{0.7em}
\begin{minipage}{0.92\textwidth}
\footnotesize\textbf{Figure 1. The demonstration-distillation loop.} A policy trained on a small demonstration set fails repeatedly. The failures are summarised and read by a language model, which prescribes the configuration at which the next demonstration is to be collected. The expert supplies that demonstration, it is added to the training set, and the policy is retrained. Each turn of the loop spends one unit of the demonstration budget.
\end{minipage}
\end{figure}

That bound would not matter if demonstrations did not help, and they do. Imitation-learning performance follows a scaling relationship in the number of demonstrations and in their diversity [51]. The demonstration is therefore both the input that most improves the policy and the input that is hardest to obtain, which is what makes it the binding cost of the enterprise. The constraint in a realistic deployment is a budget of $B$ demonstrations that has to be spent well.

The budget is the object this thesis is built on, so its parts are named once here and the names are kept. A *budget* $B$ is the total number of demonstrations the expert will supply over a run. Demonstrations are acquired in rounds, $D$ of them per round, and the policy is retrained after each round. $B$ and $D$ are symbols of the framework and of its algorithm; the values validated in this work are stated once, in the experimental setup of Section 5.1.2, because the framework does not depend on them.

Under a fixed $B$, the quantity that can still be raised is the *information content of each demonstration*: how much of what the policy does not yet know is contained in the one trajectory the expert is about to record. The claim this programme develops and tests is that a large language model, given a structured description of how the policy is failing and an explicit statement of what the environment permits, can raise it.

## 1.2 Interactive imitation learning

A policy cloned offline from expert data is trained on the expert's state distribution and deployed on its own, and the two distributions come apart as soon as the learner's actions determine what it sees next [85]. Small action errors move the learner off the demonstrated manifold, where its errors are larger, and the error compounds over the horizon [78]. Dataset aggregation removes the compounding by labelling the states the learner actually visits: roll out the current policy, ask the expert what to do at the states it reached, add those labels to the training set, retrain, and repeat [79]. The interactive branch of imitation learning that grew out of this is now a field with its own taxonomy of feedback types [14].

Its members share one skeleton and differ in one component. The skeleton is to roll out, decide whether to hand control to the expert, aggregate the expert's labels, and retrain. The component that differs is the scalar signal that opens the gate. SafeDAgger trains a classifier to predict when the policy is about to deviate from the expert by more than a tolerance [101]. DropoutDAgger reads the spread of a Monte-Carlo dropout ensemble over the novice's action [65]. EnsembleDAgger reads the variance of an ensemble of independently trained policies and combines it with the discrepancy between the novice's action and the expert's [66]. ThriftyDAgger combines a novelty estimate with a learned risk estimate and calibrates the pair against a target switching rate [35]. Diff-DAgger reads a diffusion policy's own per-step training loss, which is available for free and requires no second model [47]. Section 2.3 gives the family as instances of one template.

Every one of those gates answers the same question, which is *when* to ask the expert for help. Three consequences follow from answering only that question, and they are the opening this programme works in.

A per-state gate cannot see a batch. It fires on the state in front of it. Given twenty rollouts that all failed, it has no representation in which two of them are the same mistake and a third is a different one, so it cannot tell a redundant correction from a novel one. Under an unbounded budget this costs nothing, because every failure is eventually labelled. Under a budget of twenty demonstrations it is the whole problem.

A per-state gate has no memory across rounds. Nothing in the signal records that the region it is firing on was already corrected in the previous round and in the round before that. One failure that persists can therefore absorb a large share of a small budget, while a failure that occurs less often is never reached.

A per-state gate inherits the state that tripped it. The corrective demonstration begins wherever the policy happened to be when the signal crossed the threshold, and that state is frequently one the policy has already ruined: the object has been knocked out of reach, or the gripper has closed on nothing. The expert then spends the demonstration recovering from a situation that would not arise under a competent policy, rather than teaching the behaviour that would have avoided it.

So *when* is one decision, and it is one of three. The other two are unclaimed: *which* failure to correct, and *where* the corrective demonstration begins.

Choosing a demonstration is also not the same problem as choosing a data point, and the distinction matters because a large literature already chooses data points. Active learning, coreset and diversity selection, and demonstration curation all select from a pool that has already been collected [3, 36, 64, 83, 84], and the word *distillation* in the title of the Aim-1 paper is not the dataset-distillation sense, in which a collected dataset is compressed into a smaller synthetic one that trains as well [13]. The framework proposed here prescribes a demonstration that does not exist yet, and then has an expert produce it. Section 2.9 treats the distinction in full.

## 1.3 Central idea and thesis statement

A language model is not a controller in this work. A large language model is a model over token sequences; it does not close a control loop here, it does not output torques, and it is never in the path between an observation and an action at execution time. It is the component that reads a structured summary of the policy's own failures, together with an explicit statement of what the environment permits, and returns a request for one specific demonstration.

The division of labour follows from what these models are measured to be good at. Language and vision-language models name causes reliably when they are handed structured evidence: a vision-language model can summarise a robot's experience and say why an episode failed, and can be trained specifically to reason over manipulation failures [22, 55]. They are unreliable at metric and spatial reasoning from pixels alone [15, 28], and their proposals must be grounded in what the robot can actually do before they are acted on [1]. The framework is built around both findings. The model is handed a low-dimensional geometric descriptor of each failure rather than raw pixels, the partition of failures into modes is computed geometrically and not by the model, and every prescription the model emits is checked against an explicit store of environmental constraints and revised until it is feasible. What is left to the model is the decision that structured evidence supports, which is what kind of correction the selected region of the failure distribution needs, and where the demonstration that supplies it should begin.

The thesis statement is one sentence. *Language models can raise the information content of each demonstration under a restricted budget, and raising it is what makes imitation learning sample-efficient.*

The programme tests that statement at three levels, one per aim, and each aim removes a limitation that the previous aim's own evaluation exposed. **Aim 1** is DISEIL, named after the title of the paper that reports it, *Demonstration Distillation for Sample-Efficient Imitation Learning*. It reasons over the current round's failures, partitioned into failure modes, and decides which mode to correct and where the corrective demonstration begins, so it raises the value of a demonstration within a round. Its selector knows nothing about the demonstrations already collected. **Aim 2** supplies that record. A captioner inverts the mapping that vision-language-action models learn, turning a trajectory's observations and its executed actions back into language, and the captions accumulate into a coverage record that a new failure is checked against before a demonstration is requested, which raises the value of a demonstration across the dataset. Aim 2's record is task-local, and its supplier is a scripted expert who is always available and identically priced. **Aim 3** turns the record outward. Demand for a skill is shared across tasks, embodiments and teachers, and a demonstration is priced against the resource that is actually scarce outside simulation, which is a teacher's time, so that a generalist policy can ask a non-expert human for exactly the demonstrations it lacks.

Aim 1 is complete and is reported in this document. Aims 2 and 3 are proposed. The gap the three aims close, and the three research questions they answer, are stated in Section 3, after the literature that establishes the gap.

---
