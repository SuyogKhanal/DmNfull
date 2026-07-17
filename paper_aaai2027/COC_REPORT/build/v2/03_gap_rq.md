# 3. Gap and research questions

The literature of Chapter 2 leaves one decision unclaimed, and this chapter states it. Section 3.1 sets out the gap at the three levels at which it appears. Section 3.2 states the three research questions the programme answers, one per level. Section 3.3 states the design by which each answer is validated.

## 3.1 The gap

The programme asks one question. What is a single expert demonstration worth, and how can that worth be raised, given that the demonstration is the input whose cost does not fall when compute is bought or a simulator is downloaded. Under a fixed budget of $B$ demonstrations, the only lever is the information content of each one. The gap in the literature is that no existing framework pulls that lever, and it is absent at three levels of resolution.

The first is a gap in the interactive loop. The published query gates decide when to hand control to the expert, and they decide it from a scalar attached to a single state. Selection methods from active learning and dataset curation decide which item to take from a pool that already exists [36, 83, 84]. Between the two lies a decision nobody makes. Given a round's worth of failures, which failure mode should receive this round's demonstration, and from which configuration should that demonstration begin. A per-state gate cannot answer either question, because it has no representation in which two failures are the same mistake and a third is a different one, and because it must begin the corrective demonstration at whatever state tripped the threshold, which is frequently a state the policy has already ruined. Under an unbounded budget this costs nothing. Under a budget of twenty demonstrations it is the whole problem.

The second is a gap in what the selector knows. A gate reads the current state. A failure-reasoning model reads the current episode. Neither holds a representation of the training set assembled so far, so neither can separate a genuinely novel failure from one the dataset already covers several times over. Under a restricted budget, a demonstration spent re-teaching material the policy has already been shown is a demonstration lost.

The third is a gap in what a demonstration is taken to cost. Counting demonstrations is the right accounting in simulation, where a demonstration is a call to a motion planner and every call costs the same. Cross-embodiment pooling has made demonstration *supply* a shared object [70], and cost-sensitive acquisition is standard where the query is a label for a datum that already exists [84]. The minutes a demonstration takes to produce are recorded by the large collection efforts [42]. No framework holds demonstration *demand*: an explicit statement of which skills a policy is short of, priced against the time of the person who would have to produce them, and shared across the tasks and embodiments over which those skills recur.

The three aims close the three levels of the gap in order, and each aim raises the value of a demonstration at the level the previous aim's evaluation could not reach. Table 3 states the three levels.

| Aim | What the selector reasons over | What it decides | Level |
|:------|:----------------------------------------|:------------------------------|:-----------------|
| Aim 1 | this round's failures, partitioned into failure modes, and the environment's explicit constraints | which failure mode to correct, and where the corrective demonstration begins | within a round |
| Aim 2 | the same failures, and a language index of what the policy has already been taught | whether a failure is a coverage gap or a re-teaching of material the dataset holds | across the dataset |
| Aim 3 | the coverage of a skill inventory shared across tasks, embodiments and teachers, priced against human time | which demonstration to buy next, from whom, and what it is worth | across tasks and teachers |

**Table 3. The three aims and the level at which each raises the value of a demonstration.**

## 3.2 Research questions

**RQ1.** Under a fixed budget $B$, does choosing which failure mode to correct and where the corrective demonstration begins yield a policy with a higher final success rate than choosing only when to intervene?

The DAgger family answers *when*, and *when* is one decision of three. The other two, *which* and *where*, are the two the family leaves fixed by default: the failure it corrects is whichever rollout tripped the gate first, and the demonstration begins at the state that rollout was already in. RQ1 asks whether making those two decisions deliberately is worth anything under a budget small enough that a redundant correction cannot be recovered from. The question is answerable because the interactive loop, the policy class, the expert and the evaluation protocol can all be held fixed while the demonstration-acquisition rule alone is varied, which is the design of the Aim-1 experiments. Section 4.1 states the aim and the framework that answers the question; Section 5.1 reports the experiments and the result.

**RQ2.** Can the demonstration selector be given a model of what it has already taught, by inverting the mapping that vision-language-action models learn so that an executed trajectory is turned into language rather than language into an action, and does selection driven by a coverage gap beat selection driven by the failure in front of the policy?

The forward mapping takes vision and language to an action [11, 43, 69]. Aim 2 inverts it, so that a trajectory's frames and its executed actions produce a language description, and the descriptions accumulate into a memory of what the dataset contains. The motivation is the second level of the gap. A selector that reads only the current round will spend a demonstration on a failure the training set already covers, and it has no signal that would tell it so. The question has a second half because a memory that is merely readable is not a memory that does causal work, and the difference is measurable: a selector given the same failures and a coverage memory can be compared against the same selector given the same failures and no memory. Section 4.2 states the method and names that comparison before it is run.

**RQ3.** Can demonstration demand be made explicit, priced against a teacher's time, and satisfied across tasks and embodiments, so that a generalist policy asks a non-expert human for exactly the demonstrations it lacks?

Generalist policies already pool demonstration supply across many robots and many tasks [70]. Cost-sensitive querying is standard where the query is a label for a datum that already exists [84]. Neither prices a demonstration that has not been collected, and neither carries a demand that transfers between tasks or between teachers. RQ3 asks whether the demand can be constructed as an object: a ledger of missing skills, a price in minutes of human time, and a request rendered so that a person who is not a robotics expert can satisfy it. It is the level at which the cost the thesis is about stops being counted in demonstrations and starts being counted in the time of the person supplying them. Section 4.3 proposes the construction and the human study that tests it.

The three questions compose. Aim 1's selector reasons about the failure in front of it and knows nothing about the dataset behind it, which is the limitation RQ2 exists to remove. Aim 2's memory is task-local and its supplier is a scripted expert who is always available and identically priced, which is the limitation RQ3 exists to remove. Each aim is the correction to the defect the previous aim's own evaluation exposed, and Table 6 records which components carry across all three.

## 3.3 Validation strategy

The method of validation is the matched comparison, and it is the same at every level. The interactive loop, the policy class, the expert, the retraining schedule and the frozen held-out evaluation set are held fixed, and the only quantity that varies between arms is the rule by which the round's demonstration is chosen. Holding everything else fixed is what licenses attributing a difference in final success rate to the acquisition rule, and it is why the published query gates are re-implemented inside this project's own loop and are not compared against their reported numbers.

Four commitments follow from that design and are honoured throughout the report.

Evidence is reported at the level at which it was measured. A setting is one task under one observation modality, and the two modalities of a task share the expert, the reward structure and the reset distribution, so the settings of a task are correlated by construction and the ten are not ten independent experiments. Every result is therefore reported per setting, and the pattern across settings is read with that correlation in view.

A component is claimed only where an ablation supports it. The Aim-1 ablation programme was designed so that it could retire components of the framework, and Section 5.1.8 records what it changed: one component re-specified, and the claims made for three others reduced. The same discipline applies forward. Section 4.2.4 names, before the experiment is run, the single ablation that decides whether language does causal work in the Aim-2 selector, and pre-commits to the interpretation of a negative result.

A mechanism claim is made only where the measurement can discriminate. Where the framework and its baselines both sit at the ceiling of the success-rate metric, a null result cannot separate a component that does nothing from a component whose effect cannot be observed, so the ablation programme is run in settings that have headroom for an effect to appear in.

Symbols carry the framework and values carry the instance. The budget $B$, the per-round acquisition count $D$ and the policy $f_\theta$ appear as symbols in the method and in the algorithm. The values at which they were validated appear once, in the experimental setup of Section 5.1.2, because the framework is defined for any fixed restricted budget and the reported instance is one point in that family.
