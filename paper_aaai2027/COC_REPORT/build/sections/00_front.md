<!-- Front matter: cover page, table of contents, executive summary.
     Assembler notes:
       * The cover page carries no page number, no running head and no section number.
       * The cover mark (A2I2 logo) takes no figure number and no caption.
       * Table-of-contents page numbers, the list of figures and the list of tables are
         generated at typesetting; the entries below fix the order and the wording.
       * Heading levels here are relative; the assembler normalises them. -->

![Deakin University. Deakin Applied Artificial Intelligence Initiative.](A2I2_Logo_Stacked_2025_Keyline.png)

## Confirmation of Candidature Report

**Deakin University**
Deakin Applied Artificial Intelligence Initiative (A2I2)

### Leveraging Large Language Models for Sample-Efficient Imitation Learning

**Candidate**
Suyog Khanal
Student identifier s226137394

**Supervisors**
Associate Professor Santu Rana
Dr Arun Kumar Anjanapura Venkatesh

**Candidature start date**
13 November 2025

**Confirmation of Candidature date**
13 August 2026

---

## Table of contents

Executive summary

1. Introduction and research vision
   1.1 The demonstration is the scarce resource
   1.2 What interactive imitation learning answers, and what it leaves open
   1.3 Central idea and thesis statement
   1.4 Research questions
   1.5 Contributions to date
   1.6 Scope and constraints
   1.7 Report outline

2. Background and literature review
   2.1 Imitation learning and behaviour cloning
   2.2 Dataset aggregation and the interactive loop
   2.3 The DAgger-family query gates
   2.4 Policy classes and why the framework is agnostic to them
   2.5 Standard machinery this work uses and does not claim
   2.6 Language and vision-language models as reasoners in robotics
   2.7 Structured environmental knowledge and constraint grounding
   2.8 Demonstration selection, curation and active learning
   2.9 Vision-language-action models
   2.10 Open problems and the gap addressed by this programme

3. The research programme
   3.1 One question in three stages
   3.2 Methodology and validation strategy

4. Aim 1. DISEIL: demonstration distillation for sample-efficient imitation learning
   4.1 Motivation and problem statement
   4.2 The gap in the DAgger family
   4.3 Problem formulation
   4.4 The DISEIL framework
       4.4.1 Perceive
       4.4.2 Partition
       4.4.3 Prioritise
       4.4.4 Prescribe
       4.4.5 Algorithm
   4.5 Architecture
   4.6 Implementation and experimental setup
   4.7 Results
   4.8 Information gain, starting performance, and why the gain is real
   4.9 Ablation studies
       4.9.1 Knockouts. What is load-bearing
       4.9.2 Design choices. Is the instantiation the right one
       4.9.3 Sensitivity and diagnostics. Where the instantiation is wrong
   4.10 What the ablations changed in the framework
   4.11 Limitations
   4.12 Status and publication

5. Aim 2. Reverse-VLA: a coverage memory of what has already been taught
   5.1 The limitation Aim 1 leaves
   5.2 Core idea. Inverting the vision-language-action mapping
   5.3 Method
   5.4 Architecture
   5.5 Novelty and positioning
   5.6 Evaluation strategy
   5.7 Risks
   5.8 Relationship to Aim 1 and target venue

6. Aim 3. Demonstration demand across tasks, embodiments and teachers
   6.1 The limitation Aim 2 leaves
   6.2 The idea. A demonstration demand model
   6.3 Proposed method
   6.4 Evaluation strategy
   6.5 Risks and what completes the story

7. Coherence of the research programme
   7.1 One thesis, three levers
   7.2 What each aim contributes to the thesis chapters
   7.3 Contingencies

8. Project plan and Gantt chart
   8.1 Completed work
   8.2 Publication plan
   8.3 Milestone table
   8.4 Gantt chart

9. Ethical considerations

10. Higher-degree research training and other research activities

11. Conclusion

12. References

Appendix A. Higher-degree research training certificates
Appendix B. Full ablation tables
Appendix C. Supplementary results
Appendix D. Statistical appendix

List of figures

List of tables

---

## Executive summary

Imitation learning converts expert demonstrations into a policy, and the demonstration is the one input whose cost does not fall. Compute is bought and architectures are downloaded, while every trajectory still has to be produced by a person or by a scripted oracle, one at a time. The binding constraint on a realistic deployment is therefore not the ability to collect more data. It is a fixed budget of $B$ demonstrations that has to be spent well. Interactive imitation learning, the family of methods descended from dataset aggregation [@ross2011dagger], spends that budget on a single decision. Its members differ only in the scalar signal that decides *when* to hand control to the expert: a learned safety classifier [@zhang2017safedagger], the spread of a Monte-Carlo dropout ensemble [@menda2017dropoutdagger], ensemble variance [@menda2019ensembledagger], a novelty and risk estimate under a target switching rate [@hoque2021thriftydagger], or a diffusion policy's own per-step training loss [@lee2025diffdagger]. Two further decisions are left unclaimed. Which failure to correct, and where the corrective demonstration should begin.

This programme claims that a large language model, given a structured description of how the policy is failing together with an explicit statement of what the environment permits, can make those two decisions, and that making them raises the information content of each demonstration under a restricted budget. The language model is not a controller and is never placed in the robot's control loop. It reads a summary of the policy's own failures and returns a request for one specific demonstration. Aim 1 realises the claim as a framework named after the title of its paper, **D**emonstration d**I**stillation for **S**ample-**E**fficient **I**mitation **L**earning, and that framework is DISEIL.

Each round, DISEIL rolls out the current policy, reduces each failed rollout to a six-dimensional geometric descriptor of the state at which the policy first became unreliable, and partitions the round's failures into failure modes. A cross-round memory penalises the modes that have recently been corrected, so that one persistent mode cannot absorb the whole budget. A language model then prescribes a single demonstration inside the selected mode. The prescription is screened twice before any expert time is spent on it, once against a store of explicit environmental constraints, and once against the current policy, because a scenario the policy can already solve teaches it nothing.

A setting is one task under one observation modality. DISEIL was evaluated on five tasks, a 5×5 grid-world, Push-T [@mu2021maniskill] and the Lift, Wipe and Door manipulation tasks [@zhu2020robosuite], each under state and image observations, which gives ten settings. Every setting compares DISEIL against five methods: the published query gates of the DAgger family, with a uniform-random query control standing in on the grid-world for the gate that requires a diffusion policy. Nine seeds were run on the grid-world and five on the robot tasks. DISEIL attains the best mean success rate in all ten settings. The mean margin over the strongest competing method in each setting is 3.7 points. The two observation modalities of a task share the expert, the reward structure and the reset distribution, so the ten settings are not ten independent experiments, and the claim of record is the conservative one: collapsed to five task means the sweep holds at five wins from five, with a one-sided sign test at $p = 0.031$ and a paired $t$-test at $t(4) = 4.15$, $p = 0.014$.

The ablations say where the advantage lives, and the answer is more specific than "the method works". Removing the partition costs 4.01 success-rate points while the per-demonstration information gain does not fall, which establishes that a demonstration can be individually informative and jointly redundant with the one collected in the round before. Replacing the prescription model with a deterministic heuristic costs 1.08 points, and removing the vision-language model costs 1.01, both of which lie inside the seed standard deviation of the corresponding full run. The reason is structural and is reported rather than argued away. The partition is geometric and consumes no output from any foundation model, so by the time the language model is called, the decision that matters has already been made. DISEIL's selector reasons about the failure in front of it and knows nothing about the dataset behind it, and that is the limitation the rest of the programme exists to remove.

Aim 2 gives the selector a memory of what it has already taught. A captioner inverts the mapping that vision-language-action models learn, turning a trajectory's observations and its executed actions back into language, and the captions accumulate into a coverage record that a new failure is checked against before a demonstration is requested. Aim 3 turns that record outward. Demand for a skill is shared across tasks and embodiments, and a demonstration is priced against the resource that is actually scarce outside simulation, which is a teacher's time, so that a generalist policy can ask a non-expert human for exactly the demonstrations it lacks.

Two defects in the Aim-1 instantiation are known, and this report states them where they arise rather than deferring them to a future-work paragraph. The memory's Gaussian kernel width is a single global constant and is mis-scaled for the tasks whose reset ranges are narrow, so the memory term is close to inert on six of the ten settings; a per-task width is the identified fix and it has not yet been run. The Lift task saturates at 100.0 ± 0.0 under DISEIL, with neither headroom nor seed variance, so it carries no information about any mechanism and is excluded from every mechanism claim in this report.

The Aim-1 manuscript was submitted to the AAAI 2027 main track in July 2026. Aim 2 targets CoRL 2027 and Aim 3 targets CoRL 2028, with thesis submission in November 2028.
