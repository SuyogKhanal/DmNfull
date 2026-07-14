---

# 7. Ethical considerations

## 7.1 The position of the programme to date

No human participant has taken part in this research, and no human-subject data has been collected. Every experiment reported in Section 5.1 ran in simulation on a shared high-performance computing cluster, and the demonstrations that enter the training sets are produced either by a scripted or planner-based expert or, on the GridWorld task, by the candidate. No third party supplied a demonstration, and no personal information was recorded at any point.

The work carries no immediate deployment risk of its own. The framework decides which demonstration to collect next. It does not control a robot, and the language model is never in the path between an observation and an action at execution time. The policies trained in the evaluation are simulated manipulators and a grid agent, and none of them has been transferred to hardware.

## 7.2 The Aim-3 human study

Aim 3 is the point at which people enter the programme, and it does so in a bounded way. Non-expert participants will be asked to satisfy machine-generated demonstration requests in simulation, using a teleoperation interface. An application to the Deakin University human-research ethics committee will be prepared and approved before any participant is approached, and the target date for approval is recorded as milestone M10 in Table 11.

Four commitments constrain the design of that study, and they are stated here so that the panel can hold the eventual application to them.

Participation is voluntary and informed. Participants will be told what the demonstrations are used for, that the data trains a simulated policy, and that they may withdraw.

Data collection is minimal. The only data recorded is the demonstration itself, together with the time taken to produce it, because teacher time is the quantity Aim 3 measures. No personal information beyond what the consent process requires is collected, and demonstrations are stored against a participant identifier and not a name.

No request is issued that has not passed the feasibility check. The constraint store exists to reject prescriptions the environment cannot instantiate and the robot cannot reach, and with a person on the other end of the request an infeasible prescription is not a wasted round but wasted human time. The policy-solvability screen serves the same purpose, because a request the policy can already satisfy wastes a participant's time by construction.

Participants are not evaluated. The study measures the demand model and not the person. Demonstration quality is scored only to test the quality filter described in Section 4.3.2, and it is not reported per participant.

## 7.3 Data management, research integrity and the use of generative models

Primary data is retained and traceable to the claim it supports. Every figure and every table in Section 5.1 is generated from the recorded run outputs by script and not transcribed by hand, and the results workbook is the single source of every number in this report. Where a measurement was not taken, no placeholder is substituted for it, and the report says the measurement is missing. The items that remain open in the record are listed in Section 6.1 rather than resolved by assumption.

Two claims proposed during the study were not sustained by it, and both are retracted in the text where they arose: the halved-budget headline of Section 5.1.7.3, and the framing of the cluster memory as a principal driver of the framework's advantage.

The programme uses open-weight language and vision-language models as components [4, 96]. Their outputs are prescriptions and root-cause labels, and every prescription is verified against an explicit store of environmental constraints before it is acted on. The models are not used to generate the research claims, and the numbers in this report come from measured runs. The known failure mode of these models, the confident assertion of geometry they cannot perceive [15, 28], is the reason the framework hands them a computed descriptor and checks what they return.

Two wider issues are recorded, without overstating the programme's proximity to either. Automating the selection of what a person is asked to demonstrate is a form of task allocation to that person, and the participant on the other end of an Aim-3 request is being directed by a model. The mitigations are the ones above: the request is readable, it carries the reason it was issued, and it has been checked for feasibility before it is issued. The rationale the Aim-2 selector emits is what makes that direction auditable by the person who holds the budget, and it is scored as a deliverable. Separately, sample-efficient imitation learning lowers the cost of teaching a robot a task, and the tasks in this programme are manipulation benchmarks with no dual-use character. The framework is not task-specific and could in principle be applied to a task with a different character, which is a property it shares with imitation learning in general, and the programme develops no capability specific to a harmful application.
