## Research programme coherence, project plan and Gantt chart

### One question, three stages

The programme asks one question, and the three aims are three answers to it at three levels of resolution. The question is what a single expert demonstration is worth and how that worth can be raised, because the demonstration is the input whose cost does not fall when compute is bought or a simulator is downloaded. Under a fixed budget of $B$ demonstrations the number of demonstrations is not the lever. The information content of each one is.

Aim 1 raises that quantity within a round. DISEIL partitions the current policy's failures into failure modes over a geometric descriptor, rotates the target mode under a cross-round memory, and prescribes where the next corrective demonstration should begin, subject to a feasibility check against the constraint store and a check that the current policy cannot already solve the prescribed scenario. The two decisions it claims, which failure to correct and where the demonstration starts, are the two that the DAgger family leaves unmade, since every member of that family decides only when to hand over. The evidence for Aim 1 is set out in the Aim-1 chapter and is not repeated here.

Aim 1's own ablations locate its limit, and the limit is what Aim 2 exists to remove. Replacing the prescription model with a deterministic heuristic costs 1.08 success-rate points and retains 76.7% of the margin over the strongest baseline; removing the vision-language model costs 1.01 points and retains 78.0%. Both effects are small because both components act downstream of the decisive step. The partition is geometric and consumes no output from any foundation model, so the question of which region of the failure distribution receives this round's demonstration has been settled before the language model is called. The selector reasons about the failures in front of it and holds no representation of the training set behind it. It can identify a grasp failure and cannot report that the dataset already contains six demonstrations of one. Aim 2 gives the selector a language-indexed memory of what it has been taught, by inverting the vision-language-action mapping so that a trajectory's observations and its executed actions are turned into captions, and by weighting each stored skill by the policy's measured competence on it, so that a skill present in the dataset but not yet learned still reads as a gap.

Aim 2's memory is task-local, and its supplier is a scripted expert who is always available, always correct, and charges the same price for every request. Outside simulation the budget is not a count of demonstrations. It is the hours of the person who must produce them, and those hours differ by an order of magnitude between an easy request and a hard one. Aim 3 makes demonstration demand an explicit object: a cross-task, embodiment-annotated skill inventory, a shortfall per skill, a predicted information gain and a predicted cost in teacher time for every candidate request, and a request rendered in language that a person who has never seen the policy can satisfy. The verification machinery of Aim 1 survives into Aim 3 and becomes load-bearing there, because a request issued to a human must be one the environment can instantiate and the robot can reach.

The through-line is a single quantity. The value of one demonstration is measured after the fact in Aim 1, as the policy's per-step loss on a newly acquired demonstration before retraining on it. It is contextualised in Aim 2, where the same demonstration is worth less if the dataset already holds its content. It is priced in Aim 3, against the minutes of human time it costs, and Aim 1's measurements are the training data for the predictor that Aim 3 needs. Each aim is the correction to the limitation that the previous aim's own evaluation exposed, which is why the three papers compose into one thesis and not into a portfolio.

### Completed work

The first nine months of candidature delivered Aim 1 in full. The literature review covered interactive imitation learning and the DAgger-family query gates, language and vision-language models as reasoners in robotics, structured environmental knowledge, and demonstration selection and curation, and it produced the statement of the gap that the programme works in. The DISEIL framework was specified and implemented as a single module with one entry point, so that one command runs one cell of the experimental matrix.

The experimental programme covers five tasks under two observation modalities, which is ten settings, against six baselines drawn from the DAgger family, at nine seeds on GridWorld and five seeds on the robot tasks, under a budget of twenty demonstrations acquired one per round. Fifteen ablation studies and five diagnostic studies were run on top of that matrix, and the statistical analysis was carried out as a separate, scripted pass over the results workbook rather than by hand. DISEIL attains the best mean success rate in all ten settings, with a mean margin of 3.71 points over the strongest baseline in each. The ten settings are not ten independent experiments, because the two modalities of a task share the expert, the reset distribution and the reward structure, so the claim of record is the conservative one: collapsed to five task means the sweep holds at five out of five, one-sided sign test $p = 0.031$, paired $t(4) = 4.15$, $p = 0.014$.

The Aim-1 manuscript, *Demonstration Distillation for Sample-Efficient Imitation Learning*, was submitted to the AAAI 2027 main track in July 2026. This report was drafted alongside it, for the Confirmation of Candidature on 13 August 2026.

Four items of Aim-1 work are outstanding and are scheduled before the AAAI author-response window rather than deferred. The compute and token-cost measurement is not complete: the workbook sheet has row labels and no numbers, and the per-round cost of the reasoning stack is a figure a reviewer will ask for. The failure-count diagnostic is instrumented on one setting only and should be run on all three primary settings. The per-task kernel width identified as the fix for the mis-scaled cluster memory has not been run, and the honest position in the report is that the memory is mis-scaled for the narrow-reset tasks and that the fix is identified and untested. The bridging diagnostic and the bridging knockout disagree with the prose written around them, and the prescription logs must be inspected before that ablation is written up.

### Publication plan

Aim 1 is submitted to the AAAI 2027 main track. The review cycle, the author-response window and the camera-ready deadline follow the venue's published timetable, and the chart in Figure 8.1 shows them at their indicative positions rather than at dates this report asserts. If the paper is not accepted, the manuscript is revised against the reviews and resubmitted to the next available main-track venue in the same field, and the schedule for Aims 2 and 3 does not move, because Aim 2's development begins in September 2026 and does not depend on the outcome of Aim 1's review.

Aim 2 targets CoRL 2027, with the abstract and the full paper due in late May 2027 and the conference held in October or November 2027. Aim 3 targets CoRL 2028, with the abstract and the full paper due in late May 2028 and the conference held in early November 2028. The choice of venue is deliberate for both. Aim 2 and Aim 3 are robot-learning contributions whose evaluation is a policy's sample efficiency on manipulation suites, and CoRL reviews that kind of claim on its own terms. The CoRL 2028 conference falls in the same month as the thesis submission, which is the tightest coupling in the plan and is addressed in the schedule risks below.

Continuity of comparison constrains the publication plan as much as it constrains the science. Push-T carries through from Aim 1 into Aim 2 as the head-to-head setting, and Aim 2's single-task selector is one of the controls in Aim 3, so the chain of comparisons runs unbroken from the first paper to the third. Aim 2's breadth evaluation adds a manipulation suite that ships language instructions, which supplies free ground truth for scoring captions and a defined skill taxonomy against which coverage can be measured [@liu2023libero; @mees2022calvin]. Aim 3's cross-embodiment evaluation draws on pooled multi-robot data [@oneill2023openx; @khazatsky2024droid].

### Milestones

Table 8.1 lists the milestones of the candidature. Each row is either a deliverable with a date, or an institutional checkpoint, and each one appears as a numbered diamond in Figure 8.1 at the same date. Any disagreement between the table and the chart is a defect.

| # | Milestone | Date | Status |
|---|---|---|---|
| M1 | Candidature start | 13 Nov 2025 | Complete |
| M2 | Aim-1 framework specified and implemented | Apr 2026 | Complete |
| M3 | Aim-1 experimental matrix complete (10 settings, 6 baselines, 15 ablations, 5 diagnostics) | Jun 2026 | Complete |
| M4 | Aim-1 manuscript submitted, AAAI 2027 main track | Jul 2026 | Complete |
| M5 | Confirmation of Candidature | 13 Aug 2026 | This report |
| M6 | Aim 1 complete (review outcome resolved, camera-ready or revision and resubmission) | Feb 2027 | Planned |
| M7 | Mid-candidature progress review | May 2027 | Planned |
| M8 | Aim-2 abstract and full paper submitted, CoRL 2027 | late May 2027 | Planned |
| M9 | Aim 2 complete (conference presented, thesis chapter drafted) | Nov 2027 | Planned |
| M10 | Human-research ethics approval for the Aim-3 teaching study | Mar 2028 (target) | Planned |
| M11 | Aim-3 abstract and full paper submitted, CoRL 2028 | late May 2028 | Planned |
| M12 | Full thesis draft to supervisors | Sep 2028 | Planned |
| M13 | Aim 3 complete (human study analysed, chapter final) | Oct 2028 | Planned |
| M14 | Thesis submission | Nov 2028 | Planned |

**Table 8.1.** Candidature milestones. Milestones M1 to M4 were achieved in the first nine months and are described in the completed-work subsection above. M5 is this report. The remaining ten are targets. The two dates fixed by the university are M5 and M14; the two fixed by a venue are M8 and M11; M10 is a target because the approval date is set by the ethics committee and not by the candidate, and the contingency if it slips is stated below.

### Gantt chart

![Candidature Gantt chart, November 2025 to November 2028.](figures_generated/gantt_chart.pdf)

**Figure 8.1.** Project plan for the full candidature at month resolution, from the candidature start on 13 November 2025 to thesis submission in November 2028. Bars are coloured by workstream. Hatched bars are low-intensity or venue-scheduled activity, where the candidate's own effort is intermittent and the dates are set by a review timetable rather than by the plan. Numbered diamonds are the milestones of Table 8.1, filled where the milestone has been achieved and open where it is planned. The vertical rule marks the Confirmation of Candidature on 13 August 2026 and separates completed work from planned work. The three aims overlap by design and not by accident: Aim 2's problem formulation begins in September 2026, while the AAAI review of Aim 1 is still running, and Aim 3's formulation begins in October 2027, while the CoRL 2027 review of Aim 2 is still running, so that no development period is spent waiting on a review outcome. Thesis writing begins in November 2027 and draws its chapters from this report and from the three papers, so that the final year holds integration and revision rather than first drafting.

Three properties of the schedule are worth stating explicitly, because they are the properties a panel should test it against.

The critical path runs through the two CoRL deadlines and terminates at the thesis submission. Each aim allocates about four months to problem formulation and methodology, four to five months to implementation, three months to experimentation, and two to three months to writing, with the writing overlapping the tail of the experimentation. That shape is the one that produced Aim 1 in nine months from a standing start, and it is repeated with more slack rather than less, because the candidate no longer has to build the experimental infrastructure from nothing.

The human study of Aim 3 is the only part of the programme that depends on an external approval, and it is scheduled to survive that dependency. Ethics preparation begins in November 2027 and the application is lodged well before the Aim-3 experimentation window opens, with approval targeted for March 2028. The scripted-teacher experimentation runs from February 2028 and does not require approval, so the CoRL 2028 submission in late May 2028 rests on the scripted-teacher validation of the demand model with cost models drawn from measured demonstration-time distributions. The human study runs from March to August 2028 and strengthens the paper if it completes in time and the thesis chapter if it does not. The economic claim of Aim 3 is weaker under a modelled cost than under a measured one, and the plan states that trade rather than betting the submission on the approval date.

Thesis submission in November 2028 coincides with the CoRL 2028 conference, which is the one point where two commitments land in the same month. The full thesis draft is therefore scheduled for September 2028, two months ahead of submission, and the pre-submission review and revision window runs from September to mid-November. Examination preparation begins in mid-September 2028 and continues past submission. The chapters themselves are not written in that window. They are written progressively from November 2027 onward, out of material that has already been through peer review.

### What could move, and what would move with it

The schedule has three identified failure points, and each has a response that does not remove a rung from the programme.

If Aim 1 is not accepted at AAAI 2027, the manuscript is revised and resubmitted, and the Aim-1 chapter of the thesis is written from the same results either way. The result of record does not depend on the acceptance.

If Aim 2's matched-information ablation shows that a learned trajectory embedding matches generated captions at equal capacity, then language is not buying sample efficiency and the honest reframing is that Aim 2's contribution is interpretability and cross-task composability. That interpretation is pre-committed here, before the experiment is run. Aim 3 then proceeds on an embedding inventory, which composes across tasks as required, and loses the non-expert teaching interface, which is replaced by an interface that renders a request from a retrieved exemplar rather than from a caption. The schedule does not move, because the substitution is at the level of the index and not of the loop.

If ethics approval for the Aim-3 teaching study is delayed beyond March 2028, the demand model is validated against scripted teachers with simulated cost models, as described above, and the human study moves into the thesis as a chapter contribution rather than a paper contribution. The critical path in that case runs through the thesis draft in September 2028 and not through the CoRL 2028 deadline, and the November 2028 submission date holds.
