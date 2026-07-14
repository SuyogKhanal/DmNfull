---

# 6. Project plan

## 6.1 Completed work

The first nine months of candidature delivered Aim 1 in full. The literature review covered interactive imitation learning and the query gates of the DAgger family, language and vision-language models as reasoners in robotics, structured environmental knowledge, and demonstration selection and curation, and it produced the statement of the gap in Section 3.1. The DISEIL framework was then specified and implemented as a single module with one entry point, so that one command runs one cell of the experimental matrix.

The experimental programme covers five tasks under two observation modalities, which is ten settings, against six comparison methods, with nine seeds on GridWorld and five seeds on each robot task. An ablation programme of eighteen studies, A1 to A18, was run on top of that matrix and is reported on the three ablation settings. The statistical analysis was carried out as a scripted pass over the results workbook and not by hand, and every figure and table in Section 5.1 is generated from the recorded run outputs. DISEIL attains the best mean success rate in all ten settings, with a mean margin of 3.71 points over the strongest baseline in each. Collapsed to five task means, the sweep holds at five from five, paired $t(4) = 4.15$, $p = 0.014$. The Aim-1 manuscript, *Demonstration Distillation for Sample-Efficient Imitation Learning*, was submitted in July 2026 and is under review. This report was drafted alongside it.

Four items of Aim-1 work remain outstanding and are scheduled before the author-response window rather than deferred. The cluster memory is re-run with a per-task kernel width, defined as a fraction of each task's reset range, which is the identified fix for the mis-scaling reported in Section 5.1.9. The compute and token-cost measurement is repeated at further seeds, because no cost figure in Section 5.1.7.5 carries cross-seed variance. The failure-count diagnostic is instrumented on one setting only and is extended to the three ablation settings. The prescription logs are inspected to resolve the disagreement between the account of bridging in Section 4.1.3 and the bridged share the diagnostics record. The kernel-width re-run is taken first, because it is the one defect whose fix is specified and whose outcome changes what the work is permitted to claim about its own memory.

## 6.2 Updated project plan table

Year 1, from November 2025 to August 2026, is complete. It carried the literature review, the specification and implementation of the framework, the full experimental matrix, the ablation programme, the Aim-1 manuscript, the compulsory higher-degree-research training and this report.

Year 2, from September 2026 to October 2027, is Aim 2. Problem formulation and the captioner begin in September 2026, while the Aim-1 review is still running, so that no development period waits on a review outcome. Implementation and the matched-information ablation of Section 4.2.4 run to the paper submission in late May 2027, and the mid-candidature progress review falls in the same month. Thesis writing begins in November 2027 and draws on material that has already been through peer review.

Year 3, from November 2027 to November 2028, is Aim 3 and the thesis. Ethics preparation begins in November 2027 and the application for the teaching study is lodged well before the experimentation window opens. The scripted-teacher validation of the demand model does not require approval and runs from February 2028, so the Aim-3 submission in late May 2028 does not rest on the approval date. The full thesis draft goes to the supervisors in September 2028, two months before submission.

Table 11 lists the milestones, the training items and the target venues, and Figure 19 draws the same schedule.

| ID | Milestone | Venue | Date |
|:-----|:--------------------------------------------------|:-------------------|:---------------|
| | *Year 1 (November 2025 to August 2026): Aim 1, completed* | | |
| M1 | Candidature start | | 13 Nov 2025 |
| T1 | Research integrity training | | 2 Dec 2025 |
| T2 | Respectful behaviour module, higher-degree research | | 2 Dec 2025 |
| T3 | Research induction training | | 10 Dec 2025 |
| M2 | Aim-1 framework specified and implemented | | Apr 2026 |
| M3 | Aim-1 experimental matrix complete, ten settings and six baselines | | Jun 2026 |
| T4 | SSC900 Academic Writing and Communication, passed | | Trimester 1, 2026 |
| T5 | Reproducibility and integrity audit of the Aim-1 results workbook | | Jul 2026 |
| M4 | Aim-1 paper submitted | AAAI 2027, main track | Jul 2026 |
| **M5** | **Confirmation of Candidature** | | **13 Aug 2026** |
| | *Year 2 (September 2026 to October 2027): Aim 2* | | |
| M6 | Aim 1 complete, review outcome resolved | AAAI 2027 | Feb 2027 |
| T6 | Reproducibility and integrity audit of the Aim-2 results | | May 2027 |
| **M7** | **Mid-candidature progress review** | | **May 2027** |
| M8 | Aim-2 paper submitted | CoRL 2027 | late May 2027 |
| M9 | Aim 2 complete, paper presented and thesis chapter drafted | CoRL 2027 | Nov 2027 |
| | *Year 3 (November 2027 to November 2028): Aim 3 and thesis* | | |
| T7 | Human-research ethics training and application preparation | | Nov 2027 |
| M10 | Human-research ethics approval, Aim-3 teaching study | | Mar 2028, target |
| T8 | Reproducibility and integrity audit of the Aim-3 results | | May 2028 |
| M11 | Aim-3 paper submitted | CoRL 2028 | late May 2028 |
| M12 | Full thesis draft to supervisors | | Sep 2028 |
| M13 | Aim 3 complete, human study analysed and chapter final | CoRL 2028 | Oct 2028 |
| **M14** | **Thesis submission** | | **Nov 2028** |

**Table 11.** Updated project plan.

Two dates are fixed by the university, M5 and M14, and two are fixed by a venue, M8 and M11. M10 is a target and not a commitment, because the approval date is set by the ethics committee and not by the candidate; the schedule is built so that the Aim-3 submission survives a delay in it.

## 6.3 Thesis plan

The thesis is written from the three papers and from this report. Table 12 gives the chapter list in binding order and the year in which each chapter is drafted, updated and finalised. Chapters 1, 2 and 3 exist in draft, in the form of this report and the submitted Aim-1 manuscript.

| Chapter | 2026 | 2027 | 2028 |
|:-------------------------------------------------------------------|:-----------|:-----------|:-----------|
| 1. Introduction | draft | update | final |
| 2. Background and literature review | draft | update | final |
| 3. Demonstration distillation under a fixed budget (Aim 1) | draft | update | final |
| 4. Reverse vision-language-action, a coverage memory (Aim 2) | | draft | final |
| 5. Demonstration demand across tasks, embodiments and teachers (Aim 3) | | | draft, final |
| 6. Discussion and conclusion | | | draft, final |
| 7. References | draft | update | final |
| 8. Appendices (ablation record, prompts, constraint stores, certificates) | draft | update | final |

**Table 12.** Thesis plan. Each chapter is marked by the year in which its draft is written and the year in which it is finalised; the chapters drawn from Aim 1 and from this report already exist in draft, and the Aim-2 and Aim-3 chapters are written from the papers as those aims complete.

The same schedule is drawn as a chart in Figure 19.

## 6.4 Gantt chart

![](figures_generated/gantt_chart.pdf)

**Figure 19. Project plan for the full candidature.**
