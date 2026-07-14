# Restructure plan for CoC_Report.md

Binding spec: `build/supervisor_revision_spec.md`. Every instruction below traces to a spec item.
Source of truth for numbers: `ablations_results/DISTIL_ablation_results.xlsx` and
`../context/results_data.md`. No number may be invented, and no number may be carried over from the
current text without being checked against those two sources.

Method name is **DISEIL** everywhere. "DISTIL", "PACE" and "P4" must not appear. "A2I2" must not
appear; write *Deakin Applied Artificial Intelligence Initiative* in full (spec A3; the two current
occurrences are the title page, line 6, and Section 10.4).

---

## 0. Two decisions the architect has taken, flagged for the supervisor

**0.1 Ethical considerations is retained as Section 7.** No spec item deletes the current Chapter 9,
and a Confirmation of Candidature report that proposes a human study cannot omit it. The section
outline in the commissioning brief does not mention it, which the architect reads as an omission
rather than an instruction. It is retained, trimmed from three pages to two, and placed after the
project plan. If the supervisor intends it deleted, remove Section 7 and reclaim two pages.

**0.2 The prose "Publication plan" subsection is deleted, not kept.** Spec G1 deletes the paragraphs
before the old 8.2 as redundant with 8.2, which implies 8.2 survives; spec C4 forbids target venues
"not as a paragraph, not as a sentence, anywhere" outside the project-plan table. C4 is the more
specific and more emphatic instruction, so it governs: venue names live only in the *Venue* column of
the updated project plan table (Section 6.2) and in the Gantt, which is that table drawn. The prose
publication plan is deleted, which satisfies G1 as well.

---

## 1. The new outline

```
Title page                       (A1: "Leveraging" stays. A2: no "Planned thesis submission".
                                  A3: Deakin Applied Artificial Intelligence Initiative, in full.)
Table of contents                (A4)
Abstract                         (A5: renamed from "Executive summary")

1.  Introduction                                                  (B1, B4)
    1.1  The demonstration is the scarce resource
    1.2  What interactive imitation learning answers, and what it leaves open
    1.3  Central idea and thesis statement
    (no "Contributions to date"  — B2)
    (no "Scope and constraints"  — B3)
    (no "Report outline"         — B3)

2.  Background and literature review
    2.1  Imitation learning and behaviour cloning
    2.2  Dataset aggregation and the interactive loop
    2.3  The query gates of the DAgger family
    2.4  Uncertainty estimation                                   (B5)
    2.5  Policy classes                                           (B6)
    2.6  Standard machinery                                       (B7)
    2.7  Language and vision-language models                      (B8)
    2.8  Structured environmental knowledge and constraint grounding   (B9, unchanged)
    2.9  Demonstration selection, curation and active learning         (B9, unchanged)
    2.10 Vision-language-action models                                 (B9, unchanged)
    2.11 Open problems                                            (B10: gap removed to Section 3)

3.  Gap and research questions                                    (B11)
    3.1  The gap
    3.2  Research questions          (RQ1, RQ2, RQ3, brought down from the Introduction)
    3.3  Validation strategy         (the matched comparison; from the old 3.2)

4.  Aims and approaches                                           (B12)
    4.1  Aim 1. Demonstration distillation under a fixed budget
         4.1.1  Motivation and problem statement   (refers briefly to RQ1)
         4.1.2  Problem formulation
         4.1.3  Methodology                        (B13: SHORT — the gist plus some detail)
                • the DISEIL framework: perceive, partition, prioritise, prescribe
                • the geometric descriptor (Table 4)
                • the cluster memory, the context set, the two screens
                • Algorithm 1                      (H5: numbered block, atomic steps)
                • Figure 2, the architecture
    4.2  Aim 2. Reverse vision-language-action, a coverage memory     (C1: ≤ ~4 pp, ONE figure)
         4.2.1  The limitation Aim 1 leaves
         4.2.2  Core idea and proposed method       (captioner, coverage memory, selector)
         4.2.3  Architecture                        (Figure 3 — the one figure)
         4.2.4  Evaluation strategy                 (Table 5, the matched-information ablation)
         (no "Risks and mitigations" — C3;  no target venue — C4)
    4.3  Aim 3. Demonstration demand across tasks, embodiments and teachers   (C2: ≤ ~3 pp)
         4.3.1  The limitation Aim 2 leaves, and the gap
         4.3.2  Proposed method                     (skill inventory, priced demand, transfer credit)
         4.3.3  Component lineage across the three aims   (Table 6 — C5, kept)
         4.3.4  Evaluation strategy, including the human study
         (no "Risks and mitigations" — C3;  no target venue — C4)

5.  Progress report
    5.1  Progress on M1                                           (B13)
         5.1.1  Implementation
         5.1.2  Experimental setup   (tasks, modalities, settings; policies; B, D and seeds;
                                      initial demonstrations; baselines; metrics.
                                      All concrete values live here and nowhere else.)
         5.1.3  Results: the main comparison            (Table 7, Figures 4 and 5)
         5.1.4  Per-demonstration information gain      (Table 8, Figure 6)
         5.1.5  Prescription confidence as an in-round predictor   (Figure 7)
         5.1.6  The failure modes the framework discovers          (Figure 8)
         5.1.7  Ablation studies                        (E1: three settings only)
                5.1.7.1  Scope and conventions
                5.1.7.2  Knockouts: what carries the result   (Figs 9–14)
                5.1.7.3  Design choices                        (Figs 15–18)
                5.1.7.4  Sensitivity and diagnostics           (Table 9, Table 10, Figs 19–20)
                5.1.7.5  Computational cost of the reasoning pipeline   (Table 11)
         5.1.8  What the ablations changed in the framework
         5.1.9  Limitations
    5.2  M1 conclusion                                            (B15: ≤ 1 page)

6.  Project plan
    6.1  Completed work
    6.2  Updated project plan table                (G2; incl. HDR training rows G3; bold G4;
                                                   Venue column is the only place venues appear, C4)
    6.3  Thesis plan                               (G5; modelled on Table 19 of the Vignesh sample)
    6.4  Gantt chart                               (G6: no examination-preparation bar;
                                                   caption is the title only)

7.  Ethical considerations                          (see 0.1 above)
    7.1  The position of the programme to date
    7.2  The Aim-3 human study
    7.3  Data management and research integrity

Appendix. Higher-degree research training           (G7, G8)
    One short statement: the compulsory training is complete; Figure 22 is the evidence.
    A.1  Certificate of completion, Deakin Safety and Research Integrity Training
    A.2  Certificate of completion, Respect at Deakin, graduate research and supervision module
    A.3  Statement of results, SSC900 Academic Writing and Communication

References                                          (G9: LAST. Nothing after them.)
```

Deleted outright: the old Section 7 "Coherence of the research programme" (B14), the old top-level
Conclusion (B15), the old Section 10 prose (G8), Appendices B, C and D (G9), and the trailing
"List of figures" and "List of tables" (G9 and A4 — front matter is title page, contents, abstract).

### Narrative rule for Section 5.1 (H4)

Section 5.1 is written as the story that was actually run: the framework was implemented, it was
evaluated, and then components were added and the evaluation was repeated. It is not written as
"what the evaluation says about the framework". The method still comes first: Section 4.1.3 states
the framework, and Section 5.1 reports the build-up. Concretely, 5.1.7 opens on the allocation ladder
(the system with nothing but a random allocation, then the fallback rule, then the partition) and
works upward to the full system, rather than opening on a table of everything that was removed.

---

## 2. Content migration map

Every section of the current report. "→" means the content moves; "DELETED" means it is removed, with
the spec item that removes it.

### Front matter

| Current | Destination | Note |
|---|---|---|
| Title page, lines 1–18 | Title page | Delete the `Planned thesis submission` row (A2). Delete `(A2I2)` from the institute line and write the name in full (A3). Keep the title as registered (A1). Font: Liberation Serif (A6). |
| `\tableofcontents`, line 20 | Table of contents | Order is title → contents → abstract (A4). |
| Executive summary, 25–43 | **Abstract** | A5. Rewrite: bold the acronym letters once at first mention — **D**emonstration d**I**stillation for **S**ample-**E**fficient **I**mitation **L**earning. Delete the closing sentence naming AAAI, CoRL 2027 and CoRL 2028 (C4). Keep the conservative aggregate claim and the two known defects. |

### Chapter 1, Introduction and research vision

| Current | Destination | Note |
|---|---|---|
| Heading "1. Introduction and research vision" | **1. Introduction** | B1. |
| 1.1 The demonstration is the scarce resource | 1.1, trimmed | B4. Keep Figure 1 (the teaser). Keep the definition of the budget in symbols: B and D are framework symbols, the validated instance appears only in 5.1.2. |
| 1.2 What interactive IL answers and leaves open | 1.2, trimmed | B4. Keep the three consequences of a per-state gate and the *when / which / where* framing. Cut the paragraph distinguishing demonstration distillation from dataset distillation down to two sentences; the full treatment is in 2.9. |
| 1.3 Central idea and thesis statement | 1.3, trimmed | B4. Keep the thesis statement and the "a language model is not a controller" commitment. |
| 1.4 Research questions (RQ1–RQ3) + Table 1 | **3.2** (RQs) and **3.1** (table, renamed) | B11. The RQs are brought down out of the Introduction. Table 1 is renamed (F1: "One quantity, three levels" is wordplay). New title: *The three aims and the level at which each raises the value of a demonstration.* |
| 1.5 Contributions to date | **DELETED** | B2. |
| 1.6 Scope and constraints | **DELETED** as a section | B3. Its two load-bearing facts survive elsewhere: the definition of a *setting* and the mode/modality convention go to 5.1.2; the no-human-participants statement goes to 7.1. |
| 1.7 Report outline | **DELETED** | B3. |

### Chapter 2, Background and literature review

| Current | Destination | Note |
|---|---|---|
| 2.1 – 2.3 | 2.1 – 2.3, unchanged in scope | Keep Table 2 (query gates) → renumbered **Table 1**. |
| 2.4 Uncertainty estimation and the limits of a scalar | **2.4 Uncertainty estimation** | B5. |
| 2.5 Policy classes and why the framework does not depend on them | **2.5 Policy classes** | B6. The independence is evident from the text. |
| 2.6 Standard machinery this work uses and does not claim | **2.6 Standard machinery** | B7. |
| 2.7 What language and VLMs are and are not reliable at | **2.7 Language and vision-language models** | B8. |
| 2.8, 2.9, 2.10 | unchanged | B9. Table 3 (selection methods) → **Table 2**. |
| 2.11 Open problems and the gap this programme addresses | **2.11 Open problems** (the three open problems only) | B10. The three paragraphs that state the gap move to **3.1**. |

### Chapter 3, The research programme

| Current | Destination | Note |
|---|---|---|
| 3.1 One question in three stages | **3.1 The gap**, merged with the gap paragraphs from the old 2.11 | B11. |
| 3.2 Methodology and validation strategy | **3.3 Validation strategy** | Keep the four commitments (matched comparison, evidence at the level measured, a component is claimed only where an ablation supports it, symbols carry the framework and values carry the instance). Delete the "a task with no headroom is evidence about nothing" paragraph as a *statement about Lift*: Lift appears in no ablation and the report does not explain its absence (standing fact + D-G3). The commitment survives in general form, without naming Lift. |

### Chapter 4, Aim 1 (the chapter that is split)

| Current | Destination | Note |
|---|---|---|
| 4.1 Motivation and problem statement | **4.1.1** | Add one sentence referring to RQ1 (B12). |
| 4.2 The gap in the DAgger family | absorbed into **4.1.1** | It restates Section 2.3 and the old 1.2; two sentences suffice. |
| 4.3 Problem formulation | **4.1.2** | Equations 1–4 kept. B and D stay symbolic. |
| 4.4 The DISEIL framework (4.4.1–4.4.5) | **4.1.3 Methodology** | B13. Must be SHORT. Keep: the four stages, the flagged step (Eq. 5), the descriptor (Eq. 6, Table 4 → **Table 3**), the partition (Eq. 7), the memory (Eq. 8), the context set (Eq. 9), the two prescription arms, feasibility verification (Eq. 10), the solvability check (Eq. 11), and the naming pipeline in three sentences. Cut: every measured number (the 0.78–0.93 purity range, the 19–30% bridging shares, the fallback measurements, the σ-degeneracy paragraph) — all of it is evidence and belongs in 5.1. Cut every forward reference of the form "and ablation Ax measures it". |
| 4.4.6 Algorithm | **4.1.3**, as Algorithm 1 | H5. Rebuild as a numbered algorithm block. Short. One action per line: no line may carry two verbs. Split the current line 4 (roll out + record loss), line 12 (standardise + argmax + cluster), lines 15–16 (three context rules), line 26 (aggregate + append to memory). Loop header reads `for r = 1 to B`, never `B = 20`. |
| 4.5 Architecture (Figure 3) | **4.1.3** | Figure → **Figure 2**. Keep the block-by-block reading, cut the ablation numbers baked into it ("removing the cluster engine costs 4.01 points" etc.), which now live in 5.1.7. |
| 4.6 Representative prompts | **5.1.1 Implementation** | Cut from eight pages to about three: keep one perception call, one reasoning call and one prescription call, each with one logged reply. Delete the second worked example (keep the Wipe round, drop the GridWorld round, or the reverse — one is enough). |
| 4.7 Representative environmental constraints | **5.1.1 Implementation** | Cut to about two pages: keep the Push-T workspace nodes, the Door bound, and the Wipe `select_only` implication, which is the one that shows the store doing something a bounding box cannot. Drop the GridWorld BFS predicate quotation and paraphrase it in one sentence. |
| 4.8 Implementation and experimental setup (4.8.1–4.8.6) | **5.1.2 Experimental setup** | Unchanged in substance. This is where every concrete value lives (B = 20, D = 1, seeds, initial demonstration counts, the six baselines, the three metrics). Delete the sentence pointing at ablation A12 as the evidence for D = 1 (E2: A12 is removed). D = 1 is stated as the validated instance of the framework symbol D, with no ablation citation. |
| 4.9 Results (4.9.1–4.9.3) | **5.1.3** | Table 5 → **Table 7**, rebuilt (F2). Figure 4 → **Figure 5**, with the paragraph text removed from the artwork (D row, Fig 4). Figure 5 → **Figure 6**. |
| 4.10 Information gain (4.10.1–4.10.3) | **5.1.4** | Table 6 → **Table 8**, rebuilt (F3). Figure 6 → **Figure 7**. |
| 4.11 Prescription confidence | **5.1.5** | Figure 7 → **Figure 8**. |
| 4.12 The failure modes the framework discovers | **5.1.6** | Figure 2 (`clustering_modes_pushT.pdf`) moves here from the method and becomes **Figure 4**: it is a result, not a method statement, and moving it shortens 4.1.3. |
| 4.13 What the results establish, and what they do not | **5.2 M1 conclusion** | It is already a conclusion. Merged with the old Chapter 11. |
| 4.14.1 Scope, conventions and what is reported here | **5.1.7.1** | Rewrite under E1: the ablations are run and reported on three settings, GridWorld (image), Push-T (state) and Door (image). Delete the "eight settings with headroom" / "eight robot settings" apparatus and every sentence that explains why Lift is excluded (standing fact: Lift appears in no ablation and we do not explain its exclusion). Delete the "fifteen studies and five diagnostics" count and replace it with the post-renumbering count (see §4). |
| 4.14.2 Knockouts | **5.1.7.2** | Figures 8–13 → **Figures 9–14**, with the changes in the D row of the spec. |
| 4.14.3 Design choices | **5.1.7.3** | A12 (demonstrations per round) and its Figure 18 are **DELETED** (E2). Figures 14–17 → **Figures 15–18**. |
| 4.14.4 Sensitivity and diagnostics | **5.1.7.4** | Table 7 → **Table 9**. Figure 19 → **Figure 19**. Figure 20 (cluster purity) is **DELETED and replaced by a table** → new **Table 10** (spec, Fig 20 row). Figure 21 → **Figure 20**, with the in-figure text boxes removed. Delete the paragraph beginning "One cross-check passes cleanly…" and ending "…is reported as an internal consistency check" (H1). |
| 4.14.5 Computational cost | **5.1.7.5** | Table 8 → **Table 11**, rebuilt (F4). Figure 22 **DELETED** (spec, Fig 22 row). Delete the SLURM job identifiers and all infrastructure detail (H2): no job numbers, no cluster names, no telemetry-reconstruction procedure. Delete the single-seed caveats (H3): the sentences "Every cell in Table 8 comes from one seed, seed 1", "Seed 1 in every arm", and the equivalent caveat in the limitations. |
| 4.15 What the ablations changed in the framework | **5.1.8** | Keep. The two retired components (the visual-embedding clustering branch; the single global kernel width) are the point. Clustering is geometric in every run: no R3M/PCA branch is described anywhere, not even as a retired one — state the descriptor as geometric and move on. |
| 4.16 Limitations (4.16.1–4.16.9) | **5.1.9** | Compress nine subsections to about three pages. Keep, as prose paragraphs and not as numbered subsections: the selector knows nothing about the dataset (this is the motivation for Aim 2); the descriptor is hand-designed and recovers cause only where configuration determines cause; the memory kernel width is mis-scaled; each language-model component is worth about one point, and the reason is structural; the allocation machinery is active early and idle late; the reasoning pipeline costs seconds and tokens; the experiments are in simulation. **Delete**: the single-seed caveat (H3), the Lift blind-spot paragraph (Lift appears in no ablation and is not discussed), and the "open items in the ablation record" subsection, whose four items are engineering notes and belong in 6.1. |
| 4.17 Status and publication | **6.1 Completed work** | Delete the venue name from the prose (C4). The fact that the manuscript is submitted and under review survives; the venue appears only in the project-plan table. |
| 4.18 From Aim 1 to Aim 2 | **4.2.1** | It is the motivation for Aim 2 and belongs at the head of Aim 2. Compress to two paragraphs. |

### Chapter 5, Aim 2 → Section 4.2, at most four pages, one figure

| Current | Destination | Note |
|---|---|---|
| "Target venue: CoRL 2027…" line | **DELETED** | C4. |
| 5.1 The limitation Aim 1 leaves | **4.2.1**, merged with the old 4.18 | |
| 5.2 The research gap | absorbed into **4.2.1** | one paragraph |
| 5.3 Core idea | **4.2.2** | |
| 5.4 Proposed method (5.4.1–5.4.5) | **4.2.2**, compressed | Keep the captioner, the training signal in one paragraph, the coverage memory with Eq. 13, the memory-conditioned selector with Eq. 14, and one round of the loop as a short list. Cut the five paragraphs on caption-training sources to one. |
| 5.5 Architecture (Figure 23) | **4.2.3** | Figure 23 → **Figure 3**. This is the ONE figure Aim 2 is allowed (C1). |
| 5.6 What Aim 2 subsumes from Aim 1 (Table 9) | **DELETED** | Page budget (C1). Its content is subsumed by the component-lineage table, which is kept by C5 and now covers all three aims. |
| 5.7 Expected contributions | absorbed into **4.2.2**, as one short paragraph | |
| 5.8 Novelty and positioning | absorbed into **4.2.1**, two sentences | The literature is already in Chapter 2. |
| 5.9 Evaluation strategy (Table 10) | **4.2.4** | Keep Table 10 → **Table 5**: it is the pre-committed decisive experiment and the strongest thing in the chapter. Keep the pre-commitment to the interpretation of a negative result, in two sentences. |
| 5.10 Risks and mitigations | **DELETED** | C3. |
| 5.11 Relationship to Aim 1, and the limitation Aim 2 will leave | one closing paragraph of **4.2.4** | The limitation is what motivates Aim 3, so it survives as a link and not as a subsection. |

### Chapter 6, Aim 3 → Section 4.3, at most three pages

| Current | Destination | Note |
|---|---|---|
| 6.1 From Aim 2 to Aim 3, 6.2 The limitation Aim 2 leaves | **4.3.1** | |
| 6.3 Research question and the idea | **4.3.1** | RQ3 is stated in 3.2 and is referred to here, not restated in full. |
| 6.4 The gap (Table 11) | **4.3.1**, prose | Table 11 **DELETED** for the page budget (C2); the four rows become two sentences. |
| 6.5 Proposed method | **4.3.2** | Keep the four components: skill inventory, priced demand model, non-expert teaching interface, transfer credit. Keep the closing loop statement. |
| 6.6 What each aim contributes (Table 12) | **4.3.3** | Table 12 → **Table 6**. **KEEP** — C5 says so explicitly. |
| 6.7 Novelty | absorbed into **4.3.2**, one paragraph | |
| 6.8 Evaluation strategy (Table 13) | **4.3.4** | Table 13 **DELETED** for the page budget (C2); the five metrics become a single paragraph naming teacher-time-to-threshold as the primary metric and the calibration curve as the successor of Aim 1's confidence correlation. Keep the human-study paragraph. |
| 6.9 Risks and mitigations | **DELETED** | C3. The one item that is not a risk but a design commitment — that a non-expert demonstration breaks Aim 1's information-gain argument, so a quality filter is required — moves into 4.3.2 as one sentence. |
| 6.10 Relationship to Aims 1 and 2, and target venue | closing paragraph of **4.3.4**, with the venue removed | C4. |

### Chapter 7, Coherence of the research programme

| Current | Destination | Note |
|---|---|---|
| 7.1 One thesis, three levers | **DELETED** | B14. |
| 7.2 What each aim contributes to the thesis chapters | **DELETED** as prose | B14. Its content is exactly the new thesis-plan table (G5), which is where a reader will now find it. |
| 7.3 Contingencies | **DELETED** | B14. |

### Chapter 8, Project plan → Section 6

| Current | Destination | Note |
|---|---|---|
| 8.1 Completed work | **6.1** | Keep, and add the four outstanding Aim-1 items from the old 4.16.9. Remove the venue name (C4). |
| 8.2 Publication plan | **DELETED** as prose | C4, and see §0.2. Its content becomes the *Venue* column of the table in 6.2. |
| 8.3 Milestones (Table 14) | **6.2 Updated project plan table** | G2. Table 14 → **Table 12**. Add HDR training rows with completion dates (G3): research integrity training, research induction, respectful-behaviour module, the reproducibility and integrity audits, and SSC900 Academic Writing. Bold the important tasks (G4). Add a *Venue* column so that C4 is satisfied. |
| — | **6.3 Thesis plan** (new **Table 13**) | G5. Model it on Table 19 of the Vignesh sample report in `other_students_coc_sample_ref/`: one row per thesis chapter, with the source material and the target completion date. Rows: Introduction; Literature review; Aim 1 (from the submitted manuscript and this report); Aim 2; Aim 3; Conclusion. |
| 8.4 Gantt chart (Figure 24) | **6.4** | Figure 24 → **Figure 21**. Regenerate without the examination-preparation bar (G6). Caption is the title only (G6): *Figure 21. Project plan for the full candidature.* Delete the three "properties of the schedule" paragraphs that follow it — the chart speaks for itself, and the examination-preparation sentence in the third of them dies with the bar. |

### Chapter 9, Ethical considerations → Section 7

| Current | Destination | Note |
|---|---|---|
| 9.1 The position of the programme to date | **7.1** | Absorbs the no-human-participants sentence from the deleted 1.6. |
| 9.2 The Aim-3 human study | **7.2** | Keep the four commitments. |
| 9.3 Data management, research integrity and generative models | **7.3** | Keep, trimmed to one page. |
| 9.4 Broader considerations | absorbed into **7.3**, one paragraph | |

### Chapter 10, HDR training → Appendix

| Current | Destination | Note |
|---|---|---|
| 10 (all prose), 10.1.1, 10.1.2, 10.1.3, 10.2, 10.3, 10.4 | **DELETED** | G8. Replaced by one short statement in the Appendix: the compulsory training is complete, Figure 22 is the evidence, and the certificates are A.1, A.2, A.3. Nothing more. |
| Table 15 (compulsory modules) | **DELETED** | G8 ("nothing more"). The same three items now appear as dated rows of the project-plan table (G3), which is where the panel will look for them. |
| Figure 25 (Compulsory Training Status) | **Appendix**, as **Figure 22** | G8 keeps it as the evidence. Spec G8 refers to it by its current number, 25; after renumbering it is 22. |

### Chapter 11, Conclusion

| Current | Destination | Note |
|---|---|---|
| 11.1 What Aim 1 established | **5.2 M1 conclusion** | B15. |
| 11.2 What the evidence did not support | **5.2** | B15. This is the half that must not be lost: the two design claims the ablations did not sustain. |
| 11.3 What the programme will have shown by November 2028 | **DELETED** | B15 caps 5.2 at one page, and the forward look is already in 4.2 and 4.3. Venue names would violate C4 in any case. |
| 11.4 The next step | one sentence at the end of **5.2**, or **6.1** | The four outstanding items belong in 6.1 Completed work. |

### Chapter 12 and the back matter

| Current | Destination | Note |
|---|---|---|
| 12. References | **References**, last thing in the document | G9. |
| Appendix A (certificates) | **Appendix A.1–A.3**, placed BEFORE the references | G7. No longer a numbered section. |
| Appendix B (full ablation tables) | **DELETED** | G9. The three-setting ablation numbers are in 5.1.7; the ten-setting matrix is out of scope under E1. |
| Appendix C (supplementary results) | **DELETED** | G9. D1's content becomes the new Table 10; D2 and D3 are quoted in 5.1.7; A11's table becomes Figure 18. |
| Appendix D (statistical appendix) | **DELETED** | G9. The two conventions that survive (the unit of analysis; the resolution floor of a paired test) are stated in one paragraph of 5.1.7.1. |
| List of figures, List of tables | **DELETED** | G9 and A4. |

---

## 3. Page budget

Current: 150 pages. Target: ~100. Budgeted: **102**, with three pages of slack against the figure
rebuilds. Depth in the Progress report is preserved: the evidence is not cut, the ceremony is.

| New section | Old material | Old pp | New pp | Δ |
|---|---|---|---|---|
| Title page | title page | 2 | 1 | −1 |
| Table of contents | contents | 2 | 2 | 0 |
| Abstract | Executive summary | 2 | 2 | 0 |
| 1. Introduction | 1.1–1.7 | 9 | **5** | −4 |
| 2. Background and literature review | 2.1–2.11 | 16 | **14** | −2 |
| 3. Gap and research questions | old 3, plus 1.4 and the gap half of 2.11 | 3 | **3** | 0 |
| 4.1 Aim 1 | old 4.1–4.5 | 17 | **9** | −8 |
| 4.2 Aim 2 | old 5 | 14 | **4** | −10 |
| 4.3 Aim 3 | old 6 | 10 | **3** | −7 |
| 5.1 Progress on M1 | old 4.6–4.17 | 50 | **36** | −14 |
| 5.2 M1 conclusion | old 4.13 and old 11 | 4 | **1** | −3 |
| 6. Project plan | old 8 | 6 | **6** | 0 |
| 7. Ethical considerations | old 9 | 3 | **2** | −1 |
| — | old 7, Coherence | 4 | **0** | −4 |
| Appendix (HDR training + certificates) | old 10 + old Appendix A | 8 | **4** | −4 |
| References | old 12 | 10 | **10** | 0 |
| — | old Appendices B, C, D | 11 | **0** | −11 |
| — | List of figures, list of tables | 2 | **0** | −2 |
| **Total** | | **150** | **102** | **−48** |

### Where the fifty pages come from, itemised

1. **Deleted sections — 21 pages.** The Coherence chapter (4 pp, B14); the top-level Conclusion, which
   becomes one page inside the Progress report (3 pp net, B15); Appendices B, C and D (11 pp, G9); the
   trailing lists of figures and tables (2 pp, G9); and the "Contributions to date", "Scope and
   constraints" and "Report outline" subsections of the Introduction (part of the intro trim below).
2. **Aim-2 and Aim-3 compression — 17 pages.** Aim 2 goes from 14 pages and one figure plus two tables
   to 4 pages and one figure plus one table (C1). Aim 3 goes from 10 pages and three tables to 3 pages
   and one table (C2). The risks-and-mitigations subsections (C3) and every target-venue paragraph (C4)
   are deleted outright; the subsumption table, the four-literatures table and the Aim-3 metrics table
   are cut for the budget.
3. **The A12 study and the removed figures — 4 pages.** A12, demonstrations per round, is deleted in
   full: the study, its figure (old Figure 18) and its discussion (E2), including the sentence in the
   setup that cited it as the justification for D = 1. Old Figure 20 is replaced by a compact table, and
   old Figure 22 is deleted as unnecessary. Their surrounding discussion contracts with them.
4. **The Introduction trim — 4 pages.** Nine pages to five (B4). Sections 1.5, 1.6 and 1.7 go entirely
   (B2, B3); the remaining three subsections are tightened, and the research questions are moved down
   rather than duplicated.
5. **The HDR training chapter — 4 pages.** Four pages of prose become one short statement in the
   appendix (G8), and the training items reappear as dated rows of the project-plan table (G3).
6. **The Aim-1 restructure — 14 pages, and no loss of evidence.** The method chapter's ceremony is what
   goes: the eight pages of verbatim prompts become three; the four pages of knowledge-graph JSON become
   two; the ablation scope is restricted to three settings (E1), which removes the ten-setting apparatus,
   the two competing aggregate definitions and the recurring explanations of why a saturated task carries
   no weight; the nine limitation subsections become three pages of prose; the SLURM job identifiers and
   the infrastructure detail go (H2); the single-seed caveats go (H3); and the paragraph above old
   Figure 20 goes (H1). Every table, every figure and every measured number in the Progress report
   survives, and the setup section is not shortened at all.

---

## 4. Ablation renumbering (E2, E3, E4)

A12 (demonstrations per round) is removed entirely. The remaining A-series is renumbered, and the
D-series is folded into it as a continuation, so that there is one series and not two.

| Old | New | Study |
|---|---|---|
| A1 | **A1** | cluster memory off (λ = 0) |
| A2 | **A2** | uniform-random allocation of the budget over recorded failures |
| A3 | **A3** | clustering off, greedy worst-loss selection |
| A4 | **A4** | prescription model replaced by the dominant-representative heuristic |
| A5 | **A5** | vision-language model removed |
| A6 | **A6** | knowledge-augmented graph removed from the prompts |
| A7 | **A7** | bridging placement disabled |
| A8 | **A8** | deterministic nearest-untried fallback rule promoted to the whole method |
| A9 | **A9** | composition of the context set |
| A10 | **A10** | width of the geometric descriptor |
| A11 | **A11** | the budget sweep |
| ~~A12~~ | **REMOVED** | demonstrations per round — deleted with its figure and its discussion (E2) |
| A13 | **A12** | the three memory constants (γ, σ, λ) |
| A14 | **A13** | the cluster count |
| A15 | **A14** | number of cited episodes and the selection rule |
| D1 | **A15** | cluster purity and geometric separation |
| D2 | **A16** | the distribution of the selected cluster count |
| D3 | **A17** | targeted and bridged prescriptions |
| D4 | **A18** | failures per round |
| D5 | **A19** | computational cost of the reasoning pipeline |

Post-renumbering the programme is **nineteen studies, A1 to A19**, with no D-series. Every sentence
that counts them ("fifteen ablation studies and five diagnostics", "fifteen studies and five
diagnostics", "an ablation programme of fifteen studies") must be rewritten to that count. The word
*diagnostic* may still be used descriptively for A15–A19, but they are numbered as A's.

### Every place in the current text that must be updated

Line numbers are those of the current `CoC_Report.md`.

| Lines | What is there now | Action |
|---|---|---|
| 123, 1243, 1615, 1750 | "fifteen studies and five diagnostics" (four separate occurrences) | Replace with the post-renumbering count. Line 123 is inside "Contributions to date", which is deleted; line 1750 is inside the HDR chapter, which is deleted. |
| 419, 778 (×2), 1061, 1065 | A12, demonstrations per round | **Delete.** Line 419 forward-references it from the framework; line 778 cites it as the evidence for D = 1; lines 1061–1065 are the study itself. Old Figure 18 (line 1057–1059) goes with them. |
| 479, 1075, 1107, 1109, 1168, 1325, 1372, 1436, 1501, 1551, 1579, 1786, 2188, 2192 | A13, the memory constants | Renumber to **A12**. |
| 1041 (×2) | A14, the cluster count | Renumber to **A13**. |
| 1037, 1039 (×2) | A15, the context set size | Renumber to **A14**. |
| 461, 532 (×2), 922 (×2), 939, 1115 (×3), 1117, 1323, 1361, 1371, 1428, 1436, 2087, 2102, 2253 | D1, cluster purity | Renumber to **A15**. |
| 461, 939 (×2), 1041, 1047 (×2), 1125, 2104, 2120, 2192, 2254 | D2, cluster-count distribution | Renumber to **A16**. |
| 495, 939 (×2), 2122, 2137, 2255 | D3, targeted and bridged prescriptions | Renumber to **A17**. |
| 461, 939, 1125, 1127 | D4, failures per round | Renumber to **A18**. |
| 939 (×2), 947, 1131, 1227, 2194 | D5, compute cost | Renumber to **A19**. |
| 953 | "the phrase *reasoning stack*, used below and in Section 4.14.5" | Keep the phrase; update the section cross-reference to 5.1.7.5. |
| 2019–2156 | Appendices B and C, which are built entirely on the A/D labels | Deleted (G9); no renumbering needed, but the numbers they carry must be checked against the workbook before they are used anywhere in 5.1.7. |

Composite references must be updated as units, not by find-and-replace: line 1041 reads "**A14 and D2**,
the cluster count" and becomes "**A13 and A16**"; line 1037 reads "**A9 and A15**, the context set" and
becomes "**A9 and A14**"; line 1115 reads "**D1**, and the limit of a geometric descriptor" and becomes
"**A15**". Run the renumbering high-to-low (A15 → A14 → A13 first, then D5 → D4 → … → D1) so that no new
label collides with an old one that has not yet been rewritten.

### The scope restriction that governs every ablation number (E1)

Ablations are reported on **three settings only: GridWorld (image), Push-T (state), Door (image)**.
Consequences the rewriter must apply consistently:

- Every per-setting ablation number is one of those three. Every aggregate is the mean over those
  three, recomputed from `DISTIL_ablation_results.xlsx`, and is labelled "mean over the three ablation
  settings". The existing aggregates ("−4.01 points over the eight settings with headroom", "11.3% of
  the margin retained", "−1.08", "−1.01", "−0.75", "+10.35 / +4.49 / +2.68", "mean 24.4%", "0.877",
  "r = 0.93, p = 0.0008", "r = −0.23, p = 0.58") are **all** computed over eight or ten settings and
  **none of them may be carried over unchanged.**
- The two aggregate scopes now vanish from the text: "the eight settings with headroom" and "the eight
  robot settings" are deleted as phrases, along with every sentence that defines them.
- Statistical tests over settings (Wilcoxon, Friedman, sign test) have three paired observations in the
  ablation section, which is below the resolution of any of them. Report the three per-setting values
  and their common sign; do not report a p-value that the three-setting design cannot produce. Where a
  test cannot be recomputed at the three-setting scope, **delete the test rather than invent a number**,
  and raise it back to the orchestrator rather than substituting a value.
- The main comparison in 5.1.3 and 5.1.4 is **not** an ablation and keeps all ten settings, its five
  task means and its aggregate tests. E1 restricts the ablation programme, not the sweep.
- Lift appears in no ablation table, no ablation figure and no ablation sentence, and no sentence
  explains its absence (D-G3, standing fact).

---

## 5. Figure and table renumbering

### Figures

Three figures are deleted (18, 20, 22). One new table replaces old Figure 20. Old Figure 2 moves from
the method to the Progress report, which reorders the front of the sequence. Numbering is continuous.

| New | Old | Source file | Where it sits now | Change required |
|---|---|---|---|---|
| **1** | 1 | `../figures/Teaser_Diagram.pdf` | 1.1 | Rotate one more step clockwise (spec, Fig 1). |
| **2** | 3 | `../figures/Architectural Diagram.pdf` | 4.1.3 | Remove all orange text (D-G1). The file is currently referenced with a space in the name; rename to `Architectural_Diagram.pdf`. |
| **3** | 23 | `figures_generated/aim2_architecture.pdf` | 4.2.3 | The one figure Aim 2 is allowed (C1). Remove orange text (D-G1). |
| **4** | 2 | `../figures/clustering_modes_pushT.pdf` | 5.1.6 | Moved out of the method (it is a result). Caption trimmed; no verdict inside the artwork (D-G2). |
| **5** | 4 | `figures_generated/F14_aggregate_significance.pdf` | 5.1.3 | Remove the paragraph text baked into the image (spec, Fig 4). |
| **6** | 5 | `../figures/all_5_task_comparison.pdf` | 5.1.3 | Remove orange text (D-G1). |
| **7** | 6 | `../figures/info_gain_boxplot.pdf` | 5.1.4 | Remove orange text (D-G1). |
| **8** | 7 | `../figures/confidence_vs_success.pdf` | 5.1.5 | Remove orange text (D-G1). |
| **9** | 8 | `figures_generated/F1_allocation_ladder.pdf` | 5.1.7.2 | Remove the "A3 falls below the best baseline" annotation and its arrow (spec, Fig 8). This is the reference style for Figures 10, 11 and 12. |
| **10** | 9 | `figures_generated/F2_gain_without_allocation.pdf` | 5.1.7.2 | Redraw as a grouped bar chart in the Figure-9 style (spec, Fig 9). |
| **11** | 10 | `figures_generated/F5_grounding_and_feasibility.pdf` | 5.1.7.2 | Redraw as a bar chart in the Figure-9 style (spec, Fig 10). |
| **12** | 11 | `figures_generated/F4_reasoning_and_vision_small.pdf` | 5.1.7.2 | Redraw as a bar chart in the Figure-9 style (spec, Fig 11). |
| **13** | 12 | `figures_generated/F6_bridging.pdf` | 5.1.7.2 | Delete the left panel; keep the right panel only, as a single-panel figure, on the three ablation settings (spec, Fig 12). |
| **14** | 13 | `figures_generated/F3_knockout_summary.pdf` | 5.1.7.2 | Restrict to the three ablation settings; no Lift, no ten-setting sweep (spec, Fig 13; E1). |
| **15** | 14 | `figures_generated/F7_descriptor_dimensionality.pdf` | 5.1.7.3 | Remove the "argmax in 10/10 settings" and "chosen descriptor 6-D" annotations and remove Lift (spec, Fig 14). |
| **16** | 15 | `figures_generated/F11_context_and_selection.pdf` | 5.1.7.3 | Restrict to the three ablation settings; remove annotation prose (spec, Fig 15). |
| **17** | 16 | `figures_generated/F12_cluster_count_distribution.pdf` | 5.1.7.3 | Three ablation settings only (spec, Fig 16). |
| **18** | 17 | `figures_generated/F8_budget_sweep.pdf` | 5.1.7.3 | Delete the left panel; make the right panel three panels — Push-T (state), GridWorld (image), Door (image); move the in-figure sentence about the baseline catching up into the body text (spec, Fig 17). |
| — | ~~18~~ | `figures_generated/F9_demos_per_round.pdf` | — | **DELETED** with A12 (spec, Fig 18; E2). |
| **19** | 19 | `figures_generated/F10_memory_constants.pdf` | 5.1.7.4 | Keep; apply E1 (three settings), E3 and E4 (the study is now A12). |
| — | ~~20~~ | `figures_generated/F15_cluster_purity.pdf` | — | **DELETED, replaced by new Table 10** (spec, Fig 20). |
| **20** | 21 | `figures_generated/F13_failures_over_budget.pdf` | 5.1.7.4 | Remove the descriptive text boxes inside the figure (spec, Fig 21). |
| — | ~~22~~ | `figures_generated/F16_compute_cost.pdf` | — | **DELETED** (spec, Fig 22). |
| **21** | 24 | `figures_generated/gantt_chart.pdf` | 6.4 | Remove the examination-preparation bar; caption is the title only (spec, Fig 24; G6). |
| **22** | 25 | `Compulsory Training Status.png` | Appendix | Keep; moves to the Appendix (spec, Fig 25; G7, G8). Spec G8 calls it "Figure 25"; after renumbering it is Figure 22. |
| **A.1** | A.1 | `certs_png/cert_research_integrity.png` | Appendix | Unchanged. |
| **A.2** | A.2 | `certs_png/cert_respect_at_deakin.png` | Appendix | Unchanged. |
| **A.3** | A.3 | `SSC900 Academic Writing Result.pdf` | Appendix | Unchanged. |

Global figure rules, to be applied to every regenerated file without exception: **no orange text
anywhere** (D-G1); **no prose, verdict or interpretation inside the artwork** (D-G2); **Lift in no
ablation figure, and no sentence saying it was excluded** (D-G3).

### Tables

| New | Old | Table | Where it sits now | Change required |
|---|---|---|---|---|
| **1** | 2 | The query gates of the DAgger family | 2.3 | Unchanged. Trim the caption; it is currently five lines. |
| **2** | 3 | Selection methods, by what they choose and where the data comes from | 2.9 | Unchanged. |
| **3** | 1 | The three aims and the level at which each raises the value of a demonstration | 3.1 | Renamed. "One quantity, three levels" is wordplay and reads as machine-written (F1). |
| **4** | 4 | The six-dimensional geometric descriptor, per task | 4.1.3 | Unchanged. |
| **5** | 10 | The matched-information ablation (Aim 2) | 4.2.4 | Unchanged. |
| **6** | 12 | The lineage of each component across the three aims | 4.3.3 | **KEEP** (C5). |
| **7** | 5 | Final held-out success rate, ten settings | 5.1.3 | **REBUILD** (F2). Every cell on one line: mean and ± and standard deviation must not break across three lines. Transpose, or abbreviate the headers (Safe / Dropout / Ensemble / Thrifty / Diff / Stagger / DISEIL), or drop to a smaller font. Verify in the built PDF, not in the Markdown (F5). |
| **8** | 6 | Per-demonstration information gain, ten settings | 5.1.4 | **REBUILD** (F3). The caption currently overflows and cells overflow in many places. Same layout treatment as Table 7. |
| **9** | 7 | Memory-constant sweep (now study A12) | 5.1.7.4 | Recompute over the three ablation settings (E1). If the Holm-corrected p-values cannot be recomputed at that scope, delete the p-value column rather than invent one. |
| **10** | — | **NEW: cluster purity against geometric separation** | 5.1.7.4 | Replaces old Figure 20 (spec, Fig 20). Three rows, one per ablation setting; columns: mean cluster purity, mean distinct root causes per cluster, mean silhouette. Values from the workbook. |
| **11** | 8 | Per-round wall-clock and token cost (now study A19) | 5.1.7.5 | **REBUILD** (F4): the current table is cluttered and hard to read. Use the SafeDAgger numbers now on disk; a Diff-DAgger version is being computed separately and will be swapped in later — **do not wait for it** (F4, I). Remove the SLURM job identifiers from the surrounding prose (H2). |
| **12** | 14 | **Updated project plan table** | 6.2 | Renamed from "Milestones" (G2). Add the HDR training rows with completion dates (G3). Bold the important tasks (G4). Add a *Venue* column, which is the only place a target venue may appear (C4). |
| **13** | — | **NEW: thesis plan** | 6.3 | G5. Modelled on Table 19 of the Vignesh sample report. Placed immediately after Table 12. |
| — | ~~9~~ | Aim-1 to Aim-2 subsumption | — | **DELETED** (page budget, C1; superseded by Table 6). |
| — | ~~11~~ | The four literatures Aim 3 draws on | — | **DELETED** (page budget, C2). |
| — | ~~13~~ | The Aim-3 metrics | — | **DELETED** (page budget, C2). |
| — | ~~15~~ | The three compulsory HDR modules | — | **DELETED** (G8; the same items become rows of Table 12). |
| — | ~~B.1–B.4, C.1–C.4~~ | Appendix tables | — | **DELETED** (G9). C.1's content survives as the new Table 10. |

No table may overflow its page width, and this must be verified in the built PDF and not in the
Markdown (F5).

---

## 6. Standing constraints for every rewriter

- Method is **DISEIL**. The strings DISTIL, PACE and P4 appear nowhere.
- The acronym is bolded once, at first mention in the Abstract: **D**emonstration d**I**stillation for
  **S**ample-**E**fficient **I**mitation **L**earning. Not in the title, not anywhere else.
- **A2I2** appears nowhere. Write *Deakin Applied Artificial Intelligence Initiative* in full.
- Ablations use three settings: GridWorld (image), Push-T (state), Door (image). Lift appears in no
  ablation, and no sentence explains why.
- Clustering is geometric in every run. There is no R3M or PCA clustering branch, not even a retired
  one. R3M is the image-modality *policy's* visual encoder and nothing else.
- B and D are framework symbols. B = 20 and D = 1 are the validated instance and appear only in the
  experimental setup (5.1.2).
- Every number is checked against `ablations_results/DISTIL_ablation_results.xlsx` or
  `../context/results_data.md`. If a number cannot be found there, it does not go in the report.
- Style: `Non-AI content.md`. Formal academic tone. Minimal em dashes. No AI vocabulary. No wordplay in
  headings or captions. No rule-of-three padding. Numeric citation style is correct and does not change
  (A8).
- Font is Liberation Serif, metrically identical to Times New Roman, and the Greek glyphs (σ, λ, γ, χ)
  must survive in the built PDF (A6).
- Edit with targeted `Edit` calls against text that has been read. Do not overwrite a file blind.
