# Supervisor revision spec — binding checklist for the CoC report

Source: supervisor review of `CoC_Report.pdf` (150 pp). Every item below is MANDATORY.
Figure/table numbers refer to the CURRENT document; the mapping to source files is given
so there is no ambiguity.

---

## A. Front matter

| # | Item |
|---|---|
| A1 | **"Leveraging" in the thesis title STAYS.** It is the registered thesis title, not paper prose. Do not change it. It is exempt from the non-AI vocabulary rule. |
| A2 | **Remove "Planned thesis submission" from the title page.** |
| A3 | **The string "A2I2" must appear NOWHERE in the document.** Always write it in full: *Deakin Applied Artificial Intelligence Initiative*. Sweep every occurrence, including the cover, headers and captions. |
| A4 | Order: **Title page → Table of contents → Abstract**. |
| A5 | **Rename "Executive summary" to "Abstract".** It is the first section after the contents. Do this even though one of the sample reports uses a different name. |
| A6 | **Font: Times New Roman.** Times New Roman is not installed on this machine; use **Liberation Serif**, which is metrically identical to Times New Roman AND carries the Greek glyphs (σ, λ, γ, χ) this report needs. Verify the glyphs survive in the built PDF. |
| A7 | **Length: bring 150 pp down to ~100 pp.** Trim by deleting the material listed below, not by compressing prose into fragments. |
| A8 | Citation style (numeric) is correct. **Do not change it.** |

## B. Section structure and naming

| # | Item |
|---|---|
| B1 | "1. Introduction and research vision" → **"Introduction"**. |
| B2 | **Delete "Contributions to date"** as a section. Contributions must be evident from the report itself. |
| B3 | **Delete "Scope and constraints"** and **"Report outline"**. Both are unnecessary; a careful reader infers them. |
| B4 | **Trim the Introduction substantially.** It is far too long. |
| B5 | 2.4 "Uncertainty estimation and the limits of a scalar" → **"Uncertainty estimation"**. |
| B6 | 2.5 "Policy classes and why the framework does not depend on them" → **"Policy classes"**. (The independence is evident from the text.) |
| B7 | 2.6 "Standard machinery this work uses and does not claim" → rename to something plain, e.g. **"Standard machinery"**. |
| B8 | 2.7 "What language and vision-language models are and are not reliable at" → rename plainly, e.g. **"Language and vision-language models"**. |
| B9 | 2.8, 2.9, 2.10 headings are fine — leave them. |
| B10 | 2.11 "Open problems and the gap this programme addresses" → **"Open problems"** only. The gap moves to the next section. |
| B11 | **Add a new section: "Gap and research questions."** The gap goes here, and **RQ1, RQ2, RQ3** (currently stated in the Introduction) are brought down into this section. |
| B12 | **Section 4 must be titled "Aims and approaches"** (NOT "Aim 1"). Then: **4.1 Aim 1**, **4.1.1 Motivation and problem statement** (which refers briefly to the research questions), then **Problem formulation**, then **Methodology**, and the **DISEIL framework sits inside Methodology**. Aim 2 and Aim 3 become 4.2 and 4.3. |
| B13 | **Aim 1's methodology must be SHORT** — the overall idea with some detail, not exhaustive detail. All the detailed work, implementation, experiments, results and ablations **move into a PROGRESS REPORT section** ("Progress on M1"). |
| B14 | **Delete section 7 "Coherence of the research programme" entirely** (7.1, 7.2, 7.3 — all unnecessary). |
| B15 | **The Conclusion (current section 11) is deleted as a top-level section.** Its content moves inside the Progress report, under Progress on M1, as **"M1 conclusion"**, and must not exceed **one page**. |

## C. Aim 2 and Aim 3 (over-written — cut hard)

| # | Item |
|---|---|
| C1 | **Aim 2: at most ~4 pages, using ONE figure.** (A sample report does exactly this.) |
| C2 | **Aim 3: at most ~3 pages.** |
| C3 | **Delete all "Risks and mitigations"** paragraphs/subsections from Aim 2 and Aim 3. |
| C4 | **Delete every mention of target venues** from prose. Target venues appear **only** in the project-plan table. Not as a paragraph, not as a sentence, anywhere. |
| C5 | Table 12 (lineage of each component across the three aims) is good — **keep it**; it summarises Aim 3 well. |

## D. Figures — exact changes (figure number → source file)

| Fig | File | Required change |
|---|---|---|
| 1 | `../figures/Teaser_Diagram.pdf` | **Rotate one more step CLOCKWISE.** Currently mis-rotated. |
| 4 | `F14_aggregate_significance.pdf` | **Remove the paragraph text baked into the image** ("10 of 10 settings favour DISEIL", "setting-level, n = 10", "Lift is at a 100% ± 0 ceiling, uninformative about any mechanism"). No prose inside the figure. |
| 8 | `F1_allocation_ladder.pdf` | **Remove the "A3 falls below the best baseline" annotation and its arrow** (Door image). |
| 9 | `F2_gain_without_allocation.pdf` | **Redraw as a grouped BAR CHART in the same style as Figure 8** (that style tells the story better). |
| 10 | `F5_grounding_and_feasibility.pdf` | **Redraw as a bar chart in the Figure-8 style.** |
| 11 | `F4_reasoning_and_vision_small.pdf` | **Redraw as a bar chart in the Figure-8 style.** |
| 12 | `F6_bridging.pdf` | **Delete the LEFT panel.** Keep only the right panel, as a **single-panel figure**, showing only the three ablation settings. |
| 13 | `F3_knockout_summary.pdf` | Restrict to the **three ablation settings only** (see E1). No Lift, no ten-setting sweep. |
| 14 | `F7_descriptor_dimensionality.pdf` | **Remove the "argmax in 10/10 settings" and "chosen descriptor 6-D" annotations** — the dashed line at 6 already makes the point. **Remove Lift (the grey lines).** |
| 15 | `F11_context_and_selection.pdf` | Restrict to the three ablation settings; remove annotation prose. |
| 16 | `F12_cluster_count_distribution.pdf` | **Show only the settings we ablate** (the three). |
| 17 | `F8_budget_sweep.pdf` | **Delete the LEFT panel.** The right panel is good but is currently Push-T (state) only → make it **three panels: Push-T (state), GridWorld (image), Door (image)**. **Move the in-figure sentence** ("the margin shrinks because the baseline catches up, not because DISEIL degrades") **out of the figure and into the body paragraph.** |
| 18 | `F9_demos_per_round.pdf` | **DELETE the figure entirely** — it is A12, which is removed (see E2). |
| 19 | `F10_memory_constants.pdf` | Keep (memory-constant sweep), but apply E1/E3/E4. |
| 20 | `F15_cluster_purity.pdf` | **DELETE the figure. Replace it with a TABLE.** |
| 21 | `F13_failures_over_budget.pdf` | **Remove the descriptive text boxes inside the figure.** |
| 22 | `F16_compute_cost.pdf` | **DELETE the figure entirely** — unnecessary. |
| 24 | `gantt_chart.pdf` | **Remove the "examination preparation" bar.** Caption = **title only**; the chart speaks for itself. |
| 25 | `Compulsory Training Status.png` | Keep. Moves to the Appendix (see G). |

### Global figure rules
| # | Item |
|---|---|
| D-G1 | **Remove ALL orange text from EVERY figure.** No exceptions. |
| D-G2 | **No prose, verdicts or interpretation inside any figure.** Explanations belong in the body text or the caption, never baked into the artwork. |
| D-G3 | **Lift must not appear in any ablation figure.** Do not add a sentence saying "Lift is excluded" either — its absence is self-explanatory given its 100% success rate. Just omit it. |

## E. Ablation scope and numbering

| # | Item |
|---|---|
| E1 | **Ablations use ONLY three settings: GridWorld (image), Push-T (state), Door (image).** This was already agreed. Enforce it everywhere — text, tables and figures. |
| E2 | **A12 is REMOVED entirely** (demonstrations-per-round). The workbook itself says "Hold on A12 for now". Delete the study, its figure and its discussion, and **renumber the remaining A's**. |
| E3 | **The D-series are extensions of the A-series, not a separate series.** They must be renumbered as A's, continuing the sequence. |
| E4 | **Canonical renumbering** (after removing A12): A1–A11 unchanged; **A13→A12, A14→A13, A15→A14**; then **D1→A15, D2→A16, D3→A17, D4→A18, D5→A19**. Apply this map consistently in all text, captions, tables and cross-references. Any place that currently reads e.g. "A14 and D2" must be updated to the new numbers. |

## F. Tables

| # | Item |
|---|---|
| F1 | **Table 1 "One quantity, three levels"** — this is wordplay and reads as machine-written. **Rename or remove.** |
| F2 | **Table 5 (the Aim-1 results table) is BROKEN.** Text overlaps; the mean lands on one line, the "±" on the next, the standard deviation on a third. **Rebuild it so every cell renders on one line.** Use a layout that fits the page (fewer columns, abbreviated headers, smaller font, or transpose). |
| F3 | **Table 6 is CORRUPT** — the caption overflows and cells overflow in many places. **Rebuild it.** |
| F4 | **Table 8 is cluttered and hard to read.** Rebuild it cleanly. (Note: a Diff-DAgger version of this table is being computed separately — do NOT wait for it; use the current SafeDAgger numbers and it will be swapped in later.) |
| F5 | **No table may overflow its page width.** Verify in the built PDF, not in the Markdown. |

## G. Project plan, Gantt, appendix, end matter

| # | Item |
|---|---|
| G1 | The paragraphs before 8.2 describing where the Aim 1/2/3 papers are targeted are **redundant with 8.2** — delete them. |
| G2 | **8.3 "Milestones" → rename "Updated project plan table".** |
| G3 | That table must **also include the HDR training items**: research integrity training, induction training, reproducibility and integrity audits, and the academic-writing course, each with its completion date. |
| G4 | **Important tasks must be bold** in the updated project plan table. |
| G5 | **Add a thesis plan table** immediately AFTER the updated project plan table, modelled on the thesis plan (Table 19) in the Vignesh sample report. |
| G6 | Gantt: **remove "examination preparation"**. |
| G7 | **Training certificates move to the APPENDIX**, placed BEFORE the references. They are no longer a numbered section (currently section 10). |
| G8 | **The HDR training section must be short**: state that the compulsory training is complete, cite Figure 25 as the evidence, and then list the certificates as **A.1, A.2, A.3**. Nothing more. Delete the long prose currently in section 10. |
| G9 | **References are the LAST thing in the document. Full stop.** Delete everything currently placed after the references (the extra evidence/study-results pages) — that material is already covered in the body. |

## H. Body-text fixes (by current page)

| # | Item |
|---|---|
| H1 | **p. 80** — delete the paragraph immediately above Figure 20 (the one running from "one cross-check passes cleanly …" to "… is reported as an internal consistency check"). |
| H2 | **4.14.5 / Measurement** — **remove SLURM job IDs and all infrastructure detail.** This is a research report, not an engineering report. The research paper will come out of it. |
| H3 | **p. 86** — **remove the single-seed caveats** ("every run comes from one seed, seed 1", etc.). Do not mention them. |
| H4 | **Reframe the evaluation narrative.** Do not write it as "what the evaluation says about the framework". Write it as the story we actually ran: **we evaluated, and then we kept adding components** — an incremental build-up. (This does NOT mean putting evaluation before the method; the method still comes first.) |
| H5 | **The algorithm (currently 4.4.6) must look like a real algorithm** — a proper numbered algorithm block, not prose. Per the supervisor's earlier paper feedback: it must be **short**, must **not cram several actions into one step**, and must **segregate the steps** into atomic lines. |

## I. Out of scope for this pass
Two SLURM studies are running (Table 8 with Diff-DAgger as the baseline; the σ calibration).
**Do not wait for them.** Use the numbers currently on disk. Their results will be slotted into
the relevant sections afterwards.
