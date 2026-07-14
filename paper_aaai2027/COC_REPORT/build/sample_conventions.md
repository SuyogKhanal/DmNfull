# Conventions extracted from the two Deakin sample Confirmation-of-Candidature reports

Purpose: structure and conventions only, to guide the DISEIL CoC revision. No sentence is
reproduced from either sample. Numbers below refer to the samples' own numbering.

## 0. Which report is which

| Label | File | Author | Date | Length | Notes |
|---|---|---|---|---|---|
| **S-V** | `other_students_coc_sample_ref/ConfirmationofCandidatureReport.pdf` | **Vignesh Senthilnathan** (s223617105) | July 2026 | 86 pp (letter) | This is the **Vignesh report**. It contains the thesis-plan table the supervisor calls **Table 19**, and the project-plan table (its Table 18). It is the model for our revision. |
| **S-D** | `other_students_coc_sample_ref/COC_Report_Hung_Du-151024.pdf` | **Vinh Hung Du** (224119729) | 31 Oct 2024 | 112 pp (A4) | The second sample. Older, chapter-based, thesis-style. It has **no** project-plan table and **no** thesis-plan table. Useful only as a contrast. |

The supervisor's "Table 19" is unambiguously **S-V**.

---

## 1. The THESIS PLAN table (S-V, Table 19) — the kind of table we must reproduce

**Column list, exactly as printed:**

```
Chapter | 2026 | 2027 | 2028
```

- **Row granularity: one row per THESIS CHAPTER**, numbered `1.`–`8.`, in the order the
  thesis will be bound. Eight rows. The list includes the non-research chapters:
  `7. References` and `8. Appendices (...)` are rows in the table, as is
  `6. Discussion and Conclusion`.
- Each aim chapter's row carries the chapter's full working title followed by the aim tag in
  parentheses, e.g. `... (Aim 1)`, `... (Aim 2)`, `... (Aim 3)`. Long titles simply wrap onto a
  second line inside the cell.
- **Cell vocabulary is a tiny closed set:** `draft`, `update`, `final`, `draft, final`, or blank.
  Nothing else. A chapter written and finished in the same year gets `draft, final` in that one
  year column. A blank means no work on that chapter that year.
- Rules: booktabs (top rule, rule under the header, bottom rule). No vertical rules. Header row
  bold. Year columns centred; the Chapter column left-aligned. The table is narrower than the
  text block, centred.
- **Caption is explanatory, two clauses, placed ABOVE the table**: it states that each chapter is
  marked by the year its draft is written and the year it is finalised, and notes which chapters
  already exist in draft. Roughly two lines. (Contrast the project-plan caption, which is a bare
  title.)
- **Placement: immediately after the project-plan table**, and immediately followed by a single
  sentence pointing at the Gantt chart. Nothing else on the page.

**How it differs from the project-plan / milestone table** (this distinction is the whole point):

| | Project-plan table (S-V Table 18) | Thesis-plan table (S-V Table 19) |
|---|---|---|
| Unit of a row | a *task / milestone / deliverable* | a *thesis chapter* |
| Time | a **date or date range** per row (month, month-range, or quarter) | **year columns**; a matrix, no dates |
| Ordering | chronological, grouped by candidature year | thesis binding order |
| Content | work items, training items, submissions, reviews, **target venues** | writing state only (`draft`/`update`/`final`) |
| Answers | "what am I doing and when" | "what will the thesis contain and when is each chapter written" |
| Caption | title only | two-clause explanatory caption |

The two tables are **not** substitutes; S-V prints both, back to back, then the Gantt.

---

## 2. The project-plan / milestone table (S-V, Table 18)

**Column list, exactly as printed:**

```
Important Milestones | Completion / Target
```

- Two columns only. Header row bold, booktabs rules, no vertical rules.
- **Rows are grouped by candidature year using an italic full-width group row** that carries
  the year, the date span and the theme, e.g. a row reading *Year 1 (October 2025 to July 2026),
  completed*, then *Year 2 (August 2026 to October 2027): Aim 2*, then *Year 3 (November 2027 to
  October 2028): Aim 3 and thesis*. The group row has no date in the right column and no rule
  under it, only a small vertical space.
- **What is bolded (G4 in our spec): the formal candidature gates, bold in BOTH cells.** In S-V
  exactly four rows are bold: *Confirmation of Candidature*, *Mid-candidature review (end of year
  two)*, *Pre-submission review (end of year three)*, *Thesis submission*. Every other row,
  including the paper submissions, is roman. The italic year group-rows are italic, not bold.
- **Training items are ordinary rows in the same table**, not a separate list, e.g. a
  "Research integrity and induction training" row with a month-year in the right column. This is
  the pattern our G3 requires (integrity training, induction training, reproducibility and
  integrity audits, academic-writing course, each with its completion date).
- **Target venues appear ONLY here**, embedded in the milestone text of the submission rows
  (e.g. a row of the form "Aim *n* (Paper *n*) to be submitted to <VENUE>"), with the date or
  quarter in the right column. This is exactly the convention our C4 mandates.
- Right-column values are heterogeneous by design: `October 2025`, `Jul–Aug 2026`,
  `Nov 2026–Jan 2027`, `Q3 2027`. Long milestone text wraps to a second line; the date stays on
  one line.
- **Caption is TITLE ONLY, above the table**: the words "Updated Project Plan" and nothing more.
  Our G2 rename ("Milestones" → "Updated project plan table") matches this.
- The narrative before the table is **three short paragraphs, one per candidature year**, and ends
  by pointing at the table and the Gantt. Not a page of prose.
- The Gantt (S-V Figure 17) is a plain bar chart with quarter columns spanning the candidature,
  row labels only, no annotations, and a **title-only caption**. This matches our G6/D-row-24.

---

## 3. Page budget for the later aims — supervisor's claim CONFIRMED

S-V, section 4 "Aims and Approaches", printed pages:

| Aim | Pages | Figures | Tables |
|---|---|---|---|
| 4.1 Aim 1 | 17–20 (~3 pp) | **1** (the action-space cube) | 1 |
| 4.2 Aim 2 | 20–24 (**4 pp**) | **exactly 1** (a block diagram of the proposed model) | 0 |
| 4.3 Aim 3 | 24–27 (**3 pp**) | **0** | 0 |

So: **Aim 2 = one figure, four pages. Aim 3 = three pages, no figure.** Confirmed.

**How the compression is achieved** (these are the devices we must copy):

1. **Only two subsections per aim**: `4.n.1 Problem Statement.` and `4.n.2 Methodology.` Nothing
   else. No "Risks and mitigations", no "Expected outcomes", no "Contributions", no venue
   paragraph, no separate related-work.
2. **Bold run-in lead-ins instead of further subsections.** Each paragraph opens with a short bold
   sentence-fragment that acts as its heading and is followed on the same line by the paragraph.
   Aim 2's problem statement is four such paragraphs, each naming one deficiency of Aim 1; the
   methodology is four or five such paragraphs (proposed approach, training, analysis,
   evaluation, and a final one-paragraph note on the paper). This buys heading-like navigation at
   zero vertical cost.
3. **The problem statement is derived from the previous aim's measurements, by reference.** It
   cites Aim 1's existing tables and figures by number rather than restating any numbers, and each
   observation is a limitation that motivates exactly one component of the next aim.
4. **Research questions are one paragraph, not a section.** A single bold run-in "Research
   questions." paragraph decomposes RQ*n* into RQ*n*a / RQ*n*b / RQ*n*c inline, in prose.
5. **No results, no ablations, no implementation detail in Aim 2 and Aim 3** — all of that lives
   in the Progress Report for the completed aim only.
6. **The single Aim-2 figure is a block diagram of the proposed system**, not a results plot. It
   carries the architecture that prose would otherwise take a page to describe, and its caption is
   one short clause.
7. **Evaluation is one paragraph**, stated as "the same grid/protocol as Aim 1, unchanged", which
   removes the need to re-specify baselines, metrics or environments.

For contrast, S-D gives its future aims almost no space at all: its plan chapter is four short
subsections (completed work, potential publications, target venues, timeline) and ~5 pages total,
with the aims carried by a bulleted list of six candidate paper titles. It is a weaker model.

---

## 4. Front matter and end matter

**S-V (the model):**

- **Front-matter order:** Title page → **Contents** → List of Figures → List of Tables → body.
- Title page carries, centred and in this order: thesis title, author name, student ID,
  "Supervisor(s):" with the panel listed, the institute name written **in full**, "Deakin
  University", and the month/year. No "planned thesis submission" line. (Our A2/A3 follow this.)
- **S-V has NO abstract and no executive summary at all.** This is the "sample report uses a
  different name" the supervisor refers to in A5: we still use **Abstract**, placed after the
  Contents. S-D *does* use a heading named "Abstract" — a single unnumbered paragraph of about one
  page, placed *before* the Contents. Our spec (A4) puts it *after* the Contents.
- Running header on every body page: the report type in small caps, right-aligned, above a rule.
- **Body order:** 1 Introduction · 2 Background and Literature Study · 3 Gaps and Research
  Questions · 4 Aims and Approaches · 5 Progress Report · 6 Ethical Considerations ·
  7 Publication Plan · 8 Trainings and Other Research Activities · 9 Project Plan. This is
  precisely the skeleton our B11/B12/B13/G2 impose.
- The **conclusion is not a top-level section**: it is `5.2.6 Conclusion`, inside the Progress
  Report, under the completed aim, and it runs to **one page (two paragraphs)**. This confirms our
  B15 exactly.
- **End matter, in order: Appendix A (Training Certificates) → References. References are last.**
  Appendix A is a four-line preamble followed by the certificates as `A.1`, `A.2`, `A.3`, one
  scanned certificate per page. Nothing follows the references. This confirms G7/G8/G9.
- The "Trainings and Other Research Activities" section is **half a page**: one short paragraph
  saying the compulsory modules are complete and pointing at Appendix A, then a three-item bullet
  list of the modules with completion dates.
- Flat, document-wide numbering: Figure 1–17, Table 1–19. (S-D uses chapter-scoped numbering,
  Figure 1.1 / Table A.1; we follow S-V's flat scheme, which our current draft already uses.)

**S-D (contrast):** Title page → Abstract → Contents → Ch. 1–3 → Appendices A/B/C (which are
whole papers, reproduced) → References last. Appendices before references holds in both samples.

---

## 5. Heading register and caption length

**Heading register — S-V is mixed, and deliberately so:**

- Top-level and background headings are **plain noun phrases**: "Introduction", "Gaps and Research
  Questions", "Aims and Approaches", "Progress Report", "Ethical Considerations", "Publication
  Plan", "Project Plan". Literature subsections are plain topic labels ("Dimensional Models of
  Emotion", "Open Problems"). No verbs, no questions, no wordplay, no colons-with-a-slogan.
- The **only descriptive headings are the aim headings themselves**, of the form
  `4.n Aim n: <the working paper title>`. The aim heading doubles as the paper title. Everything
  below it is plain: "Problem Statement", "Methodology".
- Progress-report subsections are plain and functional: "Problem Statement", "Proposed Method",
  "Baselines and Comparators", "Experimental Results", "Reproducibility and Integrity Audit",
  "Conclusion".
- Consequence for us: rename 2.4–2.11 to bare noun phrases (our B5–B10 already say this), keep
  "Aims and approaches" plain, and let the aim heading carry the descriptive title.

**Caption length — two registers, used consistently:**

- **Plan/administrative artefacts get a bare title caption**: "Updated Project Plan", "Gantt chart
  of the candidature (dates)". One line, no interpretation. This is what our D-row-24 requires.
- **Results figures get a caption of one to four lines** that states what is plotted and, at most,
  the single reading of it. The longest captions in S-V are ~4 lines and belong to the two case
  studies. Method figures get a single short clause ("Block diagram of the proposed model.").
- **Nothing interpretive is drawn inside the artwork.** Verdicts and readings live in the caption
  or the body paragraph, never as text baked into the figure. S-V's figures carry axis labels, a
  legend and data labels only. This is our D-G2 and it is followed strictly in the sample.
- Table captions sit **above** the table; figure captions **below** the figure.
- Captions are sentences with terminal punctuation, in plain register, with no rhetorical framing
  and no rule-of-three lists.
