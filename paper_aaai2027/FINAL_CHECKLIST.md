# PACE — Final Assembly Checklist

Target: AAAI 2027 two-column, **~7-page body** (refs/appendix excluded from the 7).
Source: `paper_aaai2027/draft/paper.tex`. No technical content was changed in this pass.

---

## 1. Page-count estimate vs the ~7-page body target

No LaTeX toolchain is installed on this node, so this is a **word/float estimate**, not a
compiled count. Compile before submission to confirm.

**Body word counts** (excludes `%` comments; abstract and appendix broken out separately):

| Section | Words | Lines |
|---|---|---|
| Abstract | 466 | 50–97 |
| Introduction | 1,927 | 99–334 |
| Related Work | 1,043 | 334–468 |
| Method | 3,258 | 468–920 |
| Experiments | 4,032 | 920–1411 |
| Conclusion (+Limitations+Future Work) | 967 | 1411–1519 |
| Appendix | 273 | 1519–1586 |
| **Body total (Intro→Conclusion)** | **~11,227** | |

**Floats/math in the body:** 4 tables (tasks, main, cost, ablation — the main and
ablation tables are large), 2 figures, ~30 numbered equations, 1 full-page algorithm.

**Estimate.** AAAI two-column runs roughly 850–950 words of prose per page. Text alone is
~11.2k words ≈ **12 pages of prose**, and the four tables + algorithm + two figures + ~30
displayed equations add on the order of **2–3 more pages** of float space.
**Projected body: ~13–15 pages — roughly DOUBLE the 7-page target. This is the single
biggest issue in the paper and must be fixed before submission.**

The prose is also unusually dense with hedging/meta-commentary ("we claim nothing
stronger", "as an organizing device, not a contribution", "we say so plainly", repeated
scope disclaimers). Much of the overage is recoverable by cutting repetition, not results.

### Suggested per-section trims (to reach ~7 pages)

- **Method (3,258 w → target ~1,800).** Biggest single lever.
  - The "unifying view: baselines as a corner of PACE" paragraph (lines 854–891) restates
    material already in the Intro, the Related Work opener, and the Contributions block. It
    appears in full **four** times across the paper. Keep one full statement (here) and
    reduce the others to one sentence + a cross-reference.
  - "What actually does the work in Choose…" (lines 775–799) and the Execute
    default/option discussion (lines 801–852) repeat the which-vs-where framing several
    times. Compress each to one paragraph.
  - The T1-grid instantiation aside inside Perceive (lines 669–689) is long; it can move to
    an appendix or a compact footnote.
- **Experiments (4,032 w → target ~2,200).** Second biggest lever.
  - Metrics/Seeds (lines 1098–1158) is very long for a protocol whose numbers are all
    placeholders. The censoring discussion and multiplicity discussion can each drop to a
    few sentences.
  - Cost Accounting (1225–1250) and Ablations (1319–1373) narrate the same
    free-resets-vs-prescribed-where separation twice; state it once.
  - The per-cell / paired-sign-test rationale is explained in Metrics, Main Results, and
    Ablations — consolidate to Metrics with one back-reference.
- **Introduction (1,927 w → target ~1,100).** The "unifying observation" and the
  two-single-axis-controls justification are each stated twice (paragraphs "PACE." and
  "Contributions."). Cut the duplication.
- **Related Work (1,043 w)** and **Conclusion (967 w)** are close to reasonable; light
  trimming only. The Conclusion re-derives the corner/monotonicity argument yet again
  (lines 1428–1436) — shorten to a pointer.
- **Abstract (466 w)** is long for AAAI (typical ~150–250). Recommend cutting to ~200:
  the four-stage walkthrough and the unifying-view caveats can be compressed.

Because the paper currently ships as method + framework + **pre-registered protocol** (all
result cells are placeholders), aggressive prose compression will NOT weaken any empirical
claim — there are none yet to protect.

---

## 2. Abstract vs. paper consistency

The abstract is **consistent** with the body. Spot-checks that pass:

- PACE expansion "Perceive→Assess→Choose→Execute" — matches title footnote, Intro, Method.
- "which is the central axis; where is an exposed/ablated option, not a load-bearing
  pillar" — matches Intro, Method Execute, Contributions, Conclusion.
- Two implemented tasks (5×5 grid + ManiSkill Push), state+image, 5 seeds — matches
  Table 1, Task Suite, Metrics.
- Three RoboSuite tasks (Lift/Wipe/Door) are "planned, no result rows" — matches Table 1
  status column and the Task Suite T3–T5 paragraph.
- Baselines: query-efficient DAgger family + two single-axis controls (reset-curriculum
  *where*-arm, IWR-style *which*-arm) — matches Baselines section.
- Unifying view "scoped to the diffusion-loss task", "K=1", "monotonicity condition",
  "claim no formal reduction" — matches Method and Conclusion.
- Explicit statement that "all reported numbers are placeholders" — matches Experiments
  and Conclusion.
- The two headline placeholders `\PH{pace-vs-diff-q-reduction}` and `\PH{mean-pace-sr}`
  used in the abstract are the same macros used in Main Results / Conclusion — consistent.

**Minor wording notes (optional, not required):**
- Abstract says "geometric (optionally visual) descriptor"; body uses geometric as the
  **default** and visual (frozen R3M) as the option — consistent, just confirm the
  parenthetical reads as "optional," which it does.
- Abstract is long (466 w); see §1 for the recommendation to trim to ~200.

No factual contradiction found between abstract and body.

---

## 3. Every remaining `\PH{}` placeholder (author must fill)

**138 unique placeholder macros, 144 total occurrences.** Grep to find them:
`grep -n '\\PH{' paper_aaai2027/draft/paper.tex`. Grouped by what result is needed:

### A. Headline numbers (appear in Abstract, Contributions, Main Results, Conclusion)
- `pace-vs-diff-q-reduction` (×4), `mean-pace-sr` (×3)

### B. Protocol / config constants (fill from the actual run configs — these are NOT run
results and can be filled *now* from the code, before any seeds finish)
- Toy: `toy-init-demos`, `toy-pool-n`, `toy-heldout-n` (×2), `toy-budget`,
  `toy-max-rounds`, `toy-max-steps`
- Robot: `robot-init-demos`, `robot-rollout-n`, `robot-nd`, `robot-min-expert-steps`,
  `robot-heldout-n`, `robot-heldout-seed`, `robot-eval-envs`, `robot-budget`,
  `robot-max-rounds`
- Stray generic: `placeholder` (line ~952, in the Experiments intro "every quantitative
  entry is a \PH{placeholder} macro" — this is rhetorical; consider replacing the macro
  with plain text so it does not read as an unfilled cell).

### C. Main results table (Table 2) — SR and Q per method × modality
- T1 Toy, State: `toy-st-{safe,dropout,ensemble,thrifty,rand,rcur,iwr,pace}-{sr,q}`
- T1 Toy, Image: `toy-im-{safe,dropout,ensemble,thrifty,rand,rcur,iwr,pace}-{sr,q}`
- T2 Push, State: `push-st-{diff,safe,dropout,ensemble,thrifty,rand,rcur,iwr,pace}-{sr,q}`
- T2 Push, Image: `push-im-{diff,safe,dropout,ensemble,thrifty,rand,rcur,iwr,pace}-{sr,q}`

### D. Learning-curve / coverage callouts (in-text)
- `push-st-pace-eff`, `push-st-diff-eff`, `push-im-pace-eff`, `push-im-diff-eff`
- `push-st-pace-cov`, `push-st-diff-cov`, `toy-st-pace-cov`, `toy-st-rand-cov`

### E. Cost-accounting table (Table 3)
- `push-diff-q`, `push-base-q`, `push-pace-q`, `pace-resets`, `pace-vlm-calls`,
  `pace-llm-calls`

### F. Ablation table (Table 4) + LLM think/no-think callout — Eff and Cov per variant
- T2 Push (image): `abl-{perceive,vlm,llm,assess,memory,choose,execute,presreset,full}-{eff,cov}`
- T1 Toy (state): `abl-toy-{perceive,vlm,llm,assess,memory,choose,execute,presreset,full}-{eff,cov}`
- Think/no-think: `abl-think-eff`, `abl-nothink-eff`

---

## 4. Author to-do list (ordered)

1. **[BLOCKER] Cut the body from ~13–15 pp to ~7 pp.** See §1 per-section trims. Priority:
   Method and Experiments, mostly by removing the repeated "which vs where" and
   "corner/unifying view" restatements (the corner argument is stated ~4×). No results are
   at risk since all cells are placeholders.
2. **[NOW] Fill the ~18 protocol/config placeholders in group B** — these come from the
   run configs, not from seed results, so they can be filled immediately and will clean up
   the Task-Suite and Active-Loop-Protocol prose.
3. **Replace the rhetorical `\PH{placeholder}` on line ~952** with plain prose so it does
   not look like a forgotten cell.
4. **Replace the two figure stubs.** `figures/lc_push_image.pdf` and `figures/qual_push.pdf`
   are 631-byte placeholder files, not real plots. Generate the real learning-curve and
   qualitative panels once runs exist.
5. **[AFTER RUNS] Fill result groups A, C, D, E, F** (120 result placeholders) from the
   five-seed T1/T2 outputs. Follow the pre-registered reporting rules already in the paper
   (Reach → median-Q among reachers; bold only among highest-Reach methods; paired
   per-seed sign test for directional claims).
6. **Shorten the abstract to ~200 words** (currently 466).
7. **Compile with `pdflatex`/`bibtex`** (not available on this node) to get the true page
   count and catch any float-overflow / undefined-reference warnings. Re-check the page
   count after trimming.
8. **Sanity checks that already pass — no action needed:** all 52 `\cite` keys resolve
   against `references.bib` (no missing references); abstract is factually consistent with
   the body; equation/table/figure/algorithm labels are referenced consistently.

---

## 5. Verified-clean items (for the record)

- Citations: 52 used, 52 defined, **0 missing** (`\cite` keys all resolve in
  `references.bib`).
- Abstract ↔ body: no factual contradiction found (§2).
- Cross-references to Eqs./Tables/Figures/Algorithm are internally consistent on read.
- Placeholder accounting: 138 unique `\PH{}` macros, 144 occurrences, all catalogued in §3.
