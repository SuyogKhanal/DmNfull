# CoC Report — Final Validation Report

Target: `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/CoC_Report.md`
Validated: 2026-07-13. Verdict: **PASS** (2 trivial failures found and fixed in-place; 1 item referred to the author).

---

## 1. Document statistics

| Quantity | Value |
|---|---|
| Words | 62,053 |
| Lines | ~2,420 |
| Figures | 25 (captions 1–25, all paired to an embedded image) |
| Tables | 15 (captions 1–15) |
| Equations | 14 tagged (1–7, 8a/8b, 9–14) |
| References | 102 (numbered 1–102, contiguous) |
| Image assets embedded | 29 = 25 figures + A2I2 logo + 3 appendix certificates |
| Em dashes | 23 — **all 23 are table "not-applicable" cell markers. Prose em-dash count is 0.** |
| DISEIL occurrences | 102 |
| DISTIL / PACE / P4 as method names | **0** |

---

## 2. Author's mandatory checklist

### 1. DISEIL used everywhere — **PASS**
102 occurrences. Acronym derived on first use (L59, L544): "*Demonstration Distillation for Sample-Efficient Imitation Learning*".

### 2. DISTIL / PACE / P4 removed entirely — **PASS. Exact hit count: 0.**
```
grep -cEi '\bDISTIL\b|\bPACE\b|\bP4\b' CoC_Report.md  ->  0
```
A case-insensitive `distil` grep returns 14 hits; **all 14 are the English word "distillation"** (e.g. "Dataset Distillation" [13], "demonstration distillation", the paper title). These are required by the acronym derivation and are correct. No all-caps `DISTIL` token exists anywhere (`grep -onE 'DISTIL[A-Za-z]*'` returns nothing).

### 3. Updated teaser figure — **PASS**
L228: `![...](../figures/Teaser_Diagram.pdf)`. File exists (272 KB, 13 Jul 15:28). The stale `Teaser Diagram.png` is deleted from the working tree and is not referenced.

### 4. Updated architecture (policy-solvability loop) — **PASS**
L747: `![The DISEIL framework.](<../figures/Architectural Diagram.pdf>)`. Text extracted from the PDF confirms the correct loop:
```
Policy Rollout on 𝑃 ... Solvable ⇒ Revise 𝑃 ... prescribe the demo configurations 𝑃
```
This is the policy-solvability loop, not an old infeasibility loop. L765 describes the arrow, and honestly notes that the drawing depicts the solvability loop while *not* drawing a return arrow for the Eq. 10 feasibility loop — a real property of the figure, disclosed rather than papered over.

### 5. Learning curves, all five tasks — **PASS**
L1019: `![...](../figures/all_5_task_comparison.pdf)` (38 KB, regenerated 13 Jul 18:48). Figure 5 caption names all five panels: GridWorld (image), Push-T (state), Lift (state), Door (state), Wipe (image).

### 6. Information-gain discussion updated — **PASS** (all four sub-parts present)
- **Pre-retrain loss argument** — §4.10.2 (L1057–1063). States the two readings of a high pre-retrain loss and rules out the "bad datum" reading by two independent constructions (feasibility check + expert-optimal demonstrations).
- **Starting performance** — §4.10.2 (L1065) and §4.8.4.
- **Initial demonstrations** — §4.8.4 (L951–959).
- **Why the count was chosen to place starting SR in a target band** — §4.8.4 (L955–957): too-weak ⇒ no structure to partition; too-strong ⇒ nothing to allocate. Implemented as a BC data-scaling sweep selecting the prefix closest to **~50 % round-zero success**, yielding 8 (Lift), 12 (Wipe), 4 (Door), 20 (GridWorld). The report is candid that the *reported* runs used a uniform 20 and states what that costs on Lift.
- §4.10.3 adds the necessary honesty: information gain is **necessary and not sufficient** (knockout leaves gain unchanged, +0.06, p=0.25, while SR collapses 4.01 points, p=0.008).

### 7. Initial-demonstration discussion — **PASS**
§4.8.4 "Initial demonstrations and starting performance" (L951). Also flagged at the architecture block (L753).

### 8. Representative prompts — **PASS**
§4.6 (L769–876). Three verbatim on-disk prompts: perception (VLM), reasoning (strict-JSON, graph-vocabulary-constrained), prescription. Includes a self-reported defect (the template says "peak-loss frame" while the frame passed is the Eq. 5 first-crossing step).

### 9. Representative KAG examples as structured key-value — **PASS**
§4.7 (L877–920). Genuine JSON key-value nodes for Push-T (`ws_tee`, `ws_tcp`, `ctrl`, `goal`), Door (`ws_door`, `succ`), plus the `workspace_constraint`, `non_emptiness` and `select_only` implications for GridWorld and Wipe. Not prose — typed nodes with key-value properties, as required.

### 10. Equation 10 as feasibility verification — **PASS**
L668–683. The full loop is presented exactly as specified: **LLM proposes** (`cmd^(j) = LLM(A, S, K, violation(ξ^(j-1)))`) → **constraints retrieved from the KAG** → **feasibility check** (`V(ξ)` = workspace ∧ reachable ∧ valid-path) → **violation fed back to the LLM** → **revised until feasible** (`ξ* = ξ^(j)` for the first `j ≤ J_max` with `V=1`, else nearest untried failure). Eq. 11 is kept clearly separate as the policy-solvability check, with an explicit statement that it is drawn but **not exercised** and that no number in the report is attributable to it.

### 11. Cluster naming explained — **PASS**
§4.4.5 "Naming the discovered failure modes" (L693–701). Three-step pipeline: modes are born nameless (integer indices from the geometric partition); each failure is assigned a root cause from the graph's *enumerated* vocabulary; the mode's name is the majority root cause. Purity is measured (0.78–0.93, mean 0.877) and the Wipe low is explained.

### 12. Humanised writing — **PASS**
- **Em dashes: 23, of which 0 are in prose.** All 23 are `—` table cells marking "not applicable" (Tables 5, 6, and the memory-constant tables). This is correct typographic usage, not the AI em-dash tell.
- **AI-tell vocabulary: 4 raw hits, 3 unavoidable, 1 referred to the author.**
  - L2051 `robust` — inside the *title* of reference [13-adjacent] "DART: Noise Injection for **Robust** Imitation Learning". Cannot be altered; it is a citation.
  - L2157 `leverage` — inside the *title* of reference [99] "How to **Leverage** Diverse Demonstrations…". Cannot be altered; it is a citation.
  - L1278 `harness` — "ablation **harness**", the standard technical sense (a test harness), not "harness the power of". Legitimate.
  - **L8 `Leveraging` — the thesis title on the cover page. See §4 below: AUTHOR DECISION.**
- No `delve`, `pivotal`, `crucial`, `showcase`, `underscore`, `comprehensive`, `nuanced`, `Moreover`, `Furthermore`, `Additionally`, `seamless`, `meticulous`, `holistic`, `myriad`, `plethora`, `realm`, `tapestry`, `testament`, `pave the way`.

### 13. References verified — **PASS. Count: 102.**
- 102 entries, numbered **1–102 with no gaps**.
- **Cited-but-undefined: 0. Defined-but-never-cited: 0.** Every one of the 102 is used in the text.
- Every entry resolves to a real BibTeX record in `build/references_coc.bib`, which itself holds exactly **102 entries**. Title-matching confirmed 101/102 automatically; the single non-match, **[39] RLBench**, is a false positive of the matcher (the bib escapes the ampersand as `\&`, the report renders it as `&`) — the entry `@misc{james2019rlbench}` exists and is correct.
- **No fabricated references.**

### 14. Cross-references consistent — **PASS**
- Figures: captions 1–25, in-text references 1–25, back-matter "List of figures" 25 entries. All match; every caption sits within 3 lines of its image embed.
- Tables: captions 1–15, in-text references 1–15, back-matter "List of tables" 15 entries. All match.
- Equations: tagged 1, 2, 3, 4, 5, 6, 7, **8a, 8b**, 9, 10, 11, 12, 13, 14 — continuous. (An initial scan for `\tag{8}` appeared to show a gap; Eq. 8 is legitimately split into 8a (memory penalty) and 8b (target selection), and the algorithm's "(Eq. 8)" correctly refers to the pair.)
- **All 29 image paths resolve on disk.** Zero broken asset links.

---

## 3. Additional verification requested

| Item | Verdict | Evidence |
|---|---|---|
| Comparison tables label the DAgger family explicitly | **PASS** | Tables 5 (L985) and 6 (L1035) both carry the bolded sentence "**The DAgger family is the set of five published query-gated methods in the columns SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger and Diff-DAgger.**" Both tables carry a `*DAgger family*` column-group header row, a separate `Control` column for Stagger and an `Ours` column. §4.8.5 (L967) states Stagger "is never labelled as a member of the DAgger family". |
| Ablations confined to 3 primary settings; rest deferred to supplementary | **PASS** | §4.14.1 (L1108): "reported on three primary settings… GridWorld (image), where the policy is a convolutional network; Push-T (state)… Door (image)… The remaining seven settings, and the studies not discussed here, are retained in full and are held for the supplementary material and for rebuttal." Full tables are in Appendix B; supplementary diagnostics in Appendix C. |
| Lift-at-ceiling caveat wherever Lift appears | **PASS (2 gaps fixed — see §5)** | Caveat present at L208, L538, L648, L959, L985, L1001, L1013, L1021, L1065, L1099, L1114. §4.14.1 states it once as a global rule: "Lift is excluded from every aggregate and from every mechanism claim in this section." |
| Clustering geometric for every run; no surviving R3M/PCA branch | **PASS** | Asserted at L448, L592, L634, L939, L1198, L1335, L1414, L1670, L1748. The retired branch is named as retired in every instance ("the frozen-embedding-plus-projection branch… has been retired, and it does not appear anywhere in this report", L448). R3M appears **only** as the image-modality *policy* encoder, with the disclaimer "It does not supply the clustering features, which are geometric in every run" (L939). PCA appears only in the past tense, as the retired branch. **No surviving branch.** |
| B and D as framework parameters; B=20 / D=1 as the validated instance | **PASS** | L925: "The framework itself is stated over symbols: a budget $B$… $D$ demonstrations acquired per round… What follows is the instance of that framework which was actually run." L943: "The validated instance is $B = 20$ and $D = 1$." L947 + L1222: A11 sweeps B over {10,20,40} — "$B$ is a symbol in the method and in the algorithm, whose loop header reads 'for $r = 1$ to $B$', and the value 20 appears only in the experimental setup." A12 justifies D=1 at fixed labour. |
| Gantt present and consistent with November 2028 | **PASS** | Figure 24 (L1823), `figures_generated/gantt_chart.pdf` exists. Spans 13 Nov 2025 → Nov 2028. Milestone table (Table 14) cross-referenced by numbered diamonds. Nov 2028 thesis submission is consistent at L25 (cover), L210, L320, L1730, L1794, L1825, L1833, L1945, L1949. The one schedule risk (CoRL 2028 lands in the submission month) is identified and mitigated at L1833. |
| Cover page: A2I2 logo + every required field | **PASS** | L1 logo `A2I2_Logo_Stacked_2025_Keyline.png`. All fields per `COC_REPORT_INSTRUCTIONS.md` §Cover Page: Deakin University (L5), A2I2 (L6), Confirmation of Candidature Report (L3), Thesis Title (L8), Student Name (L11), Student ID (L12), Supervisors (L15–16), Candidature Start Date (L19), CoC Date (L22). Planned thesis submission (L25) is an extra. |
| D5 compute analysis present | **PASS** | §4.14.5 (L1298–1329), Table 8, Figure 22. **Every number traces to `build/d5_compute.md`** — independently re-verified cell by cell (see §4 below). Limitation §4.16.7 quantified. |
| Aims 1 → 2 → 3 connect as one programme | **PASS** | §7.1 (L1738–1750) traces **one quantity — the value of one demonstration — through three levels**: *measured* in Aim 1 (pre-retrain loss), *contextualised* in Aim 2 (coverage memory), *priced* in Aim 3 (per minute of teacher time). Second thread: component survival (descriptor, constraint store, solvability check all carry forward; only the two ablation-retired components are dropped). Third thread: each aim corrects the limitation the previous aim's own evaluation *measured*. Also §4.18, §5.11, §6.1, §6.6, §6.10, §7.2. |

---

## 4. D5 honesty audit — every number re-derived from source

I re-checked all D5 figures in the report against `build/d5_compute.md` (which traces to SLURM jobs 110355–110385). **Zero discrepancies. Nothing invented.**

| Report claim (§4.14.5) | Source (`d5_compute.md`) | ✓ |
|---|---|---|
| Shared cost 783–1,491 s (P1, RoboSuite) | 783.0 / 1,180.0 / 1,491.0 | ✓ |
| Reasoning add-on 270–700 s (P1) | +270.0 / +293.0 / +700.0 | ✓ |
| Shared 548–1,333 s, add-on 233–656 s (P5) | 547.8/901.6/1,332.7; +232.8/+266.2/+655.7 | ✓ |
| Overhead 1.13×–2.75× | P5 min 1.13×, P1 max 2.75× | ✓ |
| +63 s GridWorld, +1,232 s Push-T | +62.6 / +1,232.1 | ✓ |
| 9,560–82,116 tokens/round | Wipe 3,228+6,332=9,560; Push-T 17,504+64,612=82,116 | ✓ |
| Push-T ≈7× the next largest token total | 82,116 / 11,511 = 7.13 | ✓ |
| Push-T shared denominator 652.5 s, 2nd smallest | 652.5 (GridWorld 51.1 is smallest) | ✓ |
| SafeDAgger halts at 1 episode / 6.3 s; DISEIL screens 60 | §4 of source, verbatim | ✓ |
| GridWorld round 118 s, 2.16× a 55 s baseline | 118.0 / 54.6 → 2.16× | ✓ |
| GridWorld 9,735 tokens, KAG = 54 % of prompt | 1,690+8,045=9,735; C7: GridWorld 54 % | ✓ |
| Door (state) 1,054 s (P1) → 782.6 s (P5) | 1,054.0 → 782.6 | ✓ |
| Wipe carries 3 rounds under P5 | n rounds = 3 | ✓ |
| Table 8, all 8 rows, every cell | §2, §2b, §3, §3b | ✓ |

The report also correctly imports the source's caveats: single seed (C1), P1 as an upper bound (C2), overhead ratio is the *wrong* headline and the add-on is the right one (C4), token counts not comparable across rows because the backends differ (C5). The Push-T/GridWorld P5 cells are **UNMEASURED** in the source and are correspondingly **absent** from Table 8, with the structural reason stated in the caption. No cell was extrapolated.

---

## 5. Changes I made (2 targeted edits)

Both were the same trivial failure: a Lift appearance without the at-ceiling caveat.

1. **Table 6 caption (§4.10.1)** — appended:
   > "Lift begins the budget at the ceiling and supports no mechanism claim here either; its rows are reported for completeness, and Section 4.10.2 reads its low gain as the counter-example the measure predicts."

2. **§4.11, prescription-confidence correlations** — the Lift correlations (0.88, 0.89) were quoted bare. Appended after the per-task list:
   > "The two Lift figures are excluded from the reading that follows, on the same ground as everywhere else in this chapter: Lift is at the ceiling, so the $\Delta$SR these correlations are scored against has almost no range to correlate with, and the pair is quoted only because omitting it would be selective reporting."

   This one matters on the merits, not just for consistency: a 0.82–0.89 confidence-vs-ΔSR correlation is one of the report's stronger claims and it feeds Aim 3's demand model (§7.1). Quoting a Lift correlation against a ΔSR that cannot move would have been the weakest number in the set masquerading as one of the strongest.

No other edits. The document was already in excellent condition.

---

## 6. OUTSTANDING — what the AUTHOR must supply or decide

### DECISION REQUIRED (1)

**The thesis title contains a banned AI-tell word.** Cover page, L8:

> ## Leveraging Large Language Models for Sample-Efficient Imitation Learning

`Leveraging` is on the prohibited list in `Non-AI content.md`. It is the **only** surviving AI-tell in the report's own prose (the other three hits are inside citation titles, which cannot be altered, and "ablation harness", which is the correct technical term).

**I did not change this**, because a thesis title on a Confirmation of Candidature cover page is an administrative record that may already be registered with the faculty, and silently rewriting it is not a validation agent's call. If the title is not yet locked, the minimal fixes are:

- *Large Language Models for Sample-Efficient Imitation Learning* (delete one word — my recommendation)
- *Using Large Language Models for Sample-Efficient Imitation Learning*
- *Language-Model-Guided Demonstration Selection for Sample-Efficient Imitation Learning* (more descriptive; matches the actual contribution better, since the LLM prescribes demonstrations and is never in the control loop — a point the report is careful about at L252)

If the title **is** locked with the faculty, leave it and note the exception; the style rule cannot override a registered title.

### ALREADY DISCLOSED IN-TEXT — no action needed for the CoC, but they are live research debts

These are open items the report itself states honestly (§4.16.9, §8.1). They are **not** validation failures — the report's credibility rests partly on the fact that it names them — but the panel will ask, so they are collected here:

1. **Bridging contradiction (§4.4.4, L664).** The prescription logs record bridged prescriptions on GridWorld and Wipe, where the method's own precondition says the bridging arm should not exist. Unresolved. The logs must be inspected before that ablation is written up.
2. **Information-gain record count (§4.8.6, L975).** Each Table 6 cell pools 168–184 loss records, but a robot setting acquires only 100 demonstrations (5 seeds × 20), so a robot demonstration contributes >1 record. The source does not record the decomposition. Carried, not resolved.
3. **Confidence-scatter count (§4.11).** Figure 7 reports 152 prescriptions on GridWorld (image), below the 180 that 9 seeds × B=20 supply. The shortfall is unexplained and the run logs have not been re-read.
4. **D5 single-seed (§4.16.7).** No cost figure in the report carries cross-seed variance. Should be repeated at further seeds.
5. **D4 failure-count diagnostic** is instrumented on Push-T (image) only; should be run on all three primary settings.
6. **Per-task memory kernel width** (the identified fix for the mis-scaled σ_mem, §4.16.3 / A13) **has not been run.**
7. **Policy-solvability check (Eq. 11)** is drawn in the architecture but **not implemented, not ablated**, and no number is attributable to it. The report says so plainly at L691 — keep it that way; do not let this drift into an implied result.
8. **Prompt template defect (§4.6, L791).** The VLM template says "peak-loss frame" while the frame actually passed is the Eq. 5 first-crossing step. Stale wording, correct frame. Worth fixing in the code before the camera-ready.

### VERIFIED-CLEAN — no author action

Naming, references, cross-references, figure assets, cover page, Gantt, D5 provenance, DAgger-family labelling, ablation scoping, geometric-clustering claims, B/D framing, and the Aim 1→2→3 through-line all pass without qualification.
