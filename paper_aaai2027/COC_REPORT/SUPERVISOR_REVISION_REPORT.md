# CoC Report — Supervisor Revision Report (Rounds 1 and 2)

Artefact audited: `/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT/CoC_Report.pdf`
Built from `CoC_Report.md` by `build_pdf.sh` (pandoc → XeLaTeX, three passes).
Audit date: 14 July 2026. Verdict: **10/10 round-2 items PASS, 14/14 round-1 items still hold, no regressions.**

Page numbers below are **PDF page indices** (the printed folio is one lower, because the cover is unnumbered).

---

## 1. Final document statistics

| Quantity | Value |
|---|---|
| Pages | 75 |
| Words | 41,439 |
| File size | 2.8 MB |
| Figures | 20 numbered (1–20) + 3 appendix certificates (A.1–A.3) |
| Tables | 12 (captions 1–12, contiguous) |
| Tagged equations | 14 (1–14, contiguous — **no 8a/8b**) |
| References | 100 (numbered 1–100, contiguous, every one cited, zero dangling) |
| Ablation studies | A1–A18 (18, contiguous); **D-series count: 0** |
| Algorithm floats | 1 (`\begin{algorithm}[t]`, 27 numbered steps) |
| DISEIL occurrences | 47 |
| DISTIL / PACE / P4-LLM as method names | **0** |
| Banned acronym (institute initialism) | **0** |
| Table overflows | **0** (12 tables; 1 non-fatal 1.81 pt overfull box in a *paragraph*, not a table) |
| Greek glyphs | all 15 used by the source survive into the PDF text layer |

Build gate output: `checks passed: glyphs intact, no double captions, no banned acronym, no table overflow`

---

## 2. Round-2 items (the author's ten)

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | Algorithm 1 a proper, SHORT algorithm float with atomic steps | **PASS** | Opened PDF **p.25** (folio 24). Genuine float: `\begin{algorithm}[t]` + `algpseudocode` (`CoC_Report.tex:1807–1841`; `build/preamble.tex:12–13`), ruled header "Algorithm 1 DISEIL", `Require`/`Ensure`, **27 numbered atomic lines** each one action ("train $f_\theta$…", "$t_i^\star \leftarrow$ the flagged step"), right-margin cross-refs to Eq. 2–11. Occupies ~half a page, not a page-long dump. |
| 2 | No orange **text**; coloured orange *marks* permitted | **PASS** | Source: `grep -ic orange CoC_Report.md` = **0**; no `\textcolor` in body (the `\textcolor` hits in the .tex are pandoc's unused syntax-highlight macro defs). Pixel-scanned all 75 pages for orange ink: hits only on pp. 39, 41, 42, 44, 65, 68, 69. Opened each — **every hit is a mark, not text**: p.44 Fig. 7 orange scatter *dots* (axis labels/annotations black); p.42 Fig. 6 orange *median lines* inside boxes; p.39 Fig. 4 the tan wooden *table surface* in the sim renders (row labels are blue/pink/green); p.65 Gantt orange *bars* (legend text black); pp.68–69 Deakin certificate branding. No orange caption or in-figure text anywhere. |
| 3 | Cluster memory configurable & task-dependent; Eq. 8a gone; memory-constant study/figure/table gone; not a headline contribution | **PASS** | `CoC_Report.md:435` — "the memory is a **configurable, task-dependent component** rather than a part of the core rule… Setting $\lambda=0$ switches it off and returns Equation 8 exactly… it is a component of the instantiation and **not part of the framework's contribution**". Equation tags are **1–14 contiguous — no 8a** (the Gaussian penalty is now prose, not a numbered equation). `F10_memory_constants.{pdf,png}` **deleted** from `figures_generated/`; zero `F10_` references in the source; zero "memory constant" strings. Memory now appears only as knockout **A1** (priced at 0.73 pts mean, the smallest of seven, `:955`), a re-spec note (`:1062`) and a limitation (`:1076`). |
| 4 | Abstract ≤ 1 page, ZERO citations | **PASS** | PDF **p.4** (folio 3). Opened it: begins "Abstract", ends on the same page ("…exactly the demonstrations it lacks."). **Zero** bracketed citations in the page's text layer. |
| 5 | Teaser figure small, lands on page 5 | **PASS** | PDF **p.5** (folio 4). Figure 1 "The demonstration-distillation loop" — hand-drawn sketch occupying roughly the lower third of the page, well under half the text block, with three paragraphs of §1.1 above it. |
| 6 | Architecture figure immediately after the first paragraph of the Aim-1 Methodology subsection | **PASS** | `CoC_Report.md`: `### 4.1.3 Methodology` (:379) → **exactly one paragraph** (:381, "DISEIL runs four stages once per round…") → `![](../figures/Architectural_Diagram.pdf)` (:383) → caption "**Figure 2.** The DISEIL framework." (:385). Renders on PDF **p.21**. |
| 7 | Old aggregate-significance figure deleted; its point carried by the success-rate table | **PASS** | `F14_aggregate_significance.{pdf,png}` **deleted** from `figures_generated/`; zero `F14_` and zero "aggregate significance" references in the source. The claim now rides with **Table 7** (final success rate, ten settings, DISEIL bolded and best in all ten) and the prose beneath it (`:822–826`): "Ten wins from ten is a pattern… the aggregate advantage is significant under the conservative collapsed test" — the collapsed paired $t(4)=4.15$, $p=0.014$ also stated in the abstract. |
| 8 | Per-demonstration information-gain table shows ONLY Diff-DAgger and DISEIL | **PASS** | **Table 8** (`:842`), seen rendered on PDF **p.42**. Header is exactly `\| Task \| Mod. \| Diff \| DISEIL \|` — four columns, two methods. Safe/Dropout/Ensemble/Thrifty/Stagger all absent. |
| 9 | Gain-without-allocation figure is a SINGLE panel (no information-gain panel) | **PASS** | Rendered `figures_generated/F2_gain_without_allocation.pdf`: **one metric panel** — final success rate only — drawn across the three standard ablation settings (GridWorld image, Push-T state, Door image), each with DISEIL (full) vs Clustering-off (A3) bars and a best-baseline dashed line. **The information-gain panel is gone.** Caption (Figure 9, `:913`) speaks only of success rate. |
| 10 | Three certificates smaller, each on the same page as its caption | **PASS** | Held in unbreakable `minipage` boxes so LaTeX cannot split image from caption. A.1 at `height=0.46\textheight` → PDF **p.68** with Figure A.1 caption directly beneath. A.2 at `height=0.46\textheight` and A.3 (trimmed, `width=0.94\textwidth`) **both fit on PDF p.69**, each with its own caption on that page. Opened both pages and confirmed visually. |

---

## 3. Regression checks (nothing broke)

| Check | Verdict | Evidence |
|---|---|---|
| Numbering continuous, no gaps | **PASS** | Figures 1–20 (Fig. 1's caption is `\footnotesize\textbf{Figure 1. …}`, hence a different literal from the rest); Tables 1–12; Equations 1–14. No gaps in any series. |
| No broken references | **PASS** | 100 reference entries, numbered 1–100 contiguous; every citation in the body resolves; **0 dangling**, **0 uncited**. XeLaTeX log: **0** undefined references/citations. |
| Ablations A1–A18, no D-series | **PASS** | Unique ablation IDs in source: A1…A18. `grep -cE '\bD[0-9]\b'` = **0**. |
| Banned acronym absent | **PASS** | Build gate #3: 0 occurrences in the PDF text layer (logo file is still named for it on disk, but the name is never typeset). |
| Greek glyphs intact | **PASS** | `build/check_glyphs.py`: all 15 Greek letters the source uses render, plus 2 maths symbols. |
| No table overflowing | **PASS** | `build/check_tables.py`: 12 tables, **no table overflows its page width**. The single overfull box (1.81 pt) is in a prose paragraph and is not fatal. |
| DISEIL naming clean | **PASS** | 47 DISEIL; **0** DISTIL, **0** PACE, **0** P4-LLM. |

---

## 4. Round-1 items (re-confirmed still holding)

Carried from `COC_BUILD_REPORT.md` §2 and spot-checked against the current build.

| # | Round-1 item | Verdict (round 2) |
|---|---|---|
| 1 | DISEIL used everywhere | **PASS** (47 occurrences) |
| 2 | DISTIL / PACE / P4 removed entirely | **PASS** (exact hit count 0) |
| 3 | Updated teaser figure | **PASS** (Figure 1, now shrunk — round-2 item 5) |
| 4 | Updated architecture (policy-solvability loop) | **PASS** (Figure 2, now repositioned — round-2 item 6) |
| 5 | Learning curves, all five tasks | **PASS** (Figure 5, `all_5_task_comparison.pdf`) |
| 6 | Information-gain discussion updated | **PASS** (§5.1.5; Table 8 now two-method — round-2 item 8) |
| 7 | Initial-demonstration discussion | **PASS** |
| 8 | Representative prompts | **PASS** |
| 9 | Representative KAG examples as structured key-value | **PASS** |
| 10 | Equation 10 as feasibility verification | **PASS** (still Eq. 10 after the 8a deletion; renumbering left it in place) |
| 11 | Cluster naming explained | **PASS** (naming pipeline, §4.1.3; Algorithm 1 line 10) |
| 12 | Humanised writing | **PASS** |
| 13 | References verified | **PASS** (now 100, contiguous, all cited) |
| 14 | Cross-references consistent | **PASS** (0 undefined refs in the LaTeX log) |

---

## 5. Changes made by this audit

**None.** All ten round-2 items and all regression checks passed on the build as submitted; no trivial failures were found to fix. The PDF was rebuilt once to re-run the automated gates, and the gates passed unchanged.

## 6. Notes for the author (not failures)

- **Item 9, a wording nuance.** The gain-without-allocation figure is a single *metric* panel (success rate), drawn as three small-multiple sub-axes — one per ablation setting — matching the house style of every other knockout figure. If "single panel" was meant literally as one axis, say so and the three settings can be collapsed onto one axis; as read, the requirement (kill the information-gain panel) is met.
- Certificates A.2 and A.3 share PDF p.69. Each keeps its own caption on that page, so the constraint holds, but if you want one certificate per page that is a one-line `\clearpage`.
