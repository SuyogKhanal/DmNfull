# AAAI style recon: what an accepted 7-page AAAI paper actually looks like

Source: the four accepted papers in `past_papers_aaai/`. Style only. No content taken.

| ref | paper | content pp | ref pp |
|---|---|---|---|
| P1 | Vyas et al., DRL backdoors (AAAI-26) | 26072-26078 = **7** | 26079-26080 = 2 |
| P2 | Huang et al., UNeMo VLN (AAAI-26) | 18315-18321 = **7** | 18322-18323 = 2 |
| P3 | Wan et al., MOAIR POI (AAAI-25) | 12676-12682 = **7** | 12683-12684 = 2 |
| P4 | Lei et al., STRIL multi-agent IL (AAAI-25) | 18163-18169 = **7** | 18170-18171 = 2 |

All four land on exactly 7 + 2. None manipulates layout. All hit the limit by writing less.

---

## 1. Measured section budgets in the exemplars

Pages, measured by column-inches (2 columns = 1.0 page).

| section | P1 | P2 | P3 | P4 | typical |
|---|---|---|---|---|---|
| Abstract + Intro (incl. contribution bullets) | 1.15 | 1.40 | 1.30 | 1.00 | **1.0-1.4** |
| Related work | 0.35 (late, p7) | 0.60 | 0.50 | 0.50 | **0.4-0.6** |
| Background / preliminaries / problem formulation | 1.40 | 0.60 | 0.50 | 0.95 | **0.5-1.0** |
| Method | 3.00 | 1.30 | 2.60 | 2.30 | **1.3-3.0 (largest block)** |
| Experiments (setup + main results) | 1.20 | 1.70 | 1.70 | 1.90 | **1.2-1.9** |
| Ablation / analysis | 0.10 | 1.10 | 0.30 | 0.20 | **0.1-1.1** |
| Conclusion (+ limitations/defenses) | 0.40 | 0.30 | 0.30 | 0.35 | **0.3-0.4** |

Invariants worth copying:
- **Method is the single largest block** in every paper (1.3-3.0 pp). Reviewers are paying for the idea.
- **Related work is small** (never above 0.6 pp) and is not a survey. P1 puts it on page 7, right before the conclusion, which is a legal and effective way to buy early space.
- **Conclusion is always ~0.3 pp**, one paragraph, no bullets, no restated numbers.
- **Ablations are the compression valve**: P1 spends *four lines* on ablations and defers everything to the appendix; P4 has no ablation section at all (one "sensitivity analysis" paragraph). Only P2 spends over a page, and P2 pays for it by having the shortest method section.

## 2. How the first page is used

- **P1, P3, P4: no figure on page 1.** Abstract + intro text only. The overview figure sits full-width at the **top of page 2** (P4) or **top of page 3** (P1, P2).
- **P2 is the only teaser**: Fig. 1 sits in the **right column of page 1**, level with the abstract, about **one third of a column tall** (roughly 8 cm x 8 cm). It is a two-row before/after comparison (prior method on top, theirs below), not a pipeline. It exists to make one contrast legible in five seconds.
- The rule: a page-1 figure must earn its ~0.3 pp by carrying the *contrast*, not the *architecture*. The architecture always goes later, at the head of the method.
- Contribution bullets end the intro in P1, P2, P4 (3-4 bullets, one to three lines each). P3 uses 4 bullets. This is the AAAI norm and costs ~0.25 pp.

**For DISEIL**: use `teaser.png` on page 1 only if it renders the Diff-DAgger-vs-DISEIL contrast at a glance. Otherwise drop it and put `architecture.pdf` full-width at the top of the method page. Do not pay 0.3 pp for a picture of the pipeline twice.

## 3. How results are presented

Density (main paper only):

| | tables | figures | figs that are result plots |
|---|---|---|---|
| P1 | 3 | 2 | 0 |
| P2 | 4 | 2 | 0 |
| P3 | 2 | 5 | 3 |
| P4 | 1 | 4 | 3 |

- **One dense main table beats three thin ones.** P1's Table 2 packs 4 methods x 3 metrics x 6 environments into ~0.25 pp. P4's Table 1 is 3 games x 3 algorithms x 4 filtering methods with `mean ± std` in-cell.
- **Bold marks the winner; underline marks second or an oracle.** Both P2 and P4 do this and say so in the caption.
- Tables are `\footnotesize`-class dense **inside the table only** (this is normal and legal; the ban is on shrinking body text).
- **Captions describe the artifact, not the takeaway** in P2/P3/P4 ("Performance Comparison on REVERIE Dataset", "Overview of 3PO architecture"). P1 is the exception and the better model: its Table 2 caption ends with a bolded sentence stating the takeaway ("Both our attacks rival or surpass the performance levels of TrojDRL (baseline) and BadRL despite significantly lower adversarial privileges"). P1's Fig. 1 caption also carries the definitions the body would otherwise spend lines on.
- **Take the P1 approach**: caption = one sentence naming the artifact + one sentence stating what the reviewer should conclude + any legend the body would otherwise pay for. This moves prose into caption space, which is the only free space in the paper.
- Results prose is organised under **bolded run-in headings that state the finding**, not the artifact: "InfrectroRL outperforms baselines with limited adversarial access and no training-time poisoning" (P1), "R2R Result" (P2). P1's style is stronger and self-summarising.

## 4. Sentence-level register

**Hedging.** Claims are bounded by their evidence and the mechanism is named as an attribution, not a fact:
- "This improvement primarily arises from ..." (P1)
- "This *may* require extra exploratory actions but ensures target arrival." (P2)
- "We attribute MOAIR's robustness to several key factors: ..." (P3)
- "..., *likely due to* its use of a weighted transition graph" (P3)
- Negative results are stated plainly, in the same voice as positive ones: "the RI and EL methods did not improve ILEED on Connect Four because the dataset aligns well with the assumption of ILEED" (P4); "Although Space Invaders shows a higher score, the attack does not reduce InfrectroRL to baseline PPO performance" (P1). None of them hides a loss, and none apologises for one.

**Contributions.** Verb + object + scope. "We propose X, a novel framework designed for Y." "We define the Randomness Indicator (RI) and Exploited Level (EL), which ..." "We provide theoretical guarantees on ...". No superlatives, no "significantly advances the field", no promises about impact.

**Related work positioning.** The move is always: *state what prior work does, name the structural limitation, state the contrast in one clause.* Never a quality judgement about the authors.
- "While performing well on large task-specific datasets, they show poor policy generalization in out-of-distribution environments." (P2)
- "Despite improvements, these agents still lag behind specialized VLN models in accuracy and transparency." (P2)
- "However, the aforementioned models rely heavily on the quality of training data ... In this paper, we propose ..." (P3)
- "However, this assumption cannot be satisfied in simple games like RPS ..." (P4) — the limitation is a *condition*, not a failing.
- P1 uses a numbered limitation list ("(1) cover only a subset of ..., (2) require high and unrealistic adversarial access, (3) exclusively target training phases") then contrasts once. Compact and non-hostile.

## 5. How ablations get compressed

Three working patterns, in descending aggression:
1. **P1 (four lines)**: name the ablation *dimensions* only, give no numbers, defer to the appendix. "We systematically evaluate InfrectroRL's robustness through four key ablation dimensions: (1) $\gamma$, (2) $\lambda$, (3) trigger size variations, and (4) target action selection. See Appendix for more insights." Costs 0.1 pp.
2. **P3 (one paragraph + one small figure)**: a single sentence per knockout carrying only the delta ("w/o Multi-reward: ... causing a 17% performance drop (Acc@5 in TKY); w/o BC: ... 9% ..."), plus a compact grouped bar chart. Costs 0.3 pp.
3. **P2 (a page)**: two named ablations, each with its own table. Only affordable because the method section is 1.3 pp.

What gets deferred in all four: hyperparameter sweeps, per-environment breakdowns, full proofs, sensitivity curves, implementation detail, qualitative examples.

**For DISEIL**: pattern 2. One compact ablation table or `knockout_summary.pdf`, one paragraph naming each knockout and its delta, everything A1..A18 in `supplementary.pdf`. Budget 0.4-0.65 pp, no more. The GridWorld null result and the small A4/A5 gaps are stated in that paragraph in the P4 voice (plain, same register as the wins), not buried.

## 6. Anti-patterns these papers never commit

1. **Never** a `\vspace`, a margin change, a resized column, or `\small` on body text. Zero instances in four papers.
2. **Never** a content appendix inside the main PDF.
3. **Never** a figure that repeats a table (results appear once, in the form that is denser for that data).
4. **Never** a related-work section that grades other papers ("poor", "naive", "fails badly"). Limitations are conditions and assumptions.
5. **Never** an abstract that lists ablations. All four abstracts are: problem, why current methods fall short, what we propose, what the headline result is. P1: "Empirical and analytical evaluations across six Atari environments show ...". Nothing else.
6. **Never** a number in the conclusion. All four conclusions restate the contribution qualitatively and give one forward-looking sentence.
7. **Never** a full-page figure. The largest single figure in any of the four is P4's Fig. 3 at ~0.45 pp, and it carries twelve panels.
8. **Never** an orphaned paragraph. Every subsection opens by naming what it is about to establish and closes by handing off ("this reasoning visual-state is integrated into the subsequent Hierarchical Prediction-Feedback Navigator to enhance navigation performance", P2).
9. **Never** marketing adjectives on their own method. "Novel" appears once per paper, in the abstract, attached to the artifact name.
10. **Never** two sentences where the caption could carry one.

---

## 7. Page budget for the DISEIL paper (sums to 7.0)

| # | section | pages | contents |
|---|---|---|---|
| 1 | Title, abstract, introduction | **1.20** | Abstract (~0.20, no ablations, one headline number). Intro: problem, why Diff-DAgger-style allocation falls short, the DISEIL insight, 3 contribution bullets. Teaser `teaser.png` on p1 ONLY if it carries the contrast; else spend the 0.3 pp on text. |
| 2 | Related work | **0.55** | Three run-in blocks (interactive IL / DAgger-family; demonstration selection and coverage; LLM and VLM guidance in robot learning). Structure: what it does, what condition it assumes, one contrast clause. From `02_background.md`. No survey. |
| 3 | Problem formulation | **0.45** | Budgeted interactive IL setup, the allocation problem, notation. From `04_aims.md` s4.1. |
| 4 | Method: DISEIL | **1.70** | `architecture.pdf` full-width at the head of the section (~0.35). Perceive / Partition / Prioritise / Prescribe as run-in headings. Algorithm 1 (~0.25). Cluster memory named once as configurable and task-specific, not sold. |
| 5 | Experimental setup | **0.55** | 5 tasks x 2 modalities, budget B=20, seeds, baselines, metric = mean ± SE (SE = std/sqrt(n), n=5 robot, n=9 GridWorld). Say once that the 10 settings are not 10 independent experiments. |
| 6 | Main results | **1.30** | CoC Table 7 verbatim, one dense table, bold best (~0.40). `learning_curves.pdf` 5 panels (~0.35). Prose under finding-stating bold run-ins. Collapsed task-level test: paired t(4)=4.10, p=0.015 two-sided; sign test p=0.031 one-sided. Lift at ceiling (100.0 ± 0.0) flagged as uninformative here, not in a footnote. |
| 7 | Analysis and ablations | **0.65** | Information gain: CoC Table 8 or `info_gain.pdf` (~0.25). One ablation paragraph, P3 pattern: knockout name + delta only, A4 and A5 stated as small, GridWorld null stated plainly. Pointer to supplementary for A1..A18. `confidence.pdf` only if it displaces prose. |
| 8 | Limitations and conclusion | **0.35** | Limitations in the body voice, not a defensive list. Conclusion: one paragraph, no numbers, at most one forward-looking clause. No Aim 2/Aim 3. |
| | **total** | **7.00** | |
| | references | (2.00) | separate allowance, not counted |
| | supplementary | separate PDF | A1..A18, hyperparameters, prompts, KAG examples, per-setting tables, failure cases, compute cost |

Overflow order (cut in this order, never adjust layout): teaser figure (-0.30) → `confidence.pdf` (-0.25) → related work to P1's late-and-short form (-0.15) → ablation paragraph to P1's four-line form (-0.30) → Algorithm 1 to supplementary (-0.25).

---

## 8. House style checklist (binding on writers)

**Layout**
- [ ] 7.0 content pages, references separate, appendix in `supplementary.pdf`. Fix overflow by cutting words.
- [ ] No `\vspace`, `\setlength`, margin, column, or font-size change. No `\small` on body text. No edit to `aaai2027.sty`.
- [ ] `\usepackage[submission]{aaai2027}`; author block = "Anonymous Submission".
- [ ] pdflatex only. No literal unicode Greek: `$\sigma$`, never the character.

**Prose**
- [ ] No em-dashes. Use a comma, a semicolon, or two sentences.
- [ ] No hanging word: no paragraph ends with a single short word alone on its last line. Tighten the sentence.
- [ ] No marketing language. "Novel" at most once, in the abstract, attached to the method name.
- [ ] Every paragraph answers one of: why this matters, why existing methods fall short, why this idea is needed, why it works, why to believe it.
- [ ] Every subsection ends by handing off to the next. No isolated paragraphs.
- [ ] Related work states a *condition or assumption* prior work relies on. Never a quality judgement.
- [ ] Losses, ceilings, and small gaps are stated in the same plain register as the wins.
- [ ] Hedge the mechanism, not the measurement: "this gain primarily arises from", "we attribute this to". Never hedge a number.

**Evidence**
- [ ] Every number traces to `COC_REPORT/build/v2/*.md`. Nothing invented, nothing re-rounded.
- [ ] Mean ± SE everywhere; state SE = std/sqrt(n) and n once.
- [ ] Aggregate claim = the collapsed task-level test only. Never call the 10 settings 10 independent experiments.
- [ ] Every `\cite` key exists in `references.bib`. Never fabricate.
- [ ] Bold = best, underline = second/oracle, declared in the caption.
- [ ] No result appears in both a table and a figure.

**Captions (P1 pattern)**
- [ ] Sentence 1 names the artifact. Sentence 2 states the takeaway. Legends live in the caption, not the body.
- [ ] Result-section run-in headings state the finding, not the artifact.

**Abstract and conclusion**
- [ ] Abstract: problem, gap, method, headline result. No ablations. No section list.
- [ ] Conclusion: one paragraph, no numbers, at most one forward-looking clause. No Aim 2/Aim 3, no candidature material.

**Naming**
- [ ] The method is **DISEIL**, always. Never DISTIL, PACE, or P4. Grep the source before every build.
