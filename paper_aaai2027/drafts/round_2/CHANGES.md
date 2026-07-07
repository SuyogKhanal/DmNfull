# Round 2 changes (vs drafts/round_1)

All edits are targeted string replacements in draft/paper.tex; no figure PDFs were regenerated (no plot sources/data ship with the paper; see REVIEW_LOG R3-r1-6). Build: 9 pages including references, 0 overfull boxes.

## R1 (method & soundness) — 5 new minors, all fixed
- **R1-r2-1**: Protocol fairness inventory now credits object poses "for the state-run descriptors and for the memory centroids of every run, image runs included".
- **R1-r2-2**: Lambda calibration corrected: P_mem is gamma=0.6 at a centroid covered the previous round (memory appended at round end), so lambda=1 offsets roughly half a unit of mean peak loss; "near 1" claim removed.
- **R1-r2-3**: GridWorld memory space defined: centroids in grid-cell coordinates; sigma=0.06 is a small fraction of a cell, so the kernel reduces to an identical-centroid check shaped by gamma alone.
- **R1-r2-4**: q_A defined at Eq. 9: the representative's remaining pose components (height and full orientation), carried over unchanged.
- **R1-r2-5**: The p95 threshold now has its real consumer: it selects the steps stored as failure-point candidates (peak always kept); t* is the argmax over the stored candidates.

## R2 (experiments & evidence) — 4 majors + 3 minors; 6 fixed, 1 partial
- **R2-r2-1 (major)**: Delta-SR precisely defined in the protocol: change in success rate on the round's own rollout evaluation (20-layout pool / discovery episodes) across the retraining that incorporates the demonstration; differenced per retraining, never accumulated; coarse rollout-level measurement whose per-round values run far above held-out increments (reconciles Figure 4 vs Figure 3 magnitudes). Q3 adds the non-drift argument; Figure 4 caption points at the protocol definition.
- **R2-r2-2 (major)**: Retraining regime made consistent everywhere: from scratch after every demonstration on GridWorld, after every fourth on the robot tasks, policy unchanged between retrains (up to three consecutive rounds analyze the same policy); Eq. 3 text and Algorithm 1 line 6 carry the shared per-task cadence; info-gain staleness and shared checkpoint evaluations stated; "pre-finetune" wording replaced by "pre-retrain".
- **R2-r2-3 (major)**: Table 2 caption counts relabeled as pooled loss records (168-184 per cell), explicitly not distinct demonstrations on the robot cells (5 seeds x 20 demos).
- **R2-r2-4 (major)**: Table 1 caption names the aggregation unit: mean and std over pooled per-checkpoint held-out evaluations across seeds (not per-seed means), so means need not be multiples of 0.2pp; protocol states held-out evaluation cadence (every round / every retraining checkpoint). "As produced by the evaluation logs" removed.
- **R2-r2-5 (minor)**: "Calibrated" dropped: contribution says confidence "predicts realized improvement"; Q3 retitled "Confidence as an Improvement Predictor; Discovered Modes"; abstract updated.
- **R2-r2-6 (minor)**: "Six interactive baselines (five per setting)" in abstract, contributions, and conclusion.
- **R2-r2-7 (minor, PARTIAL)**: Added recorded decoding facts (VLM low reasoning effort, LLM high, 16,384-token output cap) and the fallback exclusion count for the plotted cell (28 of 180 rounds). Per-cell fallback tallies, wall-clock numbers, and a prompt appendix are not in the results source / do not fit the 9-page cap; declined without fabrication (see REVIEW_LOG).

## R3 (presentation & style) — 1 carried minor + 4 new minors; 2 fixed, 2 partial, 1 deferred
- **R3-r1-6 (carried, DEFERRED)**: Plot PDFs cannot be regenerated (no scripts or per-record data in the paper sources); documented with evidence in REVIEW_LOG.
- **R3-r2-1**: Eq. 10 display now ends with a period; next sentence begins "Equation 10 gives the Push-T mechanism ...".
- **R3-r2-2**: 75-word roster sentence split into three sentences.
- **R3-r2-3**: Figure 2 caption expands the acronym: "task knowledge graph (knowledge-augmented generation, KAG)"; diagram-internal "Vision LLM" box label deferred with R3-r1-6.
- **R3-r2-4 (PARTIAL)**: Figure 5 widened 0.5 -> 0.68 columnwidth (page-cap bound); annotation-font regeneration blocked with R3-r1-6.

## Page-budget measures (to hold 9 pages after the additions)
- Tightened Table 1/2 captions, protocol/metric paragraph, memory and flagging paragraphs, Q2 allocation sentence, Q3, conclusion, limitations (no reviewer-required content removed).
- Cut seven single-use background citations: AggreVaTe, Deeply AggreVaTeD, Diffuser, PaLM-E, RT-2, ProgPrompt, BADGE (every related-work line keeps an anchor citation).
- Figure rescales: architecture 1.0 -> 0.88\textwidth, comparison 0.84 -> 0.78\textwidth, teaser 0.36 -> 0.33\columnwidth; Figure 4 untouched at its round-1 enlarged size; Figure 5 enlarged (see R3-r2-4).
