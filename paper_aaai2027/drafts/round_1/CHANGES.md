# Round 1 changes (vs drafts/round_0_pre_review)

All edits are targeted Edit-call replacements in draft/paper.tex (plus three entries in
draft/references.bib). Build after edits: **9 pages including references, 0 overfull boxes,
no undefined references**. Numbers unchanged and re-verified against
context/results_data.md and table_data.xlsx.

## Per-issue edits

- **R1-r1-1 / R2-r1-8**: Q2 rewritten — the two-case dichotomy became three explicit
  readings (underrepresented / suboptimal / represented-but-hard-to-fit); "measures exactly
  the diversity" deleted; metric reframed as a novelty proxy; PPO-expert caveat and the
  selection-circularity ("confirms the mechanism operates as designed; not an independent
  quality measure") added.
- **R1-r1-2**: dropped "necessary but not sufficient"; now "gain alone does not predict
  final success ... what converts gain into success rate is its allocation across failure
  modes".
- **R1-r1-3**: Baseline Query Rules states the SafeDAgger/DropoutDAgger deviations from the
  published algorithms and the budget accounting (gating expert-action access is free for
  every baseline; only recorded demonstration segments count; DISTIL's flagging never
  consults the expert).
- **R1-r1-4**: F_r defined as the round's task-failed episodes; rollouts per round stated
  (20-layout pool GridWorld / 60 episodes robot); the per-episode 95th percentile scoped to
  locating high-loss steps within an already-failed episode.
- **R1-r1-5**: memory kernel space/units defined (raw object-pose coordinates, meters;
  independent of clustering features and PCA dimension); lambda=1 scale reading; constants
  unchanged across tasks and loss families; c_C defined.
- **R1-r1-6**: online/offline bridge restated per round (contribution 4 + method
  paragraph); no batch-recording claim remains.
- **R1-r1-7**: Eq. 9 scoped to Push-T; Lift/Door described as an absolute clamp to the
  native reset range with Delta_max = clamp half-width (kag_ur5_bounds.md values).
- **R1-r1-8**: intro tempered ("rarer modes go uncorrected for many rounds"); cross-round
  rare-mode dynamic added after Eq. 8.
- **R1-r1-9**: "categorical cross-entropy"; Eq. 2 framed as the training-loss functional at
  the executed action; denoising loss averaged over sampled noise/timestep draws.
- **R1-r1-10**: defined K (patience window, K=1), a^nov, c_C; bridging-arm form of d_r
  (full demonstration from xi); "Pearson r" in prose.
- **R1-r1-11**: "Each uncertainty-gated baseline instantiates one template"; Stagger
  presented separately as the round-level random control.
- **R1-r1-12**: "no budget is ever wasted" -> "no budget is spent on an infeasible
  prescription".
- **R1-r1-13**: protocol paragraph acknowledges DISTIL's non-expert extra inputs (KAG,
  object poses, simulator resets) and why the 20-demo accounting still holds.
- **R2-r1-1**: false Stagger-roster justification deleted; true reason given (Diff-DAgger
  needs a diffusion learner); missing robot-side random-allocation control stated in
  Baseline Query Rules and Limitations. No new experimental arm (text-only revision;
  roster fixed by results_data.md).
- **R2-r1-2**: Table 1 caption corrected — the ± is the pooled cell-level std from the
  evaluation logs, not the std of per-seed means (verified in table_data.xlsx; the three
  "duplicate" stds are one-decimal rounding of distinct values 0.0446/0.0451,
  0.0599/0.060493, 0.0320/0.032195).
- **R2-r1-3**: Q1 separation claim withdrawn; evidence restated as ranking consistency
  (best mean 10/10, best info gain 10/10) with overlap acknowledged for Push-T/Wipe too.
- **R2-r1-4**: protocol details added — |D_0|=20 shared and outside the budget; held-out
  200 layouts / 100 episodes; retrain from scratch each round (GridWorld) / every 4th demo
  (robot, Diff-DAgger cadence); baseline thresholds (Dropout N=10, tau=0.1, p=0.9;
  Ensemble chi=0.05; Thrifty calibration; Diff-DAgger K=1, alpha=0.99); no-fire fallback
  behavior.
- **R2-r1-5**: model stack named in the method (Qwen3-VL-32B VLM; Qwen3-32B reasoning LLM);
  re-prescription bound (five attempts). Wall-clock cost not logged -> kept qualitative.
- **R2-r1-6**: Fig. 3 caption softened ("tracks the pack ... separates over the second
  half"); Q1 "flatten late in the budget".
- **R2-r1-7**: fallback-rounds exclusion rule stated in Fig. 4 caption and Q3 (no sample
  count in prose).
- **R3-r1-1**: slogan kept once (end of intro); abstract carries a compressed echo; Fig. 1
  caption and DISTIL Loop opening rewritten; "purest corrections" appears once.
- **R3-r1-2**: Table 2 \resizebox removed; means-only, \small, fits the column (0 overfull).
- **R3-r1-3**: Related Work ends in positive form; negative triad only at the end of
  Baseline Query Rules.
- **R3-r1-4**: terminology unified — "vision-language model (VLM)" at first use then VLM;
  "reasoning LLM"; "20-demonstration budget" in captions.
- **R3-r1-5**: UR5e + Robotiq-85 / wiping pad named for Lift/Wipe/Door.
- **R3-r1-6**: Fig. 4 panels enlarged (0.68->0.74, 0.58->0.62 columnwidth; larger sizes
  broke the 9-page cap). Fig. 3 legend reordering needs plot re-rendering — not done in
  this text revision (acknowledged in REVIEW_LOG).
- **R3-r1-7**: Qwen entries -> corporate author "Qwen Team" (2025a/b); Self-Refine title
  casing fixed.
- **R3-r1-8**: abstract's 70-word sentence split in two; two-arm parenthetical dropped.

## Space compensation (to hold 9 pages after the additions)

- Cut the intro results paragraph that duplicated contribution (3).
- Cut the "Relation to the query rules" recap paragraph.
- Removed three redundant citations (pomerleau1991efficient, mu2021maniskill,
  gu2023maniskill2 — ManiSkill3 and ALVINN'88 remain cited).
- Tightened Limitations, Conclusion, Q3 opening, Related Work FM paragraph, Algorithm 1
  lines 3/5.
- Rescaled teaser (0.45->0.36), clustering figure (0.62->0.50), comparison figure
  (0.92->0.84 textwidth).
