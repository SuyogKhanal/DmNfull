# Round 3 changes (vs drafts/round_2)

Build: 9 pages total including references, 0 overfull boxes, no undefined references.

## Fixes, by issue id

- **R1-r3-1** (near-flat memory kernel on Lift/Door): added the parallel disclosure after the GridWorld degeneracy: "Lift and Door sit at the opposite extreme: against their narrow reset ranges ($\pm 0.03$ m; roughly $\pm 0.013$ m) the kernel is nearly flat, so the memory acts there as a recency-weighted global discount rather than a spatially selective penalty"; same-constants sentence now ends "at the price of these two degenerate regimes".
- **R1-r3-2** (Eq. 4 template vs GridWorld pool form): template intro softened to "a scalar score gated by a threshold; in its per-state form the expert takes over at the first crossing"; section now closes with the two application forms (robot per-state correct-from-crossing with episode cap; GridWorld pool form scoring every layout, correcting the highest-scoring layout whose gate fires, falling back to the highest-scoring failure). Correct-from-crossing scoped to robot tasks.
- **R1-r3-3** (degenerate failure sets): added after the k_max definition: N<=3 skips the silhouette sweep (each failure its own cluster; image PCA passes through at N<=2), a lone failure is targeted directly, and a failure-free rollout set yields no demonstration and no budget charge (budget counts recorded demonstrations); the loop rolls out again. Algorithm 1 is total.
- **R1-r3-4** (orientation in memory distance): Eq. 9 kernel now written $\|c_{xy}-c_{i,xy}\|_2^2$; text states the kernel distance is planar only, stored yaw never enters P_mem, and on Push-T coverage of one mode also discounts a differently oriented mode at the same xy.
- **R2-r2-1** (carried, Delta-SR vs plotted values): withdrew the false "(multiples of 1/20 on GridWorld)" claim; added the fresh-draw facts ("a freshly sampled pool of 20 layouts" in flagging; metrics text now says Delta-SR "is measured on rollout sets drawn fresh each round"). Telescoping objection answered in REVIEW_LOG (fresh draws, paired per retraining). Scatter point data absent from source; residue folded into the figure pass.
- **R2-r2-4** (carried, Table 1 pooling): caption restated at source-supported strength: "mean and one standard deviation over each cell's held-out evaluation records at the budget, pooled across seeds by the evaluation pipeline (not per-seed means; per-cell record counts vary and are not seeds x episodes...)"; removed the unverifiable "per-checkpoint" specificity. Requested denominators not in source (log documents the evidence).
- **R2-r2-7** (carried, prompts/fallback/cost): declined again with re-verified evidence (workbook holds no fallback tallies or timings; no supplementary channel in this deliverable).
- **R2-r3-1** (MAJOR, robot correlations share checkpoint Delta-SR): Q3 now discloses the block sharing: the eight robot-task r values effectively relate per-checkpoint confidence to per-checkpoint improvement, ~25 outcome values per cell (5 seeds x 5 retrainings), not 100 independent ones; the <60%/>=90% claim is scoped to "In the plotted GridWorld cell". Checkpoint-level recomputation impossible (source holds only the ten r values).
- **R2-r3-2** (Figure 3 vertices vs cadence): deferred to the figure pass with evidence (no plotting sources or per-seed traces in the paper materials).
- **R2-r3-3** (pre-finetune axis label): Figure 4 caption now bridges it: "(the axis label pre-finetune names this pre-retrain loss; no fine-tuning is performed)".
- **R2-r3-4** (record unit): Table 2 caption now defines "A record is one logged evaluation of the policy's loss on a demonstration"; comparability evidence (similar per-method N within each cell) documented in the log.
- **R3-r1-6 / R3-r2-4** (figure regeneration): deferred again, unchanged evidence; all figure-level items consolidated into the pre-camera-ready figure pass.
- **R3-r3-1** (90-word Delta-SR sentence): split into three sentences (info gain + staleness; Delta-SR + rollout set; shared-checkpoint rule) plus a caveat sentence.
- **R3-r3-2** ("discovery episodes"): replaced with "the round's 60 rollout episodes on the robot tasks".
- **R3-r3-3** (nested parentheses): four \citep -> \citealp in Tasks and experts.
- **R3-r3-4** (page-1 blank band): rebutted; the band is aaai2027.sty's fixed 2.25in \titlebox (reserved camera-ready author block); style spacing modifications are forbidden.

## Space compensation (to hold 9 pages)
- Removed five single-use support citations, keeping an anchor for every claim: shinn2023reflexion, madaan2023selfrefine (failure-reasoning line keeps REFLECT + AHA), eysenbach2018leavenotrace (reset line keeps florensa2017reversecurriculum), gal2016dropout, lakshminarayanan2017ensembles (uncertainty gates keep the two menda citations).
- Tightened: bridging online/offline paragraph, Q2 opener ("its pre-retrain policy loss (protocol section)"), and the round-3 additions themselves.
- No figure was shrunk; no reviewer-required content removed.
