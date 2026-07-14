# Table 8 (rebuilt) — Per-round compute: **DISEIL** vs **Diff-DAgger** baseline

Replaces SafeDAgger with **Diff-DAgger** as the baseline arm. The existing `D5_Compute` sheet
(SafeDAgger) is untouched; this lands as a new sheet `D5_vs_DiffDAgger`.

`full` = DISEIL. Baseline = `diffdagger` (`distil/config.py:160` `BASELINE_ARMS` →
`distil/diffdagger.py::run_diffdagger`). Seed 1. Backend: **OpenRouter**
(VLM `qwen/qwen3-vl-30b-a3b-instruct`, LLM `qwen/qwen3-32b`).

- **P1** = the run's FIRST round (round 0).
- **P5** = mean ± **sample SD** (ddof=1) over the LLM-active rounds **0..4**.

---

## ⚠️ How to read this table

**1. Do not quote the Overhead × alone. It is close to 1 and it UNDERSTATES the cost of
reasoning.** Both arms retrain the diffusion policy from scratch and evaluate on the same fixed
100-episode held-out set every round. That shared work dominates the denominator, so the ratio
is small almost regardless of what the reasoning costs.

**2. Wall-clock seconds carry a large measurement noise floor** (~±20 %, quantified in
§Measurement validity). They are real seconds, but a cross-arm *difference* of a few tens of
seconds is not resolvable.

**3. The statistic that survives both problems is the WITHIN-RUN SHARE** — what fraction of its
own round each arm spends on its own decision machinery. Any multiplicative slowdown (a busier
GPU, a different node) cancels in a ratio taken inside a single run:

| Setting | DISEIL: screening + VLM/LLM + prescription, as a share of its own round | Diff-DAgger: uncertainty gate, as a share of its own round |
|---|---:|---:|
| Door/state | **30.3 % ± 2.7** (n=5) | 16.7 % ± 14.2 (n=5) |
| Door/image | **23.0 % ± 2.4** (n=5) | 7.9 % ± 3.7 (n=5) |
| Wipe/image | **35.1 % ± 4.6** (n=5) | 11.4 % ± 1.0 (n=2) |

**DISEIL spends 23–35 % of each round on reasoning; Diff-DAgger spends
8–17 % on its gate.** Everything else is the retrain+eval both arms pay. That
gap — not the ratio — is the cost of the method.

---

## BLOCK 1 — Protocol P1 (first round, round 0)

### Wall-clock

| Setting | Diff-DAgger s/round | DISEIL s/round | Overhead × | **Reasoning-only add-on (s)** |
|---|---:|---:|---:|---:|
| Door/state | 734.0 | 1054.4 | ×1.437 | **258.0** |
| Door/image | 1065.0 | 1474.9 | ×1.385 | **281.0** |
| Wipe/image | 1652.0 | 2194.7 | ×1.329 | **527.0** |

### Where the seconds go

| Setting | Shared train+eval (DISEIL) | Shared train+eval (Diff-DAgger) | DISEIL screening | DISEIL analysis+prescription | DISEIL-specific total | Diff-DAgger gate/screen |
|---|---:|---:|---:|---:|---:|---:|
| Door/state | 783.0 | 721.0 | 205.0 | 66.0 | 271.0 | 13.0 |
| Door/image | 1180.0 | 1052.0 | 201.0 | 93.0 | 294.0 | 13.0 |
| Wipe/image | 1491.0 | 1475.0 | 431.0 | 273.0 | 704.0 | 177.0 |

### Tokens per round

| Setting | VLM tok | LLM tok | Reasoning-LLM tok | KAG contribution | Diff-DAgger (all token cols) |
|---|---:|---:|---:|---:|---:|
| Door/state | 3258 | 8253 | 3486 | 3195 | **0 (by construction)** |
| Door/image | 3293 | 8086 | 3004 | 3195 | **0 (by construction)** |
| Wipe/image | 3228 | 6332 | 2303 | 2392 | **0 (by construction)** |

## BLOCK 2 — Protocol P5 (mean ± sample SD over LLM-active rounds 0..4)

### Wall-clock

| Setting | Diff-DAgger s/round | DISEIL s/round | Overhead × | **Reasoning-only add-on (s)** |
|---|---:|---:|---:|---:|
| Door/state | 668.8 ± 58.5 | 783.0 ± 183.7 | ×1.171 | **126.2 ± 124.4** |
| Door/image | 1094.8 ± 68.8 | 1169.3 ± 194.8 | ×1.068 | **182.6 ± 35.2** |
| Wipe/image | 1551.5 ± 142.1 | 2011.8 ± 173.3 | ×1.297 | **503.0 ± 33.9** |

### Where the seconds go

| Setting | Shared train+eval (DISEIL) | Shared train+eval (Diff-DAgger) | DISEIL screening | DISEIL analysis+prescription | DISEIL-specific total | Diff-DAgger gate/screen |
|---|---:|---:|---:|---:|---:|---:|
| Door/state | 547.8 ± 146.5 | 560.2 ± 129.8 | 158.6 ± 41.1 | 76.2 ± 9.8 | 234.8 ± 43.7 | 108.6 ± 91.8 |
| Door/image | 901.6 ± 169.6 | 1010.2 ± 95.9 | 181.0 ± 25.8 | 86.2 ± 14.2 | 267.2 ± 37.2 | 84.6 ± 34.9 |
| Wipe/image | 1302.6 ± 105.7 | 1375.0 ± 141.4 | 414.8 ± 20.0 | 294.4 ± 158.9 | 709.2 ± 141.5 | 176.5 ± 0.7 |

### Tokens per round

| Setting | VLM tok | LLM tok | Reasoning-LLM tok | KAG contribution | Diff-DAgger (all token cols) |
|---|---:|---:|---:|---:|---:|
| Door/state | 3285 ± 22 | 7644 ± 573 | 2712 ± 630 | 3195 ± 0 | **0 (by construction)** |
| Door/image | 3292 ± 9 | 7495 ± 838 | 2561 ± 917 | 3195 ± 0 | **0 (by construction)** |
| Wipe/image | 3247 ± 13 | 7551 ± 1632 | 3657 ± 1715 | 2392 ± 0 | **0 (by construction)** |

---

## Measurement validity — read before using the seconds

### Hardware matching (a confound that was found and fixed)

A first pass ran all six Diff-DAgger jobs on one node (`a100-m-01`), 4–5 concurrent, while the
reused DISEIL runs had executed a day earlier on `a100-m-02`, `a100-m-03` and — for Wipe/image —
**an H100 (`h100-m-12`)**. That is a different GPU class, and it made DISEIL look *faster than the
baseline* on Wipe/image (×0.893). The tell: round-0 **training** is provably identical work in both
arms (same demos, same windows, same 8000 steps, same architecture — all printed in the log), yet
it took 479 s for DISEIL and 883 s for Diff-DAgger. Training time cannot depend on the arm.

**Every Diff-DAgger run in this table was therefore re-run, pinned to the same node its DISEIL
counterpart used**, at the same 2-jobs-per-node co-tenancy:

| Setting | DISEIL ran on | Diff-DAgger re-run pinned to | matched? |
|---|---|---|---|
| Door/state | `a100-m-02` | `a100-m-02` | ✅ both budgets |
| Door/image | `a100-m-03` | `a100-m-03` | ✅ both budgets |
| Wipe/image | `h100-m-12` | `h100-m-12` | ✅ BUDGET=5 only (see below) |

The superseded runs are preserved at `distil/results/_compute_confounded/` rather than deleted.

**Effect of the fix (this is not a cosmetic correction).** With the confounded runs the P1
overheads read ×1.109 / ×1.254 / **×0.893**; on matched hardware they read
**×1.437 / ×1.385 / ×1.329**. The sub-1.0 ratio — DISEIL apparently *cheaper than
the baseline* — was entirely an H100-vs-A100 artifact and has disappeared. Every ratio is now
above 1, as it must be.

**⚠️ One gap, stated rather than papered over.** The hardware-matched **Wipe/image BUDGET=20**
baseline could not be scheduled (the H100 partition's queue put it ~9 h out). Its P5 baseline
therefore falls back to the **hardware-matched BUDGET=5 run, which has only 2 rounds** — so the
Wipe/image P5 baseline is an n=2 spread, not n=5, and is labelled as such in the CSV
(`baseline_source`). A BUDGET=20 Wipe/image run **does** exist on an A100
(`_compute_confounded/`), but its seconds are **not** comparable to an H100 DISEIL run, so it is
deliberately NOT substituted here. Job `110518` remains queued; when it lands, re-running
`build_table8_diffdagger.py` + `write_table8_md.py` fills this cell automatically.

### The noise floor, measured

Round 0 is **identical work** in the BUDGET=5 and BUDGET=20 baseline runs (same seed, same
bootstrap, same 8000 steps — the budget only changes when the loop *stops*). So the disagreement
between those two runs on round 0 is a direct, empirical measure of run-to-run timing noise:

| Setting | round-0 train_s (b5 / b20) | eval_s | gate_s | round total |
|---|---|---|---|---|
| Door/state | 229 / 229 | 492 / 522 | 13 / 14 | 734 / 765 |
| Door/image | 581 / 516 | 471 / 533 | 13 / 57 | 1065 / 1106 |
| Wipe/image | — | — | — | — |

**Treat differences smaller than this spread as noise.** It is the honest resolution limit on
every second in this table, and it is why the within-run share (top of page) is the primary
statistic.

### The decomposition is exhaustive

For DISEIL, `train + eval + screen + llm + prescribe` is reconstructed purely from `run.log`
timestamps, while the round total comes from `result.json` `history[i].sec` — two independent
records. They agree to within **±1 s per round** (the 1-second granularity of the log timestamps),
so there is no hidden or unattributed cost bucket. Diff-DAgger's residual is exactly 0 because its
total *is* the log span.

---

## Column definitions

- **Shared train+eval** — retrain the diffusion policy from scratch + evaluate on the fixed
  100-episode held-out set. **Both arms pay this.** It is measured as `[train]`→`[calibrate]`
  (which therefore **includes** the diffusion-loss CDF calibration that both arms run inside
  `train_and_calibrate` — it is counted **once**, inside train, not as a separate column) plus
  `[calibrate]`→`[eval]`. It is the same *protocol* for both arms, **not the same constant**: eval
  wall-clock depends on how many episodes terminate early.
- **DISEIL screening** — the 40-episode failure screen (`[eval]` → `[screen]`).
- **DISEIL analysis+prescription** — clustering + VLM + reasoning LLM + prescribed-demo collection
  (`[screen]` → `[collect a0]`).
- **Diff-DAgger gate/screen** — the arm's own when-to-query cost: the uncertainty-gated DAgger
  rollouts and the expert interventions they trigger (`[eval]` → last `[dagger ep]`).
- **Reasoning-only add-on** = DISEIL-specific − Diff-DAgger gate, computed **per round** then
  aggregated (not mean-minus-mean), so its SD is meaningful.
- **VLM tok** = `by_stage.vlm.total`. **LLM tok** = `by_stage.analysis.total + by_stage.decision.total`.
  **Reasoning-LLM tok** = `tokens.reasoning` (thinking tokens, a subset of completion).
  **KAG contribution** = `n_analysis_calls × analysis_kag_delta + n_decision_calls × decision_kag_delta`,
  from the measured paired with/without-KAG prompt-token diff in `build/kag_tokens.json`
  (Door 798/801, Wipe 598/598).

**Diff-DAgger's token columns are 0 BY CONSTRUCTION, not missing data.** The arm queries a
diffusion-loss CDF quantile (`alpha=0.99`) and never calls a foundation model — `run_diffdagger`
cannot reach `p4/llm.py`, and its `run.log` contains no LLM-client line at all.

---

## CAVEATS — stated, not buried

1. **SINGLE SEED (seed 1).** The P5 `±` is the **round-to-round spread WITHIN one run**, not a
   cross-seed error bar, and must not be read as one. Several SDs exceed their own mean.
2. **WALL-CLOCK NOISE.** See §Measurement validity. Cross-arm second-differences below the noise
   floor are not resolvable, even though every second is individually real and logged.
3. **THE RATIO UNDERSTATES THE COST — never report it alone.** Use the reasoning-only add-on and
   the within-run share.
4. **P1 is the FIRST round; it is an upper bound on the *screening* cost, not on everything.**
   The add-on is larger at P1 than at P5 in **3 of 3** settings
   (Door/state: 258→126 s, Door/image: 281→183 s, Wipe/image: 527→503 s).
   It is **not** a universal upper bound.
5. **DISEIL's TOKEN cost is FIXED per round, not policy-dependent.** Every round of every run makes
   exactly **7 LLM/VLM calls** (3 VLM + 3 analysis + 1 decision), because `p4.analyze_cap=3` caps
   the number of failures analysed. The token columns therefore barely move across rounds. What
   *does* fall as the policy improves is the **screening seconds** (Door/state 205 → 105 s), because
   a better policy's episodes succeed and terminate early, so 40 screen rollouts finish sooner.
6. **BUDGET ASYMMETRY, and the P5 baseline is matched on ROUND INDEX ONLY.** DISEIL adds **1** demo
   per round; Diff-DAgger adds `interventions_per_round=4`. The loop stops at
   `final_demos = n_init + budget`, so at BUDGET=5 DISEIL runs 5 LLM-active rounds while
   Diff-DAgger stops after **2** (`[stop] reached final_demos=9`). P5's baseline therefore comes
   from a BUDGET=20 run, which yields exactly rounds 0..4. **But the two arms then hold different
   amounts of data at the same round index** — DISEIL's rounds 0..4 hold 4,5,6,7,8 demos (Door)
   while the baseline's hold 4,8,12,16,20. The baseline is doing *more* interaction per round, which
   inflates its gate cost and so **shrinks** the measured add-on. The add-on at P5 is, in that
   sense, a conservative (lower) estimate.
7. **BACKEND.** All DISEIL runs used **OpenRouter** (hosted). Token counts are **not comparable
   across backends** — a local vLLM deployment would tokenise and account differently. Diff-DAgger
   uses no backend at all.
8. **Every second traces to a printed, timestamped event in a run that COMPLETED on this cluster.**
   Nothing is estimated. Diff-DAgger's `history` carries no per-round `sec`
   (`baselines.py::_wrap_history`), so its round wall-clock is derived from `run.log` timestamps
   (`[train]` → last `[dagger ep]`); DISEIL's comes from `result.json` `history[i].sec`. **Precision
   is not accuracy** — see caveat 2.
9. **Scope.** Push-T and GridWorld are **out**: Diff-DAgger is a diffusion-loss rule and does not
   apply to the GridWorld CNN/MLP policies (`GT_SR` marks it `–` there), and Push-T was only run at
   BUDGET=1.

---

## What these runs are — and are NOT

**These are short compute-measurement runs, not performance runs, and their success rates are
low.** The DISEIL runs timed here are BUDGET=5 (five added demos) and finish at:

| Setting | DISEIL final SR (BUDGET=5) | Diff-DAgger final SR (BUDGET=20) | Diff-DAgger's published `GT_SR` (B=20, 5 seeds) |
|---|---:|---:|---:|
| Door/state | 0.88 | 0.93 | 95.2 ± 4.3 |
| Door/image | 0.73 | 0.48 | 89.2 ± 3.5 |
| Wipe/image | 0.12 | n/a (b20 not scheduled) | 89.6 ± 3.2 |

Two things must be said plainly about this table:

- **The Wipe/image policies barely solve the task at all** (DISEIL ends at 0.12; its per-round eval
  SR never exceeds 0.15). The Wipe/image timings are therefore measured on a near-failing policy.
  That is legitimate for a *cost* measurement — the pipeline still runs every stage — but it means
  those seconds do not describe a working system.
- **The baseline reproduces its published SR on Door/state but NOT elsewhere.** At BUDGET=20 it
  reaches 0.93 on Door/state (published 95.2 ± 4.3) — a match — but only 
  0.48 on Door/image (published 89.2 ± 3.5); and an earlier A100 Wipe/image BUDGET=20 run reached
  just 0.22 against a published 89.6 ± 3.2. **One setting validates the arm; the others do not.**
  We do NOT claim the baseline is globally validated, and the Wipe/image baseline in particular is
  far below its published value at the same budget.

**No success-rate claim is made from this table.** The DISEIL runs (BUDGET=5) and the P5 baseline
runs (BUDGET=20) had **different demonstration budgets**, so their final SRs are not comparable and
must not be read as a head-to-head. The method-vs-baseline SR comparison lives in `GT_SR`, a
different experiment, and is deliberately **not** imported here.

---

## Interpretation

**Reasoning costs DISEIL roughly a fifth to a third of each round; the rest is the retrain+eval
that both arms pay anyway.**

- DISEIL spends **23–35 %** of each round on screening + VLM/LLM + prescription.
  Diff-DAgger spends **8–17 %** on its uncertainty gate. The net cost of reasoning
  is the difference, and it is the number to quote — it is invariant to which GPU ran the job.
- In absolute seconds the reasoning-only add-on is roughly **a few hundred seconds per round**, on
  top of a shared retrain+eval of many hundreds to ~2000 s. That is why the raw ratio sits near 1:
  the denominator is expensive, not the numerator cheap.
- **The add-on is not a fixed tax.** It falls from P1 to P5 on Door (the screen gets cheaper as the
  policy stops failing) but the *subtrahend* also moves: Diff-DAgger's gate **grows** across rounds
  on Door/state (13 → 326 s), because a stronger policy trips the OOD threshold less often and the
  arm must roll out more episodes to harvest its 4 interventions (4 → 22 DAgger episodes). On
  Door/image the gate grows for a *different* reason — the episode count stays at 4 but the episodes
  get longer — and on Wipe/image it is flat at ~255 s (always exactly 4 episodes; that policy is
  weak enough that essentially every rollout queries. **A single mechanism does not explain all
  three.** Much of the apparent 'fall' in the Door add-on is the baseline's gate getting more
  expensive, not DISEIL getting cheaper.
- **Token cost is dominated by the LLM, not the VLM**: 6.3–8.3k LLM tokens/round
  vs 3.2–3.3k VLM, of which 2.3–3.7k are reasoning tokens. And it is
  **fixed per round** (7 calls, always), so it does not amortise away as the policy improves.
- **The knowledge graph is a large, constant slice of the prompt**: 3195 tok/round on Door
  (**41 %** of the round's prompt budget) and 2392 on Wipe
  (**35 %**). If prompt tokens ever become the binding cost, the KAG is the first
  place to look.

