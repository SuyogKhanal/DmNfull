# Narrative Spine — PACE (AAAI-2027)

*Story-connecting audit of the six section drafts in `sections/`. This is the
single through-line the whole paper must obey, the promises each section makes and
where they get paid off, the exact transition sentences to insert, and a
prioritized list of contradictions / undefined terms / dangling claims to fix.*

Method name is **PACE = Perceive → Assess → Choose → Execute** everywhere. Stage
map (authoritative): Perceive = VLM failure perception + peak-diffusion-loss
descriptor featurization; Assess = clustering failures into modes; Choose =
diversity / k-center (coreset) selection of which failures; Execute = LLM-prescribed
corrective reset / sub-task-entry, collect demo, retrain.

---

## 1. The one-sentence spine

> Query-efficient interactive imitation learning has, for a decade, answered only
> **when** to call the expert; PACE shows that **which** failures to correct and
> **where** to begin the correction are the decisions that actually buy sample
> efficiency, and it recovers the whole DAgger family as the degenerate
> "when-only" special case.

Everything in the paper is a service to that sentence. If a paragraph does not
advance "when → when+which+where", or does not pay off the "strict generalization"
promise, or does not feed the four experimental questions (Q1 queries, Q2
where/coverage, Q3 modality+embodiment, Q4 stage ablation), it is off-spine.

## 2. The through-line, stage by stage (problem → insight → method → evidence)

1. **Problem.** BC → covariate shift → compounding error (quadratic in H).
   [intro ¶1; method Preliminaries; related §BC/DAgger]
2. **Partial fix.** IIL/DAgger aggregates labels on learner-visited states →
   error linear in H, but at the cost of expert effort → **query efficiency** is
   the axis. [intro ¶1; related §DAgger]
3. **The gap (the insight).** Every DAgger descendant is one decision rule —
   a per-state predicate `Query(s_t)=1[score ⋛ thresh]` — that answers only
   **when**. This leaves **which** (per-round redundancy), **where** (reset fixed
   to rollout terminal state), and cross-round **memory** unaddressed. Budget is
   spent on easiest-to-detect uncertainty, not on a diverse cover of failure modes.
   [intro ¶2 "DAgger-family gap"; method §Unified + "What the framework leaves on
   the table"; related §DAgger closing ¶]
4. **The method.** PACE = a per-round pipeline that treats failures as a **batch
   to curate**, not a stream to gate: Perceive (VLM + 6-D descriptor at peak-loss
   t*), Assess (cluster into modes, silhouette-k, dominant mode, cross-round
   memory), Choose (k-center/farthest-point coreset with dominant rep forced in +
   worst-loss seed), Execute (LLM SceneCommand → L2-capped, workspace-clamped
   corrective reset ξ → one expert demo → retrain). [method §PACE; algorithm box]
5. **The unifying claim.** PACE ⊃ baselines: set `|S|=1`, `C_tgt=C*`, `g=on-policy
   identity` and PACE collapses to a Diff-DAgger round (or any baseline by swapping
   the t* rule). So any gain over Diff-DAgger under the identical 1-demo/round
   budget is *attributable to which+where*. This is the paper's rhetorical
   keystone — repeated in abstract, intro, method, experiments, conclusion.
6. **The evidence.** 5 tasks × 2 modalities × 5 seeds vs 6 baselines, shared
   learner: (Q1) fewer queries to SR=0.90; (Q2) higher Eff + Cov because
   where-prescription front-loads mode coverage; (Q3) holds across state/image and
   grid/Panda/UR5; (Q4) stage ablation — removing Execute ≈ largest Eff drop,
   removing Assess ≈ largest Cov drop. [experiments Q1–Q4; conclusion]

Each link is currently present in the drafts. The connective tissue between them
(the transitions in §5 below) is what makes it read as one argument rather than six
essays.

## 3. Promises the intro makes → where each MUST be paid off

| Intro promise | Payoff site | Status |
|---|---|---|
| "when + which + where" generalization | method §Unified→§PACE; abstract; conclusion | present; keep the exact 3-word triple verbatim everywhere |
| Baselines = single predicate `Query(s_t)` (Eqs 8–13) | method §Unified (Eqs safe..stagger); related §DAgger | present |
| PACE recovers Diff-DAgger as special case (|S|=1, C_tgt=C*, identity g) | method "Baselines as special cases"; Eq.prescribe-demo | present; **must be stated identically** in all 5 places |
| VLM perceives at *peak diffusion-loss frame* = the Diff-DAgger flag | method Perceive; related §Diffusion | present — good bridge, keep it explicit |
| cross-round memory prevents re-covering solved modes | method Choose Eq.memory; experiments Q4 (implicitly) | present in method; **NOT ablated** — see gap G7 |
| k-center/coreset diversity picks non-redundant modes | method Choose Eq.diversity; experiments −Choose ablation | present |
| LLM prescribes *where* (reset/sub-task entry), not whole plan/reward | method Execute; related §LLM-for-robots | present; the "narrows generativity" framing in related is the payoff |
| 5 tasks, 2 modalities, shared learner, isolates query efficiency | experiments Task Suite + Policies + Protocol | present |
| headline: `\PH{pace-vs-diff-q-reduction}` fewer queries; `\PH{mean-pace-sr}` mean SR | experiments Main Results; conclusion | present (placeholders) |
| Q1–Q4 preview | experiments opening ¶ | present |

Two intro promises are **thin on payoff** and must be reinforced (see §6, G6/G7):
the *cross-round memory* claim and the per-task *multi-modality → uncertainty is
unreliable* claim (intro ¶2 and experiments Task Suite assert it; nothing measures
it directly — soften to "expected to help" or add to ablation discussion).

## 4. Recurring motifs to keep verbatim (consistency anchors)

Use these exact strings so the reader hears one voice:
- The triple: **"when + which + where"** (not "when/which/where", not reordered).
- The reduction clause: **"|S|=1, the target is the dominant mode/cluster, and the
  prescribe map is the on-policy identity."** (Abstract, intro, method, experiments,
  conclusion all state this — keep wording aligned; conclusion says "dominant
  cluster", method says "dominant mode" in one place and "C*=dominant cluster"
  elsewhere — pick ONE of {mode, cluster} and use it globally. Recommend "dominant
  mode" in prose, `C^\star` in math.)
- "failure-mode curation" (abstract "batch to be curated"; conclusion final line).
  Introduce it in the abstract, reuse in conclusion — do NOT invent new synonyms
  ("failure triage", "failure harvesting") midway.
- "one-demo-per-round" as the fairness invariant (method Preliminaries;
  experiments Protocol). Always call it the "shared sample-efficiency axis".
- The four stage verbs Perceive/Assess/Choose/Execute in **that fixed order**, and
  the ablation rows named exactly `−Perceive / −Assess / −Choose / −Execute`.

## 5. Transition sentences to INSERT (between and within sections)

These are the seams that currently read as hard cuts. Insert (or adapt) verbatim.

**Abstract → Intro (contributions ¶ already bridges; no change).**

**Intro → Related** (add as the first sentence of Related, before "PACE sits at…"):
> "The gap identified above — that query-efficient IIL fixes *when* but not *which*
> or *where* — is visible across every branch of the literature PACE touches; we
> review each and mark, in each, the axis PACE adds."

**Within Related, before §Diffusion Policies:**
> "The learner we adopt is itself the source of one baseline's query signal, which
> is why we treat diffusion policies not as a competitor but as shared
> infrastructure."

**Within Related, before §Active Learning:**
> "The *which* decision PACE introduces is, formally, batch active learning over
> failures — so we borrow its coreset and diversity machinery, transported from
> label acquisition to demonstration targeting."

**Related → Method** (last sentence of Related, or first of Method):
> "Having placed PACE against each thread, we now define it formally and show that
> the entire query-based family is one restriction of it."

**Within Method, §Unified → §PACE** (this is the pivot of the paper; the current
"What the framework leaves on the table" ¶ is the correct hinge — end it with):
> "PACE keeps the loop of Eq.~\ref{eq:iil-loop} unchanged and replaces only the
> single-scalar query with a four-stage pipeline; the next subsection defines each
> stage and Algorithm~\ref{alg:pace} assembles them."

**Within Method, Choose → Execute:**
> "Choose fixes *which* failure anchors the round; Execute now decides *where* the
> single corrective demonstration for that anchor begins."

**Method → Experiments** (the method already previews the headline; open
Experiments by *earning the right* to the four questions):
> "The formal reduction of §\ref{sec:method} makes the empirical question sharp:
> if PACE is Diff-DAgger plus *which* and *where*, then any measured gain isolates
> the value of those two decisions. We test this along four axes."
(This sentence also repairs the fact that Experiments currently restates Q1–Q4
without re-grounding them in the generalization claim.)

**Within Experiments, Main Results → Learning Curves:**
> "Endpoint SR and Q answer *whether* PACE is more query-efficient; the curves and
> coverage answer *why* — that the efficiency comes from front-loaded failure-mode
> coverage, the mechanism Choose and Execute were designed to produce."

**Experiments → Conclusion** (first sentence already re-states the frame; keep,
but make the ablation the bridge):
> "The stage ablation closes the argument: each of Perceive, Assess, Choose, and
> Execute carries measurable weight, so the gain is the *joint* when+which+where
> decision, not any single trick."

## 6. CONTRADICTIONS, DANGLING REFS, and UNDEFINED TERMS (fix list, prioritized)

### BLOCKERS (compile-breaking or reader-breaking)

- **G1 — Duplicate `\label{tab:tasks}` defined 3×** (intro, method ×1 each region,
  and referenced 3×). LaTeX will emit "multiply-defined labels" and `\ref{tab:tasks}`
  becomes nondeterministic. There are **two different task tables** both labeled
  `tab:tasks`: intro's (Task/Domain/Expert/Action, 4 cols) and method's (Task/Sim/
  Action/Expert, 4 cols) — and experiments has a **third** task table also labeled
  `tab:tasks` (with `dim(A)` + Diff-DAgger note + `\PH{lift-adim}` etc.).
  FIX: keep exactly ONE task-suite table (recommend the experiments one — it is the
  richest and carries the Diff-DAgger-is-robot-only caveat). Delete the intro and
  method copies; have intro/method just `\ref{tab:tasks}` it. When merged into one
  `.tex` file this is mandatory.

- **G2 — Dangling table refs `tab:main-push` and `tab:main-robosuite`** are
  `\ref`'d in **conclusion** but never `\label`'d anywhere. The actual result tables
  are `tab:push, tab:lift, tab:wipe, tab:door` (per-task). FIX: change conclusion's
  refs to `Tables~\ref{tab:push}--\ref{tab:door}` (or introduce the two grouped
  tables — but the per-task tables already exist, so just repoint the refs).

- **G3 — Dangling `\ref{sec:method}`.** Conclusion and method-internal prose
  `\ref{sec:method}` but the Method section has no `\label{sec:method}` (it labels
  `sec:unified` and `sec:pace` only). FIX: add `\label{sec:method}` right after
  `\section{Method}`.

- **G4 — Figures referenced, never defined.** `\ref{fig:lc}` and `\ref{fig:qual}`
  are used in experiments; no `\begin{figure}…\label{fig:lc/qual}` exists in any
  section. FIX: add the two figure floats (learning-curve grid `fig:lc`,
  qualitative failure→prescription `fig:qual`) with `\includegraphics` of the
  planned `figures/lc_*.pdf` and `figures/qual_*.pdf`, or the refs will read "??".

### CONTENT CONTRADICTIONS

- **G5 — Stage-name / equation-mapping mismatch inside Method.** Method's opening
  paragraph maps "Assess = clustering (Eqs.~\ref{eq:partition}--\ref{eq:dominant})"
  and "Choose = memory + diversity (Eqs.~\ref{eq:memory}--\ref{eq:diversity})".
  Good. BUT the equation *labels* are still the OLD stage names
  (`eq:perceive/partition/dominant/memory/diversity/prescribe`). That is fine as
  internal label strings (readers never see them), EXCEPT the abstract/intro/related
  prose must never surface "Partition/Prioritize/Prescribe" as PACE stage names.
  Audit result: **prose is clean** — every visible stage word is
  Perceive/Assess/Choose/Execute. The only leaks are (a) conclusion final line
  "perceives, groups, prioritizes, and corrects" (verb "prioritizes" ≈ old P3 —
  acceptable as a plain verb, but consider "…groups, selects, and corrects" to
  avoid echoing the retired stage name), and (b) related/experiments use "partition"
  as a common noun ("failure-mode partitioning") — acceptable, but be consistent:
  either always gloss it as "(Assess)" on first use or drop it. Recommend: on first
  use in Related and Experiments write "partitioning (Assess)".

- **G6 — "Dominant mode" vs "dominant cluster" vs "target".** Method uses all three;
  conclusion says "dominant cluster". Equation eq:dominant defines `C^\star`
  (dominant), eq:memory defines `C_tgt` (memory-rotated target). The reduction claim
  must say the target equals the dominant: state it as **"C_tgt = C^\star (the
  dominant mode)"** every time. Currently abstract says "the target is the dominant
  mode", intro says "target mode is the dominant mode", method says "C_tgt=C^\star",
  experiments says "C_tgt=C^\star", conclusion says "target is the dominant cluster".
  Standardize the noun to **mode** in prose.

- **G7 — Cross-round memory is claimed but never measured.** Intro promise + method
  Eq.memory + Choose ¶ all sell cross-round memory as a distinguishing axis, and
  method's "Baselines as special cases" ¶ lists "(ii) disable memory" as one of the
  three reductions. But the **ablation table has only 4 rows (−Perceive/−Assess/
  −Choose/−Execute)** — memory is folded into Choose, so the paper never isolates the
  memory contribution it advertised. FIX (choose one): (a) add a `−Memory` ablation
  row (disable cross-round rotation, keep within-round diversity), OR (b) explicitly
  state in method+experiments that "memory is evaluated as part of the Choose stage
  (−Choose disables both the k-center selection and the cross-round rotation)". (b)
  is cheaper and honest; do (b) unless a memory-off run exists.

- **G8 — Expert count / "six baselines" arithmetic.** Abstract & intro say
  "six baselines". Roster = SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger,
  Random, Diff-DAgger = 6 — but **Diff-DAgger is robot-only (not on Toy)**. So on
  T1 there are only 5 baselines. FIX: say "up to six baselines (Diff-DAgger on robot
  tasks only)" on first mention, and keep the Toy table's caption note that
  Diff-DAgger is omitted (it already has this — good).

- **G9 — HG-DAgger / LazyDAgger / DART / IWR / Sirius / AggreVaTe(D) cited in
  Related but NOT in any results table.** This is fine (they are context, not run
  baselines) but the reader may expect them. FIX: one sentence in Related or
  Experiments Baselines: "We compare empirically against the query-efficient,
  robot-gated subset that shares our one-demo-per-round budget; human-gated
  (HG-/Lazy-DAgger), noise-injection (DART), and deployment-time reweighting
  (IWR, Sirius) methods are discussed but not run, as they assume a different
  interaction budget." Prevents a reviewer "why isn't X a baseline?" ding.

### UNDEFINED / USED-BEFORE-DEFINED TERMS

- **G10 — "KAG" appears undefined.** Method Execute (`cmd=VLM/LLM(A,frames,KAG)`),
  algorithm box, and conclusion Limitations ("hard KAG workspace clamps") all use
  KAG with no expansion. It is never spelled out in any section. FIX: on first use
  in method Perceive, write "a per-task knowledge-augmented-generation (KAG) context
  — a knowledge graph of workspace bounds and a failure taxonomy". Method Perceive
  already describes "a per-task knowledge graph of workspace bounds and failure
  taxonomy" but never names it KAG at that point; bind the acronym there.

- **G11 — "tee", "TCP", `p^{tee}`, `q^{full}` used before defined in prose.**
  Method Execute and Eq.phi use `p^{tee}` (T-block pose) and `p^{tcp}` (end-effector)
  and `q^{full}`; these are PushT-specific and only defined in the equations.tex
  symbol table, not in section prose. The 6-D descriptor φ is PushT-shaped
  (tee_x, tee_y, sinθ, cosθ, ρ, δ) but the paper claims 5 tasks. FIX: in method
  Perceive, add one sentence generalizing: "For manipulation tasks the object pose
  `p^{tee}` and end-effector pose `p^{tcp}` instantiate the descriptor; for the toy
  grid the analogue is the agent/goal cell geometry, and for Lift/Wipe/Door the
  manipulated-object pose." Otherwise a reader on the Door task cannot parse φ.
  (This also quietly handles the dossier fact that only Push is fully instantiated
  and T3–T5 are upcoming — do NOT say "upcoming" in the paper, but DO make φ
  task-agnostic in wording.)

- **G12 — `\PH{drop-N}` in experiments vs concrete "N" elsewhere.** Experiments
  Baselines writes "DropoutDAgger N=\PH{drop-N}/p=0.9". Dossier default is N=10
  (robot) / mc_N=16 (maze). Keeping it a placeholder is consistent with the rules
  (never invent a number) — acceptable, but note p=0.9 IS hard-typed. Either make p
  a placeholder too (`\PH{drop-p}`) or accept p=0.9 as a fixed hyperparameter, not a
  result (it is a config, not a measured number — fixed config values are allowed).
  Recommend: leave p=0.9, α=0.99, K=1, M=5, τ=0.1, α_h=0.1 as literal config
  constants (they are settings, not results); this is the correct reading of the
  \PH rule. Flag only for internal consistency — do NOT \PH-ify config constants.

### DANGLING / UNSUPPORTED CLAIMS (soften or support)

- **G13 — Abstract "broadest demonstration coverage" and "highest held-out success
  rate".** These are asserted as facts; results are pending. Since numbers are \PH,
  the *comparative* claim ("highest", "broadest") is an un-evidenced superlative.
  FIX: the abstract is allowed forward-looking framing, but soften to "and, in our
  experiments, the highest held-out success rate and broadest coverage among the
  compared methods" — or ensure every such superlative is backed by a \PH in the
  tables (Cov currently only has `push-st-pace-cov` and `toy-st-pace-cov` \PH's; the
  "broadest coverage" claim needs at least the baseline Cov columns to compare
  against — see G14).

- **G14 — Coverage (Cov) claim lacks baseline comparators in the tables.** Method
  Eq.cov and experiments both promise Cov "above the strongest baseline", and the
  ablation table has a Cov column, but the **main results tables (toy/push/…)** have
  only SR and Q columns — no Cov, no Eff. So the headline "broadest coverage" /
  "higher demo efficiency" claims have nowhere to live except two loose inline \PH's.
  FIX: either (a) add Eff and Cov columns (or a companion table) so the coverage/
  efficiency claims are table-backed, or (b) restrict the prose claims to the two
  inline \PH points that exist (push-st, toy-st) and the ablation table. Reviewers
  will look for the Cov numbers behind "broadest coverage".

- **G15 — "first IIL method to make which+where first-class" (intro ¶ end).** Strong
  novelty claim. It is defensible against the run baselines, but IWR/Sirius
  (prioritize interventions) and reverse-curriculum/leave-no-trace (reset states)
  are adjacent. FIX: qualify — "the first IIL method to make *which failures* and
  *where to correct them* jointly first-class, learned per-round decisions under a
  fixed query budget." The words "jointly" and "per-round under a fixed budget"
  fence off the adjacent work.

- **G16 — Experiments "$-$Execute … recovers a Diff-DAgger-like special case".**
  Consistent with the generalization theorem — good, keep. But note −Execute alone
  (with Assess+Choose still on) is NOT exactly Diff-DAgger (that also needs |S|=1
  and C_tgt=C*). Say "−Execute reverts the *where* decision to the on-policy
  intervention point (the Execute-side restriction of the reduction)" to avoid
  implying the single ablation equals the full reduction.

- **G17 — Numeric hyperparameters in Method equations (γ=0.6, σ_mem=0.06, λ=1,
  Δ_max=0.06, θ_max=0.4, κ=3) are hard-typed.** These are config constants, not
  results — allowed. But they are PushT-tuned (0.06 m caps). For a 5-task paper,
  state once that "(caps are given for Push; per-task values are in the appendix/
  config)" so a Door reader does not think 0.06 m applies to a door hinge.

## 7. Quick "is the argument closed?" checklist (for the assembling agent)

- [ ] The triple "when + which + where" appears in abstract, intro, method pivot,
      experiments opener, conclusion — identical wording.
- [ ] The reduction clause (|S|=1, C_tgt=C*, identity g) appears identically in all
      5 sections; noun standardized to "dominant mode".
- [ ] One and only one `tab:tasks`; conclusion result refs repointed to
      `tab:push`..`tab:door`; `sec:method` labeled; `fig:lc`/`fig:qual` floats added.
- [ ] KAG expanded on first use; φ made task-agnostic in prose (tee/tcp glossed).
- [ ] Memory contribution either ablated (−Memory row) or explicitly folded into
      −Choose in text.
- [ ] Cov/Eff superlatives are table-backed (add columns) or scoped to the existing
      inline \PH points.
- [ ] "six baselines" qualified as "Diff-DAgger robot-only"; non-run related methods
      given a one-line "discussed not run" note.
- [ ] Config constants (α, K, M, τ, γ, caps, κ) are NOT \PH-ified; only measured
      quantities are \PH.
