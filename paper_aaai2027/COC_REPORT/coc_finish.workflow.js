export const meta = {
  name: 'coc-finish',
  description: 'Insert the completed D5 compute analysis (+figure) into the CoC report, then run the final 14-point validation gate',
  phases: [ { title: 'D5Insert' }, { title: 'Validate' } ],
}

const ROOT   = '/weka/s226137394/DmNfull'
const COC    = ROOT + '/paper_aaai2027/COC_REPORT'
const BUILD  = COC + '/build'
const GFIG   = COC + '/figures_generated'
const XLSX   = COC + '/ablations_results/DISTIL_ablation_results.xlsx'
const REPORT = COC + '/CoC_Report.md'

const FACTS = `NAMING: the method is DISEIL everywhere in prose. DISTIL / PACE / P4 must appear NOWHERE in the report (the workbook and code identifiers may still say DISTIL/p4_* — that is fine, but never carry those names into the report text).
Acronym derivation (one letter per title word): **D**emonstration d**I**stillation for **S**ample-**E**fficient **I**mitation **L**earning -> DISEIL.
STYLE: obey ${COC}/Non-AI content.md strictly — no AI vocabulary (delve, pivotal, crucial, robust, leverage, showcase, underscore, comprehensive, nuanced), no filler connectives (Moreover/Furthermore/Additionally), minimal em dashes, no negative parallelism, no rule-of-three padding, no copula avoidance, no "-ing" tack-on clauses, no formulaic conclusions. Formal academic tone, varied sentence rhythm.
HONESTY: never invent a number. Every value must come from ${BUILD}/d5_compute.md (which traces to real SLURM jobs).
EDIT DISCIPLINE: modify ${REPORT} ONLY with targeted Edit calls. NEVER overwrite the whole file with Write.`

// ---------------- Phase 1: put D5 into the report ----------------
phase('D5Insert')
await agent(`You are the COMPUTE-COST agent. The CoC report is missing its computational-cost analysis: the D5 SLURM jobs finished only after the section writers had run. Add it now.
${FACTS}
READ: ${BUILD}/d5_compute.md (the authoritative, complete five-setting matrix with both protocols, caveats and job-ID provenance), and the D5_Compute sheet of ${XLSX}.
Then read ${REPORT} and find the ablation chapter (the diagnostics/ablation part of Aim 1, where D1-D5 style diagnostics belong).

STEP 1 — FIGURE. Generate a publication-quality figure at ${GFIG}/F16_compute_cost.pdf (+ .png) with matplotlib (Agg). It must show, per setting, the decomposition of a round's wall-clock into (a) the shared train+eval cost that BOTH arms pay and (b) the DISEIL-specific reasoning add-on — a stacked or grouped horizontal bar works well; annotate the Overhead x. Use only Protocol P1 (first round) so all five settings are comparable, and say so in the caption. Colourblind-safe, no chartjunk, >=9pt at print size. Write the generating script to ${GFIG}/make_compute_figure.py so it is reproducible.

STEP 2 — TEXT. Insert a subsection (e.g. "### Computational cost of the reasoning pipeline") into the ablation chapter of ${REPORT} with targeted Edit calls. It must contain:
 - Motivation: per-round reasoning cost is listed as a limitation, so quantify it before a reviewer asks.
 - Setup: five settings; baseline arm = SafeDAgger; seed 1; the RoboSuite settings ran a five-round budget, Push-T and GridWorld a single round; measured from SLURM job telemetry (give the job IDs).
 - The Protocol P1 matrix (all five settings) as a table, and the P5 mean +/- SD for the three RoboSuite settings, clearly labelled as two different protocols that must not be mixed.
 - The central finding, stated plainly: a round's wall-clock is dominated by the from-scratch policy retrain and the held-out evaluation, and BOTH arms pay all of it. The raw Overhead x (1.13x-2.75x) therefore UNDERSTATES the cost of the reasoning stack, because the large shared denominator dilutes it. The honest number is the reasoning-only add-on that the baseline never spends: +63 s (GridWorld) to +1,232 s (Push-T) per round, and 9.6k-82k tokens per round.
 - Why Push-T is the outlier (its screening rollout and its far larger VLM/LLM token draw), and why GridWorld is cheap in seconds yet still spends ~9.7k tokens, with the KAG block accounting for 54% of its prompt budget.
 - The caveats, stated explicitly and not buried: single seed; P1 is the FIRST round and therefore an UPPER bound on steady-state cost (weakest policy -> most failures -> largest cluster set -> most calls); token counts are NOT comparable across rows because the backends differ (OpenRouter qwen3-32b for Door/Wipe/GridWorld, local vLLM qwen3-32b + qwen3-vl-32b for Push-T); the SD in P5 is a round-to-round spread within one run, not a seed-to-seed spread.
 - Implications for the framework: the LLM stack runs only at demonstration-selection time and never in the control loop, so the resource actually traded is GPU inference against expert demonstration time. Tie it to the limitation already stated in the report.
 - Reference the new figure.
Number the new figure and table consistently with the rest of the document and fix any downstream numbering it disturbs.

STEP 3 — Also correct any place in the report that previously said the per-round cost was unmeasured or pending.
Return: the subsection heading you inserted, the figure/table numbers used, and a 6-line summary of the finding.`, {label:'d5-insert', phase:'D5Insert', effort:'high'})

// ---------------- Phase 2: final validation ----------------
phase('Validate')
const GATE = {type:'object', additionalProperties:false, properties:{
  passed:{type:'boolean'}, words:{type:'integer'}, figures:{type:'integer'}, tables:{type:'integer'}, refs:{type:'integer'},
  em_dashes:{type:'integer'}, distil_hits:{type:'integer'},
  checklist:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    item:{type:'string'}, pass:{type:'boolean'}, note:{type:'string'}
  }, required:['item','pass','note']}},
  outstanding:{type:'string'}, summary:{type:'string'}
}, required:['passed','words','figures','tables','refs','em_dashes','distil_hits','checklist','outstanding','summary']}

const gate = await agent(`You are the FINAL VALIDATION AGENT for the Confirmation of Candidature report at ${REPORT}.
${FACTS}
Run the author's mandatory checklist. For EACH item report pass/fail WITH EVIDENCE (a grep result or a document location). Fix trivial failures yourself with targeted Edit calls; report anything you cannot fix.
 1. DISEIL used everywhere.
 2. DISTIL removed entirely (also PACE / P4). Report the exact hit count.
 3. Updated teaser figure used (Teaser_Diagram.pdf).
 4. Updated architecture used ("Architectural Diagram.pdf" — the one showing the policy-solvability loop "Solvable => Revise P", not an old infeasibility loop).
 5. Learning curves for all five tasks included (all_5_task_comparison.pdf).
 6. Information-gain discussion updated: pre-retrain loss argument, starting performance, initial demonstrations, and why the initial demonstration count was chosen to place the starting success rate in a target range.
 7. Initial-demonstration discussion present.
 8. Representative prompts included.
 9. Representative KAG examples included, as structured key-value environmental constraints.
 10. Equation 10 presented as feasibility verification: LLM proposes -> constraints retrieved from KAG -> feasibility check -> feedback to the LLM on violation -> revised prescription until feasible.
 11. Cluster naming explained.
 12. Humanized writing applied: report the em-dash count and any surviving AI-tell vocabulary.
 13. References verified: every citation resolves to ${BUILD}/references_coc.bib; nothing fabricated; report the count.
 14. Cross-references consistent: figures/tables/equations numbered continuously and every reference resolves.
ALSO verify: comparison tables label the DAgger family explicitly; ablations are confined to the three primary settings (GridWorld image, Push-T state, Door image) with the rest deferred to supplementary; the Lift-at-ceiling caveat is present wherever Lift appears; the clustering is described as geometric for every run with NO surviving R3M/PCA branch; B and D are presented as framework parameters with B=20 / D=1 as the validated instance; the Gantt chart is present and consistent with the November 2028 thesis submission; the cover page carries the A2I2 logo and every required field; the D5 compute analysis is now present; and Aims 1 -> 2 -> 3 connect as one programme.
Write ${COC}/COC_BUILD_REPORT.md: the checklist with evidence, document statistics, the D5 status, and a clear list of anything the AUTHOR must still supply or decide.
Return the structured verdict.`, {label:'final-validation', phase:'Validate', schema:GATE, effort:'high'})

return { gate }
