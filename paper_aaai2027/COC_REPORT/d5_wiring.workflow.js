export const meta = {
  name: 'd5-pusht-gridworld',
  description: 'Wire Push-T (pool_rl_robo) and GridWorld (pool_x_selector) to emit D5 per-round compute telemetry, run 1 round each (DISEIL + SafeDAgger baseline), and complete the D5 matrix',
  phases: [
    { title: 'Recon' },
    { title: 'Run' },
    { title: 'Merge' },
  ],
}

const ROOT  = '/weka/s226137394/DmNfull'
const BASE  = ROOT + '/Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4'
const ROBO  = BASE + '/pool_rl_robo'
const XSEL  = BASE + '/pool_x_selector'
const COC   = ROOT + '/paper_aaai2027/COC_REPORT'
const BUILD = COC + '/build'
const LOGS  = ROOT + '/distil/slurm_logs'

const GUARD = `HARD RULES:
 - The method is DISEIL in all prose (code identifiers p4_subtask / p4_top3_rotate / safe_dagger are code only).
 - NEVER fabricate a timing or a token count. If a quantity genuinely cannot be measured, write UNMEASURED and explain exactly why. A missing number is acceptable; an invented one is not.
 - This repository is shared and pushed to git. Instrumentation must be ADDITIVE and NON-BREAKING: do not change any existing behaviour, default, threshold or output schema. Prefer writing a new telemetry/JSON side-file over editing existing logic. If you must touch existing code, make the smallest possible additive change (e.g. record usage that is already returned by the API client) and never alter control flow.
 - BUDGET = 1 round (the author's explicit instruction). One round is enough to time a round.
 - Baseline arm = SafeDAgger, to match the three RoboSuite D5 jobs already running (which used ablation=safe).
 - Do not touch, cancel or interfere with the currently RUNNING SLURM jobs (110355-110360, the Door/Wipe D5 runs).`

const NEED = `D5 needs, per (task, modality): Baseline s/round, DISEIL s/round, VLM tokens/round, LLM tokens/round, Overhead x, KAG token contribution, Reasoning LLM tokens/round.
Note for interpretation (established from the running Door job): a round's wall-clock is dominated by the from-scratch policy RETRAIN, which BOTH DISEIL and the baseline pay. So also record, separately, the DISEIL-specific REASONING time per round (rollout analysis + VLM + reasoning LLM + prescription + feasibility), because that is the number that actually characterises the overhead. Report both: total s/round and reasoning-only s/round.`

// ============================ PHASE 1: RECON ============================
phase('Recon')
const recon = await parallel([
  () => agent(`You are the PUSH-T D5 RECON agent.
${GUARD}
${NEED}
GOAL: determine EXACTLY how to run ONE round of (a) DISEIL and (b) SafeDAgger on Push-T, IMAGE modality, in ${ROBO}, emitting per-round wall-clock and per-round VLM/LLM token counts.
READ: ${ROBO}/orchestrator/run_one.py (METHOD_SPEC), ${ROBO}/orchestrator/_common.py, ${ROBO}/run_experiment.py, ${ROBO}/config.yaml, ${ROBO}/p4_subtask/pipeline.py, ${ROBO}/p4_subtask/telemetry.py, ${ROBO}/p4/vlm.py, ${ROBO}/p4/kag.py, ${ROBO}/selection/iil_baselines.py, and the fork's LLM client (search for 'responses.create' / the OpenAI client wrapper) to see whether token USAGE is already captured.
DETERMINE:
 1. The exact command to run Push-T image with method p4_subtask (DISEIL) and with safe_dagger, for exactly ONE round (budget/max_rounds = 1). Find the right config keys (budget, max_rounds, initial_demos, heldout_n, use_vlm) and the right env (conda env / PYTHONPATH). Check ${ROOT}/.env for model config (never print secrets).
 2. Whether per-round wall-clock is already recorded (telemetry tic/toc events) and whether VLM / reasoning-LLM token usage is already recorded. The OpenAI Responses API returns a usage object; check whether the client stores it.
 3. If token usage is NOT recorded: add the SMALLEST additive instrumentation that records, per LLM/VLM call, the model name and the prompt/completion/total tokens into a side JSON file (e.g. results/<run>/telemetry/d5_tokens.jsonl). Do not change any prompt, any control flow, or any existing output. Also record the KAG block's token contribution (count the tokens of the rendered KAG text that is injected into the prompt — use the same tokenizer the API reports against, or tiktoken; if only an approximation is possible, label it clearly as an approximation).
 4. Whether a SLURM sbatch wrapper exists for this suite; if not, write one at ${ROOT}/distil/scripts/run_pusht_d5.sbatch modelled on ${ROOT}/distil/scripts/run_distil.sbatch (same partition/qos/resources pattern), parameterised by env vars, writing logs into ${LOGS}.
WRITE ${BUILD}/d5_pusht_plan.md: the exact commands, the instrumentation diff you made (if any), and the sbatch path. Do NOT submit jobs in this phase.
Return: the two exact sbatch commands (DISEIL + SafeDAgger), and whether tokens are natively logged or you instrumented them.`, {label:'recon-pusht', phase:'Recon', effort:'high'}),

  () => agent(`You are the GRIDWORLD D5 RECON agent.
${GUARD}
${NEED}
GOAL: determine EXACTLY how to run ONE round of (a) DISEIL and (b) SafeDAgger on GridWorld 5x5, IMAGE modality (the image policy is a plain CNN), emitting per-round wall-clock and per-round VLM/LLM token counts.
READ: ${XSEL}/config.yaml, ${XSEL}/config_baselines.yaml, ${XSEL}/p4/pipeline_p4.py, ${XSEL}/p4/prompts.py, ${XSEL}/p4/demo_collector.py, ${XSEL}/selection/iil_baselines.py, ${XSEL}/selection/baseline_dagger.py, and any runner/orchestrator entrypoint in ${XSEL}. Also check ${ROOT}/distil/gridworld/ (loop.py, rgb_policy.py, encoder_rgb.py) in case the consolidated module can run GridWorld image directly — if it can, prefer it and say so.
DETERMINE the same four items as the Push-T agent: (1) exact one-round commands for the DISEIL arm (p4_top3_rotate / the p4 arm) and the SafeDAgger arm, image modality; (2) whether per-round wall-clock and VLM/LLM token usage are already recorded; (3) the smallest additive token instrumentation if not (side JSON file; include the KAG token contribution); (4) an sbatch wrapper at ${ROOT}/distil/scripts/run_gridworld_d5.sbatch if none exists (GridWorld is cheap; a CPU or small-GPU partition is fine).
WRITE ${BUILD}/d5_gridworld_plan.md. Do NOT submit jobs in this phase.
Return: the two exact sbatch commands, and whether tokens are natively logged or you instrumented them.`, {label:'recon-gridworld', phase:'Recon', effort:'high'}),
])
log(`Recon done: ${recon.filter(Boolean).length}/2`)

// ============================ PHASE 2: RUN ============================
phase('Run')
await agent(`You are the D5 RUN agent. Execute the plans and collect the measurements.
${GUARD}
${NEED}
READ ${BUILD}/d5_pusht_plan.md and ${BUILD}/d5_gridworld_plan.md.
STEP 1 — SMOKE FIRST. For each of the four runs (Push-T image DISEIL, Push-T image SafeDAgger, GridWorld image DISEIL, GridWorld image SafeDAgger), do a fast local sanity check that the command starts correctly and that the telemetry/token side-file is being written. Fix any breakage before submitting. Do not proceed to sbatch until the command is known-good.
STEP 2 — SUBMIT the four SLURM jobs with BUDGET/max_rounds = 1. Job names: d5_PushT_image_full, d5_PushT_image_safe, d5_GridWorld_image_full, d5_GridWorld_image_safe. Record the job IDs.
STEP 3 — POLL squeue until they finish (they are short: GridWorld is cheap; Push-T is one round). Read the logs in ${LOGS}. If a job fails, read the error, fix it, and resubmit (up to 3 attempts per job).
STEP 4 — EXTRACT, per run: total wall-clock for the round; the DISEIL-specific reasoning time within the round; VLM tokens; reasoning-LLM tokens; prescription-LLM tokens; the KAG token contribution; and the baseline's per-round wall-clock. Compute Overhead x = DISEIL s/round divided by Baseline s/round, AND the reasoning-only overhead in seconds.
STEP 5 — WRITE ${BUILD}/d5_rows_pusht_gridworld.md and ${BUILD}/d5_rows_pusht_gridworld.csv with the two completed D5 rows (Push-T/image, GridWorld/image), the job IDs, the exact commands, and every caveat (single round; round 1 is the worst case because the policy is weakest and produces the most failures, so these values are an UPPER bound on steady-state per-round cost — state this explicitly).
If anything is genuinely unmeasurable, mark it UNMEASURED with the reason. Never invent a value.
Return: the two D5 rows and the job IDs.`, {label:'run-d5', phase:'Run', effort:'high'})

// ============================ PHASE 3: MERGE ============================
phase('Merge')
const OUT = {type:'object', additionalProperties:false, properties:{
  rows_measured:{type:'integer'}, rows_unmeasured:{type:'integer'},
  table_markdown:{type:'string'}, caveats:{type:'string'}, summary:{type:'string'}
}, required:['rows_measured','rows_unmeasured','table_markdown','caveats','summary']}

const merged = await agent(`You are the D5 MERGE agent. Assemble the complete D5_Compute matrix.
${GUARD}
${NEED}
SOURCES:
 - The three RoboSuite settings already measured by the main build: ${BUILD}/d5_compute.md and ${BUILD}/d5_compute.csv (Door/state, Door/image, Wipe/image), run with BUDGET=5. Their SLURM logs are in ${LOGS} (jobs 110355-110360).
 - The two new settings: ${BUILD}/d5_rows_pusht_gridworld.md (Push-T/image, GridWorld/image), run with BUDGET=1.
PROTOCOL RECONCILIATION (important, do this carefully): the RoboSuite rows average over 5 rounds while the new rows are a single round. Round 1 is the worst case (weakest policy, most failures, largest cluster set, highest reasoning cost). So ALSO extract the ROUND-1-ONLY values for the three RoboSuite settings from their telemetry/logs, and present BOTH:
   (i) a "round 1" column set — apples-to-apples across all five settings; and
   (ii) a "mean over 5 rounds (+/- spread)" column set for the three RoboSuite settings, which is the better estimate of steady-state cost.
State plainly which protocol each number came from. Do not silently mix them.
INTERPRETATION: make explicit that a round's wall-clock is dominated by the from-scratch policy retrain, which DISEIL and the baseline both pay, so the raw Overhead x is close to 1 and UNDERSTATES the picture; the honest characterisation is the reasoning-only add-on (seconds + tokens per round). Report both.
WRITE the final ${BUILD}/d5_compute.md (overwriting it with the complete, merged, five-row matrix + both protocols + caveats + job IDs + exact commands) and ${BUILD}/d5_compute.csv.
ALSO update the workbook: write the completed D5 matrix into the D5_Compute sheet of ${COC}/ablations_results/DISTIL_ablation_results.xlsx using pandas/openpyxl, preserving every other sheet byte-for-byte (load with openpyxl, write only the D5_Compute sheet cells, save). Verify afterwards by re-reading all sheet names and confirming none were lost.
Return the structured verdict with the final table.`, {label:'merge-d5', phase:'Merge', schema:OUT, effort:'high'})

return { merged }
