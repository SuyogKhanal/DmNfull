export const meta = {
  name: 'coc-revision-round3',
  description: 'New ablation run: rebuild Table 7 (mean+-SE, Ni, init SR) and Table 8, drop in new Fig 5 and Fig 7, strip Cluster Memory from Fig 2, add SE error bars to all ablation figures (fix Fig 10 A6), sweep DISTIL->DISEIL, rebuild and audit',
  phases: [ { title: 'Data' }, { title: 'Figures' }, { title: 'Text' }, { title: 'Build' } ],
}

const COC    = '/weka/s226137394/DmNfull/paper_aaai2027/COC_REPORT'
const BUILD  = COC + '/build'
const V2     = BUILD + '/v2'
const GFIG   = COC + '/figures_generated'
const FIGS   = '/weka/s226137394/DmNfull/paper_aaai2027/figures'
const SHEETS = COC + '/ablations_results/sheets'
const UPD    = COC + '/updated_figures'
const SPEC   = BUILD + '/update_spec.md'

const CORE = `BINDING SPEC: ${SPEC}. Read it fully first; it contains the authoritative Table 7 (already computed) that you must reproduce exactly.
Method is DISEIL (never DISTIL/PACE/P4 in prose, captions, legends or column headers). Data source of truth: ${SHEETS}/*.csv (the new run).
SE rule: SE = std / sqrt(n), n=5 for robot tasks (Push-T, Lift, Wipe, Door), n=9 for GridWorld. Round SE to one decimal (pp). Error bars are SE, minimal.
The report is generated: ${COC}/CoC_Report.md is assembled from ${V2}/*.md by ${BUILD}/assemble.py, then built by ${COC}/build_pdf.sh. EDIT THE CHAPTER FILES in ${V2}/, never CoC_Report.md. Modify files with targeted Edit calls.
NEVER invent a number; every value traces to a CSV. Style: ${COC}/Non-AI content.md.`

// ===================== PHASE 1: DATA =====================
phase('Data')
await agent(`You are the DATA agent. Turn the new ablation CSVs into tidy, SE-augmented data and the exact Table 7 and Table 8 markdown.
${CORE}
1. Parse every CSV in ${SHEETS}/ (they have messy "Unnamed" columns and note rows above the real header row that begins "Task,Obs,..."). For each sheet, extract the tidy table of per-method mean and std, and compute SE = std/sqrt(n) with the correct per-task n. Relabel DISTIL -> DISEIL in every column header/label.
2. WRITE a single clean machine-readable file ${GFIG}/ablation_se.json: {sheet_name: {rows: [{task, obs, method, mean, std, se}], meta:{...}}} covering GT_SR, GT_InfoGain, and every A* / D* sheet. Also write a tiny loader ${GFIG}/ablation_data.py exposing load(sheet)->list of rows, so the figure agent imports it instead of re-parsing.
3. WRITE ${BUILD}/tables_v2.md containing TWO ready-to-paste Markdown tables:
   - Table 7 exactly as in the spec (mean±SE, with the Ni and Init SR columns; DISEIL bold and best per row; "—" where not applicable). Cells written WITHOUT spaces around ± (e.g. "92.4±0.4") so LaTeX cannot break a number across lines. VERIFY every cell against the spec's authoritative Table 7; if any cell disagrees, recompute from the CSV and report the discrepancy.
   - Table 8: per-demonstration information gain, Diff-DAgger and DISEIL columns only, three-decimal means from GT_InfoGain.csv, DISEIL label.
4. Rebuild ${COC}/ablations_results/DISTIL_ablation_results.xlsx from the CSVs with DISEIL labels (openpyxl), so make_figures.py's source stays consistent; verify all sheets are present after writing.
Return: the Table 7 you produced (paste it) and any cell that differed from the spec.`, {label:'data', phase:'Data', effort:'high'})

// ===================== PHASE 2: FIGURES + TEXT (parallel) =====================
phase('Figures')
await parallel([

  // --- drop-in the two author-supplied figures ---
  () => agent(`You are the FIGURE DROP-IN agent.
${CORE}
1. Copy ${UPD}/selected_tasks_SE.pdf -> ${FIGS}/selected_tasks_SE.pdf (this becomes report Figure 5).
2. Copy "${UPD}/confidence_vs_success (1).pdf" -> ${FIGS}/confidence_vs_success_v2.pdf (report Figure 7; the source name has a space and parenthesis, the destination must not).
3. VIEW both destination files with the Read tool to confirm they copied intact (Fig 5 = five learning-curve panels titled mean±SE; Fig 7 = confidence scatter, r=0.82, n=180).
Do NOT edit any .md here (the text agent updates captions/paths). Return confirmation.`, {label:'dropins', phase:'Figures', effort:'low'}),

  // --- architecture: remove Cluster Memory ---
  () => agent(`You are the ARCHITECTURE-FIGURE agent. Remove the "Cluster Memory" box and its connecting arrow from the DISEIL framework figure (report Figure 2).
${CORE}
Target: ${FIGS}/Architectural_Diagram.pdf. "Cluster Memory" is real text in the PDF (upper area, near "Cluster Engine", with a dashed arrow to Cluster Engine). An editable source exists at "/weka/s226137394/DmNfull/clean_working_with ablations/Architectural Diagram.drawio (1).html" if you can use it.
Steps:
 1. Back up the original: copy ${FIGS}/Architectural_Diagram.pdf -> ${FIGS}/Architectural_Diagram_withmem.pdf.
 2. Remove the "Cluster Memory" box AND the dashed arrow that connects it to "Cluster Engine", leaving nothing dangling. Preferred: edit the drawio XML (extract the mxGraphModel from the .drawio.html, delete that node + its edge) and re-export to PDF if a drawio/mxgraph renderer is available. If not, do a surgical overlay: cover the box and its arrow with white matching the background (install pypdf+reportlab or pikepdf via pip if needed; or, as a last resort, rasterize at 300 dpi with pdftoppm, white-out the region with PIL, and save a PNG the report can include). Whatever method, change NOTHING else in the figure.
 3. Write the cleaned figure to the SAME path ${FIGS}/Architectural_Diagram.pdf (or a .png sibling if you had to rasterize — if so, report the new filename so the text agent can repoint).
 4. VIEW the result with the Read tool and confirm: "Cluster Memory" is gone, its arrow is gone, and every other box/label/arrow is intact and undamaged.
Return: the method you used, the output filename, and your visual confirmation.`, {label:'architecture', phase:'Figures', effort:'high'}),

  // --- regenerate ablation figures with SE ---
  () => agent(`You are the ABLATION-FIGURE agent. Regenerate the ablation figures from the new data with SE error bars.
${CORE}
Read ${GFIG}/make_figures.py and ${GFIG}/ablation_data.py (the tidy SE data the DATA agent wrote). Point the figures at the NEW data (via ablation_data.py / ablation_se.json, or the rebuilt xlsx) and regenerate.
Requirements:
 - EVERY bar/point carries an SE error bar (small), never std or variance.
 - report Figure 10 = ${GFIG}/F5_grounding_and_feasibility.pdf (the A6 KAG-off figure) was MISSING its A6 error bar — it MUST now have the SE error bar on the A6 bars (data: A6_KAG_Off sheet).
 - Relabel DISTIL -> DISEIL in every legend/label.
 - Regenerate only the figures that CURRENTLY EXIST in the report: F1_allocation_ladder, F2_gain_without_allocation, F3_knockout_summary, F4_reasoning_and_vision_small, F5_grounding_and_feasibility, F6_bridging, F7_descriptor_dimensionality, F8_budget_sweep, F11_context_and_selection, F12_cluster_count_distribution, F13_failures_over_budget. Do NOT resurrect the round-2 deletions (F10_memory_constants, F14_aggregate_significance, F15_cluster_purity, F16_compute_cost).
 - Keep: three ablation settings only (GridWorld image, Push-T state, Door image), no Lift, no orange text, no prose/verdict text inside the artwork.
 - Regenerate both .pdf and .png for each.
 - Then VIEW at least F5_grounding_and_feasibility (confirm the A6 error bar is present) and F3_knockout_summary (confirm SE bars, DISEIL labels, new values) with the Read tool.
Return the list of regenerated figures and your verification of the A6 error bar.`, {label:'ablation-figs', phase:'Figures', effort:'high'}),
])

// ===================== PHASE 3: TEXT =====================
phase('Text')
await agent(`You are the TEXT agent. Update the report chapters for the new run.
${CORE}
Read ${BUILD}/tables_v2.md (the new Table 7 and Table 8), and the chapter files in ${V2}/ (Tables 7 and 8 live in 05_progress.md; the methodology / Figure 2 live in 04_aims.md).
Apply, with targeted Edit calls:
 1. Replace Table 7 with the new one from tables_v2.md (mean±SE, Ni and Init SR columns). Update its caption: "mean ± standard error over 5 seeds (robot tasks), 9 seeds (GridWorld); Ni = initial demonstrations; Init SR = round-0 held-out success rate; best per row in bold."
 2. Replace Table 8 with the new Diff-DAgger vs DISEIL info-gain table.
 3. Update EVERY number in the prose that quotes a success rate, margin, or gain to the new values (e.g. old GridWorld-state 89.9 -> 92.4; Door-image 99.2 -> 88.6; recompute any "+X pp over the best baseline" deltas from the new Table 7). Do not leave a stale number anywhere. Do not explain away the Door-image change; report it as it is.
 4. Figure 5: repoint to ../figures/selected_tasks_SE.pdf and update the caption (five panels: GridWorld image, Push-T state, Lift state, Door state, Wipe image; mean ± SE).
 5. Figure 7: repoint to ../figures/confidence_vs_success_v2.pdf; caption states r = 0.82, n = 180. Ensure "n = 152" appears nowhere.
 6. Figure 2 (architecture): the Cluster Memory box was removed from the figure. Add ONE sentence in the methodology (4.1.3) stating the cluster memory is a configurable, task-specific component, active only when a task exhibits recurring failure clusters, and therefore not drawn in the framework figure. Keep it consistent with the round-2 framing that the memory is not a headline contribution. (If the architecture agent produced a rasterized .png instead of a .pdf, repoint the Figure 2 include to that filename.)
 7. Initial-demonstration discussion: update to the actual Init SR values (all > ~45%): GridWorld 47.0, Push-T 46.2, Lift 67.2, Door 56.8, Wipe 45.2; Ni = 20/20/8/12/4 respectively. State the initial demonstrations were chosen so each task's round-0 success rate exceeds ~45%.
 8. Everywhere the text says the errors are "standard deviation" / "std", change to "standard error" and make sure it says 5 seeds (robot) / 9 seeds (GridWorld), not the old counts.
 9. Sweep any remaining "DISTIL" in the chapter text -> DISEIL.
Return a bullet list of every edit, and the list of prose numbers you changed.`, {label:'text', phase:'Text', effort:'high'})

// ===================== PHASE 4: BUILD + AUDIT =====================
phase('Build')
const A = {type:'object', additionalProperties:false, properties:{
  pages:{type:'integer'}, built:{type:'boolean'},
  checks:{type:'array', items:{type:'object', additionalProperties:false, properties:{
    id:{type:'string'}, pass:{type:'boolean'}, evidence:{type:'string'}
  }, required:['id','pass','evidence']}},
  outstanding:{type:'string'}
}, required:['pages','built','checks','outstanding']}

const gate = await agent(`You are the BUILD + AUDIT agent.
${CORE}
1. Run \`python3 ${BUILD}/assemble.py\` then \`bash ${COC}/build_pdf.sh\` (it hard-fails on Greek loss, pandoc captions, the banned acronym, a scanner watermark, and table overflow). Fix any failure and rebuild.
2. VERIFY in the built PDF (pdftotext + Read the relevant pages/figures), each with evidence:
   - Table 7 shows mean±SE with the Ni and Init SR columns, values matching the spec's authoritative table; caption says standard error, 5/9 seeds.
   - Table 8 is Diff-DAgger vs DISEIL info gain with the new values.
   - Figure 2 no longer contains a "Cluster Memory" box or its arrow (open the page and look), and the methodology has the one task-specific-memory sentence.
   - Figure 5 is the five-panel mean±SE learning-curve figure; Figure 7 is the new confidence plot (r=0.82, n=180).
   - Figure 10 (grounding/feasibility) now has the A6 SE error bar; all ablation figures show SE bars and DISEIL labels; no Lift and no orange text in ablation figures.
   - "DISTIL" appears nowhere in the PDF; "n = 152" appears nowhere; error language says standard error not std.
   - No table overflows; page count reported.
   Fix trivial failures yourself and rebuild.
3. Write ${COC}/UPDATE_R3_REPORT.md with the check table and final document statistics.
Return the structured verdict.`, {label:'build-audit', phase:'Build', schema:A, effort:'high'})

return { gate }
