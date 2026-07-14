# Figures v2 — rebuild against the supervisor revision spec

Generator: `figures_generated/make_figures.py` (rewritten). All numbers are parsed from
`ablations_results/DISTIL_ablation_results.xlsx` at run time; none is hand-entered.

**Verification method.** Every regenerated PDF/PNG was opened and read visually, not inferred from
the code. In addition, two mechanical sweeps were run over the outputs:

1. **Orange-ink sweep** — pixel-level scan of every rendered PNG for any pixel in the
   orange/vermillion region of colour space (catches text, arrows, dashed lines, fills and legend
   patches alike). Result: **0 orange pixels in all 13 figures.**
2. **String sweep** — `pdftotext` over every figure PDF for `Lift`, `DISTIL`, `PACE`, `A2I2`,
   `argmax`, `uninformative`, `ceiling`, `retracted`, and the old verdict sentences. Result: **clean
   in all 12 ablation figures.** `F14` retains the word `Lift` as a *row label*, which is correct:
   F14 is the headline ten-setting comparison, not an ablation figure, and the spec's instruction for
   it is only to remove the baked-in paragraph.

---

## Global changes applied to every figure

| Rule | What was done |
|---|---|
| D-G1 | Both orange hues (`#D55E00`, `#E69F00`) are withdrawn from the working palette. The palette is now Okabe-Ito minus orange: blue, sky, green, purple, grey, ink. Every orange dashed reference line, arrow, annotation, series and legend patch is recoloured to ink, grey or green. |
| D-G2 | Every `fig.text(...)` footnote and every sentence-bearing `ax.text` / `ax.annotate` is deleted. Numeric data labels, axis labels and legend keys stay; sentences, verdicts and parenthetical interpretation are gone. |
| D-G3 / E1 | Every ablation figure iterates the three ablation settings only: GridWorld (image), Push-T (state), Door (image). The whole Lift apparatus (`IS_LIFT`, `NONLIFT`, `LIFT_GREY`, `lift_note()`) is deleted from the source. No figure says that Lift was left out. |
| E4 | Displayed study labels renumbered: A14→**A13**, A15→**A14**, D1→**A15**, D2→**A16**, D3→**A17**, D4→**A18**, A13→**A12**. A12 (demonstrations per round) is removed. |
| Type | Liberation Serif (metrically identical to Times New Roman; Greek glyphs γ σ λ κ φ verified present). |
| Print size | **Every figure is now built at the text-block width (6.38 in = A4 minus 2.4 cm margins).** Previously the canvases were 9.6–12.2 in wide, so pandoc scaled them to ~0.65 and 9 pt type printed at ~5.9 pt. All 13 PDFs are now ≤ 459 pt wide, so they are placed at natural size and the 9 pt floor is a real 9 pt on the page. |

---

## Per-figure record

Figure numbers are those of the **current** document (spec section D); the new numbers are in
`restructure_plan.md`.

| Fig | File | Change required | What was done | Verified |
|---|---|---|---|---|
| 4 | `F14_aggregate_significance.pdf` | Strip the embedded paragraph | Four-line footnote deleted. Both summary diamonds recoloured off orange (pooled → ink, collapsed → grey). Ten settings retained (headline result, not an ablation). Rows, margins and the two `+3.71` diamonds are unchanged data. | Read; no prose; no orange |
| 8 | `F1_allocation_ladder.pdf` | Remove the "A3 falls below the best baseline" arrow + annotation | Annotation and arrow deleted. A2 bar recoloured orange→grey, A8 bar amber→green. Seed/budget footnote removed (it belongs in the caption). The best-baseline value is now a legend key rather than a floating label, which was colliding with the bar values. **This figure's grammar is factored into a `bar_panel()` helper and reused by Figs 9, 10, 11.** | Read; ladder intact, no annotation |
| 9 | `F2_gain_without_allocation.pdf` | Redraw as a grouped bar chart in the Fig-8 style | Arrow-vector scatter, slope panel, "Gain does not fall" verdict title, the "Excluding Lift…" text box and all ten settings are **replaced** by a 2×3 grouped bar chart: top row final success rate (DISEIL vs clustering off, with the best-baseline line), bottom row per-demonstration information gain (same two arms). The success rate falls while the gain does not: the two rows carry that with no sentence. | Read; bars only |
| 10 | `F5_grounding_and_feasibility.pdf` | Redraw as a bar chart in the Fig-8 style | Scatter replaced. Top row: three panels of success rate (DISEIL vs knowledge graph off, A6) with the best-baseline line. Bottom: the fallback rate with the graph removed, one bar per setting (27.1 / 27.0 / 34.8). The **orange italic author's note** ("the reference line this figure wants cannot be drawn") and the three-sentence text box are deleted. | Read; no orange, no prose |
| 11 | `F4_reasoning_and_vision_small.pdf` | Redraw as a bar chart in the Fig-8 style | Horizontal dot plot replaced by three panels of three bars each: DISEIL (full), reasoning → heuristic (A4), vision-language model off (A5), with ±1 s.d. error bars and the best-baseline line. The A5 series is no longer amber. The overlapping error bars now show that both knockouts lie inside the seed noise, so the sentence that said so is deleted, along with the Lift rows and the `lift_note`. | Read; three panels, no Lift |
| 12 | `F6_bridging.pdf` | Delete the left panel; single panel, three settings | Left scatter (with its Pearson-r box, its fitted line and its `lift_note`) deleted; the `scipy` dependency went with it. Single stacked-bar panel, three rows. The two-line **orange** footnote about stale workbook prose is deleted. | Read; one panel, 3 rows |
| 13 | `F3_knockout_summary.pdf` | Three settings only | Heat map is now 7 rows × **3** columns. Lift columns, the hatched `n/a` branch and the "Lift: at ceiling" legend entry are gone, as are the `*` stars and the bold-primary tick styling. The below-baseline cell (A3 × Door image, −6 % / −6.8) is recoloured vermillion→dark grey. Footnote deleted. | Read; 7×3, no Lift |
| 14 | `F7_descriptor_dimensionality.pdf` | Remove the argmax / chosen-descriptor annotations; keep the dashed line at 6; remove Lift | Both annotations deleted. The dashed line at 6 is recoloured vermillion→black. Lift's grey lines removed; under E1 the figure now draws the three ablation settings plus their mean, and the printed mean values are **recomputed over those three** (0.373, 0.507, 0.557, **0.593**, 0.550, 0.490, 0.423). 6-D remains the argmax in each of the three settings and in the mean. Friedman/Wilcoxon footnote deleted. | Read; no annotation, no Lift |
| 15 | `F11_context_and_selection.pdf` | Three settings; no annotation prose | Three per-panel notes and the bottom footnote (including "Lift is omitted (at ceiling)") deleted. Push-T series recoloured amber→green. Panels retitled for the renumbering: **A13** cluster-count selection, **A14** number of cited episodes (A9 unchanged). | Read; no prose |
| 16 | `F12_cluster_count_distribution.pdf` | Only the three ablated settings | Three rows, not ten. Stars and bold-primary styling dropped. The two-line pooled footnote (an all-ten-setting quantity) deleted. | Read; 3 rows |
| 17 | `F8_budget_sweep.pdf` | Delete the left panel; make the right panel three panels | Left panel deleted entirely (its orange mean line, orange value labels, text box and Lift lines with it). The right panel is now **three panels: Push-T (state), GridWorld (image), Door (image)**, each DISEIL against the best DAgger-family baseline at B = 10/20/40 with the gap printed. The baseline series is recoloured amber→grey. The in-panel sentence ("the margin shrinks because the baseline catches up…") and the orange retraction footnote are deleted; **both are now body-text matters.** Y-limits are computed per panel. | Read; 3 panels |
| 19 | `F10_memory_constants.pdf` | Keep; three settings, no orange, no prose | Three lines per panel, not ten. The vermillion reference-value lines are now black dashed. Deleted: the orange σ note, the two bold orange `ns` markers, the green λ-vs-A1 note, the ink λ = 0.5 annotation with its arrow, and the four-line footnote. **The Friedman panel titles are also deleted**: they were computed over ten settings in `stats_results.csv` and cannot be recomputed at the three-setting scope (restructure plan §4 forbids inventing one). Y-limits refitted to the three retained lines. | Read; no titles, no prose |
| 21 | `F13_failures_over_budget.pdf` | Remove the in-figure text boxes | The three-line description box, the "N ≤ 3: clustering sweep skipped" band caption and the orange "rounds 18–20" annotation with its arrow are deleted. The rings on rounds 18–20 are recoloured vermillion→black and are now explained by a legend key. The grey band is kept. | Read; no text boxes |
| 18 | `F9_demos_per_round.pdf` | **DELETE** (A12 removed) | Function `fig_demos_per_round()` and both output files deleted. | Absent |
| 20 | `F15_cluster_purity.pdf` | **DELETE**, replace with a table | Function `fig_purity()` and both output files deleted. Table data supplied below. | Absent |
| 22 | `F16_compute_cost.pdf` | **DELETE** | `make_compute_figure.py` and both output files deleted. | Absent |

---

## Data for the table that replaces old Figure 20 (study A15, was D1)

From sheet `D1_Cluster_Purity`, restricted to the three ablation settings.

| Setting | Mean cluster purity | Mean root causes per cluster | Mean silhouette |
|---|---|---|---|
| GridWorld (image) | 0.89 | 1.62 | 0.58 |
| Push-T (state) | 0.91 | 1.35 | 0.64 |
| Door (image) | 0.84 | 1.86 | 0.56 |

The caption must record that purity is measured against the reasoning model's own root-cause labels,
so it reports agreement between two components of the same system and not agreement with ground truth.

---

## Facts moved out of the artwork, which the captions or body must now carry

These were deleted from the figures under D-G2 and are not stated anywhere else unless the writer
places them.

- **F1 / F2 / F4 / F5.** The dashed line is the best DAgger-family baseline **in that setting**:
  Thrifty for GridWorld (image) and Door (image), Diff-DAgger for Push-T (state). Error bars are
  ±1 s.d. over seeds (9 seeds on GridWorld, 5 on the robot tasks); the budget is B = 20, D = 1.
- **F5.** The A6 means are taken from the sheet's numeric helper column because the A6 display strings
  are stale; the workbook records no standard deviation that can be trusted for A6, so **the A6 bars
  carry no error bar**. The caption should say the A6 bar is a mean over seeds.
- **F8 (budget).** "The margin shrinks because the baseline catches up, not because DISEIL degrades"
  is now a body sentence, as the spec directs. The retraction ("DISEIL at B = 10 does not match the
  best baseline at B = 20") is likewise body text.
- **F3 (heat map).** Margin retained = (ablated − best baseline) / (full DISEIL − best baseline);
  the small number in each cell is the Δ success rate in points; rows are ordered by mean damage.
- **F7.** Silhouette is a criterion of geometric separation only and is independent of success rate.
  Clustering is geometric for every run.
- **F13 (failures).** The figure is Push-T **image**, the only setting the workbook instruments — see
  the open item below.

---

## Open items for the orchestrator

1. **The report still cites the three deleted figures and will not build.**
   `CoC_Report.md:1057` (`F9_demos_per_round.pdf`), `:1111` (`F15_cluster_purity.pdf`), `:1150`
   (`F16_compute_cost.pdf`), and `build/sections/05_aim1_abl.md:119, :171`. These are the writer's
   deletions (E2, and the Fig-20/Fig-22 rows of the spec), not the figure engineer's.
2. **F13 is Push-T image, not Push-T state.** Sheet `D4_FailureCount` (study A18) instruments only
   that one setting, so this diagnostic sits outside the three ablation settings. No data exists on
   disk for Push-T state, and none was invented. Either state the setting in the caption and accept
   it as a diagnostic of the loop rather than an ablation arm, or re-run the instrumentation. This is
   flagged, not decided.
3. **The workbook's derived columns are formulas with no cached values.** `Δ vs full`, `Margin
   retained %`, the A11 margins and D2's `Total rounds` / `N ≤ 3` all read as empty through pandas.
   They are now **re-evaluated in the generator from the workbook's own formulas** (delta = ablated −
   full; margin = (ablated − best) / (full − best); total rounds = seeds × B, i.e. the 180 and 100
   constants written into the sheet's own cells). The recomputed values reproduce the numbers
   transcribed in `figure_audit.md` exactly, including the A3 × Door image cell at −6 % / −6.8. No
   number is invented, but the workbook should be re-saved with cached results.
4. **`build_pdf.sh` still sets `mainfont="DejaVu Serif"`.** Spec A6 requires Liberation Serif. The
   figures are now Liberation Serif; the body text is not. Outside this task's scope, but it will read
   as a mismatch.
