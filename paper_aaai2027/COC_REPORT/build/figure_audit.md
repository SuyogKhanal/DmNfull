# Figure audit — CoC report

Every figure below was opened and read visually (PNG twins of the PDFs, byte-identical content).
Text strings are transcribed verbatim from the rendered artwork, not inferred from the source.
Colour names map to the palette declared at `make_figures.py:63-66`:

| name | hex | reads as |
|---|---|---|
| `BLUE` | `#0072B2` | blue |
| `SKY` | `#56B4E9` | light blue |
| `VERM` | `#D55E00` | **orange-red (vermillion) — counts as orange** |
| `ORANGE` | `#E69F00` | **amber — counts as orange** |
| `GREEN` | `#009E73` | green |
| `INK` | `#222222` | near-black |
| `MUTED` | `#666666` | grey |
| `LIFT_GREY` / `GREY` | `#B0B0B0` / `#999999` | grey (Lift) |

Under spec item **D-G1** ("remove ALL orange text"), both `VERM` and `ORANGE` text are in scope.
Under **D-G2** ("no prose, verdicts or interpretation inside any figure"), every `fig.text(...)`
footnote, every `ax.text(...)` sentence box and every interpretive panel title is in scope, whatever
its colour.

Structural constants in the script that drive the Lift/ten-setting problem:

- `SETTINGS` (`make_figures.py:71-77`) — all ten (task, modality) pairs, Lift included.
- `PRIMARY` (`make_figures.py:85`) — `[("GridWorld 5x5","image"), ("Push-T","state"), ("Door","image")]`.
  This is exactly the three ablation settings of spec item **E1**.
- `IS_LIFT`, `NONLIFT`, `LIFT_GREY`, `lift_note()` (`make_figures.py:86, 199, 66, 202-204`) — the
  whole Lift-annotation apparatus, which **D-G3** deletes.

The single highest-leverage change is to make every ablation figure iterate over `PRIMARY`
instead of `SETTINGS`, and to delete `lift_note()`, `IS_LIFT`, `NONLIFT` and `LIFT_GREY` outright.

---

## Renumbering map that the figures must absorb (spec E2/E4)

The figures label their arms with the workbook's A/D identifiers. After the spec's renumbering
these labels are wrong and must be rewritten in the artwork. Sheet names in the workbook stay as
they are; only the **displayed** label changes.

| workbook sheet | current label in artwork | new label | figures affected |
|---|---|---|---|
| `A12` | A12 | *removed* | F9 (figure deleted) |
| `A13` | A13 | **A12** | F10 |
| `A14` | A14 | **A13** | F11 (middle panel title) |
| `A15` | A15 | **A14** | F11 (right panel title) |
| `D1_Cluster_Purity` | D1 | **A15** | F15 (becomes a table) |
| `D2` | D2 | **A16** | F12; also cited in F11's middle-panel note |
| `D3_Bridge_Split` | D3 | **A17** | F6 |
| `D4_FailureCount` | D4 | **A18** | F13 |
| `D5` | D5 | **A19** | — |

A1–A11 are unchanged, so F1 (A2/A3/A8), F3 (A1/A3–A8), F4 (A4/A5), F5 (A6), F6 (A7), F7 (A10) and
F8 (A11) keep their A-labels.

---

# Per-figure audit and change order

## Fig 1 — `../figures/Teaser_Diagram.pdf`

**Contents.** A single raster scan of a hand drawing (embedded JPEG, CamScanner origin). Four
stations of the loop, drawn in black ink on white. As stored, the whole drawing sits **rotated 90°
counter-clockwise** relative to the page: every handwritten caption reads bottom-to-top rather than
left-to-right.

In-figure text (all handwritten, all **black**; no orange anywhere):

- `"policy keeps failing"` — beside a robot arm at a table with a cube, drawn with motion lines, a
  `"?!"` mark, and ghost arms struck through with `X`.
- `"LLM prescribes demos"` — beside a brain-shaped character labelled `"LLM"` operating a printing
  press that emits a scroll.
- `"expert gives the demonstration"` — beside a human with a tablet teleoperating the arm along a
  dashed arc.
- `"new demo added"` — beside a sheet of paper, reached by an arrow.
- `"update the policy"` and `"policy improved"` — beside a table with a small neural-network glyph
  inside a circular-arrow loop, the arm now grasping the cube cleanly.

No task/setting labels appear, so E1/D-G3 do not bite. The scan carries a grey cast, a blue-white
edge tint on one side and a slight skew.

**Change order (spec D, Fig 1).**

1. Rotate the raster **90° clockwise**, baked in, so the captions read left-to-right. Verify the
   `/Rotate` flag ends at 0 and the media box matches the rotated content, e.g.
   `qpdf --rotate=+90:1 Teaser_Diagram.pdf Teaser_Diagram_rot.pdf` then confirm by re-reading the
   PDF. Baking it in is safer than `\includegraphics[angle=...]`, because the report is assembled
   more than once.
2. While the file is open, strip the `Title: CamScanner …` / `Author: CamScanner` metadata.
3. No text changes; the drawing has no orange text and no prose verdicts.

---

## Fig 4 — `F14_aggregate_significance.pdf` (`fig_forest`, `make_figures.py:958-1007`)

**Contents.** Single-panel forest plot. y-axis: the **ten** settings, top to bottom
`GridWorld state`, `GridWorld image*`, `Push-T state*`, `Push-T image`, `Lift state`, `Lift image`,
`Wipe state`, `Wipe image`, `Door state`, `Door image*` (the three `PRIMARY` labels bold and
starred; the two **Lift** labels greyed). x-axis: `"Margin of DISEIL over the best DAgger-family
baseline (success-rate points)"`. Each row is a dot with a horizontal standard-error bar, coloured
`BLUE` (primary), `SKY` (other), `LIFT_GREY` (Lift), with the numeric margin printed at the right
margin: `+3.1  +2.5  +5.4  +4.9  +0.8  +0.4  +4.7  +5.7  +3.2  +6.4` (the two Lift values in grey).

Below the rows, two summary diamonds:

- `"pooled, 10 settings"` (label bold `INK`), diamond filled **`VERM` (orange-red)**, value `+3.71`.
- `"collapsed, 5 tasks"` (label bold `INK`), diamond filled **`ORANGE` (amber)**, value `+3.71`.

Four-line footnote, colour `INK`, verbatim:

> `"10 of 10 settings favour DISEIL. Setting level (n = 10): sign test and Wilcoxon, two-sided p = 0.002 — the floor for 10 pairs."`
> `"Task level (n = 5, which absorbs the correlation between the two modalities of a task): 5/5, paired t(4) = 4.15, p = 0.014."`
> `"The conservative task-level result is the one to lead with. Bars are the standard error of the paired difference; they are not the test."`
> `"Lift is at the 100.0 ± 0.0 ceiling and is uninformative about any mechanism."`

**Note.** This is the **headline result figure, not an ablation figure**. E1/D-G3 restrict the
*ablations*; the spec's own instruction for Fig 4 is only to remove the baked-in prose, so the ten
settings (Lift included) **stay** here.

**Change order (spec D, Fig 4 + D-G1 + D-G2).**

1. Delete the entire `fig.text(0.5, -0.06, …)` block (`make_figures.py:1001-1006`). All four
   sentences move to the body paragraph / caption. This removes `"10 of 10 settings favour DISEIL"`,
   `"Setting level (n = 10)"` and `"Lift is at a 100% ± 0 ceiling…"` as required.
2. Recolour both diamonds off the orange ramp so no orange ink remains: `pooled` → `INK`,
   `collapsed` → `MUTED` (or `BLUE` / `GREY`). Update the two `diamond(...)` calls at
   `make_figures.py:992-995`.
3. Keep the diamonds, the `+3.71` values and the row labels — those are data, not prose.

---

## Fig 8 — `F1_allocation_ladder.pdf` (`fig_allocation_ladder`, `make_figures.py:208-256`)

**Contents.** Three panels, titled `GridWorld image`, `Push-T state`, `Door image` — already
exactly the three ablation settings. Each panel is a four-bar grouped ladder, x tick labels
`"Random\nalloc. (A2)"`, `"Fallback\nonly (A8)"`, `"Clustering\noff (A3)"`, `"DISEIL\n(full)"`, with
black error bars (± 1 s.d.) and the mean printed above each bar in `INK`:

- GridWorld image: `86.6`, `87.8`, `87.4`, `89.6`
- Push-T state: `82.3`, `92.5`, `92.0`, `96.1`
- Door image: `84.0`, `94.8`, `92.4`, `99.2`

Bar fills: A2 = **`VERM` (orange-red)**, A8 = **`ORANGE` (amber)**, A3 = `SKY`, DISEIL = `BLUE`.
A black dashed horizontal line per panel carries a white-boxed `INK` label:
`"best baseline: Thrifty 87.1"`, `"best baseline: Diff-DAgger 90.7"`, `"best baseline: Thrifty 92.8"`.
y-axis label: `"Final success rate (%), mean ± 1 s.d. over seeds"`.

**Orange text, panel 3 only:** the two-line annotation `"A3 falls BELOW\nthe best baseline"` in
**`VERM`**, with a `VERM` arrow pointing at the A3 bar (`make_figures.py:250-252`).

Bottom footnote in `MUTED`:
`"Seeds: 9 (GridWorld), 5 (robot tasks).  Budget B = 20, D = 1 demonstration per round."`

**This figure is the reference style** that Figs 9, 10 and 11 must be rebuilt into.

**Change order (spec D, Fig 8 + D-G1).**

1. Delete the `axes[2].annotate("A3 falls BELOW\nthe best baseline", …)` call
   (`make_figures.py:250-252`) — text **and** arrow.
2. Recolour the two orange bar fills so no orange ink survives anywhere in the report's figure set:
   A2 → `MUTED`/grey, A8 → `SKY`-dark or `GREEN` (the `arms` list, `make_figures.py:221-222`). The
   four arms must stay visually ordered and colourblind-separable.
3. Keep the seed/budget footnote: it is measurement metadata, not prose or interpretation. (If a
   stricter reading of D-G2 is preferred, move it to the caption.)
4. Extract the shared drawing code into a reusable helper (e.g. `ladder_panels(axes, arms, values)`)
   because Figs 9, 10 and 11 must reuse this exact style.

---

## Fig 9 — `F2_gain_without_allocation.pdf` (`fig_gain_vs_success`, `make_figures.py:260-334`)

**Contents.** Two panels.

*Left (large).* Arrow/vector scatter. x-axis: `"Per-demonstration information gain\n(pre-retrain
policy loss on the new demonstration)"`. y-axis: `"Δ-SR against full DISEIL\n(success-rate points,
round-level rollout evaluation)"`. **All ten settings** are drawn: an open circle (full DISEIL, at
Δ-SR = 0) with an arrow to a filled circle (clustering off, A3). Arrow colours `BLUE` (primary),
`SKY` (other), **`LIFT_GREY` (Lift)**. Point labels in `INK` (`MUTED` for Lift), bold for primaries:
`"Lift state"`, `"Lift image"`, `"GridWorld image*"`, `"GridWorld state"`, `"Push-T state*"`,
`"Push-T image"`, `"Wipe state"`, `"Wipe image"`, `"Door state"`, `"Door image*"`.

White text box, `INK`, bordered `#CCCCCC`:
> `"Excluding Lift:   mean ΔIG = +0.06  (Wilcoxon p = 0.23)"`
> `"                       mean Δ-SR = −4.01  (Wilcoxon p = 0.002)"`

Legend (below the axes): `"full DISEIL"`, `"clustering off (A3)"`, `"primary setting"`,
`"other setting"`, **`"Lift (at ceiling, uninformative)"`**.

*Right (small).* Slope panel, ten lines, x tick labels `"full\nDISEIL"` and
`"clustering\noff (A3)"`, y-axis `"Information gain"`. Panel title, `INK`: **`"Gain does not fall"`**
— an interpretive verdict.

No orange ink. But it carries Lift, three prose blocks and a verdict title.

**Change order (spec D, Fig 9).**

1. **Replace the whole function** with a grouped bar chart in the Fig 8 style: three panels
   (`GridWorld image`, `Push-T state`, `Door image`), and per panel two grouped bars from the same
   source columns already parsed — **information gain** (`IG[s]` from `GT_InfoGain`, and `ig_abl[s]`
   from `A3_Clustering_Off` col 7) for full DISEIL vs clustering-off. Print the value above each
   bar, exactly as Fig 8 does. This carries the "gain does not fall while success rate does"
   result without any sentence in the artwork.
2. Drop all ten-setting iteration; drop `LOFF`, the Lift arrows, the Lift legend entry and the
   `lift_note` styling.
3. Delete the `"Excluding Lift: mean ΔIG …"` text box (`make_figures.py:307-311`) and the
   `"Gain does not fall"` title (`make_figures.py:332-333`). Both go to the body text.
4. If the paired Δ-SR must also be shown, add a second row of panels in the same style rather than
   a scatter; do not reintroduce arrow-vector geometry.

---

## Fig 10 — `F5_grounding_and_feasibility.pdf` (`fig_kag_fallback`, `make_figures.py:451-505`)

**Contents.** Single-panel scatter. x-axis: `"Fallback rate with the knowledge graph removed (% of
rounds)"`. y-axis: `"Δ success rate against full DISEIL (points)"`. **All ten settings** as dots:
`BLUE` (primary, larger), `SKY` (other), **`LIFT_GREY`** (Lift). Point labels in `INK`/`MUTED`:
`"Lift image"`, `"Lift state"`, `"GridWorld image*"`, `"GridWorld state"`, `"Wipe state"`,
`"Door state"`, `"Push-T state*"`, `"Wipe image"`, `"Door image*"`, `"Push-T image"`.

White text box, `INK`:
> `"Mean Δ-SR (excl. Lift) = -2.44 points; margin retained 44.2 %."`
> `"A fallback round is not a wasted round: the fallback rule alone (A8)"`
> `"still retains 31 % of the margin, which is why A6 costs 2.4 points and not 6."`

Legend: `"primary setting"`, `"other setting"`, **`"Lift (at ceiling, uninformative)"`**.

**ORANGE TEXT (worst offender in the set).** Bottom footnote, **`VERM`, italic**
(`make_figures.py:501-504`):
> `"The full-DISEIL fallback rate is not recorded in the workbook, so the reference line this figure wants cannot be drawn."`

This is an author's note to himself, in orange, printed in the report.

**Change order (spec D, Fig 10).**

1. **Replace the whole function** with a Fig-8-style grouped bar chart over the three ablation
   settings only. Suggested content, all already parsed: per setting, bars for
   **full DISEIL success rate** (`GT[s]["diseil"]`), **A6 knowledge-graph off** (`KO["A6"][s]["mean"]`)
   and the **best-baseline dashed line** (`GT[s]["best"]`), exactly the Fig 8 grammar. Keep the
   fallback-rate number (`a6` col 7) either as a small secondary panel of three bars or drop it
   into the caption; do not keep it as the x-axis of a scatter.
2. Delete the `VERM` italic footnote (`make_figures.py:501-504`) entirely. **D-G1.**
3. Delete the `INK` text box (`make_figures.py:490-495`). All three sentences move to the body. **D-G2.**
4. Delete the Lift dots and the Lift legend entry. **D-G3.**

---

## Fig 11 — `F4_reasoning_and_vision_small.pdf` (`fig_a4_a5_dotplot`, `make_figures.py:410-447`)

**Contents.** Single-panel horizontal dot plot. y-axis: **all ten settings**, Lift rows greyed,
primaries bold and starred. x-axis: `"Δ success rate against full DISEIL (points)"`. Each row shows
a grey band (`#EFEFEF`) spanning ± 1 seed s.d. of the full DISEIL run, plus two markers:

- circle, **`BLUE`** = A4 (reasoning model → fixed heuristic)
- square, **`ORANGE` (amber)** = A5 (vision-language model off)
- both markers **`LIFT_GREY`** on the two Lift rows.

Legend, verbatim:
- `"A4  reasoning model → fixed heuristic  (mean −1.08)"`
- `"A5  vision-language model off  (mean −1.01)"`
- `"± 1 seed s.d. of the full DISEIL run"`

Inside the axes, bottom right, `MUTED` italic: `"Lift: 100.0 ± 0.0 (ceiling) — uninformative"`
(the `lift_note()` helper).

Footnote, `MUTED`: `"Means exclude Lift. Every gap lies inside the seed noise of its own setting."`

**Change order (spec D, Fig 11).**

1. **Replace the whole function** with a Fig-8-style grouped bar chart: three panels
   (`GridWorld image`, `Push-T state`, `Door image`), each with three bars — `DISEIL (full)`,
   `Reasoning → heuristic (A4)`, `VLM off (A5)` — plotted as **absolute success rate** (`GT[s]["diseil"]`,
   `KO["A4"][s]["mean"]`, `KO["A5"][s]["mean"]`) with ± 1 s.d. error bars, the value printed above
   each bar, and the best-baseline dashed line, exactly as Fig 8. That makes the "both knockouts
   land inside the seed noise" point visible from the error bars alone, with no sentence needed.
2. Do **not** use `ORANGE` for the A5 bar. **D-G1.**
3. Delete `lift_note(ax, …)` (`make_figures.py:444`) and the footnote (`make_figures.py:445-446`).
   Delete the two Lift rows. **D-G2, D-G3.**
4. Keep the `(mean −1.08)` / `(mean −1.01)` figures out of the artwork; they go to the caption.

---

## Fig 12 — `F6_bridging.pdf` (`fig_bridging`, `make_figures.py:509-584`)

**Contents.** Two panels.

*Left panel (to be DELETED).* Scatter with a fitted dashed line. x-axis: `"Bridged share of accepted
prescriptions (%)"`. y-axis: `"Δ success rate with bridging off (points)"`. All ten settings as
dots (`BLUE` / `SKY` / **`LIFT_GREY`**), labelled `"Lift state"`, `"Lift image"`, `"Push-T image"`,
`"Door image*"`, `"GridWorld image*"`, `"Push-T state*"`, `"Wipe state"`, `"Door state"`,
`"GridWorld state"`, `"Wipe image"`. Black dashed regression line. White text box, `INK`:
> `"Pearson r = 0.23, p = 0.58 (excl. Lift)"`
> `"Bridging is not more valuable where it is used more often:"`
> `"what matters is which rounds use it, not how many."`

Plus the `MUTED` italic `lift_note`: `"Lift: 100.0 ± 0.0 (ceiling) — uninformative"`.

*Right panel (to be KEPT).* Horizontal 100 %-stacked bar, one row per setting, **all ten**. x-axis:
`"Share of accepted prescriptions (%)"`. Segments: `SKY` = `"targeted in-place correction"`,
`BLUE` = `"bridged placement"`; the split printed inside each segment (`INK` on the light segment,
white on the dark one):

| setting | in-place | bridged |
|---|---|---|
| GridWorld state | 70 | 30 |
| **GridWorld image\*** | 76 | 24 |
| **Push-T state\*** | 72 | 28 |
| Push-T image | 81 | 19 |
| *Lift state* | 82 | 18 |
| *Lift image* | 81 | 19 |
| Wipe state | 72 | 28 |
| Wipe image | 73 | 27 |
| Door state | 70 | 30 |
| **Door image\*** | 79 | 21 |

**ORANGE TEXT.** Bottom footnote, **`VERM`**, two lines (`make_figures.py:580-583`):
> `"Bridging is recorded as active on all five tasks (mean 24.4 %, range 18–30 %), including GridWorld and Wipe,"`
> `"where the workbook's own prose predicts 0 %. The data are the source of truth; the prose is stale and must be corrected."`

**Change order (spec D, Fig 12).**

1. **Delete the left scatter panel entirely** (`make_figures.py:517-559`), together with its
   `stats.pearsonr` / `np.polyfit` computation, its text box and its `lift_note`. Rebuild the figure
   as a single-panel `plt.subplots(figsize=(6.4, 2.6))`.
2. Restrict the stacked bar to the **three ablation settings** — iterate `PRIMARY`, not `SETTINGS`.
   Lift, Wipe and the four non-ablation rows all go. **D-G3, E1.**
3. Delete the `VERM` footnote (`make_figures.py:580-583`). **D-G1, D-G2.** The observation that
   bridging fires on every task belongs in the body text.
4. Relabel the source study `D3` → **`A17`** wherever it is named (caption/body).

---

## Fig 13 — `F3_knockout_summary.pdf` (`fig_knockout_heatmap`, `make_figures.py:338-406`)

**Contents.** Single-panel heat map, **7 rows × 10 columns**.

Rows (ordered by mean damage, top to bottom), `INK`:
`"A3  clustering off"`, `"A8  fallback only"`, `"A6  knowledge graph off"`, `"A7  bridging off"`,
`"A4  reasoning model → heuristic"`, `"A5  vision-language model off"`, `"A1  cluster memory off"`.

Columns: **all ten settings**, rotated 38°, primaries bold + starred, **the two Lift columns greyed**.

Each cell prints two numbers: the margin retained (`%`, large) and the Δ success rate (small,
signed). Fill = `Blues_r` sequential ramp, 0–100 %. **The two Lift columns are hatched grey
(`#E8E8E8`, `///`) and print `"n/a"` in `MUTED`.** One cell — A3 × Door image, `-6%` / `-6.8` — is
filled **`VERM` (orange-red)** with white text, marking a fall below the baseline.

Colour bar label: `"Margin over the best baseline retained (%)"`.

Legend: `Patch(VERM)` `"ablated system falls below the best baseline"`;
`Patch(#E8E8E8, ///)` **`"Lift: at ceiling, uninformative"`**.

Footnote, `MUTED`:
> `"Cell: margin retained (%); small number: Δ success rate (points). Margin retained = (ablated − best baseline) / (full DISEIL − best baseline). Rows ordered by mean damage.  * primary setting."`

**Change order (spec D, Fig 13 + E1).**

1. Change the column set from `SETTINGS` to `PRIMARY` (`make_figures.py:345-346, 376-383`): the map
   becomes **7 rows × 3 columns**. Shrink `figsize` to about `(5.6, 3.6)`.
2. With Lift gone, delete the hatched-`n/a` branch (`make_figures.py:359-364`), the
   `Patch(#E8E8E8, ///)` legend entry and the greyed tick-label styling. **D-G3.**
3. Drop the `*` star and the bold-primary styling from the tick labels — with only the three
   ablation settings present, the distinction is empty.
4. Keep the `VERM` cell fill (it is a data-encoded fill, not text) but **recolour it off orange** to
   satisfy a strict reading of D-G1 if the auditor prefers; the safe option is a dark grey or the
   `Blues_r` under-range colour with a white outline. Recolour the matching legend patch.
5. Reduce the footnote to the definition of "margin retained" only, or move it wholesale to the
   caption. **D-G2.**

---

## Fig 14 — `F7_descriptor_dimensionality.pdf` (`fig_descriptor_dims`, `make_figures.py:588-626`)

**Contents.** Single-panel line plot. x-axis: `"Descriptor dimensionality (number of features in φ)"`,
ticks `2, 4, 5, 6, 8, 10, 12`. y-axis: `"Mean silhouette of the resulting failure clusters"`.
Ten thin lines, one per setting: `SKY` for the eight non-Lift settings, **`LIFT_GREY` for the two
Lift lines** (they sit at the bottom of the bundle). A thick `BLUE` line with markers is the mean
over the ten settings, with its value printed at each x in `BLUE`:
`0.354, 0.485, 0.532, 0.567, 0.524, 0.469, 0.402`.

**ORANGE.** A dashed vertical line at x = 6 in **`VERM`**, plus a two-line annotation beside it in
**`VERM`** (`make_figures.py:603-605`):
> `"chosen descriptor: 6-D"`
> `"argmax in 10 of 10 settings"`

Legend: `"mean over the 10 settings"`, `"individual setting"`,
**`"Lift (silhouette is unaffected by the ceiling)"`**.

Footnote, `MUTED`, two lines:
> `"Friedman across the seven dimensionalities: χ²(6) = 59.52, p = 5.6 × 10⁻¹¹.  Wilcoxon 6-D vs 5-D: p = 0.002.  6-D vs 8-D: p = 0.002."`
> `"Silhouette is a criterion of geometric separation only and is independent of success rate. Clustering is geometric for every run, state and image alike."`

**Change order (spec D, Fig 14).**

1. Delete the `ax.text(6.2, 0.316, "chosen descriptor: 6-D\nargmax in 10 of 10 settings", …)` call
   (`make_figures.py:604-605`). The dashed line at 6 carries the point on its own. **D-G1, D-G2.**
2. Recolour the `axvline(6, …)` from `VERM` to `INK` dashed (`make_figures.py:603`). **D-G1.**
3. Remove the Lift lines: iterate the eight non-Lift settings, or, if E1 is applied strictly, only
   the three ablation settings plus the mean. Delete the `LIFT_GREY` legend entry.
   The spec's Fig-14 row explicitly says *"Remove Lift (the grey lines)"*. **D-G3.**
4. Update the mean-line label and legend from `"mean over the 10 settings"` to match whatever set
   survives, and recompute `M.mean(axis=0)` over that set so the printed values stay true to the
   data.
5. Move the Friedman/Wilcoxon footnote to the caption. **D-G2.**

---

## Fig 15 — `F11_context_and_selection.pdf` (`fig_context_and_selection`, `make_figures.py:815-869`)

**Contents.** Three panels; **already restricted to the three ablation settings**, drawn as grouped
bars with ± 1 s.d. error bars. Series colours: **`BLUE` = `"GridWorld image"`, `ORANGE` (amber) =
`"Push-T state"`, `GREEN` = `"Door image"`** (shared legend, bottom centre). y-axis (left panel
only): `"Final success rate (%), mean ± 1 s.d."`, limits 85–105.

Panel titles and x tick labels:

- `"A9  context-set composition"` — `Full S`, `− worst-loss seed`, `− FPS (random fill)`,
  `− forced rep.`, `Random 3`.
- `"A14  cluster-count selection"` — `silhouette (adaptive)`, `fixed k = 3`, `fixed k = 4`,
  `fixed k = 2`, `fixed k = 5`.
- `"A15  number of cited episodes"` — `Full S (κ = 3)`, `Top-3 by loss`, `Top-5 by loss`,
  `Top-2 by loss`, `all in cluster`, `Random 3`.

Three per-panel notes below the axes, `INK` (`make_figures.py:854-863`):
> `"Arms ordered by effect. Spread from Full S to Random 3: 2.25 points.\nThe ordering, not the individual gap, is the result."`
> `"Silhouette selection beats the best fixed k (k = 3) by 0.34 points.\nThe adaptivity is real (D2) but its value is small."`
> `"Top-3-by-loss against Full S: −0.40 points. n = 5 buys nothing over n = 3.\nTop-1 omitted: bridging requires at least two cited failures."`

Bottom footnote, `MUTED`:
> `"Three step-level ablations with the same answer: each is worth roughly half a point. They are shown together so that none is oversold. Lift is omitted (at ceiling)."`

**Change order (spec D, Fig 15).**

1. The three-setting restriction is already satisfied — **no data change needed**. `PRIMARY` is
   already the iteration set (`make_figures.py:839`).
2. Delete all three per-panel `INK` notes (`make_figures.py:854-863`) and the `MUTED` footnote
   (`make_figures.py:865-868`). The footnote's `"Lift is omitted (at ceiling)"` is exactly the
   sentence **D-G3** forbids. **D-G1, D-G2, D-G3.**
3. Recolour the `Push-T state` series off `ORANGE` — swap the palette list
   `[BLUE, ORANGE, GREEN]` (`make_figures.py:841`) for `[BLUE, SKY, GREEN]` or
   `[BLUE, GREEN, PURPLE]`. **D-G1.**
4. Retitle the panels for the renumbering (**E4**): `"A14  cluster-count selection"` → **`"A13  cluster-count selection"`**;
   `"A15  number of cited episodes"` → **`"A14  number of cited episodes"`**. `"A9"` is unchanged.

---

## Fig 16 — `F12_cluster_count_distribution.pdf` (`fig_kstar`, `make_figures.py:873-921`)

**Contents.** Single-panel horizontal 100 %-stacked bar, **one row per setting, all ten** (primaries
bold + starred; `Lift state` and `Lift image` present as ordinary rows). x-axis:
`"Share of all rounds (%)"`, ticks `0 20 40 60 80 100`. Segments are the chosen cluster count
`k = 2 … 6`, filled from the sequential blue ramp `SEQ`, each labelled with its `k` (`"2"`, `"3"`,
`"4"`, `"5"`, `"6"`) inside the segment when the segment exceeds 6 % (white text for k ≤ 3, `INK`
otherwise). A trailing hatched grey segment (`#E8E8E8`, `///`) is `"clustering skipped (N ≤ 3)"`,
labelled with its own percentage in `INK`: `17%`, `21%`, `15%`, `29%`, `31%`, `23%`, `19%`, `25%`,
`34%`, `20%`. At the right of each row, in `MUTED`: `"180 rounds"` (GridWorld ×2) or `"100 rounds"`
(the other eight).

Legend: `"k = 2"` … `"k = 6"`, `"clustering skipped (N ≤ 3)"`.

Footnote, `MUTED`, two lines:
> `"Pooled over the 896 clustered rounds: k = 3 is the mode at 25.1 %, k = 4 second at 23.8 %; every k from 2 to 6 is selected in at least 15 % of rounds."`
> `"Total rounds = seeds × B (180 for GridWorld at 9 seeds, 100 for the robot tasks at 5 seeds), which confirms the stated seed counts."`

No orange ink.

**Change order (spec D, Fig 16).**

1. Iterate `PRIMARY` instead of `SETTINGS` (`make_figures.py:885, 902-905`): three rows, not ten.
   Shrink `figsize` to about `(6.6, 2.4)`. **E1, D-G3.**
2. Drop the `*` star and bold-primary tick styling (meaningless once only the three remain).
3. Delete the two-line footnote (`make_figures.py:915-920`). The "k = 3 is the mode" claim must be
   recomputed over the three retained settings before it is restated in the body text — **do not
   carry the 896-round pooled numbers across unchanged**, they are ten-setting quantities. **D-G2.**
4. Relabel the source study `D2` → **`A16`** in the caption/body. **E4.**

---

## Fig 17 — `F8_budget_sweep.pdf` (`fig_budget`, `make_figures.py:630-705`)

**Contents.** Two panels.

*Left panel (to be DELETED).* x-axis `"Budget B (demonstrations)"`, log scale, ticks `10 20 40`.
y-axis `"Margin over the best baseline (success-rate points)"`. Ten lines (`BLUE` primary, `SKY`
other, **`LIFT_GREY` Lift**), with the three primaries labelled at the right edge in bold `INK`:
`"Door image"`, `"Push-T state"`, `"GridWorld image"`. A thick dashed **`VERM`** line is the mean
over the eight non-Lift settings, with its values printed in **`VERM`**: `10.35`, `4.49`, `2.67`.
White text box, `INK`: `"The margin roughly doubles\nwhen the budget is halved."`
Legend: `"mean over the 8 non-Lift settings"`, `"primary setting"`, `"other setting"`,
**`"Lift (at ceiling)"`**.

*Right panel (to be KEPT and expanded).* Title `"Push-T state"` — **one setting only**. Two series:
`BLUE` circles `"DISEIL"`, **`ORANGE` (amber) squares `"best DAgger-family baseline"`**. Grey
double-headed arrows between them with the gap printed in `MUTED`: `+12.4`, `+5.4`, `+3.2`.
Axes: `"Budget B (demonstrations)"` / `"Final success rate (%)"`.
White text box, `INK`:
> `"The margin shrinks because the baseline\ncatches up, not because DISEIL degrades."`

**ORANGE TEXT.** Bottom footnote, **`VERM`** (`make_figures.py:701-704`):
> `"DISEIL at B = 10 does NOT match the best baseline at B = 20: it is below it in 7 of the 10 settings. That claim is retracted."`

**Change order (spec D, Fig 17).**

1. **Delete the left panel entirely** (`make_figures.py:642-676`), with its `VERM` mean line, its
   `VERM` value labels, its `INK` text box and its Lift lines. **D-G1, D-G2, D-G3.**
2. **Expand the right panel to three panels** — `Push-T state`, `GridWorld image`, `Door image` —
   i.e. loop `for ax, s in zip(axes, PRIMARY)` over the existing `dis[s]` / `base[s]` series (both
   are already parsed for all ten settings from sheet `A11`, so no new data is needed). Keep the
   DISEIL-vs-best-baseline pair, the `+Δ` gap annotations and the per-panel title.
3. Recolour the baseline series off `ORANGE` (`make_figures.py:680`) → `MUTED` or `GREEN`. **D-G1.**
4. Delete the in-panel sentence
   `"The margin shrinks because the baseline catches up, not because DISEIL degrades."`
   (`make_figures.py:698-700`) and hand it to the body paragraph, as the spec explicitly directs.
5. Delete the `VERM` bottom footnote (`make_figures.py:701-704`). The retraction is a body-text
   statement, not artwork.
6. Note the y-limits are currently hard-coded for Push-T (`ax2.set_ylim(73.5, 101.5)`,
   `make_figures.py:694`); with three panels they must be computed per panel.

---

## Fig 19 — `F10_memory_constants.pdf` (`fig_memory_constants`, `make_figures.py:748-811`)

**Contents.** Three panels, one per memory constant, **each drawing all ten settings** as lines
(`BLUE` primary, `SKY` other, **`LIFT_GREY` Lift, pinned flat at 100 %**). Shared y-axis
`"Final success rate (%)"`, limits 87.8–101.4.

x-axes:
- `"γ  recency discount\n(reference value 0.6, dashed)"` — ticks `0.3 0.5 0.6 0.7 0.9`
- `"σ  kernel width\n(reference value 0.06, dashed)"` — ticks `0.02 0.04 0.06 0.1 0.2`
- `"λ  memory penalty weight\n(reference value 1, dashed)"` — ticks `0 0.5 1 2 4`

A **`VERM` dashed vertical line** marks the reference value in each panel.

Per-panel titles, `INK` (read from `build/stats_results.csv`):
`"Friedman χ²(4) = 33.13, p = 1.1e-06"`, `"Friedman χ²(4) = 28.84, p = 8.4e-06"`,
`"Friedman χ²(4) = 34.36, p = 6.3e-07"`.

**ORANGE TEXT.**
- σ panel, **`VERM`**: `"σ is LIVE only on Push-T and Wipe\n(the six flat lines are ties)"`
  (`make_figures.py:790-791`).
- σ panel, **`VERM` bold**, two markers on the axis: `"ns"` at σ = 0.04 and `"ns"` at σ = 0.1
  (`make_figures.py:787-789`).

**GREEN text.** λ panel: `"λ = 0 reproduces A1\n(memory off) exactly,\nin all 10 settings"`
(`make_figures.py:796-797`).

**`INK` annotation with a leader arrow.** λ panel:
`"λ = 0.5 is worse than λ = 0:\na half-weight penalty deflects\nwithout rotating"`
(`make_figures.py:793-795`).

Legend: `"primary setting"`, `"other setting"`, **`"Lift (ceiling: flat by construction)"`**.

Footnote, `MUTED`, four lines — the longest prose block in the whole figure set
(`make_figures.py:804-810`):
> `"γ and λ are significantly best at their reference values (Holm-corrected p < 0.01 against every swept value). σ = 0.06 is directionally best but NOT statistically distinguishable from its neighbours σ = 0.04 and σ = 0.1 (Holm p = 0.125, marked ns): the kernel is degenerate on GridWorld and Door and ceiling-masked on Lift, and those six tied settings strip the paired test of its power. A per-task σ, set as a fraction of each task's reset range, is the fix. This is a limitation, not a virtue."`

**Change order (spec D, Fig 19 — keep, apply E1/E3/E4 + globals).**

1. Iterate `PRIMARY` instead of `SETTINGS` (`make_figures.py:763-768`): three lines per panel, not
   ten. Delete the `LIFT_GREY` legend entry. Re-fit `set_ylim` to the retained lines — the current
   hard-coded `(87.8, 101.4)` was chosen to span GridWorld to Lift and will leave dead space.
   **E1, D-G3.**
2. Delete the `VERM` σ note and the two bold `VERM` `"ns"` markers (`make_figures.py:786-791`).
   **D-G1.** (The σ non-significance is a body-text limitation, per H-series discipline.)
3. Delete the `GREEN` λ-vs-A1 note (`make_figures.py:796-797`) and the `INK` λ = 0.5 annotation with
   its arrow (`make_figures.py:793-795`). **D-G2.**
4. Recolour the reference-value `axvline` from `VERM` to `INK` dashed (`make_figures.py:770`). **D-G1.**
5. Delete the four-line `MUTED` footnote (`make_figures.py:804-810`); it moves to the body. Keep the
   three Friedman panel titles — they are statistics, not prose — but note they are computed over the
   ten settings in `stats_results.csv`; if E1 is applied to the *statistics* as well as the artwork,
   they must be recomputed over the three settings or dropped from the artwork.
6. Relabel the study `A13` → **`A12`** and the cross-reference `"reproduces A1"` stays `A1`. **E4.**

---

## Fig 20 — `F15_cluster_purity.pdf` (`fig_purity`, `make_figures.py:1011-1067`)

**Contents.** Single-panel scatter with a fitted dashed line. x-axis: `"Mean silhouette (geometric
separation of the clusters)"`. y-axis: `"Mean cluster purity\n(fraction sharing the dominant
root-cause label)"`. **All ten settings**, marker shape by modality (`o` = state, `^` = image),
colour `BLUE` (primary) / `SKY` (other) / **`LIFT_GREY` (Lift)**. Labels in `INK`/`MUTED`, primaries
bold: `"Lift image"`, `"Lift state"`, `"GridWorld state"`, `"Push-T state*"`, `"GridWorld image*"`,
`"Push-T image"`, `"Door state"`, `"Door image*"`, `"Wipe state"`, `"Wipe image"`.

White text box, `INK`:
> `"Pearson r = 0.18, p = 0.62"`
> `"Geometric separation and semantic purity are uncorrelated:"`
> `"A10 and D1 are independent checks, not two views of one quantity."`

Legend: `"state modality"`, `"image modality"`, `"primary setting"`, **`"Lift (at ceiling)"`**.

Footnote, `MUTED`, two lines:
> `"Purity ranges from 0.78 (Wipe image) to 0.93 (Lift state), mean 0.877. It is measured against the reasoning model's own root-cause labels, so it records agreement between two components of the same system, not agreement with ground truth."`

No orange ink, but Lift, a verdict box and a prose footnote.

**Change order (spec D, Fig 20).**

1. **Delete the figure.** Remove `fig_purity()` (`make_figures.py:1011-1067`) and its entry in the
   `__main__` list (`make_figures.py:1077`). Delete `F15_cluster_purity.pdf` / `.png`.
2. **Replace it with a table** in the report body, built from `D1_Cluster_Purity` (renumbered
   **A15**), restricted to the three ablation settings. Columns:
   `Setting | Mean cluster purity | Root-cause labels per cluster | Mean silhouette`, i.e. workbook
   columns 2, 3 and 4 for the rows `GridWorld 5x5/image`, `Push-T/state`, `Door/image`.
   The caveat that purity is measured against the reasoning model's own labels goes in the caption.

---

## Fig 21 — `F13_failures_over_budget.pdf` (`fig_failure_count`, `make_figures.py:925-954`)

**Contents.** Single-panel line plot. x-axis: `"Round r  (budget B = 20)"`, ticks 2…20. y-axis:
`"Mean number of recorded failures N"`, 0–46. One `BLUE` line with markers, falling monotonically
from 42 at round 1 to 2 at round 20. A light grey band (`#F2F2F2`) spans y ∈ [0, 3.5]. The last
three markers (rounds 18, 19, 20) are ringed in **`VERM`**.

**Setting flag.** The figure is **Push-T image only** — this is *not* one of the three ablation
settings (`Push-T state` is). Sheet `D4_FailureCount` instruments only that one setting, so under a
strict reading of **E1** this figure shows a setting we do not ablate.

In-figure text:
- **`INK` text box, top right (three lines):**
  > `"Push-T image, mean over 5 seeds — the only setting instrumented."`
  > `"The clustering and memory do their work in the first two thirds of the budget;"`
  > `"the last rounds are effectively running the fallback rule."`
- **`MUTED`, inside the grey band:** `"N ≤ 3: clustering sweep skipped\n(each failure becomes its own cluster)"`
- **`VERM` (ORANGE), with a `VERM` leader arrow:** `"rounds 18–20"`

**Change order (spec D, Fig 21).**

1. Delete the three-line `INK` text box (`make_figures.py:948-953`). **D-G2** — this is the spec's
   "descriptive text boxes inside the figure".
2. Delete the `MUTED` band caption `"N ≤ 3: clustering sweep skipped …"` (`make_figures.py:933-934`).
   Keep the grey band itself; define it in the caption. **D-G2.**
3. Delete the `"rounds 18–20"` annotation and its arrow (`make_figures.py:946-947`). **D-G1.**
4. Recolour the ringed markers for rounds 18–20 from `VERM` to `INK` (`make_figures.py:938`), or drop
   the rings and let the grey band mark the region. **D-G1.**
5. **Decision required from the author:** either (a) accept that this diagnostic is Push-T *image*
   and state the setting in the caption (it is a diagnostic of the loop, not an ablation arm, so E1
   is arguably not binding), or (b) re-run the `D4`/`A18` instrumentation on Push-T **state** so the
   figure lands inside the three ablation settings. Option (a) requires no new compute; option (b)
   does. **Flagging, not deciding.**
6. Relabel the source study `D4` → **`A18`** in the caption/body. **E4.**

---

# Figures the spec deletes outright

| file | function | action |
|---|---|---|
| `F9_demos_per_round.pdf/.png` | `fig_demos_per_round` (`make_figures.py:709-744`) | Delete the function, its `__main__` entry (`:1075`) and both output files. A12 is removed (**E2**). |
| `F16_compute_cost.pdf/.png` | `make_compute_figure.py` (whole script) | Delete the figure and the outputs (**spec D, Fig 22**). |
| `F15_cluster_purity.pdf/.png` | `fig_purity` (`make_figures.py:1011-1067`) | Delete; replaced by a table (**spec D, Fig 20**). |

`gantt_chart.pdf` (`make_gantt.py`) is out of this audit's scope but carries the spec's Fig-24 items:
remove the `"examination preparation"` bar, and note `STREAMS["thesis"]` is currently
`ORANGE (#E69F00)` with the label `"Thesis and examination"` (`make_gantt.py:49, 62`), and the CoC
marker line and its label are drawn in `VERM` (`make_gantt.py:183-185`) — both fall under **D-G1**.

---

# Cross-cutting code changes to `make_figures.py`

1. **Delete the Lift apparatus.** `IS_LIFT` (`:86`), `NONLIFT` (`:199`), `LIFT_GREY` (`:66`) and the
   `lift_note()` helper (`:202-204`), plus every call site and every legend entry naming Lift.
   Also delete the Lift sentence in the module docstring (`:15-17`).
2. **Delete the orange palette entries from use.** `VERM` and `ORANGE` must not colour any text,
   annotation, arrow, dashed reference line, series or legend patch. The remaining safe set is
   `BLUE`, `SKY`, `GREEN`, `PURPLE`, `INK`, `MUTED`, `GREY`, plus the `SEQ` blue ramp. Keep the
   variables defined if a fill is still wanted for the below-baseline heat-map cell, but audit every
   use.
3. **Iterate `PRIMARY`, not `SETTINGS`,** in every ablation figure: F1, F2, F3, F4, F5, F6, F7, F8,
   F10, F12, F15. F14 (the aggregate significance forest) keeps all ten — it is the headline result,
   not an ablation.
4. **Strip every `fig.text(...)` footnote and every sentence-bearing `ax.text(...)` / `ax.annotate(...)`.**
   Numeric data labels (bar values, split percentages, margins) stay; sentences, verdicts and
   parenthetical interpretation go.
5. **Apply the A/D renumbering** to every displayed arm label and panel title, per the map above.
6. **Factor out the Fig-8 grouped-bar style** into one helper, since Figs 9, 10 and 11 must all be
   rebuilt into it.
