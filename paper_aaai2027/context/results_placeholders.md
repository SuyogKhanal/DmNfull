# Results placeholders — results are NOT final yet (5-seed runs pending)

**Rule:** no numeric result is ever typed into the paper. Every quantitative value is a
`\PH{...}` macro so nothing can ship un-filled. Define once in the preamble:

```latex
\usepackage{xcolor}
% visible red placeholder — grep for "\PH{" to find every un-filled result
\newcommand{\PH}[1]{{\color{red}\textbf{[#1]}}}
```

## Naming scheme

`\PH{<task>-<modality>-<method>-<metric>}`

- **task**: `toy` | `push` | `lift` | `wipe` | `door`
- **modality**: `st` (state) | `im` (image)
  - toy: `st` = equivariant MLP, `im` = plain CNN
  - push/lift/wipe/door: `st` = state diffusion policy, `im` = image diffusion policy
- **method**: `pace` (ours) | `safe` | `dropout` | `ensemble` | `thrifty` | `rand` | `diff`
  - `diff` = Diff-DAgger, **robot tasks only** (push/lift/wipe/door); not on toy
- **metric**:
  - `sr`  = final held-out success rate, mean±std over 5 seeds
  - `q`   = expert queries to reach target SR (0.90); if not reached, `budget`
  - `eff` = demo efficiency = area under the SR-vs-#demos curve
  - `cov` = state/space coverage of collected demos

Examples: `\PH{push-im-pace-sr}`, `\PH{toy-st-thrifty-q}`, `\PH{door-st-diff-eff}`.

Aggregates: `\PH{mean-pace-sr}` (mean over tasks), `\PH{pace-vs-diff-q-reduction}` (headline delta).

## Main results table skeleton (repeat one block per task)

```latex
\begin{table}[t]\centering\small
\caption{Held-out success rate (SR, \%) and expert queries to reach 0.90 SR (Q),
mean$\pm$std over 5 seeds. Best per column in \textbf{bold}. Task: PUSH.}
\begin{tabular}{l cc cc}
\toprule
 & \multicolumn{2}{c}{State} & \multicolumn{2}{c}{Image} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Method & SR$\uparrow$ & Q$\downarrow$ & SR$\uparrow$ & Q$\downarrow$ \\
\midrule
Diff-DAgger    & \PH{push-st-diff-sr}     & \PH{push-st-diff-q}     & \PH{push-im-diff-sr}     & \PH{push-im-diff-q} \\
SafeDAgger     & \PH{push-st-safe-sr}     & \PH{push-st-safe-q}     & \PH{push-im-safe-sr}     & \PH{push-im-safe-q} \\
DropoutDAgger  & \PH{push-st-dropout-sr}  & \PH{push-st-dropout-q}  & \PH{push-im-dropout-sr}  & \PH{push-im-dropout-q} \\
EnsembleDAgger & \PH{push-st-ensemble-sr} & \PH{push-st-ensemble-q} & \PH{push-im-ensemble-sr} & \PH{push-im-ensemble-q} \\
ThriftyDAgger  & \PH{push-st-thrifty-sr}  & \PH{push-st-thrifty-q}  & \PH{push-im-thrifty-sr}  & \PH{push-im-thrifty-q} \\
Random         & \PH{push-st-rand-sr}     & \PH{push-st-rand-q}     & \PH{push-im-rand-sr}     & \PH{push-im-rand-q} \\
\textbf{PACE (ours)} & \PH{push-st-pace-sr} & \PH{push-st-pace-q} & \PH{push-im-pace-sr} & \PH{push-im-pace-q} \\
\bottomrule
\end{tabular}
\end{table}
```

Notes:
- **Toy table**: drop the `Diff-DAgger` row (robot-only). Columns "State (eq-MLP)" / "Image (CNN)".
- **Lift / Wipe / Door**: identical block, swap the task prefix. These are RoboSuite/UR5 and may
  be *upcoming* — placeholders stand in until the runs land.

## Figure placeholders

- Learning curves (SR vs #demos), one per task×modality:
  `figures/lc_<task>_<modality>.pdf` (e.g. `lc_push_image.pdf`). Reference callouts with `\PH{fig-lc-push-im}`.
- Ablation of PACE components (perceive / assess / choose / execute):
  `figures/abl_components.pdf`; values `\PH{abl-<component>-<metric>}`.
- Qualitative failure→prescription examples: `figures/qual_<task>.pdf`.
