% =====================================================================
%  ABSTRACT + CONTRIBUTIONS  (PACE, AAAI-2027)
%  Body LaTeX only -- no preamble. Assumes the standard AAAI-2027 kit
%  preamble, \newcommand{\PH}[1]{{\color{red}\textbf{[#1]}}}, and
%  references.bib. AAAI rule: NO citations inside \begin{abstract};
%  citations appear only in the (post-abstract) contributions paragraph.
% =====================================================================

\begin{abstract}
Interactive imitation learning corrects the compounding covariate shift of
behavior cloning by aggregating fresh expert demonstrations on the states a
learner actually visits. The dominant DAgger family reduces to a single
question---\emph{when} to query the expert---answered by a per-state uncertainty
or safety gate (action discrepancy, dropout or ensemble variance, task risk, or
diffusion loss). Answering only \emph{when} leaves two decisions implicit and
uncontrolled: \emph{which} of the many observed failures deserve scarce expert
effort, and \emph{where} a corrective demonstration should begin. We introduce
\textbf{PACE} (\textbf{P}erceive~$\rightarrow$~\textbf{A}ssess~$\rightarrow$~%
\textbf{C}hoose~$\rightarrow$~\textbf{E}xecute), an interactive-imitation loop
for diffusion policies that treats each round's rollout failures as a batch to be
curated rather than a stream to be gated. PACE \emph{perceives} failures with a
vision--language model and a compact geometric (optionally visual) failure
descriptor anchored at each rollout's peak diffusion-loss step; \emph{assesses}
the round by clustering descriptors into failure modes and scoring their
dominance and cross-round novelty; \emph{chooses} a small, diverse subset via
farthest-point (k-center coreset) selection with the dominant mode's
representative forced in; and \emph{executes} an LLM-prescribed corrective
reset / sub-task-entry scenario from which one expert demonstration is collected
before retraining. We show that PACE strictly generalizes the query-based
baselines: it recovers a Diff-DAgger round when the selected set is a singleton,
the target is the dominant mode, and the prescribe map is the on-policy identity.
Across five tasks spanning a toy $5\times5$ grid, ManiSkill Push, and RoboSuite
Lift, Wipe, and Door (state and image observations, five seeds each), PACE
reaches the target success rate with \PH{pace-vs-diff-q-reduction} fewer expert
queries than the strongest baseline while attaining the highest held-out success
rate and the broadest demonstration coverage. Ablating any of Perceive, Assess,
Choose, or Execute degrades query efficiency, confirming that \emph{which} and
\emph{where}---not only \emph{when}---drive sample-efficient interactive
imitation learning.
\end{abstract}

% ---------------------------------------------------------------------
%  CONTRIBUTIONS  (place at the end of the Introduction; citations allowed
%  here, unlike inside the abstract). Referenced eqs: \eqref{eq:iil-loop},
%  \eqref{eq:diffdagger}, \eqref{eq:perceive}--\eqref{eq:prescribe-demo}.
% ---------------------------------------------------------------------
\paragraph{Contributions.}
We make four contributions.
\textbf{(1)} We reframe interactive imitation learning
\cite{ross2011dagger,celemin2022iil} as a per-round failure-curation problem
rather than a per-state query-gating problem, and cast the safe/query-efficient
DAgger family---SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, and the
diffusion-native Diff-DAgger
\cite{zhang2017safedagger,menda2017dropoutdagger,menda2019ensembledagger,hoque2021thriftydagger,lee2025diffdagger}---%
as the special case that answers only \emph{when} to query
(Eq.~\eqref{eq:iil-loop}, Eq.~\eqref{eq:diffdagger}).
\textbf{(2)} We propose \textbf{PACE}, a Perceive--Assess--Choose--Execute loop
for diffusion-policy learners \cite{chi2023diffusionpolicy,ho2020ddpm} that
perceives failures with a vision--language model
\cite{bai2025qwen3vl,liu2023reflect,duan2025aha}, partitions them into modes by
clustering \cite{lloyd1982kmeans}, prioritizes a diverse coreset by
farthest-point / k-center selection \cite{sener2018coreset,eldar1997fps}, and
prescribes an LLM-generated corrective reset / sub-task-entry scenario
\cite{yang2025qwen3,florensa2017reversecurriculum,eysenbach2018leavenotrace}
(Eqs.~\eqref{eq:perceive}--\eqref{eq:prescribe-demo}).
\textbf{(3)} We prove PACE is a strict generalization of the query-based
baselines, recovering a Diff-DAgger round as the singleton, dominant-mode,
on-policy-identity limit (Eq.~\eqref{eq:prescribe-demo}), so any gain over
Diff-DAgger is attributable to the added \emph{which} and \emph{where} decisions
under an identical one-demo-per-round budget.
\textbf{(4)} On five tasks (toy grid, ManiSkill Push, RoboSuite Lift/Wipe/Door)
in both state and image modalities \cite{tao2024maniskill3,zhu2020robosuite,%
florence2021implicitbc,nair2022r3m}, evaluated over five seeds against six
baselines under a matched demonstration budget, PACE cuts expert queries to reach
$\mathrm{SR}_{\mathrm{target}}{=}0.90$ by \PH{pace-vs-diff-q-reduction} versus the
best baseline and achieves a mean held-out success rate of \PH{mean-pace-sr},
with component ablations isolating the contribution of each PACE stage.
