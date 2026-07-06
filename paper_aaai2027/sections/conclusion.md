\section{Conclusion}
\label{sec:conclusion}

We introduced \textbf{PACE} (Perceive $\rightarrow$ Assess $\rightarrow$ Choose $\rightarrow$ Execute),
an interactive imitation-learning loop that reframes the central question of the DAgger
family. Where safe and query-efficient DAgger variants
decide only \emph{when} to query the expert from a per-state uncertainty or safety
gate \cite{zhang2017safedagger,menda2017dropoutdagger,menda2019ensembledagger,hoque2021thriftydagger,lee2025diffdagger},
PACE additionally decides \emph{which} failures matter and \emph{where} the corrective
demonstration should begin. Concretely, a vision-language model perceives and describes
rollout failures \cite{bai2025qwen3vl}, the failures are assessed by clustering them into
failure modes in a shared feature space
(Eqs.~\eqref{eq:perceive}--\eqref{eq:partition}), a diverse and representative subset is
chosen by a k-center / farthest-point coreset rule with a memory-rotated target
(Eqs.~\eqref{eq:diversity}--\eqref{eq:memory}), and an LLM \cite{yang2025qwen3} executes a
prescription that maps each chosen failure to a corrective reset / sub-task-entry state
from which a single expert demonstration is collected
(Eqs.~\eqref{eq:prescribe}--\eqref{eq:prescribe-demo}). This construction is a strict
generalization of prior interactive IL: PACE recovers the diffusion-loss query rule of
Diff-DAgger \cite{lee2025diffdagger} when the selected set has size one, the target is the
dominant cluster, and the prescription is the on-policy identity (\S\ref{sec:method}).

Across the five-task suite of Table~\ref{tab:tasks}---a toy $5\times5$ grid, ManiSkill
Push \cite{florence2021implicitbc,tao2024maniskill3}, and RoboSuite Lift, Wipe, and Door on a
UR5/UR5e arm \cite{zhu2020robosuite}---and both state and image modalities, PACE reaches the
$0.90$ target success rate with substantially fewer expert queries than the DAgger-family
baselines while matching or exceeding their final success rate. Averaged over tasks, PACE
attains a held-out success rate of \PH{mean-pace-sr} and a headline query reduction of
\PH{pace-vs-diff-q-reduction} relative to Diff-DAgger (Tables~\ref{tab:main-push}
and~\ref{tab:main-robosuite}). The component ablation (Table~\ref{tab:ablation}) shows that
each PACE stage contributes: removing failure clustering (Assess) costs
\PH{abl-assess-eff}, removing diversity selection (Choose) costs \PH{abl-choose-eff}, and
replacing the prescribed reset with an on-policy correction (Execute) costs
\PH{abl-execute-eff} in demo efficiency. The gains are largest in coverage
(Eq.~\eqref{eq:cov}), consistent with our central hypothesis that spending expert effort on
curated, diverse failure modes---rather than on every uncertain state---yields more
sample-efficient corrective data.

\subsection{Limitations}
\label{sec:limitations}

PACE inherits several assumptions worth making explicit. \emph{(i) Perception fidelity.}
The Perceive and Execute stages rely on a VLM and an LLM
\cite{bai2025qwen3vl,yang2025qwen3}; a mis-described failure or an out-of-bounds
prescription degrades the corrective demo. We mitigate this with hard KAG workspace clamps
(\S\ref{sec:method}, Eq.~\eqref{eq:prescribe}), a non-empty prescription floor, and a
deterministic escalation ladder that falls back to selecting a real, untried failure, but
the pipeline is only as reliable as its foundation-model backbones and their prompts.
\emph{(ii) Resettable environments.} Execute prescribes corrective reset / sub-task-entry
states, which presumes the ability to instantiate a chosen scene---natural in simulation
but harder on hardware without a reset mechanism
\cite{eysenbach2018leavenotrace,florensa2017reversecurriculum}. \emph{(iii) A privileged
scripted expert.} Our expert is an A$^\star$/BFS oracle (toy) or a PPO / motion-planner
oracle (robot tasks); results with a noisier human expert may differ, as
human-in-the-loop studies suggest \cite{mandlekar2020iwr,liu2023sirius}. \emph{(iv)
Descriptor and clustering design.} Assess operates on a hand-designed $6$-D geometric
descriptor or a frozen R3M visual embedding \cite{nair2022r3m} reduced by PCA, and the
partition uses standard clustering \cite{lloyd1982kmeans}; the selection budget
$\kappa$ (Eq.~\eqref{eq:diversity}) and memory bandwidth are fixed hyperparameters rather
than learned. \emph{(v) Compute and cost.} Each round issues VLM/LLM calls; although the
VLM stage is cached and the query metric $q$ counts only expert demonstrations, the
wall-clock and monetary cost of foundation-model inference is a real overhead not captured
by $q$ (Eq.~\eqref{eq:queries}). \emph{(vi) Evaluation scope.} We report five tasks and
five seeds per cell; broader task diversity and real-robot transfer remain open.

\subsection{Future Work}
\label{sec:future-work}

Several directions follow directly. First, closing the loop between Perceive and Assess by
letting the VLM's semantic failure taxonomy \emph{define} the clustering feature space---
rather than clustering hand-designed geometric or frozen-encoder features---could make mode
discovery more robust, drawing on failure-reasoning VLMs
\cite{liu2023reflect,duan2025aha}. Second, the Choose stage is a natural site for richer
active-learning objectives that trade off predictive uncertainty against diversity
\cite{ash2020badge,sener2018coreset,settles2009active}; learning the selection budget and
the memory rotation on-line, instead of fixing them, is a concrete next step. Third,
Execute currently emits a single reset pose; extending prescriptions to full sub-task
curricula, and validating them on physical hardware with human experts
\cite{kelly2019hgdagger,liu2023sirius}, would test whether the resettability assumption can
be relaxed. Fourth, our LLM is run with reasoning enabled versus disabled as two setups
\cite{yang2025qwen3}; a systematic study of how reasoning depth affects prescription
quality is warranted. Finally, because PACE is agnostic to the learner, applying it to
other generative policy classes---trajectory-level diffusion planners \cite{janner2022diffuser}
and vision-language-action models \cite{brohan2023rt2,driess2023palme}---may extend
failure-driven, prescription-based data collection well beyond the diffusion policy studied
here \cite{chi2023diffusionpolicy}. More broadly, PACE reframes interactive imitation
learning as \emph{failure-mode curation}: an agent that perceives, groups, prioritizes, and
corrects its own mistakes spends its expert budget where it is most needed.
