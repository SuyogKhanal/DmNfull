\section{Introduction}

Imitation learning trains a control policy $\pi_\theta$ to reproduce an
expert oracle $\pi^\star$ from demonstrations, and behavior cloning does
so by regressing the expert's actions on the states the expert
visits~\citep{pomerleau1988alvinn,pomerleau1991efficient}. The
well-known failure mode is \emph{covariate shift}: because the learner's
own actions determine the states it subsequently encounters, small
prediction errors push the agent off the demonstrated support, where
further errors compound over the horizon~\citep{ross2011dagger}.
Interactive imitation learning (IIL) confronts this by aggregating
expert labels on states the \emph{learner} actually visits, reducing
imitation learning to no-regret online learning and bounding error
linearly rather than quadratically in the horizon~\citep{ross2011dagger,
celemin2022iil}. The price is expert effort: vanilla DAgger asks the
expert to relabel every rolled-out state, which is prohibitive when the
expert is a human teleoperator or an expensive planner~\citep{
zhang2017safedagger,hoque2021thriftydagger}. \emph{Query efficiency} ---
reaching a target success rate with as few expert queries as possible ---
is therefore the central axis along which IIL methods are judged, and the
one we study here.

\paragraph{The DAgger-family gap.}
Every query-efficient descendant of DAgger addresses covariate shift by
deciding \emph{when} to hand control to the expert during a rollout.
SafeDAgger learns a safety classifier that predicts where the novice will
deviate~\citep{zhang2017safedagger}; DropoutDAgger and EnsembleDAgger
gate on predictive uncertainty from Monte-Carlo dropout and ensemble
variance~\citep{menda2017dropoutdagger,menda2019ensembledagger,
gal2016dropout,lakshminarayanan2017ensembles}; HG-DAgger and LazyDAgger
gate on learner--expert discrepancy~\citep{kelly2019hgdagger,
hoque2021lazydagger}; ThriftyDAgger budgets interventions by novelty and
risk~\citep{hoque2021thriftydagger}; and Diff-DAgger --- designed for the
multimodal diffusion policies we adopt as our learner --- reuses the
diffusion training loss itself as the query signal~\citep{
lee2025diffdagger,chi2023diffusionpolicy}. Despite their different
signals, all of these methods share one decision structure: a per-state
binary predicate $\mathrm{Query}(s_t)$ fires the first time a scalar
score crosses a threshold, and the expert takes over from that
step~(Eq.~\ref{eq:iil-loop}). Formally they differ only in the score
inside the indicator (Eqs.~\ref{eq:safe}--\ref{eq:diffdagger}). This
answers a single question --- \emph{when} in a rollout to query --- and
leaves three others unasked. It does not decide \emph{which} of the many
distinct failures a policy exhibits per round deserve correction, so
redundant failures from the same underlying failure mode are corrected
over and over while rarer modes are starved. It does not decide
\emph{where} a corrective demonstration should begin, so the reset
distribution is fixed to whatever state the rollout happened to reach. And
it treats each query as an isolated event with no memory across rounds, so
a mode that was already fixed can be re-queried indefinitely. The result
is that expert budget is spent on the \emph{easiest-to-detect}
uncertainty rather than on a diverse, non-redundant cover of the policy's
actual failure modes.

\paragraph{PACE.}
We propose \textbf{PACE} (\textbf{P}erceive $\to$ \textbf{A}ssess $\to$
\textbf{C}hoose $\to$ \textbf{E}xecute), an interactive imitation-learning
loop that generalizes the DAgger-family query from ``\emph{when}'' to
``\emph{when} $+$ \emph{which} $+$ \emph{where}.'' Each round, PACE rolls
the current diffusion policy to collect a \emph{set} of failures rather
than a single intervention point. (i)~\emph{Perceive}: a vision-language
model~\citep{bai2025qwen3vl} inspects each failed rollout at its peak
diffusion-loss frame --- exactly the state the Diff-DAgger signal would
flag~(Eq.~\ref{eq:diffdagger}) --- and each failure is turned into a
compact geometric (or visual) descriptor~(Eq.~\ref{eq:phi}). (ii)~\emph{Assess}:
the round's descriptors are clustered into failure
\emph{modes}~\citep{lloyd1982kmeans}, with the number of modes chosen by
silhouette~(Eq.~\ref{eq:partition}); a cross-round memory penalizes modes
already covered in earlier rounds~(Eq.~\ref{eq:memory}). (iii)~\emph{Choose}:
a farthest-point / $k$-center coreset selection picks a small, diverse,
representative subset of failures that maximally covers the failure-mode
space~\citep{sener2018coreset,eldar1997fps} (Eq.~\ref{eq:diversity}).
(iv)~\emph{Execute}: a language model~\citep{yang2025qwen3} prescribes a
corrective reset / sub-task-entry scenario for the chosen mode --- the
\emph{start state} of the single new demonstration --- which is projected
onto the task's workspace bounds, rolled by the expert, aggregated, and
the policy is retrained~(Eqs.~\ref{eq:prescribe}--\ref{eq:prescribe-demo}).
This four-stage decomposition is deliberately a \emph{strict generalization}
of the query-based baselines: PACE recovers a Diff-DAgger-style query when
the selected subset has size one, the target mode is the dominant mode, and
the prescribe map is the on-policy identity~(Eq.~\ref{eq:prescribe-demo}).
The extra machinery --- partitioning the round's failures, selecting a
diverse cover, and prescribing where the correction begins --- is what lets
PACE spend each unit of expert budget on a previously-uncorrected failure
mode.

Our design draws two threads together. From the DAgger family we keep the
sound principle of correcting on the learner's own state
distribution~\citep{ross2011dagger,laskey2017dart}, and we reuse the
diffusion loss as the state-level failure signal~\citep{lee2025diffdagger}.
From foundation-model robotics we borrow the ability to \emph{perceive}
and \emph{describe} failures~\citep{liu2023reflect,duan2025aha,
shinn2023reflexion} and to \emph{prescribe} structured control targets ---
here, reset states and sub-task entry points rather than whole
plans or reward functions~\citep{liang2023codeaspolicies,ma2024eureka,
yu2023language2rewards,florensa2017reversecurriculum,eysenbach2018leavenotrace}.
The Assess and Choose stages are, in effect, batch active
learning over failures: cluster into modes and cover them with a
diversity-aware coreset under a query budget~\citep{settles2009active,
ash2020badge,sener2018coreset}. To our knowledge, PACE is the first IIL
method to make \emph{which failures} and \emph{where to correct them}
first-class, learned decisions alongside \emph{when} to query.

\paragraph{Evaluation.}
We evaluate PACE on five tasks spanning discrete navigation and
continuous robot manipulation, each in both a state and an image modality,
against the full DAgger family and a uniform-random query control
(Table~\ref{tab:tasks}). The toy $5{\times}5$ grid is a multimodal
navigation task with a shortest-path expert; Push is ManiSkill
PushT~\citep{mu2021maniskill,tao2024maniskill3,florence2021implicitbc};
and Lift, Wipe, and Door are RoboSuite manipulation tasks on a UR5/UR5e
arm~\citep{zhu2020robosuite,mandlekar2021robomimic}. All robot methods
share one diffusion-policy learner~\citep{chi2023diffusionpolicy,
ho2020ddpm} and differ only in the demonstration-acquisition rule, so the
comparison isolates query efficiency. We measure held-out success rate,
the number of expert queries to reach a target success rate of
$0.90$~(Eqs.~\ref{eq:sr}--\ref{eq:queries}), demo efficiency (area under
the success-rate-vs-demos curve, Eq.~\ref{eq:eff}), and the coverage of
the collected demonstrations over the task space~(Eq.~\ref{eq:cov}).

\begin{table}[t]
\centering
\small
\caption{The five-task evaluation suite. Every task is run in both a
state and an image modality; robot tasks share a single diffusion-policy
learner across all methods so that only the demonstration-acquisition
rule varies.}
\label{tab:tasks}
\begin{tabular}{llll}
\toprule
Task & Domain & Expert & Action \\
\midrule
Toy  & $5{\times}5$ grid            & A$^\star$/BFS       & discrete (4) \\
Push & ManiSkill PushT             & PPO                 & rel.\ joint \\
Lift & RoboSuite (UR5e)            & motion planner      & rel.\ joint \\
Wipe & RoboSuite (UR5)             & motion planner      & rel.\ joint \\
Door & RoboSuite (UR5)             & motion planner      & rel.\ joint \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Contributions.}
Our contributions are:
\begin{enumerate}
\item \textbf{A generalized IIL query.} We recast the DAgger-family query
from a single ``\emph{when}'' decision to a joint ``\emph{when} $+$
\emph{which} $+$ \emph{where}'' decision, and formalize it as PACE, a
four-stage (Perceive$\to$Assess$\to$Choose$\to$Execute) loop that provably
recovers a Diff-DAgger-style query as a special
case~(Eqs.~\ref{eq:perceive}--\ref{eq:prescribe-demo}).
\item \textbf{Failure-mode partitioning and diverse selection.} We
introduce a per-round pipeline that perceives rollout failures with a VLM,
partitions them into failure modes by clustering with a memory that
discourages re-covering old modes, and selects a diverse, representative
subset via $k$-center coreset selection --- so expert budget is spent on
non-redundant modes rather than on the most-detectable
uncertainty~(Eqs.~\ref{eq:phi}--\ref{eq:diversity}).
\item \textbf{LLM-prescribed corrective scenarios.} We use an LLM to
prescribe \emph{where} each corrective demonstration begins --- a reset /
sub-task-entry state projected onto workspace bounds --- rather than
fixing the reset to the rollout's terminal state, turning ``where to
correct'' into a controllable, task-grounded
decision~(Eqs.~\ref{eq:prescribe}--\ref{eq:prescribe-demo}).
\item \textbf{An apples-to-apples query-efficiency study.} Across five
tasks $\times$ two modalities against the full DAgger family with a shared
learner, PACE reaches the target success rate with
\PH{pace-vs-diff-q-reduction} fewer expert queries than the strongest
baseline while attaining a mean held-out success rate of
\PH{mean-pace-sr}, e.g.\ \PH{push-im-pace-sr} on Push (image) at
\PH{push-im-pace-q} queries versus \PH{push-im-diff-q} for Diff-DAgger,
with higher demo efficiency and demonstration coverage
(Eqs.~\ref{eq:eff}--\ref{eq:cov}).
\end{enumerate}
