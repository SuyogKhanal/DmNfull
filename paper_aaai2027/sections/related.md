\section{Related Work}
\label{sec:related}

PACE sits at the intersection of interactive imitation learning (IIL), generative
policy learning, uncertainty-driven active querying, and the use of large
language and vision--language models (LLMs/VLMs) for robot behavior. We organize
the discussion around these threads and, in each, contrast the axis that PACE
controls. The interactive-IL baselines all answer a single question --- \emph{when}
to invoke the expert --- through a per-state query predicate
$\mathrm{Query}(s_t)\!=\!\mathbf{1}[\,\mathrm{score}(s_t)\,\square\,\mathrm{thresh}\,]$
(Eqs.~\eqref{eq:iil-loop}, \eqref{eq:safe}--\eqref{eq:diffdagger}). PACE
generalizes this decision: rather than gating on a single state, it \emph{perceives}
and \emph{partitions} the round's failures into modes, \emph{chooses} a diverse
subset, and \emph{executes} a prescribed corrective entry state
(Eqs.~\eqref{eq:perceive}--\eqref{eq:prescribe-demo}). It thus decides \emph{when},
\emph{which} failure modes, and \emph{where} to place the corrective demonstration.

\subsection{Interactive Imitation Learning and the DAgger Family}
Behavior cloning maps observations to expert actions in a purely supervised
fashion, a paradigm dating to ALVINN's end-to-end road follower
\cite{pomerleau1988alvinn,pomerleau1991efficient}. Naive cloning suffers from
covariate shift: small action errors carry the learner into states absent from
$\mathcal{D}$, and mistakes compound over the horizon
(Eqs.~\eqref{eq:expert}--\eqref{eq:bc}) \cite{ross2011dagger,laskey2017dart}.
DAgger reframes imitation learning as no-regret online learning, aggregating
expert labels on the states the \emph{learner} visits and thereby bounding error
linearly rather than quadratically in the horizon \cite{ross2011dagger}; AggreVaTe
and Deeply AggreVaTeD extend the reduction to cost-to-go objectives and to
differentiable deep policies \cite{ross2014aggrevate,sun2017aggrevated}. Because
vanilla DAgger demands near-constant expert access, a family of \emph{safe} and
query-efficient variants gate \emph{when} to defer: SafeDAgger learns a safety
classifier over the learner--expert action discrepancy (Eq.~\eqref{eq:safe})
\cite{zhang2017safedagger}; DropoutDAgger and EnsembleDAgger threshold
Bayesian-dropout and ensemble-variance signals
(Eqs.~\eqref{eq:dropout}--\eqref{eq:ensemble})
\cite{menda2017dropoutdagger,menda2019ensembledagger}; HG-DAgger cedes control to
a human on demand \cite{kelly2019hgdagger}; LazyDAgger reduces context switching
\cite{hoque2021lazydagger}; and ThriftyDAgger budgets interventions by combined
novelty and task risk (Eq.~\eqref{eq:thrifty}) \cite{hoque2021thriftydagger}. DART
instead injects optimized noise into the expert's demonstrations so that offline
cloning is itself robust to covariate shift \cite{laskey2017dart}. A complementary
deployment-time thread learns continually from targeted human takeovers, weighting
the samples around interventions: IWR up-weights bottleneck states
\cite{mandlekar2020iwr} and Sirius reweights by an approximated trust signal
\cite{liu2023sirius}. The landscape is surveyed by Celemin et al.
\cite{celemin2022iil}.

Every method above decides a single intervention step
$t^\star\!=\!\min\{t:\mathrm{Query}(s_t)\!=\!1\}$ and collects one on-policy
expert segment per round (Eq.~\eqref{eq:iil-loop}). PACE departs on three counts.
First, it replaces per-state uncertainty/safety gating with a VLM \emph{failure
analysis} over completed rollouts (Eq.~\eqref{eq:perceive}). Second, it clusters
those failures into modes and selects a diverse subset, so expert effort is spent
on \emph{curated failure modes} rather than on every uncertain state
(Eqs.~\eqref{eq:partition}--\eqref{eq:diversity}). Third, it prescribes the
\emph{start state} of the corrective demonstration rather than always resuming
on-policy at $t^\star$ (Eqs.~\eqref{eq:prescribe}--\eqref{eq:prescribe-demo}). The
query-based rules are recovered as the special case $|\mathcal{S}|\!=\!1$,
$C_{\mathrm{tgt}}\!=\!C^\star$ with an identity (on-policy) prescribe map, so the
DAgger family is a strict restriction of PACE.

\subsection{Diffusion Policies and Generative Imitation Learning}
Denoising diffusion probabilistic models established a stable, expressive class of
generative models \cite{ho2020ddpm}, which Diffuser adapted to trajectory-level
planning \cite{janner2022diffuser} and Diffusion Policy adapted to visuomotor
control, capturing the multimodal action distributions that unimodal regression
collapses (Eqs.~\eqref{eq:forward}--\eqref{eq:vpred})
\cite{chi2023diffusionpolicy}. Implicit Behavioral Cloning captures similar
multimodality through energy-based models and contributes the PushT manipulation
task we adopt \cite{florence2021implicitbc}. Most directly related, Diff-DAgger
derives an interactive-IL query signal from the diffusion training objective
itself, thresholding the per-step diffusion loss against a quantile of the
training-loss CDF (Eqs.~\eqref{eq:diffsignal}, \eqref{eq:diffdagger})
\cite{lee2025diffdagger}. PACE uses a diffusion policy as its learner and reuses
this very loss $\ell_t$ as the analysis frame for locating each failure's peak
(Eq.~\eqref{eq:perceive}); it is therefore complementary to Diff-DAgger --- rather
than \emph{triggering} a query from model uncertainty, it \emph{groups} the
observed failures and prescribes where to collect corrections.

\subsection{Uncertainty Estimation for Querying}
The gating decisions of the safe-DAgger variants rest on predictive uncertainty,
most commonly Monte-Carlo dropout \cite{gal2016dropout} and deep ensembles
\cite{lakshminarayanan2017ensembles}, which DropoutDAgger and EnsembleDAgger
instantiate directly as safety signals
\cite{menda2017dropoutdagger,menda2019ensembledagger}, and which motivate
uncertainty-aware reset/abort schemes in autonomous RL \cite{eysenbach2018leavenotrace}.
Such action-level uncertainty is known to be brittle for expressive multimodal
policies, where legitimate mode disagreement is misread as doubt
\cite{lee2025diffdagger}. PACE deliberately avoids calibrated action-uncertainty
as its selection criterion, keying instead on observed failure \emph{outcomes}
and their geometry (Eq.~\eqref{eq:phi}).

\subsection{LLMs and VLMs for Robot Planning, Code, and Reward}
A large body of work turns foundation models into sources of robot behavior:
SayCan grounds LLM plans in learned affordances \cite{ahn2022saycan}, Inner
Monologue closes the loop with textual environment feedback
\cite{huang2022innermonologue}, and PaLM-E and RT-2 fuse perception with language
into embodied and vision--language--action models
\cite{driess2023palme,brohan2023rt2}. A complementary line has LLMs emit
\emph{executable structure} --- robot policy code, as in Code as Policies and
ProgPrompt \cite{liang2023codeaspolicies,singh2023progprompt}, or composable 3D
value maps, as in VoxPoser \cite{huang2023voxposer}. Closest to our Execute
(prescribe) stage, Eureka and Language-to-Rewards use LLMs to synthesize and
iteratively refine reward code and parameters
\cite{ma2024eureka,yu2023language2rewards}. PACE narrows this generativity to a
different target: instead of whole plans, policy code, or reward functions, its
LLM emits a compact \texttt{SceneCommand} that the prescribe map $g(\cdot)$ turns
into a corrective \emph{reset / sub-task-entry} distribution $\xi$, bounded by an
$L_2$-capped, workspace-clamped perturbation of the target anchor
(Eq.~\eqref{eq:prescribe}). The output is thus a single start state from which a
fresh expert demonstration is collected, not an executed behavior. In our
implementation the prescribing LLM is Qwen3 \cite{yang2025qwen3}, whose unified
thinking / non-thinking modes we treat as two experimental setups (reasoning
enabled vs.\ disabled).

\subsection{LLM/VLM Failure Reflection and Self-Correction}
A recent thread has models reason over their own failures. REFLECT summarizes
multimodal robot experience into a hierarchical narrative to explain and correct
task failures \cite{liu2023reflect}, and AHA is a VLM trained specifically to
detect and reason over failures in robotic manipulation \cite{duan2025aha}; in the
language domain, Reflexion and Self-Refine show that models iteratively critique
and improve their own outputs
\cite{shinn2023reflexion,madaan2023selfrefine}. These works motivate PACE's
Perceive stage, in which a VLM describes \emph{why} a rollout failed as the first
step of the pipeline (Eq.~\eqref{eq:perceive}); concretely, we use the Qwen3-VL
vision--language model \cite{bai2025qwen3vl}. Unlike this line, which targets
one-off explanation or single-trajectory recovery, PACE aggregates the round's
failure descriptions into a structured feature set that is then \emph{clustered
and prioritized} to drive interactive data collection, closing the loop back into
policy training rather than into a corrective plan for the current episode.

\subsection{Active Learning: Coreset, Diversity, and Clustering}
Pool-based active learning selects the most informative examples under a labeling
budget \cite{settles2009active}, and modern diverse-batch methods inform our
Assess (partition) and Choose (prioritize) stages. The core-set / $k$-center
formulation selects points for max-min coverage of a feature space
\cite{sener2018coreset}; BADGE unifies predictive uncertainty and diversity via
gradient embeddings \cite{ash2020badge}; and farthest-point sampling yields
deterministic, uniform min-max coverage \cite{eldar1997fps}. PACE clusters rollout
failures into modes with $K$-means over standardized geometric (or R3M-visual)
descriptors \cite{lloyd1982kmeans}, choosing $k$ by silhouette
(Eq.~\eqref{eq:partition}), then applies farthest-point / coreset selection with
the target-mode representative forced in and a worst-loss seed added
(Eq.~\eqref{eq:diversity}). Whereas classical active learning selects
\emph{unlabeled inputs} to query, PACE selects \emph{failure modes} to correct and
then prescribes the corrective state --- transporting coreset diversity from label
acquisition to demonstration targeting.

\subsection{Visual Representations for Manipulation}
Pre-trained visual encoders supply the features PACE clusters over in image mode:
R3M learns a reusable representation from egocentric human video
\cite{nair2022r3m}, MVP uses masked visual pre-training \cite{radosavovic2022mvp},
and VIP learns a value-implicit representation with a dense visual reward
\cite{ma2023vip}. We use frozen R3M embeddings, PCA-reduced, as the visual failure
descriptor $\tilde v_i$ (Eq.~\eqref{eq:phi}). These encoders are orthogonal to
PACE's contribution: they determine \emph{how} an observation is embedded, whereas
PACE determines \emph{which} corrective demonstrations to collect.

\subsection{Simulators, Benchmarks, and Reset Curricula}
Our evaluation builds on the ManiSkill line of large-scale manipulation
benchmarks \cite{mu2021maniskill,gu2023maniskill2,tao2024maniskill3}, from which
we take the Push task; on robomimic's systematic study of learning from offline
human demonstrations \cite{mandlekar2021robomimic}; and on the robosuite framework
\cite{zhu2020robosuite}, which supplies the Lift, Wipe, and Door tasks we run on a
UR5/UR5e arm; the PushT task originates in Implicit BC \cite{florence2021implicitbc}.
Finally, PACE's Execute stage is conceptually related to reset-state and
curriculum generation: reverse curriculum generation grows a distribution of
start states outward from the goal \cite{florensa2017reversecurriculum}, and
learning-to-reset enables autonomous operation \cite{eysenbach2018leavenotrace}.
PACE differs in that its prescribed entry states are not driven by a difficulty
curriculum but are \emph{anchored to perceived, clustered failure modes} and
realized as a VLM/LLM-specified corrective distribution
(Eqs.~\eqref{eq:prescribe}--\eqref{eq:prescribe-demo}), tying reset-state choice
directly to the learner's observed failure structure.
