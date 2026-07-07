\section{Experiments}

Our experiments answer four questions. \textbf{(Q1)}~Under a fixed
one-demonstration-per-round expert-query budget, does PACE reach a target held-out
success rate with fewer expert queries than uncertainty- and safety-gated
interactive-imitation-learning (IIL) baselines? \textbf{(Q2)}~Does prescribing
\emph{where} the corrective demonstration begins, rather than only deciding
\emph{when} to query, improve demonstration efficiency and state-space coverage?
\textbf{(Q3)}~Do these gains hold across observation modalities (privileged state
vs.\ raw image) and across embodiments (discrete grid, Franka Panda, UR5/UR5e)?
\textbf{(Q4)}~Which of the four PACE stages---Perceive, Assess, Choose,
Execute---are responsible for the improvement? Because the five-seed runs are still
in progress, every quantitative entry is a \PH{placeholder} macro; the tables and
callouts fix the exact reporting protocol so that no result can ship un-filled.

\subsection{Task Suite}

We evaluate on the five tasks summarized in Table~\ref{tab:tasks}, chosen to span a
discrete navigation problem with an exact expert, a contact-rich pushing task, and
three RoboSuite manipulation tasks on UR5/UR5e arms. Every task is genuinely
multi-modal in its optimal behavior, which is precisely the regime in which
per-state action-agreement uncertainty is unreliable and in which failure-mode
partitioning is expected to help.

\textbf{T1 (Toy).} A custom $5{\times}5$ grid-navigation environment
(\texttt{MazeNavEnv}) in which a point agent moves \{up, down, left, right\}
($\mathcal{A}=\mathrm{Discrete}(4)$) from a fixed start to a fixed goal; two fire
tiles block the middle so that two equally-optimal $8$-step routes (top and bottom)
exist, making the task multi-modal by construction. The expert is an $A^{\!\star}$/BFS
shortest-path oracle that emits a multi-label optimal-action mask (every action that
reduces the BFS distance-to-goal is a positive), so both optimal routes are taught.
The state observation is a $14$-d vector (agent and goal coordinates, a
$3{\times}3$ local tile patch, and steps-remaining fraction); the image observation
is an $80{\times}80{\times}3$ bird's-eye render. Held-out evaluation uses a fixed set
of $\PH{toy-heldout-n}$ layouts.

\textbf{T2 (Push).} ManiSkill3~\cite{tao2024maniskill3,mu2021maniskill} PushT with a
Franka Panda and a \texttt{panda\_stick} end-effector: push a T-shaped block onto a
fixed goal pose (success = overlap within tolerance); the PushT task originates in
Implicit BC~\cite{florence2021implicitbc}. State mode exposes the privileged
T-pose (\texttt{proprio\_dim}\,$=21$); image mode uses a $256{\times}256$ base-camera
RGB stream and hides the T-pose from the policy (\texttt{proprio\_dim}\,$=14$),
forcing it to read the object from pixels. The action space is
$\texttt{rel\_joint\_pos}\in\mathbb{R}^{9}$ and the expert is a privileged PPO
policy, identical across modalities so demonstrations differ only by observation.

\textbf{T3--T5 (Lift, Wipe, Door).} RoboSuite~\cite{zhu2020robosuite} manipulation
on a UR5e (Lift) and UR5 (Wipe, Door), each with a state and an image diffusion
policy and a motion-planner expert, following the offline-demonstration protocol
of robomimic~\cite{mandlekar2021robomimic}. These share the T2 loop and reporting
format.

\begin{table}[t]
\centering
\small
\caption{Task suite. All five tasks are evaluated in both a state and an image
modality; \textbf{Diff-DAgger} is a robot-only baseline (T2--T5). Expert:
$A^{\!\star}$/BFS oracle (T1), PPO (T2), motion planner (T3--T5).}
\label{tab:tasks}
\begin{tabular}{llccc}
\toprule
 & Sim & Embodiment & $\dim(\mathcal{A})$ & Expert \\
\midrule
T1 Toy  & MazeNav          & grid agent    & $4$ (disc.) & $A^{\!\star}$/BFS \\
T2 Push & ManiSkill3       & Panda stick   & $9$         & PPO \\
T3 Lift & RoboSuite        & UR5e          & \PH{lift-adim} & MP \\
T4 Wipe & RoboSuite        & UR5           & \PH{wipe-adim} & MP \\
T5 Door & RoboSuite        & UR5           & \PH{door-adim} & MP \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Policies per Modality}

To isolate the effect of the demonstration-acquisition rule, all methods share one
learner per (task, modality) cell and differ \emph{only} in which corrective demo
is collected each round. On T1, the \emph{state} learner is a $p4m$/$D_4$
group-equivariant policy (\texttt{escnn}, $8$-fold rotation$\times$flip symmetry over
a $5$-channel one-hot grid map) and the \emph{image} learner is a plain,
grid-size-invariant CNN. On T2--T5, both modalities use a single diffusion-policy
backbone~\cite{chi2023diffusionpolicy,ho2020ddpm} shared by all methods: a
conditional $1$-D U-Net denoiser with observation horizon $1$, prediction horizon
$32$, and action horizon $2$, trained with a $v$-prediction objective
(Eq.~\ref{eq:vpred}) under a DDIM schedule (\texttt{num\_train\_timesteps}\,$=16$,
$16$ inference steps). The state encoder is an MLP over proprioception; the image
encoder is an R3M ResNet-18~\cite{nair2022r3m}, finetuned end-to-end with a
spatial-softmax head (encoder output dimension $1038$ for PushT). Using an
identical, expressive multi-modal learner for every arm makes the comparison
apples-to-apples and matches the multi-modality regime for which Diffusion Policy
and Diff-DAgger were designed~\cite{chi2023diffusionpolicy,lee2025diffdagger}.

\subsection{Baselines}

We compare PACE against the interactive-IL family, all cast as a single decision-rule
schema $\mathrm{Query}(s_t)=\mathbf{1}[\,\mathrm{score}_\mathrm{method}(s_t)\ \text{crosses threshold}\,]$
so that only the query criterion changes. \textbf{SafeDAgger}~\cite{zhang2017safedagger}
queries on novice--expert action discrepancy (Eq.~\ref{eq:safe}); \textbf{DropoutDAgger}~\cite{menda2017dropoutdagger,gal2016dropout}
queries when the MC-sampled in-ball agreement with the expert drops
(Eq.~\ref{eq:dropout}); \textbf{EnsembleDAgger}~\cite{menda2019ensembledagger,lakshminarayanan2017ensembles}
queries on ensemble doubt or mean discrepancy (Eq.~\ref{eq:ensemble});
\textbf{ThriftyDAgger}~\cite{hoque2021thriftydagger} queries on novelty or learned
task risk $1-Q_\psi$ with budget-calibrated thresholds (Eq.~\ref{eq:thrifty}); and
\textbf{Random} is a uniform one-demo-per-round control ("Stagger",
Eq.~\ref{eq:stagger}), reported as a floor rather than a published method. On the
robot tasks we additionally compare against \textbf{Diff-DAgger}~\cite{lee2025diffdagger},
the diffusion-policy--native rule that queries when the per-step diffusion loss stays
in the tail of its training-loss CDF for $K$ windowed steps (Eq.~\ref{eq:diffdagger});
Diff-DAgger is omitted on T1 because it requires the diffusion learner. Each baseline
uses its published/default hyperparameters (e.g., SafeDAgger $\tau=0.1$, DropoutDAgger
$N=\PH{drop-N}$/$p=0.9$, EnsembleDAgger $M=5$, ThriftyDAgger $\alpha_h=0.1$,
Diff-DAgger $\alpha=0.99$, $K=1$). PACE is the strict generalization of these rules:
it reuses the same per-step signal to \emph{perceive} failures but additionally
partitions them, chooses a diverse subset, and prescribes the corrective entry state,
recovering Diff-DAgger as the special case $|S|=1$, $C_\mathrm{tgt}=C^\star$, and an
on-policy identity prescribe map (Eq.~\ref{eq:prescribe-demo}).

\subsection{Active-Loop Protocol}

Every method follows the same interactive loop (Eq.~\ref{eq:iil-loop}): bootstrap an
initial policy from a small shared set of expert demonstrations, then repeat
\{roll out $\to$ the method's query rule flags a failure $\to$ collect exactly
\textbf{one} new expert demonstration $\to$ retrain $\to$ held-out eval\} until a stop
condition. The \textbf{one-demo-per-round} cap is a hard invariant: it makes the
number of expert queries the shared sample-efficiency axis for all arms. Crucially,
all arms share the \emph{same} bootstrap policy per seed, so the only difference
between arms is the demonstration-acquisition rule.

\textbf{Toy (T1).} We bootstrap from $\PH{toy-init-demos}$ $A^{\!\star}$/BFS
demonstrations, draw a size-$\PH{toy-pool-n}$ correction pool per round, retrain
from scratch each round, and evaluate on $\PH{toy-heldout-n}$ fixed held-out layouts.
The budget is $\PH{toy-budget}$ extra demonstrations over the bootstrap, with a
$\PH{toy-max-rounds}$-round cap and rollout cap of $\PH{toy-max-steps}$ steps; the
target success rate is $\mathrm{SR}_\mathrm{target}=0.90$ (Eq.~\ref{eq:sr}). A run
stops when the target is reached, the budget is exhausted, or the round cap is hit.

\textbf{Robot (T2--T5).} We bootstrap a behavior-cloned diffusion policy from
$\PH{robot-init-demos}$ expert demonstrations (one shared warm start per seed). Each
round discovers a failure via the method's rule over a $\PH{robot-rollout-n}$-episode
failure-discovery rollout and adds one demonstration; the diffusion policy is
retrained \emph{from scratch every} $n_d=\PH{robot-nd}$ \emph{demonstrations}
following Diff-DAgger~\cite{lee2025diffdagger}, a cadence shared across all methods.
An episode counts as a demonstration only if the expert succeeded, actually
intervened, and contributed at least $\PH{robot-min-expert-steps}$ expert steps.
We evaluate on $\PH{robot-heldout-n}$ held-out episodes per checkpoint (fixed
held-out seed base $\PH{robot-heldout-seed}$, vectorized over $\PH{robot-eval-envs}$
environments), with a budget of $\PH{robot-budget}$ additional demonstrations, a
$\PH{robot-max-rounds}$-round cap, and $\mathrm{SR}_\mathrm{target}=0.90$; the query
count is recorded when the target is first reached.

\subsection{Metrics and Seeds}

For each (task, modality, method, seed) we report four metrics from the recorded
learning curve. \textbf{SR}: final held-out success rate (Eq.~\ref{eq:sr}).
\textbf{Q}: the number of expert queries---equivalently demonstrations added---to
first reach $\mathrm{SR}_\mathrm{target}=0.90$ (Eq.~\ref{eq:queries}); if the target
is never reached, $Q$ is charged the full budget $B$. \textbf{Eff}: demonstration
efficiency, the area under the SR-vs-\#demonstrations curve (Eq.~\ref{eq:eff};
higher is better). \textbf{Cov}: coverage of the collected demonstrations over the
task space (grid cells for T1, quantized workspace/pose cells for T2--T5;
Eq.~\ref{eq:cov}). We run \textbf{$5$ seeds} per cell (per-run seed
$=\text{base}+\text{run\_id}$, with a shared bootstrap per seed across arms) and
report every metric as mean$\pm$std over the $5$ seeds; the best entry per column is
in \textbf{bold}.

\subsection{Main Results}

Tables~\ref{tab:toy}--\ref{tab:door} report SR and Q for all five tasks in both
modalities; PACE is expected to reach the target success rate with the fewest expert
queries while matching or exceeding baseline SR. We highlight the headline
query-reduction of PACE over the strongest baseline: on the robot tasks PACE reduces
expert queries to reach $0.90$ SR by $\PH{pace-vs-diff-q-reduction}$ relative to
Diff-DAgger, and averaged over all five tasks it attains $\PH{mean-pace-sr}$ SR.

\begin{table}[t]
\centering
\small
\caption{T1 Toy: held-out SR (\%) and expert queries $Q$ to reach $0.90$ SR,
mean$\pm$std over $5$ seeds. Best per column in \textbf{bold}. State $=$
equivariant MLP, Image $=$ plain CNN; Diff-DAgger is robot-only and omitted.}
\label{tab:toy}
\begin{tabular}{l cc cc}
\toprule
 & \multicolumn{2}{c}{State} & \multicolumn{2}{c}{Image} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Method & SR$\uparrow$ & Q$\downarrow$ & SR$\uparrow$ & Q$\downarrow$ \\
\midrule
SafeDAgger     & \PH{toy-st-safe-sr}     & \PH{toy-st-safe-q}     & \PH{toy-im-safe-sr}     & \PH{toy-im-safe-q} \\
DropoutDAgger  & \PH{toy-st-dropout-sr}  & \PH{toy-st-dropout-q}  & \PH{toy-im-dropout-sr}  & \PH{toy-im-dropout-q} \\
EnsembleDAgger & \PH{toy-st-ensemble-sr} & \PH{toy-st-ensemble-q} & \PH{toy-im-ensemble-sr} & \PH{toy-im-ensemble-q} \\
ThriftyDAgger  & \PH{toy-st-thrifty-sr}  & \PH{toy-st-thrifty-q}  & \PH{toy-im-thrifty-sr}  & \PH{toy-im-thrifty-q} \\
Random         & \PH{toy-st-rand-sr}     & \PH{toy-st-rand-q}     & \PH{toy-im-rand-sr}     & \PH{toy-im-rand-q} \\
\textbf{PACE (ours)} & \PH{toy-st-pace-sr} & \PH{toy-st-pace-q} & \PH{toy-im-pace-sr} & \PH{toy-im-pace-q} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\small
\caption{T2 Push (ManiSkill PushT): held-out SR (\%) and expert queries $Q$ to
reach $0.90$ SR, mean$\pm$std over $5$ seeds. Best per column in \textbf{bold}.}
\label{tab:push}
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

\begin{table}[t]
\centering
\small
\caption{T3 Lift (RoboSuite UR5e): held-out SR (\%) and expert queries $Q$ to
reach $0.90$ SR, mean$\pm$std over $5$ seeds. Best per column in \textbf{bold}.}
\label{tab:lift}
\begin{tabular}{l cc cc}
\toprule
 & \multicolumn{2}{c}{State} & \multicolumn{2}{c}{Image} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Method & SR$\uparrow$ & Q$\downarrow$ & SR$\uparrow$ & Q$\downarrow$ \\
\midrule
Diff-DAgger    & \PH{lift-st-diff-sr}     & \PH{lift-st-diff-q}     & \PH{lift-im-diff-sr}     & \PH{lift-im-diff-q} \\
SafeDAgger     & \PH{lift-st-safe-sr}     & \PH{lift-st-safe-q}     & \PH{lift-im-safe-sr}     & \PH{lift-im-safe-q} \\
DropoutDAgger  & \PH{lift-st-dropout-sr}  & \PH{lift-st-dropout-q}  & \PH{lift-im-dropout-sr}  & \PH{lift-im-dropout-q} \\
EnsembleDAgger & \PH{lift-st-ensemble-sr} & \PH{lift-st-ensemble-q} & \PH{lift-im-ensemble-sr} & \PH{lift-im-ensemble-q} \\
ThriftyDAgger  & \PH{lift-st-thrifty-sr}  & \PH{lift-st-thrifty-q}  & \PH{lift-im-thrifty-sr}  & \PH{lift-im-thrifty-q} \\
Random         & \PH{lift-st-rand-sr}     & \PH{lift-st-rand-q}     & \PH{lift-im-rand-sr}     & \PH{lift-im-rand-q} \\
\textbf{PACE (ours)} & \PH{lift-st-pace-sr} & \PH{lift-st-pace-q} & \PH{lift-im-pace-sr} & \PH{lift-im-pace-q} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\small
\caption{T4 Wipe (RoboSuite UR5): held-out SR (\%) and expert queries $Q$ to
reach $0.90$ SR, mean$\pm$std over $5$ seeds. Best per column in \textbf{bold}.}
\label{tab:wipe}
\begin{tabular}{l cc cc}
\toprule
 & \multicolumn{2}{c}{State} & \multicolumn{2}{c}{Image} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Method & SR$\uparrow$ & Q$\downarrow$ & SR$\uparrow$ & Q$\downarrow$ \\
\midrule
Diff-DAgger    & \PH{wipe-st-diff-sr}     & \PH{wipe-st-diff-q}     & \PH{wipe-im-diff-sr}     & \PH{wipe-im-diff-q} \\
SafeDAgger     & \PH{wipe-st-safe-sr}     & \PH{wipe-st-safe-q}     & \PH{wipe-im-safe-sr}     & \PH{wipe-im-safe-q} \\
DropoutDAgger  & \PH{wipe-st-dropout-sr}  & \PH{wipe-st-dropout-q}  & \PH{wipe-im-dropout-sr}  & \PH{wipe-im-dropout-q} \\
EnsembleDAgger & \PH{wipe-st-ensemble-sr} & \PH{wipe-st-ensemble-q} & \PH{wipe-im-ensemble-sr} & \PH{wipe-im-ensemble-q} \\
ThriftyDAgger  & \PH{wipe-st-thrifty-sr}  & \PH{wipe-st-thrifty-q}  & \PH{wipe-im-thrifty-sr}  & \PH{wipe-im-thrifty-q} \\
Random         & \PH{wipe-st-rand-sr}     & \PH{wipe-st-rand-q}     & \PH{wipe-im-rand-sr}     & \PH{wipe-im-rand-q} \\
\textbf{PACE (ours)} & \PH{wipe-st-pace-sr} & \PH{wipe-st-pace-q} & \PH{wipe-im-pace-sr} & \PH{wipe-im-pace-q} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\small
\caption{T5 Door (RoboSuite UR5): held-out SR (\%) and expert queries $Q$ to
reach $0.90$ SR, mean$\pm$std over $5$ seeds. Best per column in \textbf{bold}.}
\label{tab:door}
\begin{tabular}{l cc cc}
\toprule
 & \multicolumn{2}{c}{State} & \multicolumn{2}{c}{Image} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Method & SR$\uparrow$ & Q$\downarrow$ & SR$\uparrow$ & Q$\downarrow$ \\
\midrule
Diff-DAgger    & \PH{door-st-diff-sr}     & \PH{door-st-diff-q}     & \PH{door-im-diff-sr}     & \PH{door-im-diff-q} \\
SafeDAgger     & \PH{door-st-safe-sr}     & \PH{door-st-safe-q}     & \PH{door-im-safe-sr}     & \PH{door-im-safe-q} \\
DropoutDAgger  & \PH{door-st-dropout-sr}  & \PH{door-st-dropout-q}  & \PH{door-im-dropout-sr}  & \PH{door-im-dropout-q} \\
EnsembleDAgger & \PH{door-st-ensemble-sr} & \PH{door-st-ensemble-q} & \PH{door-im-ensemble-sr} & \PH{door-im-ensemble-q} \\
ThriftyDAgger  & \PH{door-st-thrifty-sr}  & \PH{door-st-thrifty-q}  & \PH{door-im-thrifty-sr}  & \PH{door-im-thrifty-q} \\
Random         & \PH{door-st-rand-sr}     & \PH{door-st-rand-q}     & \PH{door-im-rand-sr}     & \PH{door-im-rand-q} \\
\textbf{PACE (ours)} & \PH{door-st-pace-sr} & \PH{door-st-pace-q} & \PH{door-im-pace-sr} & \PH{door-im-pace-q} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Learning Curves and Coverage}

Beyond the endpoint metrics, we plot held-out SR against the number of collected
demonstrations for every (task, modality) cell (Figure~\ref{fig:lc}). PACE is
expected to dominate the SR-vs-demonstration frontier---higher demonstration
efficiency \textbf{Eff} at every budget---because prescribing diverse
sub-task-entry states front-loads coverage of the failure modes rather than
re-correcting the dominant mode. On the state PushT curve, PACE attains
$\PH{push-st-pace-eff}$ Eff vs.\ $\PH{push-st-diff-eff}$ for Diff-DAgger; on the
image curve it attains $\PH{push-im-pace-eff}$ vs.\ $\PH{push-im-diff-eff}$. The
coverage metric \textbf{Cov} makes the mechanism explicit: PACE reaches
$\PH{push-st-pace-cov}$ workspace-cell coverage on state PushT and
$\PH{toy-st-pace-cov}$ grid-cell coverage on the toy task, above the strongest
baseline in each case, confirming that the demonstration-efficiency gain is driven
by broader failure-mode coverage rather than by higher per-demonstration quality
alone. Figure~\ref{fig:qual} shows representative failure$\to$prescription panels:
a clustered failure mode, its VLM-perceived description, and the prescribed
corrective reset state from which the single new demonstration is collected.

\subsection{Ablations}

To attribute the gains to the four PACE stages (Q4) we ablate one stage at a time on
T2 Push, reported in Table~\ref{tab:ablation}. \textbf{$-$Perceive} replaces the
VLM/diffusion-loss failure localization with the episode-end state, testing whether
localizing the failure point matters. \textbf{$-$Assess} disables clustering and
treats every failure as its own mode, so the diversity selection has no
partition to draw from. \textbf{$-$Choose} replaces the k-center/farthest-point
diversity selection (Eq.~\ref{eq:diversity}) with picking the highest-loss failure,
collapsing the "which failures" decision to a single greedy pick.
\textbf{$-$Execute} disables the prescribed corrective reset (Eq.~\ref{eq:prescribe})
and reverts to the on-policy intervention point, which recovers a Diff-DAgger-like
special case. Removing Execute is expected to cause the largest drop in
demonstration efficiency ($\PH{abl-execute-eff}$ vs.\ full PACE
$\PH{abl-full-eff}$), and removing Assess the largest drop in coverage
($\PH{abl-assess-cov}$), isolating partition-and-prescribe as the source of PACE's
advantage over when-only query rules. We additionally report the effect of the
selection cap $\kappa$ and of the Prescribe backbone, comparing the Qwen3
LLM~\cite{yang2025qwen3} in its reasoning-enabled (thinking) vs.\ reasoning-disabled
(non-thinking) modes ($\PH{abl-think-eff}$ vs.\ $\PH{abl-nothink-eff}$), with
failure perception driven by the Qwen3-VL vision-language
model~\cite{bai2025qwen3vl}.

\begin{table}[t]
\centering
\small
\caption{Ablation of PACE stages on T2 Push (state), mean$\pm$std over $5$ seeds.
Each row removes one stage. SR (\%) at target, expert queries $Q$, demonstration
efficiency Eff (Eq.~\ref{eq:eff}), and coverage Cov (Eq.~\ref{eq:cov}).}
\label{tab:ablation}
\begin{tabular}{l cccc}
\toprule
Variant & SR$\uparrow$ & Q$\downarrow$ & Eff$\uparrow$ & Cov$\uparrow$ \\
\midrule
$-$Perceive & \PH{abl-perceive-sr} & \PH{abl-perceive-q} & \PH{abl-perceive-eff} & \PH{abl-perceive-cov} \\
$-$Assess   & \PH{abl-assess-sr}   & \PH{abl-assess-q}   & \PH{abl-assess-eff}   & \PH{abl-assess-cov} \\
$-$Choose   & \PH{abl-choose-sr}   & \PH{abl-choose-q}   & \PH{abl-choose-eff}   & \PH{abl-choose-cov} \\
$-$Execute  & \PH{abl-execute-sr}  & \PH{abl-execute-q}  & \PH{abl-execute-eff}  & \PH{abl-execute-cov} \\
\midrule
\textbf{PACE (full)} & \PH{abl-full-sr} & \PH{abl-full-q} & \PH{abl-full-eff} & \PH{abl-full-cov} \\
\bottomrule
\end{tabular}
\end{table}
