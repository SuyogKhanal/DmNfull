\section{Method}

We present \textbf{PACE} (\textbf{P}erceive $\rightarrow$ \textbf{A}ssess
$\rightarrow$ \textbf{C}hoose $\rightarrow$ \textbf{E}xecute), an interactive
imitation-learning (IIL) loop for diffusion policies that decides not only
\emph{when} to ask the expert for help, but \emph{which} failures to correct and
\emph{where} the corrective demonstration should begin. We first fix the problem
setup and the diffusion-policy learner (\S\,Preliminaries), then cast the classical
DAgger-family query rules as instances of one shared decision family
(\S\,A Unified Query Framework), and finally define PACE as a strict
generalization of that family (\S\,PACE). Throughout, the four PACE stages map to
the equations as follows: \emph{Perceive} $=$ VLM/uncertainty failure localization
and featurization (Eqs.~\ref{eq:perceive}--\ref{eq:phi}); \emph{Assess} $=$
clustering the round's failures into failure modes (Eqs.~\ref{eq:partition}--\ref{eq:dominant});
\emph{Choose} $=$ memory-aware diversity/k-center selection of which failures to
correct (Eqs.~\ref{eq:memory}--\ref{eq:diversity}); \emph{Execute} $=$ prescribing
the corrective reset / sub-task-entry scenario, collecting the expert demo, and
retraining (Eqs.~\ref{eq:prescribe}--\ref{eq:prescribe-demo}).

\subsection{Preliminaries}

\paragraph{Sequential decision process.}
Each task is a finite-horizon (PO)MDP
\begin{equation}
\mathcal{M} = \big(\, \mathcal{S},\ \mathcal{A},\ P(s'\mid s,a),\ R(s,a),\ H,\ \gamma_{\mathrm{env}} \,\big),
\qquad
\pi_\theta:\ \mathcal{S}\to\Delta(\mathcal{A}),
\label{eq:mdp}
\end{equation}
where the state $s$ is privileged proprioception in the state-based setting and a
partial image observation in the image-based setting, $a$ is a discrete move (toy
grid) or a relative joint-position delta (manipulation), and $\pi_\theta$ is the
learner (novice) policy. An expert oracle $\pi^\star$ (A$^\star$/BFS on the grid,
PPO / motion-planner on the robots) induces demonstrations
\begin{equation}
\pi^\star=\arg\max_{\pi}\ \mathbb{E}_{\tau\sim\pi}\!\Big[\textstyle\sum_{t=0}^{H}\gamma_{\mathrm{env}}^{\,t} R(s_t,a_t)\Big],
\qquad
\mathcal{D}=\big\{(s_t,a_t)\ :\ a_t=\pi^\star(s_t)\big\},
\label{eq:expert}
\end{equation}
and behavior cloning fits the policy to the aggregated data,
\begin{equation}
\theta^\star=\arg\min_{\theta}\ \mathbb{E}_{(s,a)\sim\mathcal{D}}\big[\,\mathcal{L}_{\mathrm{BC}}\big(\pi_\theta(\cdot\mid s),\,a\big)\,\big],
\label{eq:bc}
\end{equation}
with $\mathcal{L}_{\mathrm{BC}}=-\log\pi_\theta(a\mid s)$ in the discrete case and
$\|\pi_\theta(s)-a\|_2^2$ in the continuous case. Because the learner visits states
its own errors produce, offline cloning suffers covariate shift
\cite{pomerleau1988alvinn,pomerleau1991efficient}; DAgger corrects this by
aggregating expert labels on \emph{learner-visited} states
\cite{ross2011dagger,ross2014aggrevate,sun2017aggrevated}.

\paragraph{Interactive-IL loop.}
All methods we study share one loop skeleton. In round $r$ the learner
$\pi_{\theta_r}$ is rolled out, a \emph{query predicate} $\mathrm{Query}(s_t)$
flags a first intervention step $t^\star$, the expert takes over from there, and
the resulting expert segment is the single new demonstration:
\begin{align}
\text{roll } \pi_{\theta_r}\ &\Rightarrow\ t^\star=\min\{t:\ \mathrm{Query}(s_t)=1\}, \nonumber\\
\mathcal{D}_{r+1}&=\mathcal{D}_r\ \cup\ \big\{(s_t,\pi^\star(s_t)):\ t\ge t^\star\big\},\quad
\theta_{r+1}=\arg\min_{\theta}\ \mathbb{E}_{\mathcal{D}_{r+1}}[\mathcal{L}_{\mathrm{BC}}].
\label{eq:iil-loop}
\end{align}
Exactly one demonstration is added per round; the policy is retrained from scratch
every $n_d$ demos; the loop stops when a held-out target success rate
$\mathrm{SR}_{\mathrm{target}}=0.90$ is reached or a budget $B$ of demos is spent.
This one-demo-per-round protocol is the fair sample-efficiency axis on which every
method is compared. The \emph{only} degree of freedom across methods is how the new
demonstration is chosen; the baselines fully specify it through
$\mathrm{Query}(\cdot)$ (a single ``\emph{when}''), whereas PACE additionally
chooses \emph{which} failures and \emph{where} to start the demo.

\paragraph{Diffusion-policy learner.}
Every robot-task method (PACE, Diff-DAgger, and all IIL baselines) shares one
diffusion-policy backbone \cite{chi2023diffusionpolicy,ho2020ddpm,janner2022diffuser}
so that only the demo-acquisition rule differs. With clean action target
$x_0=a$, the forward process noises $x_0$ over $K_{\mathrm{dif}}$ steps,
\begin{equation}
x_k=\sqrt{\bar\alpha_k}\,x_0+\sqrt{1-\bar\alpha_k}\,\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I),\quad k\in\{1,\dots,K_{\mathrm{dif}}\},
\label{eq:forward}
\end{equation}
and the network is trained with a $v$-prediction denoising objective (conditioning
on $s$ suppressed for brevity),
\begin{equation}
v_k=\sqrt{\bar\alpha_k}\,\epsilon-\sqrt{1-\bar\alpha_k}\,x_0,
\qquad
\mathcal{L}_{\mathrm{DP}}(\theta)=\mathbb{E}_{k,\epsilon,(s,a)}\big[\big\|v_\theta(x_k,k,s)-v_k\big\|_2^2\big].
\label{eq:vpred}
\end{equation}
Crucially, the \emph{per-pair denoising loss} doubles as an uncertainty signal
\cite{lee2025diffdagger}: averaged over noise draws it scores how out-of-distribution
a state--action pair is, and evaluated at the novice's own executed action along a
rollout it gives the per-step signal $\ell_t$,
\begin{equation}
L_{\mathrm{dif}}(s,a)=\mathbb{E}_{k,\epsilon}\big[\big\|v_\theta(x_k,k,s)-v_k\big\|_2^2\big],
\qquad
\ell_t \;=\; L_{\mathrm{dif}}\big(s_t,\,a^{\mathrm{nov}}_t\big).
\label{eq:diffsignal}
\end{equation}
$\ell_t$ is the trigger for Diff-DAgger and, in PACE, the analysis frame selector
that localizes each failure (\S\,Perceive).

\subsection{A Unified Query Framework for the Baselines}
\label{sec:unified}

Every IIL baseline instantiates the same template: it computes a scalar score at
each visited state and queries when that score crosses a threshold,
$\mathrm{Query}(s_t)=\mathbf{1}\big[\mathrm{score}_{\mathrm{method}}(s_t)\ \gtrless\ \tau_{\mathrm{method}}\big]$.
All action comparisons are made in the shared relative-joint-position space so that
$\|\cdot\|_2$ is consistent across methods. This framing exposes the baselines as
special cases of one decision rule, differing only in the score.
\emph{SafeDAgger}$^\star$ \cite{zhang2017safedagger} queries on novice--expert
action discrepancy,
\begin{equation}
\mathrm{Query}_{\mathrm{Safe}}(s_t)=\mathbf{1}\big[\ \|a^{\mathrm{nov}}_t-a^{\mathrm{exp}}_t\|_2 > \tau_{\mathrm{sd}}\ \big],
\label{eq:safe}
\end{equation}
(on the grid, the score is the fraction of off-optimal-mask steps). \emph{DropoutDAgger}
\cite{menda2017dropoutdagger,gal2016dropout} draws $N_{\mathrm{mc}}$ stochastic
samples and queries when the in-$\tau$-ball agreement with the expert drops below
$p$,
\begin{align}
\hat p_t=\tfrac{1}{N_{\mathrm{mc}}}\sum_{i=1}^{N_{\mathrm{mc}}}\mathbf{1}\big[\ \|a^{(i)}_t-a^{\mathrm{exp}}_t\|_2\le\tau\ \big],
\qquad
\mathrm{Query}_{\mathrm{Drop}}(s_t)=\mathbf{1}\big[\ \hat p_t<p\ \big].
\label{eq:dropout}
\end{align}
\emph{EnsembleDAgger} \cite{menda2019ensembledagger,lakshminarayanan2017ensembles}
queries on ensemble doubt (variance) \emph{or} mean discrepancy,
\begin{align}
\mathrm{doubt}_t&=\tfrac1{d_a}\!\sum_{j=1}^{d_a}\mathrm{Var}_{m}\big(a^{[m]}_{t,j}\big),
\quad
\mathrm{disc}_t=\tfrac1M\!\sum_{m=1}^{M}\big\|a^{[m]}_t-a^{\mathrm{exp}}_t\big\|_2, \nonumber\\
\mathrm{Query}_{\mathrm{Ens}}(s_t)&=\mathbf{1}\big[\ \mathrm{disc}_t>\tau\ \ \lor\ \ \mathrm{doubt}_t>\chi\ \big].
\label{eq:ensemble}
\end{align}
\emph{ThriftyDAgger} \cite{hoque2021thriftydagger} queries on novelty (doubt)
\emph{or} task risk $1-Q$, with budget-calibrated $(1-\alpha_h)$-quantile
thresholds,
\begin{align}
y_t&=\mathbf{1}[\mathrm{success}]\cdot\gamma^{\,n-1-t},
\quad
\mathcal{L}_Q=\mathrm{BCE}\big(\sigma(Q_\psi(s_t,a_t)),\,y_t\big),
\quad
\mathrm{risk}(s,a)=1-\sigma\!\big(Q_\psi(s,a)\big), \nonumber\\
\delta_h&=\mathrm{Quantile}_{1-\alpha_h}\{\mathrm{doubt}\},\
\beta_h=\mathrm{Quantile}_{1-\alpha_h}\{\mathrm{risk}\},\
\mathrm{Query}_{\mathrm{Thr}}(s_t)=\mathbf{1}\big[\ \mathrm{doubt}_t>\delta_h\ \lor\ \mathrm{risk}_t>\beta_h\ \big].
\label{eq:thrifty}
\end{align}
\emph{Diff-DAgger} \cite{lee2025diffdagger} uses the diffusion loss of
Eq.~\ref{eq:diffsignal}: with $\eta_\alpha=\hat F^{-1}(\alpha)$ the
$\alpha$-quantile of the training-loss CDF (recalibrated at each retrain), it
queries when the last $K$ steps are all in the tail,
\begin{equation}
\eta_\alpha=\hat F^{-1}(\alpha),
\qquad
\mathrm{Query}_{\mathrm{Diff}}(s_t)=\mathbf{1}\!\left[\ \sum_{u=t-K+1}^{t}\mathbf{1}\big[\ \ell_u>\eta_\alpha\ \big]\ \ge\ K\ \right].
\label{eq:diffdagger}
\end{equation}
A uniform-random control (\emph{Random}) queries at a pre-drawn step, giving a
lower bound on any informed rule,
\begin{equation}
t_{\mathrm{stag}}\sim\mathrm{Unif}\{0,\dots,H-1\},
\qquad
\mathrm{Query}_{\mathrm{Rand}}(s_t)=\mathbf{1}\big[\ t\ge t_{\mathrm{stag}}\ \big].
\label{eq:stagger}
\end{equation}

\paragraph{What the framework leaves on the table.}
Eqs.~\ref{eq:safe}--\ref{eq:stagger} answer a single question --- \emph{at which
step $t^\star$ should the expert intervene?} --- and then default to an on-policy
correction from that step. Three limitations follow. (i) They are \emph{per-state},
so within a batch of failed rollouts they cannot reason about which failures are
redundant. (ii) They are \emph{memoryless} across rounds, so a persistent failure
mode can be corrected over and over. (iii) They inherit the \emph{start state} of
whatever rollout tripped the predicate, and cannot deliberately place the demo at a
more instructive sub-task entry point. PACE keeps the shared loop of
Eq.~\ref{eq:iil-loop} but replaces the single-scalar query with a
perceive--assess--choose--execute pipeline that answers \emph{when $+$ which $+$ where}.

\subsection{PACE}
\label{sec:pace}

PACE operates on the \emph{set} of failures produced by a round's rollouts. Let
$\mathcal{F}_r=\{f_1,\dots,f_N\}$ be that set. The four stages transform
$\mathcal{F}_r$ into one corrective demonstration; Algorithm~\ref{alg:pace} gives
the full loop.

\paragraph{Perceive.}
For each failure $i$ we localize the \emph{failure point} $t^\star_i$ as the
peak-loss step of Eq.~\ref{eq:diffsignal} --- the moment of maximal diffusion
uncertainty --- and record raw geometry at that step:
\begin{equation}
\mathcal{F}_r=\{f_1,\dots,f_N\},
\quad
t^\star_i=\arg\max_t \ell^{\,(i)}_t,\quad
\mathrm{peak}_i=\max_t \ell^{\,(i)}_t,\quad
\rho_i=\frac{t^\star_i}{\max(1,T_i)},\quad
\delta_i=\big\|p^{\mathrm{tcp}}_i-p^{\mathrm{tee}}_i\big\|_2 .
\label{eq:perceive}
\end{equation}
Here $\rho_i$ is task progress at the failure and $\delta_i$ is an
end-effector-to-object contact distance. A vision-language model
\cite{bai2025qwen3vl,duan2025aha,liu2023reflect} reads three key frames around
$t^\star_i$ (start, peak, end), together with a per-task knowledge graph of
workspace bounds and failure taxonomy, and \emph{describes} why the rollout failed;
this description grounds the corrective command emitted in the Execute stage
\cite{driess2023palme,huang2022innermonologue,shinn2023reflexion,madaan2023selfrefine}.
Failures are then featurized by a quaternion-free geometric descriptor
$\phi_i\in\mathbb{R}^6$, with an optional R3M visual embedding
\cite{nair2022r3m} reduced by PCA for the image-based setting:
\begin{align}
\phi_i&=\big[\,p^{\mathrm{tee}}_{i,x},\ p^{\mathrm{tee}}_{i,y},\ \sin\theta_i,\ \cos\theta_i,\ \rho_i,\ \delta_i\,\big]\in\mathbb{R}^6,
\quad
\tilde v_i=\mathrm{PCA}_{k'}\big(v_i\big),\ \ k'=\min(16,N-1,d), \nonumber\\
\psi_i&=\begin{cases}\tilde v_i & \text{visual mode}\\[2pt]\phi_i & \text{geometric (default)}\end{cases},
\quad
X=[\psi_1;\dots;\psi_N],\quad
\tilde X_i=\big(\psi_i-\mu\big)\oslash\sigma,\quad \sigma_j\!\leftarrow\!\max(\sigma_j,10^{-8}).
\label{eq:phi}
\end{align}
The clustering feature $\psi_i$ is used only for \emph{mode discovery}; all
prescription geometry (\S\,Execute) uses the raw tee pose and full robot
configuration.

\paragraph{Assess.}
PACE clusters the standardized features into failure modes, choosing the number of
clusters by maximum mean silhouette \cite{lloyd1982kmeans} and computing per-cluster
geometry on the raw tee pose:
\begin{align}
k^\star&=\arg\max_{k\in[2,\,k_{\max}]}\ \mathrm{sil}(k),\qquad k_{\max}=\max\!\big(2,\min(6,N-1)\big),\nonumber\\
c^{xy}_{C}&=\tfrac{1}{|C|}\!\sum_{i\in C}p^{\mathrm{tee}}_{i,xy},\quad
\bar\theta_{C}=\operatorname{atan2}\!\Big(\!\sum_{i\in C}\!\sin\theta_i,\ \sum_{i\in C}\!\cos\theta_i\Big),\quad
\bar L_{C}=\tfrac{1}{|C|}\!\sum_{i\in C}\mathrm{peak}_i,\nonumber\\
\mathrm{rep}(C)&=\arg\min_{i\in C}\ \big\|\tilde X_i-\bar X_{C}\big\|_2,\qquad
\bar X_{C}=\tfrac1{|C|}\!\sum_{i\in C}\tilde X_i.
\label{eq:partition}
\end{align}
The cluster representative $\mathrm{rep}(C)$ is the member nearest the cluster mean
in feature space. The \emph{dominant} cluster --- the most prevalent, hardest
failure mode --- is selected by a deterministic lexicographic key on (size, mean
peak-loss, earliest episode),
\begin{equation}
C^\star=\operatorname*{arg\,lexmax}_{C\in\{C_1..C_m\}}\ \big(\ |C|,\ \bar L_{C},\ -\!\min_{i\in C}\mathrm{eid}_i\ \big).
\label{eq:dominant}
\end{equation}

\paragraph{Choose.}
Because the policy is retrained after every demo, the failure distribution shifts
each round and a single mode could be prescribed repeatedly. PACE therefore keeps a
cross-round \emph{memory} of already-corrected centroids and applies a
recency-discounted Gaussian coverage penalty, rotating the \emph{target} cluster
among those tied with the dominant one to avoid re-covering a solved mode
\cite{florensa2017reversecurriculum,eysenbach2018leavenotrace}:
\begin{align}
P_{\mathrm{mem}}(c)&=\sum_{i:\,(r_i,c_i)\in\mathrm{Mem}}\gamma^{\,\max(0,\,r-r_i)}\exp\!\Big(-\tfrac{\|c-c_i\|_2^2}{2\sigma_{\mathrm{mem}}^2}\Big),\nonumber\\
C_{\mathrm{tgt}}&=\arg\max_{C:\,|C|\ge|C^\star|-1}\ \Big(\ \bar L_{C}\ -\ \lambda\,P_{\mathrm{mem}}\big(c^{xy}_{C}\big)\ \Big),\qquad \gamma=0.6,\ \sigma_{\mathrm{mem}}=0.06,\ \lambda=1.
\label{eq:memory}
\end{align}
Given the target, PACE selects \emph{which} failures to analyze/correct by a
memoryless-within-round \emph{coreset / k-center} (farthest-point) rule
\cite{sener2018coreset,eldar1997fps,ash2020badge,settles2009active}, forcing the
target representative in and seeding the hardest failure:
\begin{align}
S_0&=\big\{\,\mathrm{rep}(C_{\mathrm{tgt}})\,\big\}\ \cup\ \big\{\,\arg\max_i \mathrm{peak}_i\,\big\}, \nonumber\\
S&\leftarrow S\ \cup\ \Big\{\ \arg\max_{i\notin S}\ \min_{j\in S}\ \big\|\tilde X_i-\tilde X_j\big\|_2\ \Big\}\quad\text{until}\ |S|=\kappa,\qquad \kappa=3, \nonumber\\
S^\star&=\arg\max_{S\subseteq\mathcal{F}_r,\ |S|\le\kappa}\ \min_{\substack{i,j\in S\\ i\ne j}}\ \big\|\tilde X_i-\tilde X_j\big\|_2\quad\text{s.t.}\ \ \mathrm{rep}(C_{\mathrm{tgt}})\in S.
\label{eq:diversity}
\end{align}
The last line is the max-min coverage objective PACE approximates; the first two
lines are its greedy realization. This is the coverage/diversity axis that the
per-state baselines of \S\ref{sec:unified} lack: rather than correcting whichever
failure the predicate happened to trip, PACE spends its single demo on the failure
that most enlarges coverage of the failure-mode space, anchored on the dominant
mode.

\paragraph{Execute.}
The anchor $A$ is the target representative's raw geometry. A VLM/LLM
\cite{yang2025qwen3,ma2024eureka,yu2023language2rewards,liang2023codeaspolicies}
emits a scene command $\mathrm{cmd}$ (a target object pose, TCP, and decision
label), which a prescribe map $g(\cdot)$ turns into a corrective reset /
sub-task-entry specification $\xi$ --- the \emph{start state} of the one new demo.
The anchor-relative perturbation is $L_2$-capped in position and clamped in yaw, and
the whole spec is projected onto the workspace bounds $\mathcal{W}$:
\begin{align}
A&=\big(p^{\mathrm{tee}}_A,\ \theta_A,\ q^{\mathrm{full}}_A,\ p^{\mathrm{tcp}}_A,\ t^\star_A\big)=\mathrm{rep}(C_{\mathrm{tgt}}),
\quad
\mathrm{cmd}=\mathrm{VLM/LLM}\big(A,\ \text{frames}(t^\star_A),\ \mathrm{KAG}\big),\nonumber\\
\Delta_{xy}&=\mathrm{cmd}.p^{\mathrm{tee}}_{xy}-p^{\mathrm{tee}}_{A,xy},\quad
\Delta_{xy}\leftarrow\Delta_{xy}\cdot\min\!\Big(1,\ \tfrac{\Delta_{\max}}{\|\Delta_{xy}\|_2}\Big),\quad
\Delta_\theta=\mathrm{clamp}\big(\mathrm{wrap}_\pi(\mathrm{cmd}.\theta-\theta_A),\,\pm\theta_{\max}\big),\nonumber\\
\xi&=g(\mathrm{cmd})=\mathrm{clamp}_{\,\mathcal{W}}\Big(p^{\mathrm{tee}}_A+[\Delta_{xy},0],\ \ \theta_A+\Delta_\theta,\ \ q^{\mathrm{full}}_A\Big),
\quad
(\Delta_{\max},\theta_{\max})=(0.06,0.4).
\label{eq:prescribe}
\end{align}
The prescribed round then samples the start state from the reset distribution
$\Xi(\cdot\mid\xi)$, lets the expert roll from $t^\star$, and aggregates the single
new demo:
\begin{equation}
s_0\sim\Xi(\,\cdot\mid\xi\,),
\qquad
\mathcal{D}_{r+1}=\mathcal{D}_r\ \cup\ \big\{(s_t,\pi^\star(s_t)):\ t\ge t^\star,\ s_0\sim\Xi(\cdot\mid\xi)\big\}.
\label{eq:prescribe-demo}
\end{equation}
An empty or refused prescription is never accepted (a hard non-emptiness floor with
bounded re-prescription attempts), since a wasted round is the worst outcome under a
fixed budget.

\paragraph{Baselines as special cases of PACE.}
Eq.~\ref{eq:prescribe-demo} recovers the entire unified query family of
\S\ref{sec:unified} under three restrictions: (i) select a single failure,
$|S|=1$; (ii) disable memory and diversity, $C_{\mathrm{tgt}}=C^\star$ with the
selection reduced to the peak-loss failure; and (iii) set $g$ to the on-policy
identity, so $\Xi(\cdot\mid\xi)$ replays the failing rollout and the expert corrects
in place from $t^\star$. With these three settings, PACE reduces exactly to
Diff-DAgger \cite{lee2025diffdagger} when $t^\star$ is chosen by the diffusion-loss
CDF (Eq.~\ref{eq:diffdagger}), and more generally to any baseline of
Eqs.~\ref{eq:safe}--\ref{eq:stagger} by swapping in that baseline's $t^\star$ rule.
PACE's contribution is precisely the three axes it \emph{adds} on top: a
\emph{set}-level diversity choice ($|S|>1$, Eq.~\ref{eq:diversity}), cross-round
\emph{memory} (Eq.~\ref{eq:memory}), and a learned \emph{where}-to-start
prescription ($g\neq\mathrm{identity}$, Eq.~\ref{eq:prescribe}). We ablate each of
these axes in the experiments.

\begin{algorithm}[tb]
\caption{PACE: Perceive $\rightarrow$ Assess $\rightarrow$ Choose $\rightarrow$ Execute}
\label{alg:pace}
\textbf{Input}: initial demos $\mathcal{D}_0$, expert $\pi^\star$, budget $B$,
retrain cadence $n_d$, target $\mathrm{SR}_{\mathrm{target}}$, cap $\kappa$\\
\textbf{Output}: trained policy $\pi_\theta$
\begin{algorithmic}[1]
\STATE $\theta \leftarrow \arg\min_\theta \mathbb{E}_{\mathcal{D}_0}[\mathcal{L}_{\mathrm{DP}}]$;\ \ $r\leftarrow 0$;\ \ $\mathrm{Mem}\leftarrow\varnothing$ \hfill{\footnotesize// Eq.~\ref{eq:vpred}}
\WHILE{$q<B$ \textbf{and} $\mathrm{SR}(\theta)<\mathrm{SR}_{\mathrm{target}}$}
  \STATE roll $\pi_\theta$; collect failure set $\mathcal{F}_r$; for each $f_i$ get $t^\star_i,\mathrm{peak}_i,\rho_i,\delta_i$ \hfill{\footnotesize// \textsc{Perceive}, Eq.~\ref{eq:perceive}}
  \STATE VLM describes failures around $t^\star_i$; featurize $\psi_i$; standardize to $\tilde X$ \hfill{\footnotesize// Eq.~\ref{eq:phi}}
  \STATE cluster $\{C_1,\dots\}$ at $k^\star$; compute $\mathrm{rep}(C),\bar L_C$; pick dominant $C^\star$ \hfill{\footnotesize// \textsc{Assess}, Eqs.~\ref{eq:partition}--\ref{eq:dominant}}
  \STATE $C_{\mathrm{tgt}}\leftarrow$ memory-rotated target;\ \ $S\leftarrow$ k-center select, $\mathrm{rep}(C_{\mathrm{tgt}})$ forced, $|S|\!\le\!\kappa$ \hfill{\footnotesize// \textsc{Choose}, Eqs.~\ref{eq:memory}--\ref{eq:diversity}}
  \STATE $\mathrm{cmd}\leftarrow$ VLM/LLM$(A,\mathrm{frames},\mathrm{KAG})$;\ \ $\xi\leftarrow g(\mathrm{cmd})$ (cap, clamp, project) \hfill{\footnotesize// \textsc{Execute}, Eq.~\ref{eq:prescribe}}
  \STATE $s_0\!\sim\!\Xi(\cdot|\xi)$; expert rolls from $t^\star$; $\mathcal{D}_{r+1}\!\leftarrow\!\mathcal{D}_r\cup\{(s_t,\pi^\star(s_t)):t\ge t^\star\}$ \hfill{\footnotesize// Eq.~\ref{eq:prescribe-demo}}
  \STATE append $(r,c^{xy}_{C_{\mathrm{tgt}}},\bar\theta_{C_{\mathrm{tgt}}})$ to $\mathrm{Mem}$;\ \ $q\leftarrow q+1$;\ \ $r\leftarrow r+1$
  \IF{$q \bmod n_d = 0$}
    \STATE $\theta \leftarrow \arg\min_\theta \mathbb{E}_{\mathcal{D}_r}[\mathcal{L}_{\mathrm{DP}}]$ \hfill{\footnotesize// retrain from scratch, recalibrate $\eta_\alpha$}
  \ENDIF
\ENDWHILE
\STATE \textbf{return} $\pi_\theta$
\end{algorithmic}
\end{algorithm}

\subsection{Task Suite and Evaluation}

We evaluate on five tasks spanning a discrete toy domain and four continuous
manipulation tasks, over two observation modalities each
(Table~\ref{tab:tasks}). The toy grid uses an A$^\star$/BFS oracle; Push is
ManiSkill PushT with a PPO expert \cite{mu2021maniskill,gu2023maniskill2,tao2024maniskill3};
Lift, Wipe, and Door are RoboSuite tasks with motion-planner experts
\cite{zhu2020robosuite,mandlekar2021robomimic}. All robot methods share the single
diffusion-policy backbone described above so that the only variable is the
demonstration-acquisition rule.

\begin{table}[t]\centering\small
\caption{Task suite. Modalities: state (privileged proprioception) and image
(partial visual observation). ``Action'' is the learner's action space; ``Expert''
is the demonstration oracle. All tasks use the one-demo-per-round IIL protocol with
$\mathrm{SR}_{\mathrm{target}}=0.90$.}
\label{tab:tasks}
\begin{tabular}{llll}
\toprule
Task & Sim / Embodiment & Action & Expert \\
\midrule
Toy grid & MazeNav ($5\!\times\!5$) & Discrete(4) & A$^\star$/BFS \\
Push     & ManiSkill / Panda-stick & rel.\ joint pos & PPO \\
Lift     & RoboSuite / UR5e        & rel.\ joint pos & MP \\
Wipe     & RoboSuite / UR5         & rel.\ joint pos & MP \\
Door     & RoboSuite / UR5         & rel.\ joint pos & MP \\
\bottomrule
\end{tabular}
\end{table}

We report three metrics. \emph{Held-out success rate} over a fixed evaluation set,
\begin{equation}
\mathrm{SR}(\theta)=\frac{1}{|\mathcal{E}|}\sum_{e\in\mathcal{E}}\mathbf{1}\big[\ e\ \text{succeeds under }\pi_\theta\ \big],
\qquad \mathrm{SR}_{\mathrm{target}}=0.90;
\label{eq:sr}
\end{equation}
\emph{expert queries to target}, the number of demos to first reach
$\mathrm{SR}_{\mathrm{target}}$ (else the budget $B$),
\begin{equation}
q(\mathrm{SR}_{\mathrm{target}})=\min\big\{\,q\ :\ \mathrm{SR}(\theta_q)\ge\mathrm{SR}_{\mathrm{target}}\,\big\},
\qquad q\leftarrow B\ \text{if never reached};
\label{eq:queries}
\end{equation}
and \emph{demo efficiency}, the area under the SR-vs-demos curve,
\begin{equation}
\mathrm{Eff}=\int_{0}^{B}\mathrm{SR}(\theta_{q})\ \mathrm{d}q
\ \approx\ \sum_{q=1}^{B}\tfrac12\big(\mathrm{SR}(\theta_{q-1})+\mathrm{SR}(\theta_{q})\big).
\label{eq:eff}
\end{equation}
We additionally track \emph{coverage} of the collected demos over the task space,
\begin{equation}
\mathrm{Cov}=\frac{1}{|\mathcal{Z}|}\Big|\ \big\{\,z\in\mathcal{Z}\ :\ \exists\,(s,a)\in\mathcal{D},\ \mathrm{cell}(s)=z\,\big\}\ \Big|,
\label{eq:cov}
\end{equation}
which isolates the effect of PACE's diversity/memory choice. As a headline preview,
PACE reaches the target with \PH{pace-vs-diff-q-reduction} fewer expert queries than
Diff-DAgger while attaining higher coverage; full per-task, per-modality results
appear in the Experiments section.
