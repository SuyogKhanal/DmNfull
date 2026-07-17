\begin{titlepage}
\centering
\includegraphics[width=0.80\textwidth]{Institute_Logo_Stacked_2025_Keyline.png}\\[2.2cm]
{\Large\bfseries Confirmation of Candidature Report}\\[0.35cm]
{\large Deakin University}\\
{\large Deakin Applied Artificial Intelligence Initiative}\\[1.9cm]
{\LARGE\bfseries Leveraging Large Language Models\\[0.2cm] for Sample-Efficient Imitation Learning}\\[2.2cm]
\begin{tabular}{@{}r@{\hspace{1em}}l@{}}
\textbf{Candidate} & Suyog Khanal\\
\textbf{Student identifier} & s226137394\\[0.6em]
\textbf{Supervisors} & Associate Professor Santu Rana\\
                     & Dr Arun Kumar Anjanapura Venkatesh\\[0.6em]
\textbf{Candidature start date} & 13 November 2025\\
\textbf{Confirmation of Candidature date} & 13 August 2026\\
\end{tabular}
\vfill
\end{titlepage}

\tableofcontents

\newpage


# Abstract

Imitation learning converts expert demonstrations into a policy, and the demonstration is the one input whose cost does not fall. Compute is bought and architectures are downloaded, while every trajectory still has to be produced by a person or a scripted oracle, one at a time. The binding constraint on a realistic deployment is a fixed budget of $B$ demonstrations that has to be spent well. Interactive imitation learning, the family descended from dataset aggregation, spends that budget on a single decision, and its members differ only in the signal that decides when to hand control to the expert. Two further decisions are left unclaimed. Which failure to correct, and where the corrective demonstration should begin.

This programme claims that a large language model, given a structured description of how the policy is failing together with an explicit statement of what the environment permits, can make those two decisions, and that making them raises the information content of each demonstration under a restricted budget. The model is never placed in the robot's control loop. It reads a summary of the policy's own failures and returns a request for one specific demonstration. Aim 1 realises the claim as **D**emonstration d**I**stillation for **S**ample-**E**fficient **I**mitation **L**earning, DISEIL.

Each round, DISEIL perceives the round's failures by reducing every failed rollout to a six-dimensional geometric descriptor of the state at which the policy first became unreliable, partitions those descriptors into failure modes, prioritises one mode, and prescribes the demonstrations of the round inside it. A prescription is screened before any expert time is spent on it, once for feasibility against a store of explicit environmental constraints and once against the current policy, since a scenario the policy can already solve teaches it nothing. The learner is any policy that exposes a per-step loss, and the evaluation uses a multilayer perceptron, a convolutional network and a diffusion policy without changing the selection loop.

A setting is one task under one observation modality. DISEIL was evaluated on five tasks, a 5×5 grid-world, Push-T and the Lift, Wipe and Door manipulation tasks, each under state and image observations, which gives ten settings, and in every setting against five comparison methods: four query gates of the DAgger family throughout, with Diff-DAgger as the fifth on the robot tasks and a uniform-random allocation control as the fifth on the grid-world. Nine seeds were run on the grid-world and five on each robot task. DISEIL attains the best mean success rate in all ten settings, with a mean margin of 3.71 points over the strongest competing method in each. The two modalities of a task share the expert, the reward structure and the reset distribution, so the ten settings are not ten independent experiments and the sweep should not be read as ten independent confirmations.

Ablations on three of the settings place the advantage in the allocation. Removing the partition costs 4.37 success-rate points as a mean over the three, while the per-demonstration information gain of the same runs does not fall (mean change $+0.02$), so a demonstration can be individually informative and jointly redundant with the one collected in the round before. The advantage is largest where the budget is smallest, the margin over the best baseline averaging 10.97 points at a budget of 10 demonstrations, 4.77 at 20 and 2.83 at 40.

Each foundation-model component is worth about a point, 1.33 for the prescription model and 1.33 for the vision-language model, and the cause is structural: the partition that decides which failure mode to correct is computed from the geometric descriptor and consumes no model output, so by the time a model is called the decision that carries the result has been taken. Aim 2 supplies the selector with the input it lacks, a record of what has already been taught. A captioner turns a trajectory's observations and its executed actions back into language, and the captions accumulate into a coverage record that a new failure is checked against before a demonstration is requested. Aim 3 turns that record outward, pricing a demonstration against the resource that is scarce outside simulation, a teacher's time, so that a generalist policy can ask a non-expert human for exactly the demonstrations it lacks.

---
