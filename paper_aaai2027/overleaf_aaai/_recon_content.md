# DISEIL AAAI-2027 — content reconnaissance

Source of truth: `COC_REPORT/build/v2/{02_background,03_gap_rq,04_aims,05_progress}.md`.
Every number below is copied from those files. Nothing here is inferred or rounded.
Method name is **DISEIL** everywhere (never DISTIL / PACE / P4).
All Greek must be written in math mode (`$\sigma$`, `$\theta$`, `$\lambda$`, `$\kappa$`, `$\rho$`, `$\delta$`, `$\eta$`, `$\gamma$`, `$\xi$`, `$\phi$`, `$\mu$`, `$\tau$`) because the build is pdflatex-only.

---

## 1. The scientific contribution

**One paragraph.** Interactive imitation learning corrects covariate shift by having an expert relabel the states the learner itself visits, and the query-efficient members of that family (SafeDAgger, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Diff-DAgger) all reduce to one template: compute a scalar at a visited state, compare it against a threshold, hand over at the first crossing. That template answers *when* to call the expert and leaves two decisions fixed by default: *which* of a round's failures the expert's time is spent on, and *where* the corrective demonstration begins. Under an unbounded budget those two decisions cost nothing; under a budget of twenty demonstrations they are the whole problem, because a scalar attached to one state carries no representation in which two failures are the same mistake and a third is different, and because the corrective demonstration must start at whatever state tripped the threshold, which is frequently a state the policy has already ruined. DISEIL makes all three decisions. Each round it partitions the round's failures into failure modes over a six-dimensional geometric descriptor computed at the flagged step, selects the near-dominant mode of highest mean peak loss, and prescribes the configuration of a demonstration that does not exist yet, verified against an explicit store of what the environment permits before any expert time is spent. Across five tasks under two observation modalities, DISEIL attains the best mean held-out success rate in all ten settings at a budget of twenty demonstrations, with a mean margin of 2.80 points over the strongest baseline in each setting; collapsed to five task means the paired $t$-test gives $t(4) = 4.10$, $p = 0.015$ two-sided. The ablations locate the advantage in the allocation step rather than in the language models: removing the partition costs 4.37 points while per-demonstration information gain does not fall, which shows that greedy worst-loss selection collects demonstrations that are individually informative and jointly redundant.

**Three bullets.**
1. **The decision nobody makes.** DISEIL is the first framework to decide *which* failure mode a round's demonstration is spent on and *where* that demonstration begins, rather than only *when* to hand over. It prescribes a demonstration that has not been collected, so the selection literatures (active learning, core-set, sub-trajectory retrieval), which all rank items in a pool that already exists, do not answer the question and are not baselines.
2. **A partition of the round's failures paired with a feasibility-verified prescription.** The failures of a round are grouped into modes over a hand-designed six-dimensional geometric descriptor; the prescribed configuration is checked against a knowledge-augmented graph of workspace bounds, spawn ranges, reachability and path validity, and a violation is returned to the model as feedback until a feasible prescription is produced. This is what licenses the information-gain measurement: an infeasible scenario never becomes a demonstration, and the demonstration comes from the expert whose trajectories are the fitting target, so a high pre-retrain loss cannot be read as a bad datum.
3. **Evidence that allocation, not per-demonstration informativeness, is the mechanism.** Best mean in all ten settings ($t(4) = 4.10$, $p = 0.015$ on the conservative task-level test); DISEIL acquires higher-information demonstrations than Diff-DAgger in all eight settings where Diff-DAgger runs, even though Diff-DAgger gates on the same per-step loss the metric is computed from; and removing the partition leaves information gain unchanged (mean $+0.02$) while success falls 4.37 points, which dissociates the two and shows the margin is bought by spreading the budget across modes.

---

## 2. Problem, gap, insight

**Problem.** A demonstration is the one input whose cost does not fall when compute is bought or a simulator is downloaded. Policy performance rises with the number and the coverage of the demonstrations a policy is trained on [lin2024datascaling], and under a fixed allowance $B$ the allowance cannot grow, so the only lever is what each demonstration in it contains. The question the field leaves open is what a single expert demonstration is worth and how that worth can be raised.

**Gap.** The published query gates decide when to hand control to the expert, and they decide it from a scalar attached to a single state. Selection methods from active learning and dataset curation decide which item to take from a pool that already exists [houlsby2011bald, settles2009active, sener2018coreset]. Between the two lies a decision nobody makes: given a round's worth of failures, which failure mode should receive this round's demonstration, and from which configuration should that demonstration begin. A per-state gate cannot answer either, because it holds no representation in which two failures are the same mistake and a third is a different one, and because it inherits the state that tripped it.

**Insight.** The value of a demonstration is not proportional to the frequency of the corresponding failure, nor to how badly any single failure scores. A round's failures form a small set with structure, and that structure is recoverable from geometry: a low-dimensional descriptor of the robot and object configuration at the step where the policy first becomes unreliable partitions the failures into behaviourally distinct modes. Allocating across those modes, and prescribing a fresh configuration verified against what the environment permits, buys coverage of the failure distribution at a rate a per-state scalar cannot, because peak loss is a property of one trajectory and never a property of the set of failures the policy is still producing.

---

## 3. Method

### 3.1 What is standard (belongs in Related Work / preliminaries, NOT presented as ours)

| Component | Status | Cite |
|---|---|---|
| MDP formulation, behaviour-cloning objective $\mathcal{L}_{\mathrm{BC}}$ | standard, unchanged for every arm | [bain1995cloning, pomerleau1988alvinn, argall2009survey, osa2018algorithmic] |
| Covariate shift, $O(H^2\epsilon)$ vs linear-in-$H$ | standard | [shimodaira2000covariate, ross2010reductions, ross2011dagger] |
| Dataset aggregation / the interactive loop skeleton | standard, shared by every arm | [ross2011dagger] |
| Query-gate template (score, threshold, handover) | standard; the five gates are the baselines | [zhang2017safedagger, menda2017dropoutdagger, menda2019ensembledagger, hoque2021thriftydagger, lee2025diffdagger] |
| Diffusion policy; the per-step denoising loss $\ell_t$ as a signal | **Diff-DAgger's idea**, used here for localisation AND compared against as a baseline; must be stated plainly | [chi2023diffusionpolicy, lee2025diffdagger, ho2020ddpm] |
| Agglomerative clustering (Ward) as the instantiation of a generic $\mathcal{C}$ | standard; k-means would serve | [ward1963hierarchical, lloyd1982kmeans, pedregosa2011sklearn] |
| Silhouette criterion for $k^\star$ | standard, used unmodified | [rousseeuw1987silhouette, pedregosa2011sklearn] |
| Farthest-point / k-centre selection for the context set | standard | [eldar1997fps] |
| A\* and BFS for grid path validity | standard; **never the expert** | [hart1968astar, cormen2022algorithms] |
| R3M visual encoder for image-modality policies | standard; supplies the policy's features only, **never the clustering features** | [nair2022r3m] |
| Propose-verify-revise with an external checker | standard pattern; what is new is the object verified | [liu2023llmp, chen2024autotamp] |
| PPO Push-T expert | standard, unmodified | [schulman2017ppo] |

### 3.2 What is ours

The contribution is **the pairing of a partition of the round's failures into modes with a prescription that is verified against an explicit model of what the environment permits before an expert is asked to satisfy it**, plus the second, separate solvability screen (Eq. 11) that asks whether the prescribed configuration would teach the policy anything. The cluster memory is a **configurable, task-specific component and NOT a headline contribution** (A1 evidence). The feasibility check itself is not the novel part; the novel part is that the object being verified is a request for a training demonstration rather than a plan to be executed.

### 3.3 Notation

| Symbol | Meaning |
|---|---|
| $f_\theta$ | the policy; any function exposing a per-step loss at a visited state under the executed action |
| $\pi^\star$ | the expert |
| $\mathcal{D}_0$, $\mathcal{D}_r$ | initial demonstration set; the set after round $r$ |
| $B$, $D$ | the demonstration budget; demonstrations acquired per round. Loop runs $B/D$ rounds |
| $\ell^{(i)}_t$ | per-step loss of $f_\theta$ on episode $i$ at step $t$ under the executed action |
| $\mathcal{F}_r$, $N$ | the round's failure set; $N = |\mathcal{F}_r|$ |
| $t^\star_i$ | the flagged step of failure $i$ (first crossing, not the peak) |
| $\eta$, $K$ | OOD threshold (a quantile of the training-loss distribution, recalibrated at each retrain); consecutive-step run length |
| $\phi_i \in \mathbb{R}^6$ | the geometric descriptor at $t^\star_i$ |
| $\rho_i$, $\delta_i$ | fraction of episode completed before the flag; end-effector-to-object distance |
| $\tilde{X}_i$, $\mu$, $\sigma_\phi$ | standardised descriptor; feature mean; feature std |
| $\mathcal{C}$, $k^\star$, $k_{\max}$ | generic clustering step; selected cluster count; $k_{\max} = \max(2,\min(6,N-1))$ |
| $C^\star$, $C_{\mathrm{tgt}}$, $\bar{L}_C$, $\mathrm{rep}(C)$ | dominant mode; target mode; mode mean peak loss; member nearest the cluster mean in standardised space |
| $S$, $\kappa$ | the context set of cited failures; its cap |
| $\lambda$, $\gamma$, $\sigma$ | cluster-memory weight, recency discount, kernel width ($\lambda = 0$ switches it off) |
| $\mathcal{K}$, $\mathcal{W}_{\mathcal{K}}$ | the task's knowledge-augmented graph; its workspace bounds |
| $\xi$, $g$, $V$, $J_{\max}$ | reset specification; command-to-specification map; the validity conjunction; re-prescription limit |
| $P = \xi^\star$, $\tau_{\mathrm{solve}}$ | the prescribed configuration; the solvability threshold |
| $A = (t^\star, C_{\mathrm{tgt}}, \xi)$ | the three-part acquisition rule: when, which, where |

### 3.4 Equations (exact LaTeX)

Eq. 1, the per-step loss (the only requirement on the policy):
```latex
\ell^{(i)}_t \;=\; \mathcal{L}\big(f_\theta,\; s^{(i)}_t,\; a^{(i)}_t\big)
```

Eq. 2, the round's failure set:
```latex
\mathcal{F}_r \;=\; \big\{\, f_i \;=\; (\tau_i,\; \ell^{(i)}_{1:T_i}) \;:\; \tau_i \text{ is a failed rollout of } f_\theta \,\big\},
\qquad N \;=\; |\mathcal{F}_r|
```

Eq. 3, aggregation and retraining (standard; from [ross2011dagger]):
```latex
\mathcal{D}_r \;=\; \mathcal{D}_{r-1} \,\cup\, \{d_{r,1},\dots,d_{r,D}\},
\qquad
\theta_r \;=\; \arg\min_\theta\ \mathbb{E}_{(s,a)\sim\mathcal{D}_r}\big[\mathcal{L}_{\mathrm{BC}}\big]
```
Loop stops when $|\mathcal{D}_r| - |\mathcal{D}_0| = B$.

Eq. 4, the three-part acquisition rule (**the framing equation of the paper**):
```latex
A \;=\; \big(\underbrace{t^\star}_{\text{when}},\; \underbrace{C_{\mathrm{tgt}}}_{\text{which}},\; \underbrace{\xi}_{\text{where}}\big)
```
The DAgger family fixes the second and third trivially: $C_{\mathrm{tgt}}$ is whichever rollout tripped the gate first, $\xi$ is the state the rollout was already in. DISEIL computes all three.

Eq. 5, the flagged step (first crossing, not the peak):
```latex
t^\star_i \;=\; \min\Big\{\, t \;:\; \ell^{(i)}_u > \eta \ \ \text{for all } u \in [\,t,\, t+K\,] \,\Big\},
\qquad \text{with } t^\star_i \leftarrow \arg\max_t \ell^{(i)}_t \ \text{if no crossing occurs}
```
Rationale to keep: in a failing episode the peak arrives late, so an expert who takes over at the peak inherits a badly corrupted state and has almost no episode left to correct it. The first crossing is early and the state is less corrupted. The threshold construction is Diff-DAgger's, used unchanged [lee2025diffdagger].

Eq. 6, the geometric descriptor:
```latex
\phi_i \;=\; \big[\, p_{x},\; p_{y},\; \sin\theta,\; \cos\theta,\; \rho_i,\; \delta_i \,\big],
\qquad \rho_i = \frac{t^\star_i}{T_i},
\qquad \delta_i = \big\|\, p^{\mathrm{tcp}}_i - p^{\mathrm{obj}}_i \,\big\|_2
```
Yaw enters through sine and cosine so the wrap at $\pm\pi$ does not create a false distance.

Eq. 7, the partition:
```latex
\tilde{X}_i \;=\; \frac{\phi_i - \mu}{\sigma_\phi},
\qquad \{C_1,\dots,C_{k^\star}\} \;=\; \mathcal{C}\big(\tilde{X},\, k^\star\big),
\qquad k^\star \;=\; \arg\max_{k \in [2,\,k_{\max}]} \operatorname{sil}(k)
```
with $k_{\max} = \max(2, \min(6, N-1))$. When fewer than four failures remain, the sweep is skipped and each failure becomes its own singleton, so the round is allocated by the fallback rule.

Eq. 8, the prioritisation rule:
```latex
C_{\mathrm{tgt}} \;=\; \arg\max_{C \,:\, |C| \,\ge\, |C^\star| - 1} \ \bar{L}_C
```
The size constraint keeps the target inside the bulk of the round's failures, so a mode that barely exists cannot capture the round's budget on the strength of one badly failed episode.

Eq. 9, the context set:
```latex
S_0 \;=\; \big\{\mathrm{rep}(C_{\mathrm{tgt}})\big\} \cup \big\{\arg\max_i \mathrm{peak}_i\big\},
\qquad
S \;\leftarrow\; S \cup \Big\{ \arg\max_{i \,\notin\, S} \ \min_{j \in S} \ \big\| \tilde{X}_i - \tilde{X}_j \big\|_2 \Big\}
\ \ \text{until } |S| = \kappa
```

Eq. 10, the feasibility-verification loop:
```latex
\begin{aligned}
\mathrm{cmd}^{(j)} &= \mathrm{LLM}\big(\, A,\ S,\ \mathcal{K},\ \text{violation}(\xi^{(j-1)}) \,\big),
\qquad \xi^{(j)} \;=\; g\big(\mathrm{cmd}^{(j)}\big), \\[2pt]
V(\xi) &= \mathbf{1}\big[\, \xi \in \mathcal{W}_{\mathcal{K}} \,\big] \;\wedge\;
          \mathbf{1}\big[\, \mathrm{reachable}_{\mathcal{K}}(\xi) \,\big] \;\wedge\;
          \mathbf{1}\big[\, \mathrm{valid\text{-}path}_{\mathcal{K}}(\xi) \,\big], \\[2pt]
\xi^\star &= \xi^{(j)} \ \ \text{for the first } j \le J_{\max} \text{ with } V\big(\xi^{(j)}\big) = 1,
\qquad \text{else } \xi^\star = \text{nearest untried failure}
\end{aligned}
```
A failed attempt consumes no budget: the budget counts demonstrations collected, not prescriptions proposed.

Eq. 11, the solvability screen:
```latex
\mathrm{SR}_{f_\theta}(P) \;\ge\; \tau_{\mathrm{solve}} \quad \Longrightarrow \quad \text{revise } P
```
**Honest flag:** this screen is a design element only. It is **not exercised in any reported run** and no number in the report is attributable to it. Nearest relatives: reverse curriculum [florensa2017reversecurriculum] and reset learning [eysenbach2018leavenotrace].

### 3.5 The four stages

- **Perceive.** Anchor each failure at $t^\star_i$ (Eq. 5). Compute $\phi_i$ (Eq. 6) from privileged simulator state. **In parallel and never mixed:** three rendered frames (start, $t^\star_i$, end) go to a VLM [bai2025qwen3vl], which returns a short spatial account; a text-only reasoning model [yang2025qwen3] converts it into a root cause and a trajectory phase drawn from a closed taxonomy stored in the task's graph, not invented. Justification for the division of labour: VLMs are competent at naming a cause given structured evidence [duan2025aha, liu2023reflect] and unreliable at metric/spatial reasoning from pixels [chen2024spatialvlm, fu2024blink].
- **Partition.** Eq. 7. Each mode carries its raw-pose centroid, its mean peak loss $\bar{L}_C$, and $\mathrm{rep}(C)$. Mode names are the majority root cause among members. **Clustering is geometric in every run, state and image alike; no output of any foundation model enters the partition.**
- **Prioritise.** Eq. 8, optionally minus the cluster-memory penalty. Then build $S$ by Eq. 9.
- **Prescribe.** Two arms. **Targeted correction** names one cited failure; that exact episode is re-instantiated and the expert takes over at $t^\star$. **Bridging placement** names two or three cited failures and asks for a new configuration between them, from which the expert demonstrates a complete episode. Bridging is what allows a prescription to be easier than any failure it addresses. Which arms exist is read from the graph, not hard-coded (Wipe randomises a marker path, so it is declared targeted-only). Verified by Eq. 10, screened by Eq. 11.

### 3.6 Algorithm 1 (verbatim from the CoC)

```latex
\begin{algorithm}[t]
\caption{DISEIL}
\label{alg:diseil}
\begin{algorithmic}[1]
\Require initial demonstration set $\mathcal{D}_0$; policy $f_\theta$; expert $\pi^\star$; budget $B$; demonstrations per round $D$; knowledge-augmented graph $\mathcal{K}$; context-set cap $\kappa$; re-prescription limit $J_{\max}$
\Ensure the trained policy $f_\theta$
\State train $f_\theta$ on $\mathcal{D}_0$ by behaviour cloning
\For{$r = 1$ to $B/D$}
    \State $\mathcal{F}_r \gets$ the failed rollouts of $f_\theta$ on a fresh pool of episodes \Comment{Eq. 2}
    \ForAll{failures $f_i \in \mathcal{F}_r$}
        \State $t^\star_i \gets$ the flagged step of $f_i$ \Comment{Eq. 5}
        \State $\phi_i \gets$ the geometric descriptor of $f_i$ at $t^\star_i$ \Comment{Eq. 6}
        \State assign $f_i$ a root cause from the frames at $t^\star_i$ and the taxonomy in $\mathcal{K}$
    \EndFor
    \State $\{C_1,\dots,C_{k^\star}\} \gets$ partition $\{\phi_i\}$ into failure modes \Comment{Eq. 7}
    \State name each mode by the majority root cause of its members
    \State $C_{\mathrm{tgt}} \gets$ the near-dominant mode of highest mean peak loss \Comment{Eq. 8}
    \State $S \gets$ the cited failures of $C_{\mathrm{tgt}}$, at most $\kappa$ of them \Comment{Eq. 9}
    \Repeat
        \State $\xi \gets$ a configuration prescribed from $C_{\mathrm{tgt}}$, $S$, $\mathcal{K}$ and the last violation
        \State $v \gets$ the constraint of $\mathcal{K}$ that $\xi$ violates, if any
    \Until{$V(\xi) = 1$ or $J_{\max}$ attempts are spent} \Comment{Eq. 10}
    \If{$V(\xi) = 0$}
        \State $\xi \gets$ the nearest untried failure in $\mathcal{F}_r$
    \EndIf
    \If{$f_\theta$ already solves $\xi$}
        \State revise $\xi$ \Comment{Eq. 11}
    \EndIf
    \State $d_{r,1},\dots,d_{r,D} \gets$ $D$ demonstrations from $\pi^\star$ at $\xi$
    \State $\mathcal{D}_r \gets \mathcal{D}_{r-1} \cup \{d_{r,1},\dots,d_{r,D}\}$ \Comment{Eq. 3}
    \State retrain $f_\theta$ from scratch on $\mathcal{D}_r$ \Comment{Eq. 3}
\EndFor
\State \Return $f_\theta$
\end{algorithmic}
\end{algorithm}
```

### 3.7 Per-task descriptor (CoC Table 4) — supplementary material

| Task | The six components of $\phi$ |
|---|---|
| GridWorld | agent cell (2), signed offset to goal (2), progress, Manhattan distance to goal |
| Push-T | block planar position (2), $\sin\theta$, $\cos\theta$, progress, end-effector-to-block distance |
| Lift | cube planar position (2), progress, gripper-to-cube distance, gripper height, grasp indicator |
| Door | door-frame position (2), frame yaw, normalised hinge angle, end-effector-to-handle distance, progress |
| Wipe | remaining-dirt centroid (2), proportion wiped, end-effector-to-centroid distance, markers remaining, progress |

### 3.8 Setup facts needed for the Experiments section

- **Settings:** 5 tasks x 2 observation modalities (state, image) = 10 settings. "Mode" means failure mode only; an observation modality is never called a mode.
- **Tasks:** GridWorld 5x5 (discrete, 3 obstacles); Push-T = ManiSkill3 PushT-v1 [tao2024maniskill3], task from [florence2021implicitbc], popularised by [chi2023diffusionpolicy] (cite the simulator separately: [mu2021maniskill, gu2023maniskill2] are earlier releases without Push-T); Lift, Wipe, Door are RoboSuite UR5/UR5e tasks [zhu2020robosuite].
- **Experts:** GridWorld = a human. Lift = open-loop motion planner. Door = closed-loop routine reading the hinge angle. Wipe = scripted wiping routine over the sampled marker path. Push-T = a PPO policy [schulman2017ppo], learned not scripted, competent in one rotational direction only; the other configurations are excluded by the graph's workspace constraints.
- **Policies:** GridWorld image = CNN; GridWorld state = MLP; four robot tasks = diffusion policies under both modalities, R3M encoder on the image branch [chi2023diffusionpolicy, nair2022r3m].
- **Budget:** $B = 20$, $D = 1$. Retraining from scratch every round on GridWorld; every fourth acquired demonstration on the robot tasks. Both cadences hold for every arm.
- **Seeds:** 9 on GridWorld, 5 on the robot tasks. Round accounting confirms both: clustered + skipped rounds total 180 per GridWorld setting and 100 per robot setting.
- **$N_i$ chosen by a BC data-scaling sweep** targeting a round-zero success rate near 50 per cent (the band in which a fixed budget can be spent well or badly). Below the band every configuration fails and the failure set has no structure; above it the failure set is empty and every method converges.
- **Baselines:** the five published gates plus **Stagger**, a uniform-random allocation control implemented in this project (no gate, no score, no threshold; corrects one uniformly chosen recorded failure). Stagger carries **no citation** and must **never** be labelled a DAgger-family method. Diff-DAgger runs on the robot tasks only; Stagger is reported on GridWorld only in Table 7.

---

## 4. Table 7 and Table 8, verbatim

### Table 7 — final held-out success rate (per cent), mean $\pm$ standard error

Caption (CoC wording): *Final held-out success rate (per cent), mean $\pm$ standard error over 5 seeds (robot tasks), 9 seeds (GridWorld); Ni = initial demonstrations; Init SR = round-0 held-out success rate; best per row in bold. The budget is twenty expert demonstrations in every setting. Safe, Dropout, Ensemble, Thrifty and Diff-DAgger are the five published query-gated methods of the DAgger family. Stagger is a uniform-random allocation control implemented in this project and is not a DAgger-family method. Diff-DAgger is run on the robot tasks only; Stagger is reported on GridWorld only.*

| Task | Obs | Ni | Init SR | Safe | Dropout | Ensemble | Thrifty | Stagger | Diff-DAgger | DISEIL (ours) |
|--------|------|----|------|------|------|------|------|------|------|------|
| GridWorld 5x5 | state | 20 | 48.9 | 85.3±0.9 | 84.9±0.8 | 86.2±0.7 | 86.8±0.7 | 85.7±0.5 | — | **92.4±0.4** |
| GridWorld 5x5 | image | 20 | 47.0 | 88.8±0.9 | 88.4±0.7 | 88.8±0.9 | 88.7±0.6 | 89.1±0.8 | — | **91.3±0.6** |
| Push-T | state | 20 | 46.2 | 82.0±3.0 | 84.8±2.7 | 85.9±2.6 | 83.2±3.2 | — | 94.1±2.0 | **96.1±1.6** |
| Push-T | image | 20 | 43.3 | 78.1±3.5 | 82.1±3.1 | 83.2±3.0 | 79.3±3.6 | — | 89.0±2.1 | **92.6±2.2** |
| Lift | state | 8 | 67.2 | 99.2±0.7 | 99.2±0.4 | 99.2±0.4 | 100.0±0.0 | — | 99.2±0.4 | **100.0±0.0** |
| Lift | image | 8 | 66.4 | 99.6±0.4 | 97.2±1.6 | 98.8±0.7 | 99.6±0.4 | — | 99.6±0.4 | **100.0±0.0** |
| Wipe | state | 12 | 47.7 | 88.0±1.1 | 88.6±1.8 | 86.8±1.9 | 89.0±1.1 | — | 90.4±2.7 | **93.1±1.3** |
| Wipe | image | 12 | 45.2 | 69.6±2.4 | 83.2±3.0 | 84.4±3.2 | 69.2±4.0 | — | 88.6±1.4 | **92.3±1.4** |
| Door | state | 4 | 56.8 | 91.8±2.1 | 92.5±1.2 | 88.8±3.1 | 89.6±1.7 | — | 93.2±1.9 | **96.6±1.9** |
| Door | image | 4 | 43.1 | 82.4±1.4 | 81.8±1.5 | 83.0±4.9 | 82.8±1.2 | — | 84.2±1.6 | **88.6±1.5** |

In LaTeX write every `±` as `$\pm$`. Do **not** apply the CoC's `\footnotesize`/`\scriptsize` swap: that is layout manipulation.

Derived facts (all from the CoC, use as written):
- Margin over the strongest baseline in each setting: **mean 2.80 points, sd 1.73, range 0.0 to +5.6**.
- The strongest baseline moves: Diff-DAgger on Push-T, Wipe and Door under both modalities; ThriftyDAgger on GridWorld (state); the Stagger control on GridWorld (image); ties on both Lift settings (100.0 with ThriftyDAgger on state; 99.6 on image).
- Learning curves (`figures/selected_tasks_SE.pdf`, panels: GridWorld image, Push-T state, Lift state, Door state, Wipe image): separation on Push-T opens from about the fifth demonstration and holds; on GridWorld every method rises together and finishes bunched (whole image column between 88.4 and 91.3), which is why GridWorld (image) at +2.2 is among the smallest margins.

### Table 8 — per-demonstration information gain (mean $\pm$ standard error)

Caption (CoC wording): *Per-demonstration information gain (mean $\pm$ standard error). The policy's per-step loss on each newly acquired demonstration, measured before retraining on it; the error is the standard error over 5 seeds (robot tasks) and 9 seeds (GridWorld). Diff is Diff-DAgger, which is run on the robot tasks only, so its GridWorld cells are empty.*

| Task | Obs | Diff | DISEIL |
|---|---|---|---|
| GridWorld 5x5 | state | — | **3.55±0.84** |
| GridWorld 5x5 | image | — | **3.21±0.78** |
| Push-T | state | 1.57±0.49 | **2.81±0.93** |
| Push-T | image | 1.80±0.49 | **2.82±0.77** |
| Lift | state | 1.61±0.50 | **2.64±0.74** |
| Lift | image | 1.36±0.38 | **2.93±0.75** |
| Wipe | state | 1.43±0.36 | **2.91±0.90** |
| Wipe | image | 1.95±0.52 | **3.62±0.98** |
| Door | state | 1.84±0.50 | **3.43±0.95** |
| Door | image | 1.58±0.41 | **3.00±0.89** |

Supporting facts:
- Diff-DAgger is **the** reference here because its gate signal is the same per-step diffusion loss the metric is computed from, so it is the one baseline that selects on the quantity reported and the one that could be expected to lead on it. It does not. DISEIL is higher in all eight settings where it runs, and above every other gate and the control in all ten.
- Each cell pools **between 168 and 184 loss records**. A GridWorld setting acquires 9 seeds x 20 = 180 demonstrations at one record each; a robot setting acquires 100, so a robot demonstration contributes more than one record and the source does not record the decomposition. **This discrepancy is carried as an outstanding item in the CoC and must not be resolved by an assumption.** Safest for the paper: do not quote the 168–184 range in the main text.
- The retraining cadence enters the measurement: on GridWorld the scoring policy is the policy of the round; on the robot tasks it can be up to three demonstrations stale.
- **Why a high pre-retrain loss means novelty and not a bad datum** (the argument, keep it): a high loss admits two readings, novel data or an incoherent demonstration. The second is ruled out by two independent constructions. A prescription reaches the expert only after passing the feasibility check, so an infeasible scenario never becomes a demonstration; and the demonstration comes from the expert, whose trajectories are the fitting target, so it cannot be suboptimal with respect to that target. The argument also depends on the starting-competence band, because a policy that fails uniformly produces a high loss on any demonstration whatsoever.

---

## 5. The statistics, exactly as the CoC states them

- **Setting level (10 pairs):** DISEIL attains the best mean in every one of the ten settings. A sign test and a Wilcoxon signed-rank test both reject a coin-flip ranking at **two-sided $p = 0.002$**, which is **the smallest $p$-value attainable with ten pairs** and therefore the floor of what this design can produce.
- **Why the ten settings are not ten independent experiments:** they are five tasks under two observation modalities, and **the two modalities of a task share the expert, the reward structure and the reset distribution**, so they are correlated by construction and the **effective sample size is nearer five than ten**. This is a defect in the design, stated as one; the ten-pair figure must not be led with.
- **Collapsed task-level test (the claim of record):** paired differences are **+3.90 (GridWorld), +2.80 (Push-T), +0.20 (Lift), +3.20 (Wipe), +3.90 (Door)**. The sweep holds at **five from five**. A one-sided sign test gives **$p = 0.031$**, which is its floor at $n = 5$. A paired $t$-test over the same five means gives **$t(4) = 4.10$, $p = 0.015$ two-sided**. The two-sided nonparametric test does **not** reject at $n = 5$ (**$p = 0.063$**), and it cannot, because 0.063 is the smallest value it can return with five pairs.
- **What carries the claim:** the sign of the margin across the rows, not the size of any one of them. Table 7 shows the seed standard errors of the two arms overlapping in several rows, so those rows would not support the claim on their own, and the rows are not independent of one another. The systematic direction is the whole of the evidence and the pooled tests are what convert it into a number.

---

## 6. Ablations A1..A18

Three ablation settings throughout, chosen to span the three policy classes and both modalities: **GridWorld (image)** = CNN, **Push-T (state)** = state diffusion policy, **Door (image)** = image diffusion policy. Every triple below is in that order. **No aggregate $p$-value is reported in this section**: three paired settings are below the resolution of a Wilcoxon, sign or Friedman test, so what is reported is the three per-setting values and the sign they share.

*Margin retained* $= (\text{ablated} - \text{best baseline}) / (\text{full} - \text{best baseline})$, per cent. Near 100 = decorative. Near 0 = carries the result. Negative = the ablated system fell beneath the baseline it was built to beat.

**Knockouts (A1–A8), ordered by mean damage:**

| # | What it knocks out | Result |
|---|---|---|
| **A3** | the clustering step; each round greedily corrects the single highest-loss failure (loss signal kept, mode structure removed) | **−2.2 / −4.1 / −6.8, mean −4.37 pts**; margin retained **0.0 / −105.0 / −54.5, mean −53.2 %**. Falls beneath its own best baseline on Push-T (92.0 vs 94.1) and Door (81.8 vs 84.2); lands exactly on it on GridWorld. **Largest damage of any knockout.** Meanwhile **information gain does not fall**: +0.02 / +0.16 / −0.13, **mean +0.02**. |
| **A8** | the deterministic nearest-untried fallback promoted to the whole method | reaches **89.5 / 92.5 / 84.2**; costs **1.8 / 3.6 / 4.4, mean −3.27 pts**; retains **18.2 / −80.0 / 0.0, mean −20.6 %**. Beats the strongest baseline on GridWorld only; falls below it on Push-T; lands exactly on it on Door. |
| **A6** | the knowledge-augmented graph, removed from both the VLM and the reasoning prompts, so the feasibility loop has nothing to verify against | **−1.5 / −2.7 / −2.9, mean −2.37 pts**; retains **31.8 / −35.0 / 34.1, mean 10.3 %**. **Fallback rate rises to 27.1 / 27.0 / 34.8 % of rounds**, i.e. roughly five to seven of twenty rounds spent on a fallback correction. Third most damaging; costs nearly twice what the prescription model is worth. |
| **A4** | the prescription model, replaced by "always target the dominant cluster representative" over the same geometric clusters (the root-cause reasoning call is retained) | **−0.5 / −1.9 / −1.6, mean −1.33 pts**; retains **48.6 %**. |
| **A5** | the vision-language model (reasoning model keeps the descriptor and the taxonomy) | **−0.6 / −2.0 / −1.4, mean −1.33 pts**; retains **47.0 %**. |
| **A7** | bridging placement | **−1.3 / −1.1 / −1.4, mean −1.27 pts**; retains **40.9 / 45.0 / 68.2, mean 51.4 %**. |
| **A1** | the cluster memory ($\lambda = 0$) | **−0.6 / −0.4 / −1.2, mean −0.73 pts**; retains **72.7 / 80.0 / 72.7, mean 75.1 %**. **Smallest of the seven.** Price varies by a factor of three across settings. |
| **A2** | uniform-random allocation over recorded failures (no descriptor, no partition, no memory, no reasoning model) | reaches **89.1 / 82.3 / 80.0** against DISEIL's **91.3 / 96.1 / 88.6**. Below the strongest gated baseline on both robot settings; level with the gates on GridWorld (where it is the Stagger control of Table 7) and still below DISEIL. **Settles the "any failure replay would do" objection.** |

The CoC's own ranking sentence: clustering (−4.37, −53.2 %), fallback promoted (−3.27, −20.6 %), the graph (−2.37, 10.3 %), the prescription model and the VLM (−1.33 each, 48.6 and 47.0 %), bridging (−1.27, 51.4 %), the cluster memory (−0.73, 75.1 %).

**Design choices (A9–A13):**

| # | What it varies | Result |
|---|---|---|
| **A9** | the composition of the context set $S$ (target cluster fixed by the memory in every arm, so only $S$ varies). Full-system reference **92.0** | dropping the forced representative **−3.2**; dropping the farthest-point diversity fill **−3.2**; dropping the worst-loss seed **−3.27**; three episodes drawn at random from the cluster **−3.6**. The study cannot rank the three rules; random is worse than any single-rule removal, so the rules are complementary. |
| **A10** | the width of the geometric descriptor, scored by mean silhouette (a criterion with no relationship to success rate) | a clean inverted U with a single interior maximum; **six dimensions is the highest-scoring variant in each of the three settings**. Mean silhouette: **0.373 (2-d), 0.507 (4), 0.557 (5), 0.593 (6), 0.550 (8), 0.490 (10), 0.423 (12)**. Largest single step is **+0.133** from two to four dimensions (adding orientation); progress adds **+0.050**, contact distance **+0.037**. The fall above six is distance concentration, not information loss. |
| **A11** | the budget $B \in \{10, 20, 40\}$ | margin **+9.07 at $B{=}10$, +2.87 at $B{=}20$, +2.83 at $B{=}40$**; per-setting **+5.7 / +9.0 / +12.5** at $B{=}10$ and **+1.5 / +3.2 / +3.8** at $B{=}40$. DISEIL's own rate **rises** with budget (86.8→94.0 GridWorld, 87.9→97.7 Push-T, 82.3→99.5 Door); the margin shrinks because the baseline catches up from a lower start. **Retracted claim, must not reappear:** DISEIL at $B{=}10$ does **not** match the strongest baseline at $B{=}20$ (86.8 vs 89.1; 87.9 vs 94.1; 82.3 vs 84.2). What survives: the advantage of allocation grows as the budget shrinks. |
| **A12** | silhouette selection of $k$ against fixed $k \in \{2,3,4,5\}$ | silhouette wins on each of the three settings and beats the best fixed alternative by **4.1 points** on average: fixed $k{=}2$ costs **4.1**, $k{=}3$ costs **4.6**, $k{=}4$ costs **7.4**, $k{=}5$ costs **6.7**. Well outside the seed SE (0.6 to 1.6 points). No single fixed $k$ is best across settings: best is $k{=}3$ on GridWorld, $k{=}2$ on Push-T, $k{=}5$ on Door. |
| **A13** | the number of cited episodes jointly with the selection rule | plain top-three-by-peak-loss costs **2.13 points** against the three-rule construction (the full construction wins on each of the three settings); two citations cost **1.93**; five gives no measurable gain over three; citing every failure only raises prompt length. **The single-citation arm was not run** because it is confounded: bridging needs at least two cited failures. **It is not a null result and must not be reported as one.** |

**Diagnostics (A14–A18), which measure a property of the running system rather than knock a component out:**

| # | What it measures | Result |
|---|---|---|
| **A14** | root-cause label purity per geometric cluster (CoC Table 9) | GridWorld (image): purity **0.89**, root causes per cluster **1.62**, silhouette **0.58**. Push-T (state): **0.91 / 1.35 / 0.64**. Door (image): **0.84 / 1.86 / 0.56**. Geometric separation and semantic purity rise and fall together, so A10 and A14 are **not independent audits**. **Purity is scored against the reasoning model's own labels, so it records agreement between two components of the same system, not agreement with ground truth.** |
| **A15** | the distribution of the selected cluster count, pooled over the **308 clustered rounds** of the three settings | $k{=}3$ **26.3 %**, $k{=}4$ **23.4 %**, $k{=}5$ **21.4 %**, $k{=}2$ **14.9 %**, $k{=}6$ **14.0 %**; no value from two to six falls below 14 %. **21 % of GridWorld, 15 % of Push-T and 20 % of Door rounds never cluster at all** (fewer than four failures remain). The claim permitted: the number of discovered modes varies by round and is most often three or four, **not** that there are three failure modes. |
| **A16** | the share of accepted prescriptions that are bridged | **24 % GridWorld (image), 28 % Push-T (state), 21 % Door (image)**. **Discrepancy of record:** bridging should be inapplicable on a discrete grid, yet 24 % of GridWorld prescriptions are marked bridged and A7 records an effect there. The data are the source of truth; the resolution is an outstanding item. |
| **A17** | failures per round over the budget, Push-T (image), 5 seeds (the only setting logged per round) | falls from **forty-two to two**, halving by round 8 and falling by an order of magnitude by round 17. The descriptor, clustering and memory do their work through round 17; the last three rounds run the fallback. |
| **A18** | per-round wall-clock and token cost, DISEIL against SafeDAgger, matched task/modality/hardware | P1 (first round, an upper bound, all five settings) and P5 (mean over LM-active rounds of the longer-budget runs). **Add-on** (the DISEIL-specific stages less the query-gate rollout the baseline runs in its place): **+270.0 s Door (state), +293.0 Door (image), +700.0 Wipe (image), +1,232.1 Push-T (image), +62.6 GridWorld (image)** under P1, at **9,560 to 82,116 tokens**. Shared cost (retrain + evaluation, both arms pay it) is **783 to 1,491 s** under P1 on the RoboSuite settings. Round ratio **1.13 to 2.75**. P1 and P5 rows measure different objects and **must not be read against one another**; token counts are **not comparable across rows** because the backends differ. |

### What earns main-paper space

**Main paper (a single compact ablation figure or table plus two paragraphs):**
- **A3 with the information-gain dissociation.** This is *the* argument of the paper, not one knockout among seven: the largest damage (−4.37, margin −53.2 % retained, below its own baseline on two settings) while information gain does not fall (mean +0.02). It is what converts Table 8 from a decoration into evidence, and it is the finding a reviewer cannot get anywhere else. Non-negotiable.
- **A2 and A8, the two controls, together as the allocation ladder** (`COC_REPORT/figures_generated/F1_allocation_ladder.pdf`). A2 kills the "any failure replay would do" objection; A8 kills "your fallback heuristic is doing the work". A reviewer will raise both, and they cost one figure between them.
- **A11, the budget sweep** (+9.07 at $B{=}10$ vs +2.83 at $B{=}40$). One sentence, no figure. It is the sample-efficiency claim in the title and it earns its line, including the retraction.

**Compact summary row in the main paper, one line each, no discussion:** A6 (−2.37, fallback 27–35 %), A4 and A5 (−1.33 each), A7 (−1.27), A1 (−0.73). This ordering is itself the honest statement that the language components are not the source of the advantage, and stating it in the main paper is a strength, not a concession.

**Supplementary:** everything else. A9, A10, A12, A13, A14, A15, A16, A17, A18 in full, with figures F2–F13, plus per-setting numbers, prompts, KAG examples, the cost tables, and the descriptor table.

---

## 7. Honest caveats that must survive into the paper

1. **Lift is at a ceiling and is uninformative about any mechanism.** DISEIL reaches 100.0±0.0 on both Lift settings, ThriftyDAgger reaches 100.0±0.0 on Lift (state) and the task-level paired difference is **+0.20**, the smallest of the five. Lift begins the budget furthest above the target band (67.2 and 66.4) because the smallest prefix that trains a stable policy already clears it. Where the framework and its baselines both sit at the ceiling, a null result cannot separate a component that does nothing from a component whose effect cannot be observed, which is why the ablation programme is run only in settings with headroom.
2. **A4 and A5 show small gaps, and the reason is structural.** Each is worth **1.33 points on average**, and **every individual gap is comparable to the seed standard error of the corresponding full run**. The partition is geometric and consumes no output from any foundation model, so by the time either model is called the decision that matters has already been taken by the descriptor and the memory. Read the other way this is a practical result: a deployment that cannot afford the reasoning stack can delete it, keep the geometric clustering, the memory and the deterministic heuristic, and still beat every baseline on every setting. Limitations that cut against us and must be kept: the A4 heuristic is a *strong* one (it is itself an allocation rule and it inherits the memory's rotation), it cannot bridge, so A4 and A7 are not independent; and A5 does not test whether a better visual model would help more.
3. **The cluster memory is a configurable, task-specific component, not a headline contribution.** An earlier draft advanced it as one of two contributions; A1 prices it at 0.73 points on average, the least damaging of seven, with the price varying by a factor of three across three settings, and every gap no larger than the seed SE. The kernel is inert in most rounds: the candidate set of near-dominant modes is a **singleton in 56 to 84 per cent of rounds** on the settings with enough telemetry, and the dominant mode is then returned regardless of the penalty. Its width is a single global constant while the tasks do not share a spatial scale. It is switched **on** in every run reported. It is **not drawn in the architecture figure** (`Architectural_Diagram.pdf` already has it removed).
4. **The confidence result is an observed property, not a mechanism.** Pearson $r$ between reported confidence and realised $\Delta$SR runs **0.82 to 0.89** across the ten settings (state then image: GridWorld 0.88 / 0.82, Push-T 0.87 / 0.88, Lift 0.88 / 0.89, Wipe 0.82 / 0.86, Door 0.83 / 0.82). The figure is GridWorld (image), **$r = 0.82$ over $n = 180$ prescriptions**, which is nine seeds at a budget of twenty. What makes it usable is the order of availability: the confidence is reported blind at prescription time, before the demonstration is collected, the expert is called, the policy is retrained and the re-rollout is run. Two limits: it is measured on DISEIL runs only, so it says nothing about whether a baseline's gate signal would predict its own $\Delta$SR as well; and **no experiment gates on it** (nothing is skipped, deferred or re-prescribed on a low score), so it is not yet a mechanism inside the system.
5. **Wipe (image) has not plateaued.** DISEIL and the strongest baseline are both still rising at the twentieth demonstration, so the +3.7-point claim rests on the final gap, 92.3 against 88.6, and not on a demonstrated asymptote. A longer budget could close it and it has not been shown to vanish.
6. **Eq. 11 (solvability) is not exercised** in any reported run and no number is attributable to it. It is also not ablated anywhere, which is named as a gap in the study.
7. **The A16 bridging discrepancy on GridWorld** (24 per cent bridged on a discrete grid) is unresolved and reported as active because the data are the source of truth.
8. **Diff-DAgger's per-step loss is Diff-DAgger's idea.** DISEIL uses it for localisation and also compares against it as a baseline. Both facts must be stated plainly wherever the signal appears.

---

## 8. Limitations (CoC 5.1.9, ordered by how much they constrain the claim)

1. **The selector reasons about one round and knows nothing about the dataset.** Its only representation of history is the recency-discounted penalty on already-corrected clusters, which records *where* corrections have been placed and holds no representation of *what the training set contains*. The selector can determine that the policy failed at a configuration with root cause "grasp failure"; it cannot determine that the training set already holds six demonstrations of that cause, so the failure is under-fitting and not under-coverage. Under a restricted budget, a demonstration spent re-teaching material the dataset already contains is a demonstration lost. Nothing in the instantiation prevents that expenditure and nothing measures how often it occurs. A memory indexed on geometry answers "have we placed a demonstration near here?" and not "does the dataset already contain this behaviour?", and the two come apart wherever geometry does not determine cause (A14).
2. **The descriptor is designed by hand, and geometry recovers cause only where configuration determines cause.** A10 defends its *width* on a criterion independent of success rate but says nothing about whether a learned descriptor [nair2022r3m] would separate modes better; that experiment has not been run. A14 locates the ceiling: purity 0.84 to 0.91, 1.35 to 1.86 distinct root causes per cluster, measured against the reasoning model's own labels and covarying with silhouette. A human-labelled root-cause set is the measurement that is missing.
3. **The cluster memory is worth about three quarters of a point, and how much depends on the task.** No study establishes when a task should switch it on; three settings disagree by a factor of three, three settings cannot support a rule, and the rule is not offered.
4. **Each language-model component is worth about one and a third points, for a structural reason.** The comparison that sharpens the point: removing the prescription model costs 1.33 points, removing the environmental constraints its proposals are verified against costs 2.37 and drives the fallback rate to 27–35 per cent of rounds. Two readings are admissible, that language models add little to demonstration selection, or that the language model here was given too little to reason over; the evidence supports the second. For the paper, keep this as a limitation and route the "what to do about it" to at most one forward-looking clause in the conclusion (Aim 2 material is out of scope).
5. **The allocation machinery is active early and idle late.** Failures fall from forty-two to two; below four remaining failures the sweep is skipped. Across the three settings, **15 to 21 per cent of all rounds skip clustering**. The allocation account is an account of the early and middle rounds, and the failure-count curve is instrumented on **one setting only**.
6. **The reasoning pipeline costs seconds and tokens.** Add-on the baseline never pays: **63 s per round at the cheapest setting and 1,232 s at the most expensive, at 9,560 to 82,116 tokens**. The round ratio 1.13 to 2.75 is the smaller of the two facts, because the retrain and the evaluation that both arms pay dilute it. A ratio near one is not evidence that the pipeline is cheap; it is evidence that the retrain is expensive. The models run only at demonstration-selection time, never in the control loop and never at execution. Degrading to the deterministic heuristic is a real deployment option at about 1.33 points.
7. **The experiments are in simulation and no expert is a person.** The expert answers instantly, correctly, and at the same price for any prescription. Scripted oracle on Lift, Wipe, Door; a PPO policy on Push-T [schulman2017ppo]; a human on GridWorld. The uniformity assumption is what allows the budget to be counted in demonstrations. Outside simulation the budget is a person's time, demonstrations differ by an order of magnitude in what they cost, and human demonstrators are not uniformly correct [khazatsky2024droid, mandlekar2020iwr, mandlekar2018roboturk]. **The second half of the information-gain argument fails the moment the demonstrator is a person**, because high pre-retrain loss identifies novel data only if invalid demonstrations are ruled out by construction.
8. **A3 knocks out the allocation *stack*, not the partition in isolation**, because the descriptor and the memory have nothing to operate on once modes are gone. The cleaner variant, keeping the descriptor and replacing agglomerative clustering with a random partition into $k$ groups, was not run. It is future work, not a claim.
9. **A6 does not decompose the graph**, so it cannot be said whether the workspace bounds alone would recover most of the damage.

---

## 9. Citation key mapping (CoC bracket numbers to `references.bib` keys)

The CoC uses numeric brackets. The bib has **102 verified entries**. The numbering is alphabetical by key with one known swap: **[15] = chen2024autotamp** and **[16] = chen2024spatialvlm** in key order, but the CoC's semantics require [15] to be SpatialVLM/BLINK-style spatial-reasoning work and [16] to be AutoTAMP. **Cite by semantics, never by translating a bracket number mechanically.** Every `\cite` key must be verified against `references.bib` before the build.

Frequently needed: `ross2011dagger` (DAgger), `lee2025diffdagger` (Diff-DAgger), `zhang2017safedagger`, `menda2017dropoutdagger`, `menda2019ensembledagger`, `hoque2021thriftydagger`, `chi2023diffusionpolicy`, `florence2021implicitbc`, `tao2024maniskill3`, `zhu2020robosuite`, `nair2022r3m`, `schulman2017ppo`, `ward1963hierarchical`, `lloyd1982kmeans`, `rousseeuw1987silhouette`, `pedregosa2011sklearn`, `eldar1997fps`, `hart1968astar`, `cormen2022algorithms`, `sener2018coreset`, `ash2020badge`, `settles2009active`, `houlsby2011bald`, `cazenavette2022mtt` (dataset distillation, the name clash to distinguish), `memmel2025strap`, `belkhale2023dataquality`, `lin2024datascaling`, `oneill2023openx`, `tenorth2013knowrob`, `lewis2020rag`, `edge2024graphrag`, `liu2023llmp`, `chen2024autotamp`, `chen2024spatialvlm`, `fu2024blink`, `duan2025aha`, `liu2023reflect`, `florensa2017reversecurriculum`, `eysenbach2018leavenotrace`, `bai2025qwen3vl` (the VLM), `yang2025qwen3` (the prescription model), `bain1995cloning`, `pomerleau1988alvinn`, `shimodaira2000covariate`, `ross2010reductions`.

**Name clash to handle explicitly in Related Work:** *dataset distillation* [cazenavette2022mtt] compresses a large training set into a small synthetic one, runs **after** collection, and operates on data already held. *Demonstration distillation*, the sense in this paper's title, runs **before** collection and decides which demonstration to acquire next. The two share a word and share no mechanism.
