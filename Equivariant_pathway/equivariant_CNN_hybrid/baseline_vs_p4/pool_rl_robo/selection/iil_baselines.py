"""The 5 IIL baselines on the SHARED diffusion-policy backbone.

Continuous-action-space version. Every method in this suite uses the fork's
diffusion policy as the novice π_nov; only the per-step QUERY RULE differs. The
loop is the fork's classic Diff-DAgger episode (``diffdagger_baseline._run_one_episode``):
roll the diffusion policy step by step; when the rule fires, the expert takes
over INLINE and finishes the episode; the successful expert trajectory is the
ONE corrective demo for that round; retrain FROM SCRATCH every Nd demos;
evaluate the frozen held-out set. 1 demo/round, target_sr early stop.

The 5 rules (baseline_implementations_continuous_guide.md), all computed in
joint-delta (rel_joint_pos) space where the policy's predicted action and the
expert's clip-and-scaled action both live, so ``‖a_nov − a_exp‖₂`` is consistent:

  safe     — query when ‖a_nov − a_exp‖₂ > tau                          (SafeDAgger*)
  dropout  — N MC-dropout draws; query when in-τ-ball fraction p̂ < p   (DropoutDAgger)
  ensemble — M policies; query when mean discrepancy > tau OR doubt > chi (EnsembleDAgger)
  thrifty  — query when novelty(=ensemble doubt) > δ_h OR risk(=1−Q) > β_h (ThriftyDAgger)
  stagger  — query at ONE uniform-random visited state per episode      (Stagger)

diff_dagger (the fork's native diffusion-loss CDF rule) is run separately via the
fork's run_diffdagger_baseline; p4_top3 via p4/pipeline.py. Those + these 5 = 7.

FAITHFULNESS CAVEATS (report; do not silently "fix"):
- The acting/evaluated policy is the single novice π_nov for every kind; the M
  ensemble members feed only the doubt/novelty SIGNAL (EnsembleDAgger's "act with
  the mean" is approximated by acting with the novice — cleaner for held-out eval,
  which scores one policy). Members are retrained from scratch with distinct seeds
  every round so the doubt is meaningful from round 1.
- Per-timestep rules are reduced to a per-episode decision: the FIRST step whose
  rule fires is the intervention point t*; the expert finishes from there (=1 demo).
- DropoutDAgger uses a runtime MC-dropout wrapper on the UNet global-cond path
  (the fork ships dropout but disables it in forward; we never edit the fork).
- discrepancy lives in joint-delta space, not pd_ee_delta_pose (the fork runs the
  panda joint controller).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from ..envs import env_setup as E

E.bootstrap_fork_path()

from collections import deque  # noqa: E402

from tensordict import TensorDict  # noqa: E402

# Fork engine pieces (read-only reuse).
from diffdagger.main_pipeline.diffdagger_baseline import (  # noqa: E402
    _append_episode_to_dataset, _to_bool,
)
from diffdagger.main_pipeline.sim_bridge import (  # noqa: E402
    collect_initial_demos_shared, train_policy, evaluate_heldout,
)
from diffdagger.agents.RL_agent import clip_and_scale_action  # noqa: E402

from . import uncertainty as U  # noqa: E402
from . import success_q as SQ   # noqa: E402

# ---- lockstep dispatch (kind per method dir name) ----------------------
# diff_dagger shares this unified arm too: its query is the diffusion policy's
# NATIVE rule (CDF(diffusion_loss) > alpha-quantile for K consecutive steps), and
# running it HERE (not the fork's run-to-budget baseline) makes it early-stop at
# target_sr like every other method — the apples-to-apples fix for the
# "Diff-DAgger consumed 43 demos" asymmetry (the fork baseline had no target stop).
KIND_OF = {
    "diff_dagger": "diff",
    "safe_dagger": "safe",
    "dropout_dagger": "dropout",
    "ensemble_dagger": "ensemble",
    "thrifty_dagger": "thrifty",
    "stagger": "stagger",
}
KINDS = ("diff", "safe", "dropout", "ensemble", "thrifty", "stagger")


def _info(method: str, env_id: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{method}|{env_id}] {msg}", flush=True)


# ---- per-step signal helpers (joint-delta space) -----------------------
def _full_obs(env) -> Dict[str, Any]:
    """Full (unfiltered) observation dict the expert needs — bypasses the
    policy env's FilterObservation wrapper (gymnasium forwards get_obs/get_info
    through to the ManiSkill base env)."""
    return env.get_obs(env.get_info())


def _arm_limits(env):
    cc = env.agent._controller_configs["pd_joint_delta_pos"]["arm"]
    return cc.lower, cc.upper


def _expert_delta(env, expert, full_obs, seed: int, num_joints: int) -> torch.Tensor:
    """π_exp(s) as a joint-delta vector (B, num_joints), matching the policy's
    rel_joint_pos parameterisation. Mirrors MultipleExperts.move_to_next_goal's
    clip_and_scale + wrap-to-[-π,π]."""
    n_exp = len(getattr(expert, "experts", [None]))
    raw = expert.get_action(full_obs, expert_idx=seed % max(1, n_exp),
                            deterministic=True)
    lower, upper = _arm_limits(env)
    delta = clip_and_scale_action(raw, lower, upper)
    delta[:, :num_joints] = (delta[:, :num_joints] + math.pi) % (2 * math.pi) - math.pi
    return delta[:, :num_joints]


def _policy_delta(action_seq: torch.Tensor, policy, num_joints: int) -> torch.Tensor:
    """a_nov as a joint-delta vector (B, num_joints): the first executed step of
    the policy's predicted sequence (rel_joint_pos ⇒ the raw value IS the delta)."""
    idx = policy.obs_horizon - 1
    return action_seq[:, idx, :num_joints]


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((a - b).reshape(-1)).item())


# ---- the IIL episode runner (mirrors the fork _run_one_episode) --------
def _run_one_episode_iil(env, expert, policy, cfg, seed, *, kind, hp,
                         thresholds, members=None, qnet=None, rng=None):
    """One episode with the diffusion novice + an IIL query rule. Returns the
    same dict shape as the fork baseline (intervened/success/ep_tds/...) plus
    the picked score for logging."""
    obs, info = env.reset(seed=int(seed))
    ep_info = dict(seed=int(seed))
    env.set_action_space("joint_pos")
    expert.reset(env)
    policy.reset()
    if members:
        for m in members:
            m.reset()

    num_joints = int(env.num_joints)
    obs_deque = deque([obs] * cfg.pred_horizon, maxlen=cfg.obs_horizon)
    timestep = 0
    ep_tds: List[Any] = []
    done = False
    intervention_count = 0
    conseq = 0
    max_steps = cfg.max_episode_steps
    hard_limit = cfg.max_episode_steps + cfg.expert.max_episode_steps
    picked_score = None
    # Stagger: pre-pick one uniform-random visited state to query.
    stagger_t = (rng.randint(0, max(0, max_steps - 1)) if (kind == "stagger" and rng)
                 else None)

    def _obs_seq():
        return {
            k: torch.stack([obs_deque[i][k] for i in range(cfg.obs_horizon)]
                           ).swapaxes(0, 1) for k in obs_deque[0].keys()
        }

    def _decide_query(action_seq) -> (bool, Optional[float]):
        """Apply the kind's per-step query rule. Returns (query, score)."""
        if kind == "stagger":
            return (timestep >= stagger_t, float(timestep))
        try:
            full = _full_obs(env)
            a_nov = _policy_delta(action_seq, policy, num_joints)
            a_exp = _expert_delta(env, expert, full, int(seed), num_joints)
        except Exception as exc:  # pragma: no cover - GPU-only path
            _info("iil", str(cfg.env_id), f"signal error ({exc}); no query")
            return (timestep >= max_steps, None)

        if kind == "safe":
            d = _l2(a_nov, a_exp)
            return (d > float(hp.get("tau", 0.1)), d)
        if kind == "dropout":
            N = int(hp.get("N", 10)); tau = float(hp.get("tau", 0.1))
            draws = U.mc_dropout_deltas(policy, _obs_seq(), N, num_joints)  # (N,J)
            within = (torch.linalg.vector_norm(draws - a_exp.reshape(1, -1), dim=1)
                      <= tau).float().mean().item()
            return (within < float(hp.get("p", 0.9)), 1.0 - within)
        if kind in ("ensemble", "thrifty"):
            member_deltas = U.member_deltas(members, _obs_seq(), num_joints)  # (M,J)
            a_bar = member_deltas.mean(dim=0)
            doubt = float(member_deltas.var(dim=0).mean().item())
            if kind == "ensemble":
                mean_disc = float(torch.linalg.vector_norm(
                    member_deltas - a_exp.reshape(1, -1), dim=1).mean().item())
                q = (mean_disc > float(hp.get("tau", 0.1))) or \
                    (doubt > float(hp.get("chi", 0.05)))
                return (q, doubt)
            # thrifty: novelty (doubt) OR risk (1 - Q)
            risk = 0.0
            if qnet is not None:
                try:
                    obsv = SQ.flat_obs(full, expert.obs_keys)
                    risk = 1.0 - SQ.q_value(qnet, obsv, a_nov)
                except Exception:
                    risk = 0.0
            dh = float(thresholds.get("delta_h", 0.0))
            bh = float(thresholds.get("beta_h", 1.0))
            return ((doubt > dh) or (risk > bh), max(doubt, risk))
        return (timestep >= max_steps, None)

    def _act_and_query():
        """Compute the policy action sequence + the kind's query decision. For
        kind='diff' the query is the diffusion policy's NATIVE rule (returned by
        get_action(dagger=True)); all other kinds use _decide_query."""
        if kind == "diff":
            d = policy.get_action(_obs_seq(), dagger=True, return_dict=True)
            aseq = d["action"] if isinstance(d, dict) else d
            q = _to_bool(d.get("query", False)) if isinstance(d, dict) else False
            sc = float(d.get("diffusion_loss", 0.0)) if isinstance(d, dict) else None
            return aseq, q, sc
        out = policy.get_action(_obs_seq(), dagger=False, return_dict=False)
        aseq = out["action"] if isinstance(out, dict) else out
        q, sc = _decide_query(aseq)
        return aseq, q, sc

    env.set_action_space(cfg.action_space)
    action_seq, query, sc0 = _act_and_query()
    if sc0 is not None:
        picked_score = sc0

    while timestep < hard_limit or not done:
        if query or timestep >= max_steps:
            if conseq == 0:
                intervention_count += 1
                env.set_action_space(cfg.expert.action_space)
                if timestep > 0:
                    for _ in range(int(cfg.get("wait_timestep", 0))):
                        a = expert.generate_stationary_action()
                        obs, reward, done, _, info = env.step(a)
                        obs_deque.append(obs)
            expert.setup_task()
            while True:
                conseq += 1
                td = expert.move_to_next_goal(ep_info)
                td.update(dict(episode=torch.ones(len(td["episode"])) * int(seed)))
                ep_tds.append(td)
                info = env.get_info()
                done = _to_bool(td["done"][-1])
                timestep += len(td["episode"])
                for i in range(-min(cfg.obs_horizon, len(td["episode"])), 0, 1):
                    # keep the batch dim (1, dim) to match reset/step obs so a later
                    # _obs_seq stack/swapaxes stays (B, H, dim) — td[k][i] alone is
                    # (dim,) and would transpose-corrupt the next policy forward.
                    obs = {k: td[k][i:i + 1] for k in cfg.obs_keys}
                    obs_deque.append(obs)
                if done or not timestep < hard_limit:
                    break
            if done or not timestep < hard_limit:
                break
        else:
            env.set_action_space(cfg.action_space)
            actions = env.post_process_action(action_seq)
            if conseq > 0:
                conseq = 0
            for i in range(cfg.action_horizon):
                action = actions[:, policy.obs_horizon - 1 + i]
                obs, reward, _, _, info = env.step({"action": action})
                done = _to_bool(info.get("success", False))
                obs_deque.append(obs)
                timestep += 1
                if done or not timestep < max_steps:
                    break
                action_seq, query, sc = _act_and_query()
                if sc is not None:
                    picked_score = sc
                if query:
                    break
        if done:
            break

    total = sum(len(td["episode"]) for td in ep_tds) if ep_tds else 0
    return {
        "intervened": intervention_count > 0,
        "intervention_count": intervention_count,
        "success": bool(done),
        "ep_tds": ep_tds,
        "expert_steps": total,
        "total_steps": timestep,
        "picked_score": picked_score,
    }


def run_iil_arm(*, method: str, kind: str, env, eval_env, expert, cfg,
                make_policy: Callable[[], Any], make_dataset: Callable[[], Any],
                init_ckpt: str, init_sr: float, work_dir: str,
                budget: int, nd_retrain: int, target_sr: float,
                max_rounds: int, heldout_seed_base: int, heldout_num_ep: int,
                round_epoch: Optional[int], max_train_steps: Optional[int],
                hp: Dict, seed: int) -> Dict[str, Any]:
    """Run one IIL arm to a fixed expert-demo budget. Writes the suite-format
    learning_curve.json into <work_dir> and returns the summary."""
    import random
    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    lc_path = work / "learning_curve.json"
    ah = cfg.action_horizon
    mes = cfg.env.max_episode_steps
    rng = random.Random(0xC0FFEE + seed + 1009 * (abs(hash(method)) % 997))
    M = int(hp.get("M", 5))
    curve: List[Dict[str, Any]] = []
    t0 = time.time()

    dataset = make_dataset()
    collect_initial_demos_shared(env, expert, dataset, cfg, list(range(int(hp.get(
        "num_initial_demos", 50)))))

    # Round 0: shared init for the acting novice (+ M members for ensemble/thrifty).
    policy = make_policy(); policy.load(init_ckpt); policy.to(cfg.device)
    if kind == "diff":
        # Calibrate the native Diff-DAgger query threshold from the loaded
        # training-set diffusion-loss CDF at alpha (paper rule); set K.
        policy.alpha = float(hp.get("alpha", 0.99))
        policy.patience = int(hp.get("patience", 1))
        policy.patience_window = int(hp.get("patience", 1))
        _cdf = getattr(policy, "diffusion_loss_cdf", None)
        if _cdf is not None:
            policy.diffusion_loss_threshold = _cdf.get_quantile(policy.alpha)
    policy.reset()
    members = None
    if kind in ("ensemble", "thrifty"):
        members = [make_policy() for _ in range(M)]
        for m in members:
            m.load(init_ckpt); m.to(cfg.device); m.reset()
    qnet = None  # thrifty success-Q, trained lazily after the first demos

    def _curve_point(rnd, q, sr, added, reason):
        try:
            cum = int(len(dataset.rb))
        except Exception:
            cum = -1
        tec = getattr(expert, "total_expert_calls", None)   # secondary cost axis
        curve.append({"round": rnd, "n_queries": q, "success_rate": sr,
                      "demos_added": added, "n_demos": cum, "stop_reason": reason,
                      "total_expert_calls": tec})
        with open(lc_path, "w") as f:
            json.dump({"method": method, "env": str(cfg.env_id), "seed": seed,
                       "budget": budget, "backbone": "diffusion",
                       "history": curve,
                       "final_performance": {"success_rate": sr, "n_queries": q},
                       "stopped_reason": reason}, f, indent=2, default=str)
        _info(method, str(cfg.env_id),
              f"round {rnd}: sr={sr:.3f} q={q}/{budget} added={added} stop={reason}")

    if hasattr(expert, "reset_counts"):   # count only DAgger-phase expert calls
        expert.reset_counts()
    _curve_point(0, 0, float(init_sr), 0, None)

    def _retrain(rnd):
        if round_epoch is not None:
            cfg.epoch = int(round_epoch)
        if max_train_steps is not None:
            cfg.max_train_steps = int(max_train_steps)
        nonlocal policy, members, qnet
        res = train_policy(policy, dataset, cfg,
                           checkpoint_dir=str(work / f"checkpoints/round_{rnd:03d}"),
                           log_dir=str(work / f"logs/round_{rnd:03d}"), fine_tune=False)
        if res.get("status") != "ok":
            raise RuntimeError(f"train_policy failed: {res.get('error')}")
        policy = res["policy"]
        if members is not None:                       # M independent retrains
            new_members = []
            for mi in range(M):
                U.seed_all(seed + 101 * (mi + 1))
                mres = train_policy(make_policy(), dataset, cfg,
                                    checkpoint_dir=str(work / f"members/m{mi}/round_{rnd:03d}"),
                                    log_dir=None, fine_tune=False)
                new_members.append(mres["policy"])
            members = new_members
        if kind == "thrifty":
            qnet = SQ.train_success_q_from_dataset(dataset, expert.obs_keys,
                                                   num_joints=int(env.num_joints),
                                                   gamma=float(hp.get("gamma", 0.99)),
                                                   epochs=int(hp.get("q_epochs", 50)),
                                                   device=cfg.device)

    thresholds = {"delta_h": 0.0, "beta_h": 1.0}
    nd = max(1, int(nd_retrain))
    interventions = 0
    last_trained = 0
    ep_seed = 0
    episodes = 0
    max_episodes = int(hp.get("max_episodes", 400))
    stopped = "max_rounds"

    while interventions < budget and episodes < max_episodes:
        episodes += 1
        ep = _run_one_episode_iil(env, expert, policy, cfg, ep_seed, kind=kind,
                                  hp=hp, thresholds=thresholds, members=members,
                                  qnet=qnet, rng=rng)
        ep_seed += 1
        if (not ep["success"]) or ep["expert_steps"] > cfg.expert.max_episode_steps:
            continue
        if (not ep["intervened"]) or ep["expert_steps"] < 10:
            continue
        _append_episode_to_dataset(dataset, ep["ep_tds"])
        interventions += 1
        if interventions % nd == 0 or interventions >= budget:
            added = interventions - last_trained
            _retrain(interventions)
            sr = evaluate_heldout(policy, eval_env, heldout_seed_base,
                                  heldout_num_ep, ah, mes)
            last_trained = interventions
            reason = "budget_exhausted" if interventions >= budget else None
            _curve_point(interventions, interventions, sr, added, reason)
            if sr >= target_sr:
                stopped = "target_hit"; break
            if interventions >= budget:
                stopped = "budget_exhausted"; break

    if stopped == "max_rounds" and episodes >= max_episodes:
        stopped = "max_episodes"          # honest label: backstop hit, not a real stop
    if interventions > last_trained:
        added = interventions - last_trained
        _retrain(interventions)
        sr = evaluate_heldout(policy, eval_env, heldout_seed_base, heldout_num_ep, ah, mes)
        _curve_point(interventions, interventions, sr,
                     added, "budget_exhausted" if interventions >= budget else stopped)

    final = curve[-1] if curve else {}
    return {"method": method, "env": str(cfg.env_id), "stopped_reason": stopped,
            "final_performance": {"success_rate": final.get("success_rate"),
                                  "n_queries": interventions},
            "history": curve, "elapsed_seconds": time.time() - t0,
            "total_expert_calls": interventions}
