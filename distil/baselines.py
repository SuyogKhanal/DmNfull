"""Interactive-IL baseline gate family for DISTIL's comparison (05_..md / paper).

Six when-to-query baselines on the SAME diffusion policy + retrain-from-scratch loop
as DISTIL, differing only in the per-step query gate. Paper-exact HPs (tau=0.1, N=10,
p=0.9, chi=0.05, M=5, alpha=0.99). Ported from pool_rl_robo/selection/{iil_baselines,
uncertainty,success_q}.py and rewired to distil's policy/expert/train/eval API.

Fair accounting == DISTIL: 1 successful intervention/round, stop at n_init + budget; the
same frozen held-out eval each round. The gate decides WHERE the expert takes over during
a policy rollout; the expert then completes the episode and that segment is the demo.

  diffdagger : diffusion-loss CDF quantile + K-patience (native, distil/diffdagger.py rule)
  safe       : ||a_nov - a_exp|| > tau
  dropout    : N stochastic draws; query when frac within tau-ball of a_exp < p
  ensemble   : M policies; query when mean-disc > tau OR doubt(var) > chi
  thrifty    : novelty(doubt) > delta OR risk(1-Q_psi) > beta (thresholds = pool quantiles)
  stagger    : intervene at one uniformly-random timestep (random control)
"""
from __future__ import annotations

import gc
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .collect import collect_demos
from .diffdagger import train_and_calibrate, train_bc
from .eval import evaluate_policy, make_obs_seq

HP = dict(tau=0.1, N=10, p=0.9, chi=0.05, M=5, gamma=0.99, alpha_h=0.1, q_epochs=50)


def _first_action(policy, obs_seq) -> np.ndarray:
    return policy.get_action(obs_seq)[0, 0].cpu().numpy()


class SuccessQ(nn.Module):
    """ThriftyDAgger risk head: Q(s,a) ~ success-to-go; risk = 1 - Q."""

    def __init__(self, state_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + act_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)

    def risk(self, s, a) -> float:
        with torch.no_grad():
            return float(1.0 - self.forward(s, a).item())


def _train_success_q(trajs, state_dim, act_dim, device, gamma, epochs) -> SuccessQ:
    q = SuccessQ(state_dim, act_dim).to(device)
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    S, A, Y = [], [], []
    for t in trajs:
        s, a = np.asarray(t["state"]), np.asarray(t["action"])
        L = len(a)
        succ = 1.0  # collected demos are successful
        for i in range(L):
            S.append(s[i]); A.append(a[i]); Y.append(succ * (gamma ** (L - 1 - i)))
    if not S:
        return q
    S = torch.tensor(np.array(S), dtype=torch.float32, device=device)
    A = torch.tensor(np.array(A), dtype=torch.float32, device=device)
    Y = torch.tensor(np.array(Y), dtype=torch.float32, device=device)
    for _ in range(epochs):
        idx = torch.randint(0, len(S), (min(256, len(S)),), device=device)
        loss = nn.functional.mse_loss(q(S[idx], A[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return q


def _query(arm, policies, q_head, obs_seq, a_exp, device):
    """Return True if the gate fires (expert should take over) at this step."""
    p0 = policies[0]
    a_nov = _first_action(p0, obs_seq)
    if arm == "safe":
        return float(np.linalg.norm(a_nov - a_exp)) > HP["tau"]
    if arm == "dropout":
        draws = np.stack([_first_action(p0, obs_seq) for _ in range(HP["N"])])
        within = np.mean(np.linalg.norm(draws - a_exp, axis=1) <= HP["tau"])
        return within < HP["p"]
    if arm in ("ensemble", "thrifty"):
        members = np.stack([_first_action(m, obs_seq) for m in policies])  # (M, act)
        doubt = float(members.var(axis=0).mean())
        disc = float(np.linalg.norm(members - a_exp, axis=1).mean())
        if arm == "ensemble":
            return disc > HP["tau"] or doubt > HP["chi"]
        # thrifty: novelty(doubt) OR risk
        s = torch.tensor(obs_seq["state"][0, -1], dtype=torch.float32, device=device)[None]
        a = torch.tensor(a_nov, dtype=torch.float32, device=device)[None]
        risk = q_head.risk(s, a) if q_head is not None else 0.0
        return doubt > HP.get("delta_h", 0.0) or risk > HP.get("beta_h", 1.0)
    return False


def run_baseline(cfg, arm, make_env_fn, make_expert_fn, device, log_fn=print,
                 init_trajs=None, image_size=None):
    """One baseline arm; same eval/budget as DISTIL. arm in {diffdagger,safe,dropout,
    ensemble,thrifty,stagger}. image_size>0 => image-conditioned policy (frames)."""
    from .diffdagger import run_diffdagger
    if arm == "diffdagger":                      # native distil Diff-DAgger loop
        hist = run_diffdagger(cfg, make_env_fn, make_expert_fn, device, log_fn=log_fn,
                              init_trajs=init_trajs, image_size=image_size)
        return _wrap_history(hist, cfg, arm)

    env_h = cfg["env_horizon"]
    obs_h, act_h = cfg["obs_horizon"], cfg["act_horizon"]
    budget = int(cfg["budget"]) if cfg.get("budget") is not None else cfg["final_demos"]
    env = make_env_fn(env_h); eval_env = make_env_fn(env_h); expert = make_expert_fn()
    trajs = list(init_trajs) if init_trajs is not None else collect_demos(
        env, expert, cfg["num_init_demos"], seed_base=0, max_steps=env_h, log_fn=log_fn)
    final_demos = len(trajs) + budget
    M = HP["M"] if arm in ("ensemble", "thrifty") else 1
    rng = np.random.default_rng(cfg["seed"])
    screen_base = 3_000_000
    history = []

    for rnd in range(int(cfg["max_rounds"])):
        steps = cfg["initial_train_steps"] if rnd == 0 else cfg["round_train_steps"]
        log_fn(f"\n===== {arm} Round {rnd} | dataset={len(trajs)} =====")
        # diff-dagger needs the calibrated threshold; others just BC. Ensemble/Thrifty
        # train M members from scratch (different seeds).
        policies = []
        for m in range(M):
            torch.manual_seed(cfg["seed"] * 100 + m)
            policies.append(train_bc(cfg, trajs, device, steps, log_fn if m == 0 else (lambda *a: None)))
        policy = policies[0]
        q_head = _train_success_q(trajs, policy.state_dim if hasattr(policy, "state_dim") else
                                  np.asarray(trajs[0]["state"]).shape[1],
                                  np.asarray(trajs[0]["action"]).shape[1], device,
                                  HP["gamma"], HP["q_epochs"]) if arm == "thrifty" else None

        ev = evaluate_policy(policy, eval_env, num_episodes=cfg["eval_episodes"],
                             obs_horizon=obs_h, act_horizon=act_h, device=device,
                             base_seed=cfg["eval_seed_base"], max_steps=env_h,
                             image_size=image_size)
        log_fn(f"  [eval] SR={ev['success_rate']:.3f}")
        n_at_eval = len(trajs)
        if len(trajs) >= final_demos:
            history.append({"round": rnd, "n_demos_at_eval": n_at_eval,
                            "eval_success": ev["success_rate"]})
            break

        # roll the policy on fresh seeds; the gate decides where the expert takes over.
        demo = _collect_intervention(arm, policies, q_head, env, expert, cfg,
                                     seed=screen_base + rnd, device=device, log_fn=log_fn,
                                     rng=rng, image_size=image_size)
        if demo is not None:
            trajs.append(demo)
        history.append({"round": rnd, "n_demos_at_eval": n_at_eval,
                        "eval_success": ev["success_rate"]})
        del policies; gc.collect(); torch.cuda.is_available() and torch.cuda.empty_cache()

    for e in (env, eval_env):
        e.close()
    return {"history": history,
            "final_success": next((h["eval_success"] for h in reversed(history)
                                   if h["eval_success"] is not None), None),
            "n_demos": len(trajs), "budget": budget, "arm": arm}


@torch.no_grad()
def _rollout_states(policy, env, obs_h, act_h, seed, device, max_steps):
    s = env.reset(seed=seed)
    dq = deque([s] * obs_h, maxlen=obs_h)
    states, t, done, success = [s], 0, False, False
    while t < max_steps and not done:
        obs_seq = make_obs_seq(dq, obs_h, device)
        acts = policy.get_action(obs_seq)[0].cpu().numpy()
        for i in range(act_h):
            s, r, done, success, info = env.step(acts[i]); dq.append(s); states.append(s); t += 1
            if done:
                break
    return states, success


def _collect_intervention(arm, policies, q_head, env, expert, cfg, *, seed, device,
                          log_fn, rng, max_rollouts=20, image_size=None):
    """Roll the policy; at the first gate-fire step, expert takes over to done -> demo."""
    from .eval import frame
    obs_h, act_h, env_h = cfg["obs_horizon"], cfg["act_horizon"], cfg["env_horizon"]
    policy = policies[0]
    for k in range(max_rollouts):
        s = env.reset(seed=seed + k)
        expert.reset(env)
        dq = deque([s] * obs_h, maxlen=obs_h)
        idq = deque([frame(env, image_size)] * obs_h, maxlen=obs_h) if image_size else None
        t, done, success = 0, False, False
        if arm == "stagger":                       # random intervention timestep
            fire_at = int(rng.integers(0, max(1, env_h // 2)))
        while t < env_h and not done:
            obs_seq = make_obs_seq(dq, obs_h, device, idq)
            a_exp = np.asarray(expert.get_action(env, s), dtype=np.float32)
            fire = (t >= fire_at) if arm == "stagger" else _query(arm, policies, q_head, obs_seq, a_exp, device)
            if fire:
                # expert takeover to done -> the corrective demo
                demo, succ, _ = _expert_from(env, expert, s, env_h, image_size)
                if succ and demo is not None:
                    log_fn(f"  [{arm}] intervention at t={t} (rollout {k}) len={len(demo['action'])}")
                    return demo
                break
            a_nov = policy.get_action(obs_seq)[0, 0].cpu().numpy()
            s, r, done, success, info = env.step(a_nov); dq.append(s); t += 1
            if idq is not None:
                idq.append(frame(env, image_size))
    return None


def _expert_from(env, expert, s, max_steps, image_size=None):
    """Expert takeover from state s to done. Returns (demo|None, success, len); demo =
    {state, action[, image]} (image frames aligned with each recorded state)."""
    from .eval import frame
    states, actions, images, done, success, t = [], [], [], False, False, 0
    while not done and t < max_steps:
        a = np.asarray(expert.get_action(env, s), dtype=np.float32)
        states.append(np.asarray(s, dtype=np.float32)); actions.append(a)
        if image_size:
            images.append(frame(env, image_size))
        s, r, done, success, info = env.step(a); t += 1
    if not states or not success:
        return None, success, t
    demo = {"state": np.stack(states), "action": np.stack(actions)}
    if image_size and images:
        demo["image"] = np.stack(images)
    return demo, success, t


def _wrap_history(hist, cfg, arm):
    """Adapt run_diffdagger's dict-of-lists history to the aggregate schema."""
    rounds = []
    for i in range(len(hist.get("round", []))):
        rounds.append({"round": hist["round"][i], "n_demos_at_eval": hist["n_trajs"][i],
                       "eval_success": hist["eval_success"][i]})
    return {"history": rounds,
            "final_success": hist.get("final_success", rounds[-1]["eval_success"] if rounds else None),
            "n_demos": hist.get("n_demos"), "budget": cfg.get("budget"), "arm": arm}
