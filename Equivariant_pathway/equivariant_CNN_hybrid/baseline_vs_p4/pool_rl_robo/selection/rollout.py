"""Env-walk helpers, backbone-agnostic via the ``policies.Policy`` interface.

Everything works in FEATURES: ``policy.features(obs_raw, env)`` (state-flatten for
the MLP locomotion backbone; R3M(render)+goal for the diffusion Fetch backbone).
The novice acts via ``policy.act``; uncertainty comes from ``policy.samples``; the
expert always acts on the RAW state obs. Demonstrations are stored as
``{"feat": (n, fdim), "act": (n, act_dim), "success": bool}``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..envs import env_setup as E
from ..envs.experts import expert_action

SAMPLE_KINDS = ("dropout", "ensemble", "thrifty")  # methods that need uncertainty samples


def assemble(demo_trajs: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    feat = [t["feat"] for t in demo_trajs if len(t["feat"]) > 0]
    act = [t["act"] for t in demo_trajs if len(t["act"]) > 0]
    if not feat:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
    return (np.concatenate(feat, axis=0).astype(np.float32),
            np.concatenate(act, axis=0).astype(np.float32))


def collect_expert_demos(env, env_id: str, expert, policy, n_demos: int, seed: int,
                         max_steps: int) -> List[Dict]:
    trajs: List[Dict] = []
    for i in range(int(n_demos)):
        obs_raw, _ = env.reset(seed=int(seed) + 700000 + i)
        fl, al = [], []
        term = trunc = False
        last_info: Dict = {}
        si = 0
        while not (term or trunc) and si < int(max_steps):
            fl.append(policy.features(obs_raw, env))
            a = expert_action(expert, obs_raw)
            al.append(a)
            obs_raw, _r, term, trunc, last_info = env.step(a)
            si += 1
        if fl:
            trajs.append({"feat": np.stack(fl), "act": np.stack(al),
                          "success": E.episode_success(env_id, last_info, term, trunc)})
    return trajs


def _rollout_episode(env, env_id: str, policy, expert, ep_seed: int, max_steps: int,
                     need_samples: bool, k: int) -> Tuple[List[Dict], Dict]:
    obs_raw, _ = env.reset(seed=int(ep_seed))
    g0 = np.asarray(obs_raw["desired_goal"], np.float32) if isinstance(obs_raw, dict) and "desired_goal" in obs_raw else None
    o0 = np.asarray(obs_raw["achieved_goal"], np.float32) if isinstance(obs_raw, dict) and "achieved_goal" in obs_raw else None
    recs: List[Dict] = []
    feats, nov_actions, rewards = [], [], []
    term = trunc = False
    last_info: Dict = {}
    si = 0
    while not (term or trunc) and si < int(max_steps):
        f = policy.features(obs_raw, env)
        nov = policy.act(f)
        exp = expert_action(expert, obs_raw)
        rec = {"t": si, "feat": f, "novice_action": nov, "expert_action": exp,
               "discrepancy": float(np.linalg.norm(nov - exp)),
               "goal_dist": E.goal_distance(obs_raw)}
        if need_samples:
            rec["samples"] = policy.samples(f, int(k))
        recs.append(rec)
        feats.append(f)
        nov_actions.append(nov)
        obs_raw, r, term, trunc, last_info = env.step(nov)
        rewards.append(float(r))
        si += 1
    suffix = 0.0
    run = 0.0
    n = len(rewards)
    for t in range(n - 1, -1, -1):
        suffix += rewards[t]
        recs[t]["reward_to_go"] = suffix
    for t in range(n):
        recs[t]["return_so_far"] = run
        run += rewards[t]
    succ = E.episode_success(env_id, last_info, term, trunc)
    episode = {"ep_seed": int(ep_seed),
               "actions": (np.stack(nov_actions) if nov_actions else np.zeros((0, 0), np.float32)),
               "feat": (np.stack(feats) if feats else np.zeros((0, 0), np.float32)),
               "success": succ, "ep_return": float(sum(rewards)), "goal": g0, "object": o0}
    for r in recs:
        r["is_success"] = succ
    return recs, episode


def rollout_candidates(env, env_id: str, policy, expert, ep_seed_base: int,
                       n_episodes: int, max_steps: int, need_samples: bool, k: int) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    episodes: List[Dict] = []
    for e in range(int(n_episodes)):
        recs, ep = _rollout_episode(env, env_id, policy, expert, ep_seed_base + e,
                                    max_steps, need_samples, k)
        for r in recs:
            r["ep"] = e
        records.extend(recs)
        episodes.append(ep)
    return records, episodes


def expert_demo_from_state(env, env_id: str, policy, ep_seed: int,
                           prefix_actions: np.ndarray, expert, max_steps: int) -> Optional[Dict]:
    obs_raw, _ = env.reset(seed=int(ep_seed))
    term = trunc = False
    for a in np.asarray(prefix_actions, np.float32):
        if term or trunc:
            return None
        obs_raw, _r, term, trunc, _i = env.step(a)
    if term or trunc:
        return None
    fl, al = [], []
    last_info: Dict = {}
    si = 0
    while not (term or trunc) and si < int(max_steps):
        fl.append(policy.features(obs_raw, env))
        a = expert_action(expert, obs_raw)
        al.append(a)
        obs_raw, _r, term, trunc, last_info = env.step(a)
        si += 1
    if not fl:
        return None
    return {"feat": np.stack(fl), "act": np.stack(al),
            "success": E.episode_success(env_id, last_info, term, trunc)}


def expert_demo_from_config(env, env_id: str, policy, goal, obj, expert, max_steps: int):
    """STRICT loadable+solvable check: load the prescribed {goal, object} config,
    roll the EXPERT, require success. Records (policy features, expert action) as
    the demonstration. Returns (demo|None, solved, loaded_goal, loaded_obj)."""
    from ..envs import fetch_config as FC
    obs_raw, g, o = FC.load_config(env, env_id, goal, obj)
    fl, al = [], []
    term = trunc = False
    last_info: Dict = {}
    si = 0
    while not (term or trunc) and si < int(max_steps):
        fl.append(policy.features(obs_raw, env))
        a = expert_action(expert, obs_raw)
        al.append(a)
        obs_raw, _r, term, trunc, last_info = env.step(a)
        si += 1
    solved = bool(last_info.get("is_success", 0.0))
    if not fl:
        return None, False, g, o
    demo = {"feat": np.stack(fl), "act": np.stack(al), "success": solved}
    return (demo if solved else None), solved, g, o


def eval_policy(env, env_id: str, policy, n_eval: int, eval_seed: int, max_steps: int) -> Dict:
    returns: List[float] = []
    succ: List[bool] = []
    is_goal = E.is_goal_env(env_id)
    for i in range(int(n_eval)):
        obs_raw, _ = env.reset(seed=int(eval_seed) + i)
        term = trunc = False
        last_info: Dict = {}
        si = 0
        tot = 0.0
        while not (term or trunc) and si < int(max_steps):
            a = policy.act(policy.features(obs_raw, env))
            obs_raw, r, term, trunc, last_info = env.step(a)
            tot += float(r)
            si += 1
        returns.append(tot)
        succ.append(E.episode_success(env_id, last_info, term, trunc))
    arr = np.asarray(returns, np.float32)
    return {"mean_reward": float(arr.mean()) if arr.size else 0.0,
            "std_reward": float(arr.std()) if arr.size else 0.0,
            "success_rate": (float(np.mean(succ)) if (is_goal and succ) else None)}
