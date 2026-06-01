"""Environment sourcing + path roots for the pool_rl_robo suite.

The continuous-control analogue of pool_x_selector's ``layouts/layout_setup.py``:
it owns the suite path roots (REPO_ROOT / SUITE_ROOT / RESULTS_ROOT — imported
by ``orchestrator/workspace.py``), registers the gymnasium_robotics (Fetch)
envs, resolves env-id version fallbacks, builds envs, and handles the Box vs
Dict (goal-conditioned) observation flattening that is the "MultiInputPolicy"
requirement. No layout pools here — the per-round candidate pool in this domain
is the novice's visited states (see ``selection/rollout.py``).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# This file: …/baseline_vs_p4/pool_rl_robo/envs/env_setup.py
#   parents[0]=envs [1]=pool_rl_robo [2]=baseline_vs_p4
#   [3]=equivariant_CNN_hybrid [4]=Equivariant_pathway [5]=repo root
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUITE_ROOT = Path(__file__).resolve().parents[1]      # …/pool_rl_robo
RESULTS_ROOT = SUITE_ROOT / "results"

# Canonical key order for flattening a goal-conditioned Dict observation.
FETCH_OBS_KEYS = ("observation", "achieved_goal", "desired_goal")


def is_goal_env(env_id: str) -> bool:
    return env_id.startswith("Fetch")


def register_robotics() -> None:
    """Register the gymnasium_robotics (Fetch) envs. Idempotent; tolerant of
    both the modern ``gym.register_envs`` hook and the older entrypoint."""
    import gymnasium as gym
    try:
        import gymnasium_robotics  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"gymnasium_robotics import failed: {e}") from e
    try:
        gym.register_envs(gymnasium_robotics)
    except Exception:  # noqa: BLE001
        pass
    fn = getattr(gymnasium_robotics, "register_robotics_envs", None)
    if callable(fn):
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass


def resolve_env_id(env_id: str) -> Tuple[str, bool]:
    """Return (resolved_id, substituted). Falls back to the highest registered
    version of the same base if ``env_id`` is not registered."""
    from gymnasium.envs.registration import registry
    if env_id in registry:
        return env_id, False
    m = re.match(r"^(.*)-v(\d+)$", env_id)
    if m:
        base, v = m.group(1), int(m.group(2))
        for vv in range(v, -1, -1):
            cand = f"{base}-v{vv}"
            if cand in registry:
                return cand, (cand != env_id)
    raise ValueError(f"env id {env_id!r} not registered (call register_robotics())")


def make_env(env_id: str, seed: Optional[int] = None,
             max_episode_steps: Optional[int] = None, render: bool = False):
    import gymnasium as gym
    kwargs = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    if render:
        kwargs["render_mode"] = "rgb_array"   # for image-based (R3M) policies
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def is_dict_obs(env) -> bool:
    import gymnasium as gym
    return isinstance(env.observation_space, gym.spaces.Dict)


def flatten_obs(obs) -> np.ndarray:
    """Flatten an observation to a 1-D float32 vector. Dict (goal) obs ->
    observation+achieved_goal+desired_goal; Box obs -> ravel."""
    if isinstance(obs, dict):
        keys = [k for k in FETCH_OBS_KEYS if k in obs]
        keys += [k for k in sorted(obs.keys()) if k not in keys]
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in keys])
    return np.asarray(obs, dtype=np.float32).ravel()


def obs_dim(env) -> int:
    obs, _ = env.reset()
    return int(flatten_obs(obs).shape[0])


def act_dim(env) -> int:
    return int(np.prod(env.action_space.shape))


def act_bounds(env) -> Tuple[np.ndarray, np.ndarray]:
    low = np.asarray(env.action_space.low, dtype=np.float32).ravel()
    high = np.asarray(env.action_space.high, dtype=np.float32).ravel()
    low = np.where(np.isfinite(low), low, -1.0).astype(np.float32)
    high = np.where(np.isfinite(high), high, 1.0).astype(np.float32)
    return low, high


def goal_distance(obs_raw) -> Optional[float]:
    if isinstance(obs_raw, dict) and "achieved_goal" in obs_raw and "desired_goal" in obs_raw:
        a = np.asarray(obs_raw["achieved_goal"], dtype=np.float32).ravel()
        d = np.asarray(obs_raw["desired_goal"], dtype=np.float32).ravel()
        return float(np.linalg.norm(a - d))
    return None


def episode_success(env_id: str, last_info: dict, terminated: bool, truncated: bool) -> bool:
    """Goal envs use the real ``info['is_success']``; locomotion has no goal set,
    so we use a survival proxy (episode did NOT end via early/unhealthy
    termination). HalfCheetah never terminates -> always 'survives' (documented
    caveat: ThriftyDAgger risk saturates there)."""
    if is_goal_env(env_id):
        return bool(last_info.get("is_success", 0.0))
    return not bool(terminated)
