"""Pretrained-expert registry + loader (the domain 'expert', analogous to the
maze A* expert that pool_x_selector borrows from ``Equivariant_pathway.expert``).

Owns the 5 (env -> HuggingFace SB3/SB3-contrib expert) pairs and the
environment-specific LLM prompts, plus loading (with the gym-pickle / HER /
numpy-2 fixes the Phase-1 smoke surfaced) and the π_exp(s) query.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Dict, Optional

import numpy as np

from .env_setup import make_env, resolve_env_id

ExpertSpec = namedtuple("ExpertSpec", ["repo", "filename", "algo"])

# Repo/file corrections vs the originally-specified ids (Phase-1 smoke surfaced
# the 404s): SB3 RL-Zoo MuJoCo TQC checkpoints are tagged -v3 (run fine on the
# -v4 envs; same spaces). FetchPickAndPlace's file is `sac-FetchPickAndPlace-v4.zip`
# (native v4, SAC). FetchReach's sb3 v1 checkpoint is INCOMPATIBLE with gymnasium
# FetchReach-v4 (17-dim vs 16-dim obs; v1 needs mujoco_py), so we use
# kuds/fetch-reach-dense-tqc — TQC trained natively on FetchReachDense-v4 (same
# 16-dim space; still reaches the goal on the sparse FetchReach-v4 env).
EXPERTS: Dict[str, ExpertSpec] = {
    "HalfCheetah-v4":       ExpertSpec("sb3/tqc-HalfCheetah-v3",           "tqc-HalfCheetah-v3.zip",     "TQC"),
    "Hopper-v4":            ExpertSpec("sb3/tqc-Hopper-v3",                "tqc-Hopper-v3.zip",          "TQC"),
    "Walker2d-v4":          ExpertSpec("sb3/tqc-Walker2d-v3",             "tqc-Walker2d-v3.zip",        "TQC"),
    "FetchReach-v4":        ExpertSpec("kuds/fetch-reach-dense-tqc",       "best_model.zip",             "TQC"),
    "FetchPickAndPlace-v4": ExpertSpec("IntelliGrow/FetchPickAndPlace-v4", "sac-FetchPickAndPlace-v4.zip", "SAC"),
}

ENV_PROMPTS: Dict[str, str] = {
    "HalfCheetah-v4":
        "A 2D cheetah-like body must run forward as fast as possible. "
        "Reward is forward velocity. Select demonstrations that show sustained high-speed locomotion.",
    "Hopper-v4":
        "A single-legged robot must hop forward without falling. "
        "Reward is forward progress while staying upright. Select demonstrations with stable, continuous hopping.",
    "Walker2d-v4":
        "A bipedal robot must walk forward steadily. "
        "Reward is forward velocity with balance. Select demonstrations showing smooth, stable bipedal walking.",
    "FetchReach-v4":
        "A 7-DoF Fetch robot arm must move its end-effector to a goal position in 3D space. "
        "Reward is sparse (1 if within 5cm of goal). Select demonstrations that reach the goal efficiently.",
    "FetchPickAndPlace-v4":
        "A Fetch robot must grasp a box on a table and place it at a 3D target location. "
        "Reward is sparse (1 if box is within 5cm of target). Select demonstrations showing successful grasp and place.",
}


def _algo_class(name: str):
    if name == "TQC":
        from sb3_contrib import TQC
        return TQC
    if name == "SAC":
        from stable_baselines3 import SAC
        return SAC
    raise ValueError(f"unknown expert algo {name!r}")


def _download(repo: str, filename: str) -> str:
    try:
        from huggingface_sb3 import load_from_hub
        return load_from_hub(repo, filename)
    except Exception:  # noqa: BLE001
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo, filename=filename)


# custom_objects overrides pickled schedules / old-Gym spaces / removed HER kwargs
# so a checkpoint trained under an older SB3 loads cleanly for predict-only use.
_SAFE_CUSTOM_OBJECTS = {
    "learning_rate": 0.0,
    "lr_schedule": (lambda _: 0.0),
    "clip_range": (lambda _: 0.0),
    "exploration_schedule": (lambda _: 0.0),
    "replay_buffer_kwargs": {},
}


def load_expert(env_id: str, resolved_env_id: Optional[str] = None, device: str = "cpu"):
    """Download + load the pretrained expert for ``env_id`` (keyed by the
    ORIGINAL requested id). Overrides observation/action spaces with the
    gymnasium env's (skips the old-Gym unpickle) and passes ``env=`` for HER
    models. Returns an SB3 model with ``.predict(obs, deterministic=True)``."""
    spec = EXPERTS[env_id]
    path = _download(spec.repo, spec.filename)
    cls = _algo_class(spec.algo)
    custom = dict(_SAFE_CUSTOM_OBJECTS)
    rid = resolved_env_id or resolve_env_id(env_id)[0]
    tmp = make_env(rid)
    try:
        custom["observation_space"] = tmp.observation_space
        custom["action_space"] = tmp.action_space
        model = cls.load(path, env=tmp, device=device, custom_objects=custom)
    finally:
        tmp.close()
    return model


def expert_action(expert, obs_raw) -> np.ndarray:
    """π_exp(s). Feeds the RAW (possibly Dict) observation to the SB3 policy."""
    action, _ = expert.predict(obs_raw, deterministic=True)
    return np.asarray(action, dtype=np.float32).ravel()
