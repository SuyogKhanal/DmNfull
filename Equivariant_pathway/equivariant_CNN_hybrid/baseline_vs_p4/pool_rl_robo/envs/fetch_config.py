"""Load an LLM-prescribed Fetch task CONFIGURATION into the simulator.

The full P4-LLM pipeline (Fetch goal envs) prescribes an environment
configuration — a goal position (and, for FetchPickAndPlace, an object
placement) — which must be (a) LOADABLE into the MuJoCo sim and (b) SOLVABLE by
the expert. This module does the loading + workspace-bound clipping; the strict
expert-solvable check lives in ``selection/rollout.expert_demo_from_config``
(it rolls the expert and requires ``info['is_success']``).

A config is ``{"goal": [x,y,z], "object": [x,y] | None}``. Goals/objects are
clipped into the env's own sampling ranges so they stay in-distribution (the
pretrained experts solve in-range configs with sr~1.0); the expert-solve check
is the final gate.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _u(env):
    return env.unwrapped


def get_bounds(env, env_id: str) -> Dict[str, np.ndarray]:
    u = _u(env)
    gpos = np.asarray(getattr(u, "initial_gripper_xpos", [1.34, 0.75, 0.55]), float)[:3]
    tr = float(getattr(u, "target_range", 0.15))
    orr = float(getattr(u, "obj_range", 0.15))
    hoff = float(getattr(u, "height_offset", gpos[2]))
    goal_low = gpos - np.array([tr, tr, tr])
    goal_high = gpos + np.array([tr, tr, tr])
    if env_id.startswith("FetchPickAndPlace"):
        # place targets sit on the table or up to ~45cm in the air
        goal_low[2] = hoff
        goal_high[2] = hoff + 0.45
    return {"goal_low": goal_low, "goal_high": goal_high,
            "obj_low": gpos[:2] - orr, "obj_high": gpos[:2] + orr,
            "obj_z": np.array([hoff], float)}


def clip_goal(goal, b) -> np.ndarray:
    return np.clip(np.asarray(goal, float).ravel()[:3], b["goal_low"], b["goal_high"]).astype(np.float32)


def clip_object(obj, b) -> np.ndarray:
    return np.clip(np.asarray(obj, float).ravel()[:2], b["obj_low"], b["obj_high"]).astype(np.float32)


def is_fetch_pick(env_id: str) -> bool:
    return env_id.startswith("FetchPickAndPlace")


def load_config(env, env_id: str, goal, obj: Optional[np.ndarray] = None,
                seed: Optional[int] = None) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
    """Reset, override the goal (+ object for PickAndPlace), and return the
    resulting RAW obs Dict (with the overridden desired/achieved goal) plus the
    clipped (goal, object) actually loaded. The strict expert-solve check is
    done by the caller."""
    u = _u(env)
    if seed is not None:
        env.reset(seed=int(seed))
    else:
        env.reset()
    b = get_bounds(env, env_id)
    g = clip_goal(goal, b)
    o = None
    if is_fetch_pick(env_id) and obj is not None:
        o = clip_object(obj, b)
        try:
            u._utils.set_joint_qpos(u.model, u.data, "object0:joint",
                                    np.array([o[0], o[1], float(b["obj_z"][0]), 1.0, 0.0, 0.0, 0.0]))
            u._utils.set_joint_qvel(u.model, u.data, "object0:joint", np.zeros(6))
        except Exception:  # noqa: BLE001 — caught by the expert-solve check downstream
            pass
        try:
            import mujoco
            mujoco.mj_forward(u.model, u.data)
        except Exception:  # noqa: BLE001
            pass
    u.goal = np.asarray(g, float)
    obs = u._get_obs()
    return obs, g, o
