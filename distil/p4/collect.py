"""P4-LLM demo-collection primitives on robosuite (the two V3-hybrid options).

Both return a demo in the repo's trajectory format ({"state": (N,d), "action":
(N,a)}) or None on failure, plus success + length — so a collected demo drops
straight into the Diff-DAgger `trajs` list and DemoDataset.

  collect_select : DAgger-faithful on-policy correction of ONE real failure —
                   replay the failure's recorded policy actions to t* (the
                   deterministic re-roll of that scene), then the reactive expert
                   takes over from the divergence state to completion. Only the
                   expert's (state, action) pairs are the demo (== a Diff-DAgger
                   intervention).
  collect_bridge : prescribe a middle-ground object placement, reset, and the
                   expert demonstrates the whole task from there.

Budget rule (fairness): a demo counts only if success is True; failed/empty
attempts are budget-free (the caller escalates/retries).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bridge import set_object_pose


def _run_expert_to_done(env, expert, s, max_steps, image_size=None) -> Tuple[Optional[Dict], bool, int]:
    """Record (state, action[, image]) as the reactive expert finishes from state s."""
    from ..eval import frame
    states, actions, images = [], [], []
    success, done, t = False, False, 0
    while not done and t < max_steps:
        a = np.asarray(expert.get_action(env, s), dtype=np.float32)
        states.append(np.asarray(s, dtype=np.float32))
        actions.append(a)
        if image_size:
            images.append(frame(env, image_size))
        s, r, done, success, info = env.step(a)
        t += 1
    if not states:
        return None, False, 0
    demo = {"state": np.stack(states), "action": np.stack(actions)}
    if image_size and images:
        demo["image"] = np.stack(images)
    return (demo if success else None), success, t


def collect_select(env, expert, *, seed: int, exec_actions: List[Any], t_star: int,
                   max_steps: int, image_size=None) -> Tuple[Optional[Dict], bool, int]:
    """Re-roll the failure scene (deterministic replay to t*), then expert corrects."""
    s = env.reset(seed=seed)
    k = int(min(max(0, t_star), len(exec_actions)))
    done = False
    for i in range(k):                                   # replay the policy prefix
        s, r, done, succ, info = env.step(np.asarray(exec_actions[i], dtype=np.float32))
        if done:
            break
    expert.reset(env)
    return _run_expert_to_done(env, expert, s, max_steps, image_size)


def collect_bridge(env, expert, *, task: str, x: float, y: float, yaw: float = 0.0,
                   z: Optional[float] = None, quat=None, seed: int,
                   max_steps: int, image_size=None) -> Tuple[Optional[Dict], bool, int, Dict]:
    """Prescribe a middle-ground object pose, then expert demonstrates the task.
    z/quat carry the full pose for objects that need it (Door frame body)."""
    env.reset(seed=seed)
    applied = set_object_pose(env.env, task, x, y, yaw=yaw, z=z, quat=quat)
    s = env._flatten(env.env._get_observations())        # obs AFTER placement
    expert.reset(env)
    demo, success, t = _run_expert_to_done(env, expert, s, max_steps, image_size)
    return demo, success, t, applied
