"""GridWorld failure screening: roll the classifier on N sampled layouts, score each
step with predictive entropy (self-uncertainty l_t), collect FAILED episodes (didn't
reach goal), and anchor the descriptor at t_flag = first entropy-threshold crossing
(K-patience) — the same first-crossing rule as the robot side (02_..md #1).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..p4.screen import _first_threshold_crossing
from .descriptor import GWFailureDescriptor
from .env import make_maze, reset_to, scene_of


def screen_failures_gw(policy, layouts: List[Dict], *, max_steps: int,
                       patience: int = 2) -> List[GWFailureDescriptor]:
    threshold = getattr(policy, "loss_threshold", None)
    fails: List[GWFailureDescriptor] = []
    env = make_maze()
    for layout in layouts:
        reset_to(env, layout)
        entropies: List[float] = []
        actions: List[int] = []
        cells = [tuple(env.agent_pos)]          # cell BEFORE each action
        success, t = False, 0
        while t < max_steps:
            grid, agent, goal, fires = scene_of(env)
            entropies.append(policy.uncertainty(grid, agent, goal, fires))
            a = policy.act(grid, agent, goal, fires)
            actions.append(a)
            _, _, term, trunc, info = env.step(a)
            success = bool(info["success"]); t += 1
            cells.append(tuple(env.agent_pos))
            if term or trunc:
                break
        if not success and actions:
            t_flag, _src = _first_threshold_crossing(entropies, threshold, patience)
            t_flag = min(t_flag, len(cells) - 1)
            fails.append(GWFailureDescriptor(
                episode_id=int(layout["seed"]), seed=int(layout["seed"]),
                t_star=int(t_flag), T=int(t), peak_loss=float(max(entropies)),
                layout=layout, agent_cell=list(cells[t_flag]), exec_actions=actions))
    env.close()
    return fails
