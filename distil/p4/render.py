"""Render offscreen RGB frames of screening failures for the VLM.

A failure descriptor carries {seed, exec_actions, t_star}. To show the VLM what
the failure looks like at its peak-uncertainty step, we reproduce that exact
state in a render-capable env (a RobosuiteStateEnv with offscreen=True — same
wrapper, so reset()+step() reproduce the screening dynamics byte-for-byte) and
capture the agentview camera at t*.

One offscreen env per task is cached and reused: each offscreen env holds its own
GL context, so building a fresh one per frame would leak contexts and be slow.
On a GPU node set MUJOCO_GL=egl (the envs module default); on a CPU-only login
node set MUJOCO_GL=osmesa.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..envs import make_env

# Per-task VLM camera (chosen by visual inspection so the RELEVANT scene is shown,
# not just the robot pedestal): Lift's agentview shows the cube+gripper, but for
# Door/Wipe agentview is dominated by the robot base — frontview shows the door
# (+handle) and the dirt path respectively.
TASK_CAMERA = {"Lift": "agentview", "Door": "frontview", "Wipe": "frontview"}

# task -> render-capable RobosuiteStateEnv (one offscreen GL context, reused)
_RENDER_ENV: Dict[str, Any] = {}


def _render_env(task: str, env_h: int):
    if task not in _RENDER_ENV:
        # obs keys don't affect physics; a default render env replays identically.
        _RENDER_ENV[task] = make_env(task, horizon=env_h, offscreen=True)
    return _RENDER_ENV[task]


def render_failure(task: str, env_h: int, seed: int, exec_actions: List[Any],
                   t_star: int, out_path: str, *, camera: str = None,
                   width: int = 512, height: int = 512) -> str:
    """Reproduce the failure to t* and save an RGB PNG; returns out_path."""
    cam = camera or TASK_CAMERA.get(task, "agentview")
    renv = _render_env(task, env_h)
    renv.reset(seed=int(seed))                     # == screening reset (reseed+reset)
    k = int(min(max(0, int(t_star)), len(exec_actions)))
    for i in range(k):                             # replay to the peak-loss state
        renv.step(exec_actions[i])                 # wrapper step == identical dynamics
    frame = renv.render(camera=cam, height=height, width=width)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(frame).save(out_path)
    return out_path


def render_failure_frames(task: str, env_h: int, seed: int, exec_actions: List[Any],
                          t_star: int, out_dir: str, ep_id: int, *, camera: str = None,
                          width: int = 512, height: int = 512) -> dict:
    """Render the failure's start / peak-loss(t*) / end frames (V3 3-frame VLM
    perception). Returns {'start':path,'high_loss':path,'end':path}."""
    from PIL import Image
    cam = camera or TASK_CAMERA.get(task, "agentview")
    renv = _render_env(task, env_h)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n = len(exec_actions)
    k = int(min(max(0, int(t_star)), n))

    def _save(role: str) -> str:
        f = renv.render(camera=cam, height=height, width=width)
        p = str(Path(out_dir) / f"ep{ep_id}_{role}.png")
        Image.fromarray(f).save(p)
        return p

    renv.reset(seed=int(seed))
    paths = {"start": _save("start")}
    for i in range(k):
        renv.step(exec_actions[i])
    paths["high_loss"] = _save("high_loss")
    for i in range(k, n):
        renv.step(exec_actions[i])
    paths["end"] = _save("end")
    return paths


def close_render_envs() -> None:
    for e in _RENDER_ENV.values():
        try:
            e.close()
        except Exception:
            pass
    _RENDER_ENV.clear()
