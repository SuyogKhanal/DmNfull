"""Render failure-episode frames for the P4 VLM analysis pass.

Replays a recorded novice episode in a render-enabled env and captures the
START, the FIRST-MISTAKE moment (highest action-discrepancy step), and the END
as base64 PNGs to feed Qwen3-VL. Needs an EGL/OSMesa GL backend (set on import).
"""
from __future__ import annotations

import base64
import io
import os
from typing import List, Sequence

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen render backend


def make_render_env(env_id: str):
    import gymnasium as gym
    return gym.make(env_id, render_mode="rgb_array")


def encode_png(img) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_episode_frames(render_env, ep_seed: int, actions: np.ndarray,
                          key_steps: Sequence[int]) -> List[str]:
    """Replay ``actions`` from ``reset(seed=ep_seed)``, capturing frames at the
    requested step indices plus a final frame. Returns base64 PNGs in order."""
    render_env.reset(seed=int(ep_seed))
    actions = np.asarray(actions, np.float32)
    n = len(actions)
    ks = sorted({int(k) for k in key_steps if 0 <= int(k) < n})
    frames = {}
    term = trunc = False
    for t in range(n):
        if term or trunc:
            break
        if t in ks:
            try:
                img = render_env.render()
                if img is not None:
                    frames[t] = encode_png(img)
            except Exception:  # noqa: BLE001
                pass
        _o, _r, term, trunc, _i = render_env.step(actions[t])
    try:
        img = render_env.render()
        if img is not None:
            frames[n] = encode_png(img)  # END frame
    except Exception:  # noqa: BLE001
        pass
    return [frames[k] for k in sorted(frames)]
