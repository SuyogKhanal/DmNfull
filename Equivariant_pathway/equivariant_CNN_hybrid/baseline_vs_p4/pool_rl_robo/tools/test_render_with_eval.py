"""Isolate the p4_top3 VLM render hang: does rendering the policy env work when a
(headless) eval env + reposition env also exist? The standalone render test passes
with only policy+reposition render contexts; the live job ALSO builds a render-enabled
10-env eval, and the extra Vulkan render contexts hang env.render(). This reproduces
the full-job env set on ONE GPU (no LLM) to verify the fix: make_eval_env now forces
the eval env HEADLESS (render_mode=None), which should restore rendering.

PASS ⇒ the eval-headless fix works ⇒ safe to run the 3-GPU VLM hybrid on h100.
Run on h100 (rendering is h100-only; h200 device-lost):
  python -m ...pool_rl_robo.tools.test_render_with_eval
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..envs import env_setup as E       # noqa: E402
from ..envs import maniskill_env as MS   # noqa: E402

OUT = E.SUITE_ROOT / "assets" / "render_test"


def _save(frame, name):
    if frame is None:
        print(f"  [render] {name}: FRAME IS None", flush=True)
        return False
    arr = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype("uint8")).save(str(OUT / name))
    print(f"  [render] {name}: shape={arr.shape} min={arr.min()} max={arr.max()}", flush=True)
    return True


def main() -> int:
    cfg = MS.load_cfg("StackCube-v1")
    E.register_envs()
    print(f"[test] render_mode={cfg.render_mode}; building policy + EVAL(10, headless) + reposition",
          flush=True)

    env = MS.make_policy_env(cfg)
    # The contention source: build the eval env exactly as the job does (now headless).
    eval_env = MS.make_eval_env(cfg, num_envs=10)
    reposition_env = MS.make_reposition_env(cfg, "StackCube-v1")
    print("[test] all three envs built", flush=True)

    # Use the policy env a bit (mirror a rollout), then render — this is where the job hung.
    env.reset(seed=0)
    env.set_action_space("joint_pos")
    nj = int(env.num_joints)
    for _ in range(15):
        cur = env.unwrapped.agent.robot.get_qpos()[:, :nj]
        a = cur + (torch.rand_like(cur) - 0.5) * 0.05
        if int(cfg.action_dim) > nj:
            a = torch.cat([a, torch.ones(1, 1, device=a.device)], dim=-1)
        env.step({"action": a})

    # Also exercise the eval env (forward step) to ensure its (headless) context is live.
    eval_env.reset(seed=7777)

    print("[test] rendering policy env WITH eval+reposition present (the hang point)...", flush=True)
    t0 = time.time()
    frame = env.render()
    dt = time.time() - t0
    ok = _save(frame, "render_with_eval.png")
    print(f"[test] render returned in {dt:.2f}s ok={ok}", flush=True)

    # render the reposition env too (BRIDGE-collection env), as the VLM path may use either
    reposition_env.reset(seed=0)
    ok2 = _save(reposition_env.render(), "render_repo_with_eval.png")

    passed = ok and ok2 and dt < 60
    print(f"\n[test] {'PASS' if passed else 'FAIL'} — render-with-headless-eval "
          f"{'works' if passed else 'still BROKEN (eval-headless fix insufficient)'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
