"""De-risk SAPIEN rendering for p4_top3 (the make-or-break dependency).

p4_top3 needs RGB frames of failed episodes for the VLM. select_arm is deliberately
text-only to dodge SAPIEN/Vulkan render fragility, so before building the full arm we
confirm:
  1. the policy env (StackCube-v1, render_mode=rgb_array) returns a real RGB frame,
  2. frames can be saved as PNG (the VLM input format),
  3. StackCube-Start-v0 renders AND honours set_prescription (prescribed cube poses),
  4. a replay of recorded actions renders across steps (for start/t*/end frames).

Run on a GPU node, MUST be h100 (h200 hits Vulkan device-lost):
  MODULE=...tools.test_stackcube_render LOGTAG=stackcube_render sbatch \
    --constraint=gpu-h100 tools/run_gen_normalizers.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from ..envs import env_setup as E       # noqa: E402
from ..envs import maniskill_env as MS   # noqa: E402

OUT = E.SUITE_ROOT / "assets" / "render_test"


def _to_img(frame):
    if frame is None:
        return None
    arr = frame
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def _save(frame, name):
    arr = _to_img(frame)
    if arr is None:
        print(f"  [render] {name}: FRAME IS None")
        return False
    try:
        from PIL import Image
        OUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.astype("uint8")).save(str(OUT / name))
        print(f"  [render] {name}: shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min()} max={arr.max()} -> {OUT/name}")
        return True
    except Exception as exc:
        print(f"  [render] {name}: SAVE FAILED {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    cfg = MS.load_cfg("StackCube-v1")
    print(f"[render] cfg render_mode={cfg.render_mode}")

    # ── 1) policy env (StackCube-v1) render ──────────────────────────────
    env = MS.make_policy_env(cfg)
    obs, info = env.reset(seed=0)
    ok_start = _save(env.render(), "policy_start.png")
    # step a few times, render mid + end
    env.set_action_space("joint_pos")
    nj = int(env.num_joints)
    for t in range(20):
        cur = env.unwrapped.agent.robot.get_qpos()[:, :nj]
        a = cur + (torch.rand_like(cur) - 0.5) * 0.05
        if int(cfg.action_dim) > nj:
            a = torch.cat([a, torch.ones(1, 1, device=a.device)], dim=-1)
        env.step({"action": a})
        if t == 9:
            ok_mid = _save(env.render(), "policy_mid.png")
    ok_end = _save(env.render(), "policy_end.png")
    print(f"[render] policy env: start={ok_start} mid={ok_mid} end={ok_end}")

    # ── 2) reposition env (StackCube-Start-v0) render + prescription ─────
    repo = MS.make_reposition_env(cfg, "StackCube-v1")
    assert repo is not None, "reposition env is None (reposition_env_id not wired?)"
    applied = repo.unwrapped.set_prescription(
        cubeA_xyz=[0.08, 0.12, 0.02], cubeB_xyz=[-0.06, -0.10, 0.02],
        cubeA_zrot=0.3, cubeB_zrot=0.0)
    print(f"[render] set_prescription applied: {applied}")
    obs, info = repo.reset(seed=0)
    cubeA = repo.unwrapped.cubeA.pose.p.reshape(-1)[:3].cpu().numpy().round(3).tolist()
    cubeB = repo.unwrapped.cubeB.pose.p.reshape(-1)[:3].cpu().numpy().round(3).tolist()
    print(f"[render] after prescribed reset: cubeA={cubeA} (want ~[0.08,0.12,0.02]) "
          f"cubeB={cubeB} (want ~[-0.06,-0.10,0.02])")
    ok_repo = _save(repo.render(), "repo_prescribed.png")
    # prescription honoured?
    a_ok = abs(cubeA[0] - 0.08) < 0.02 and abs(cubeA[1] - 0.12) < 0.02
    b_ok = abs(cubeB[0] + 0.06) < 0.02 and abs(cubeB[1] + 0.10) < 0.02
    print(f"[render] prescription honoured: cubeA={a_ok} cubeB={b_ok}")
    # random fallback still works after clear
    repo.unwrapped.clear_prescription()
    repo.reset(seed=1)
    _save(repo.render(), "repo_random.png")

    all_ok = ok_start and ok_mid and ok_end and ok_repo and a_ok and b_ok
    print(f"\n[render] {'PASS' if all_ok else 'FAIL'} — rendering "
          f"{'works' if (ok_start and ok_repo) else 'BROKEN'}; prescription "
          f"{'honoured' if (a_ok and b_ok) else 'NOT honoured'}")
    print("[render] DONE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
