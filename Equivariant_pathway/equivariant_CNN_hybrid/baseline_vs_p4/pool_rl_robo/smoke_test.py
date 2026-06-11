"""STEP-3 environment smoke: confirm the 4 ManiSkill tasks build + reset on a GPU.

Run on a GPU node (the SAPIEN GPU backend needs a GPU):
    /home/s226137394/.conda/envs/diffdagger/bin/python smoke_test.py

All 4 must print ✓ before the experiment smoke (submit_smoke.sh). This only
checks env construction/reset; the full 1-round-per-method + live-LLM smoke is
the sbatch (submit_smoke.sh).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # DmNfull on path

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.envs \
    import env_setup as E  # noqa: E402

TASKS = ["StackCube-v1", "PickCube-v1", "PushT-v1", "PlugCharger-v1"]


def main() -> int:
    import gymnasium as gym
    E.register_envs()
    ok = 0
    for suite_id in TASKS:
        spec = E.task_spec(suite_id)
        try:
            env = gym.make(spec.fork_env_id, obs_mode="state_dict",
                           control_mode="pd_joint_pos", sim_backend="gpu",
                           num_envs=1)
            obs, info = env.reset(seed=0)
            env.close()
            print(f"{suite_id:16s} [{spec.expert_kind}] → {spec.fork_env_id}  reset ✓")
            ok += 1
        except Exception as exc:
            print(f"{suite_id:16s} ✗  {type(exc).__name__}: {exc}")
    print(f"\n{ok}/{len(TASKS)} tasks built + reset.")
    return 0 if ok == len(TASKS) else 1


if __name__ == "__main__":
    sys.exit(main())
