"""Diagnose the PlugCharger motion-planner solve rate — THE bring-up gate.

Everything downstream (normalizers, bootstrap sweep, Diff-DAgger baseline, the
hybrid) is moot if the expert can't insert the charger. PlugCharger has the
tightest success tolerance in the suite (dist <= 5 mm AND angle <= 0.2 rad), so
we measure the planner SR FIRST and report it honestly.

Three views (mirrors tools/diag_stackcube_planner.py):
  A. fork's ``solvePlugCharger`` on a RAW GPU env — the canonical algorithm SR.
  B. verbose stage-by-stage plan (grasp / close / pre-insert / insert) with
     is_grasping + dist/angle after each phase — pinpoints WHERE it fails.
  C. ``MotionPlannerExpert.move_to_next_goal`` on the WRAPPED env (the actual
     pipeline path, incl. the closed-gripper settle) — the demo-collection SR.

No Hydra cfg needed: the wrapped env is built directly via the fork's
``wrap_env`` with action_space="joint_pos" (the planner steps absolute joint
targets), so this runs BEFORE plugcharger_state.yaml exists. render_mode=None
(no rendering is needed for planning/physics/success → no Vulkan device-lost →
runs on the idle a100 ``gpu`` partition).

Run on a GPU node:
  MODULE=Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.tools.diag_plugcharger_planner \
  LOGTAG=plugcharger_diag \
    sbatch --partition=gpu --constraint=gpu-a100 tools/run_gen_normalizers.sh
"""
from __future__ import annotations

import sys

import numpy as np

from ..envs import env_setup as E      # noqa: E402
from ..envs import experts as X        # noqa: E402

E.register_envs()


def _b(x):
    try:
        return bool(x.reshape(-1)[0]) if hasattr(x, "reshape") else bool(x)
    except Exception:
        return None


def _f(x):
    try:
        return float(x.reshape(-1)[0]) if hasattr(x, "reshape") else float(x)
    except Exception:
        return float("nan")


def _ev(base):
    ev = base.evaluate()
    return _f(ev["obj_to_goal_dist"]), _f(ev["obj_to_goal_angle"]), _b(ev["success"])


def _grasp(base):
    return _b(base.agent.is_grasping(base.charger))


def _make_raw(noise: float = 0.02):
    import gymnasium as gym
    return gym.make(
        "PlugCharger-v1", num_envs=1, obs_mode="state_dict",
        control_mode="pd_joint_pos", render_mode=None,
        max_episode_steps=400, sim_backend="gpu",
        robot_init_qpos_noise=noise)


def partA_original_solve(n: int = 10):
    """Fork solvePlugCharger on a raw GPU env (it reset()s itself each call)."""
    from mani_skill.examples.motionplanning.panda.solutions import solvePlugCharger
    print("\n==== PART A: fork solvePlugCharger on RAW GPU env ====", flush=True)
    env = _make_raw()
    ok = 0
    dists, angles = [], []
    for seed in range(n):
        try:
            solvePlugCharger(env, seed=seed, debug=False, vis=False)
            d, a, s = _ev(env.unwrapped)
        except Exception as exc:
            print(f"  seed={seed}: ERR {type(exc).__name__}: {exc}", flush=True)
            continue
        ok += int(bool(s))
        dists.append(d)
        angles.append(a)
        print(f"  seed={seed}: success={s} dist={d:.4f} angle={a:.3f} "
              f"grasp={_grasp(env.unwrapped)}", flush=True)
    if dists:
        print(f"  PART A SR={ok}/{len(dists)}  median dist={np.median(dists):.4f} "
              f"angle={np.median(angles):.3f}  (NB: no settle after insert)", flush=True)
    env.close()


def partB_verbose(n: int = 3):
    import sapien
    import trimesh
    from transforms3d.euler import euler2quat

    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )
    from mani_skill.examples.motionplanning.panda.utils import (
        compute_grasp_info_by_obb,
    )
    print("\n==== PART B: verbose stage-by-stage (.sp) on RAW env ====", flush=True)
    env = _make_raw()
    for seed in range(n):
        env.reset(seed=seed)
        base = env.unwrapped
        solver = PandaArmMotionPlanningSolver(
            env, debug=False, vis=False, base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False, print_env_info=False,
            joint_vel_limits=0.5, joint_acc_limits=0.5)
        FINGER_LENGTH = 0.025
        obb = trimesh.primitives.Box(
            extents=np.array(base._base_size) * 2,
            transform=base.charger_base_pose.sp.to_transformation_matrix())
        approaching = np.array([0.0, 0.0, -1.0])
        target_closing = base.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
        gi = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=target_closing,
            depth=FINGER_LENGTH)
        gp = base.agent.build_grasp_pose(approaching, gi["closing"], gi["center"])
        gp = gp * sapien.Pose(q=euler2quat(0, np.deg2rad(15), 0))
        reach = gp * sapien.Pose([0, 0, -0.05])
        r1 = solver.move_to_pose_with_screw(reach)
        r2 = solver.move_to_pose_with_screw(gp)
        solver.close_gripper()
        print(f"  seed={seed} reach={'ok' if r1 != -1 else 'FAIL'} "
              f"grasp={'ok' if r2 != -1 else 'FAIL'} grasping={_grasp(base)}", flush=True)
        pre = (base.goal_pose.sp * sapien.Pose([-0.05, 0, 0])
               * base.charger.pose.sp.inv() * base.agent.tcp.pose.sp)
        ins = base.goal_pose.sp * base.charger.pose.sp.inv() * base.agent.tcp.pose.sp
        solver.move_to_pose_with_screw(pre, refine_steps=0)
        solver.move_to_pose_with_screw(pre, refine_steps=5)
        r5 = solver.move_to_pose_with_screw(ins)
        d, a, s = _ev(base)
        print(f"    after insert: move={'ok' if r5 != -1 else 'FAIL'} "
              f"grasping={_grasp(base)} dist={d:.4f} angle={a:.3f} success={s}", flush=True)
        solver.close()
    env.close()


def partC_my_expert(n: int = 30):
    import gymnasium as gym

    from diffdagger.util.maniskill_env import wrap_env
    print("\n==== PART C: MotionPlannerExpert on WRAPPED env (pipeline path) ====",
          flush=True)
    base = gym.make(
        "PlugCharger-v1", num_envs=1, obs_mode="state_dict",
        control_mode="pd_joint_pos", render_mode=None,
        max_episode_steps=400, sim_backend="gpu", robot_init_qpos_noise=0.02)
    obs_keys = ["agent_qpos", "extra_tcp_pose", "extra_charger_pose",
                "extra_receptacle_pose", "extra_goal_pose"]
    env = wrap_env(base, obs_keys, "joint_pos")
    expert = X.MotionPlannerExpert("PlugCharger-v1")
    ok = done_ok = n_eff = 0
    dists, angles = [], []
    for seed in range(n):
        env.reset(seed=seed)
        env.set_action_space("joint_pos")
        expert.reset(env)
        try:
            td = expert.move_to_next_goal(dict(seed=seed))
        except Exception as exc:
            print(f"  seed={seed}: ERR {type(exc).__name__}: {exc}", flush=True)
            continue
        d, a, s = _ev(env.unwrapped)
        done = bool(td["done"][-1].item()) if td is not None else False
        steps = len(td["episode"]) if td is not None else 0
        n_eff += 1
        ok += int(bool(s))
        done_ok += int(done)
        dists.append(d)
        angles.append(a)
        print(f"  seed={seed}: steps={steps} done={done} success={s} "
              f"dist={d:.4f} angle={a:.3f} grasp={_grasp(env.unwrapped)}", flush=True)
    if dists:
        print(f"\n  PART C SR(evaluate)={ok}/{n_eff}  done[-1]={done_ok}/{n_eff}  "
              f"median dist={np.median(dists):.4f} angle={np.median(angles):.3f}  "
              f"<=5mm:{sum(d <= 5e-3 for d in dists)} <=1cm:{sum(d <= 1e-2 for d in dists)} "
              f"<=2cm:{sum(d <= 2e-2 for d in dists)}", flush=True)
    env.close()


def main():
    partA_original_solve(n=10)
    partB_verbose(n=3)
    partC_my_expert(n=30)
    print("\n[diag] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
