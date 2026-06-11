"""Diagnose WHY the StackCube motion-planner demos fail (grasp/placement).

A/B test to isolate env-config vs my port/recorder:
  A. fork's ORIGINAL solveStackCube() on a RAW env (the canonical usage).
  B. a verbose manual plan on the RAW env that checks is_grasping after each stage
     (reach/grasp/close/lift/align/open) — pinpoints WHERE the cube is lost.
  C. my MotionPlannerExpert.move_to_next_goal on the WRAPPED env (the pipeline path).

Run on a GPU node:
  PYTHONPATH=/weka/s226137394/DmNfull /home/.../diffdagger/bin/python -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.tools.diag_stackcube_planner
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from ..envs import env_setup as E      # noqa: E402
from ..envs import maniskill_env as MS  # noqa: E402
from ..envs import experts as X        # noqa: E402

E.register_envs()


def _b(x):
    try:
        return bool(x.reshape(-1)[0]) if hasattr(x, "reshape") else bool(x)
    except Exception:
        return None


def _grasp(env):
    return _b(env.agent.is_grasping(env.cubeA))


def _make_raw(backend="gpu", noise=0.0):
    import gymnasium as gym
    return gym.make("StackCube-v1", num_envs=1, obs_mode="state_dict",
                    control_mode="pd_joint_pos", render_mode="rgb_array",
                    max_episode_steps=200, sim_backend=backend,
                    robot_init_qpos_noise=noise)


def partA_original_solve():
    """The fork's original solve() has bare .numpy() on poses → crashes on GPU sim
    (it was written for CPU sim). Try CPU sim to confirm the plan logic itself works,
    so we know GPU is the only delta."""
    from mani_skill.examples.motionplanning.panda.solutions import solveStackCube
    print("\n==== PART A: original solveStackCube ====")
    for backend in ("physx_cpu", "gpu"):
        try:
            env = _make_raw(backend=backend)
        except Exception as exc:
            print(f"  backend={backend}: make ERR {type(exc).__name__}: {exc}")
            continue
        ok = 0
        for seed in range(3):
            try:
                res = solveStackCube(env, seed=seed, debug=False, vis=False)
                succ = _b(res[-1]["success"]) if isinstance(res, tuple) else None
            except Exception as exc:
                succ = f"ERR:{type(exc).__name__}:{exc}"
            print(f"  backend={backend} seed={seed}: success={succ}")
            if succ is True:
                ok += 1
        print(f"  backend={backend}: {ok}/3 succeeded")
        env.close()


def partB_verbose_plan():
    import sapien
    from transforms3d.euler import euler2quat
    from transforms3d.quaternions import quat2mat
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )
    print("\n==== PART B: verbose stage-by-stage plan on RAW env (WORLD-frame grasp) ====")
    env = _make_raw()
    for seed in range(2):
        env.reset(seed=seed)
        base = env.unwrapped
        solver = PandaArmMotionPlanningSolver(
            env, debug=False, vis=False, base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False, print_env_info=False)
        approaching = np.array([0.0, 0.0, -1.0])
        cubeA_p = base.cubeA.pose.p.reshape(-1)[:3].cpu().numpy()
        Rc = quat2mat(base.cubeA.pose.q.reshape(-1)[:4].cpu().numpy())
        tcp_y = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

        def _orth(v):
            v = v - approaching * float(v @ approaching)
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-6 else v

        closing = max([_orth(Rc[:, 0]), _orth(Rc[:, 1])],
                      key=lambda c: abs(float(c @ tcp_y)))
        grasp_pose = base.agent.build_grasp_pose(approaching, closing, cubeA_p)
        angles = np.repeat(np.arange(0, np.pi * 2 / 3, np.pi / 2), 2)
        angles[1::2] *= -1
        found = False
        for angle in angles:
            gp2 = grasp_pose * sapien.Pose(q=euler2quat(0, 0, angle))
            if solver.move_to_pose_with_screw(gp2, dry_run=True) != -1:
                grasp_pose = gp2; found = True; break
        def _p(x):
            return [round(float(v), 3) for v in x.reshape(-1)[:3].cpu().tolist()]

        def _fingers():
            return [round(float(v), 4) for v in base.agent.robot.get_qpos()[0, -2:].cpu().tolist()]

        cubeA0 = _p(base.cubeA.pose.p)
        reach = grasp_pose * sapien.Pose([0, 0, -0.05])
        r1 = solver.move_to_pose_with_screw(reach)
        print(f"  seed={seed} found_grasp={found} reach={'ok' if r1 != -1 else 'FAIL'} "
              f"cubeA0={cubeA0} tcp_after_reach={_p(base.agent.tcp.pose.p)} fingers={_fingers()}")
        r2 = solver.move_to_pose_with_screw(grasp_pose)
        print(f"    at grasp pose: move={'ok' if r2 != -1 else 'FAIL'} "
              f"tcp={_p(base.agent.tcp.pose.p)} cubeA={_p(base.cubeA.pose.p)} "
              f"tcp_to_cubeA={round(float((base.agent.tcp.pose.p - base.cubeA.pose.p).reshape(-1)[:3].norm().cpu()),4)} "
              f"fingers={_fingers()}")
        solver.close_gripper()
        print(f"    after close(6): grasping={_grasp(base)} fingers={_fingers()}")
        solver.close_gripper(t=20)
        print(f"    after close(+20): grasping={_grasp(base)} fingers={_fingers()} "
              f"cubeA={_p(base.cubeA.pose.p)}")
        lift = sapien.Pose([0, 0, 0.1]) * grasp_pose
        r3 = solver.move_to_pose_with_screw(lift)
        print(f"    after lift: move={'ok' if r3 != -1 else 'FAIL'} grasping={_grasp(base)} "
              f"cubeA_z={round(float(base.cubeA.pose.p.reshape(-1)[2].cpu()),3)} fingers={_fingers()}")
        goal = base.cubeB.pose * sapien.Pose([0, 0, float(base.cube_half_size[2]) * 2])
        offset = (goal.p - base.cubeA.pose.p).cpu().numpy()[0]
        align = sapien.Pose(lift.p + offset, lift.q)
        r4 = solver.move_to_pose_with_screw(align)
        print(f"    after align: move={'ok' if r4 != -1 else 'FAIL'} grasping={_grasp(base)} "
              f"on_cubeB={_b(base.evaluate()['is_cubeA_on_cubeB'])}")
        solver.open_gripper()
        ev = base.evaluate()
        print(f"    after open: on_cubeB={_b(ev['is_cubeA_on_cubeB'])} "
              f"grasped={_b(ev['is_cubeA_grasped'])} static={_b(ev['is_cubeA_static'])} "
              f"success={_b(ev['success'])}")
    env.close()


def partC_my_expert():
    print("\n==== PART C: MotionPlannerExpert.move_to_next_goal on WRAPPED env ====")
    cfg = MS.load_cfg("StackCube-v1")
    env = MS.make_policy_env(cfg)
    expert = X.MotionPlannerExpert("StackCube-v1")
    for seed in range(3):
        env.reset(seed=seed)
        env.set_action_space("joint_pos")
        expert.reset(env)
        td = expert.move_to_next_goal(dict(seed=seed))
        ev = env.unwrapped.evaluate()
        done = bool(td["done"][-1].item())
        print(f"  seed={seed}: steps={len(td['episode'])} done={done} "
              f"on_cubeB={_b(ev['is_cubeA_on_cubeB'])} grasped={_b(ev['is_cubeA_grasped'])} "
              f"static={_b(ev['is_cubeA_static'])} success={_b(ev['success'])}")
    env.close()


def main():
    partA_original_solve()
    partB_verbose_plan()
    partC_my_expert()
    print("\n[diag] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
