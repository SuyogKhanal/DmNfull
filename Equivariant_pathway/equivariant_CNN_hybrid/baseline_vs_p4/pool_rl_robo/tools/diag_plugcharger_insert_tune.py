"""Tune the PlugCharger insert to beat the contact-rich short-fall.

The planner-SR gate showed: grasp works, angle is fine, but the open-loop insert
lands ~1.7-1.9 cm SHORT of the socket (median dist 0.019 m vs the 5 mm threshold)
→ only ~10% SR. The standard fix for a peg that jams at the socket mouth is
OVER-INSERTION: command the charger a few mm PAST the goal so the position
controller maintains forward force and seats the peg, plus a longer refine/settle
hold at the final insert.

This sweeps (over_insert_delta, final_refine_steps) on N seeds and reports the SR +
median residual distance per setting, with a closed-gripper settle (matching the
pipeline). Pick the best (delta, refine) and bake it into experts._plan_plugcharger.

Run on a GPU node (1 GPU, no LLM):
  MODULE=...tools.diag_plugcharger_insert_tune LOGTAG=plugcharger_insert_tune \
    sbatch --partition=gpu-large --constraint=gpu-h100 --qos=batch-short --time=00:40:00 \
    tools/run_gen_normalizers.sh
"""
from __future__ import annotations

import sys

import numpy as np

from ..envs import env_setup as E      # noqa: E402

E.register_envs()

ARM_NJ = 7   # panda arm joints (qpos[:7]); qpos[7:9] are the fingers


def _f(x):
    try:
        return float(x.reshape(-1)[0]) if hasattr(x, "reshape") else float(x)
    except Exception:
        return float("nan")


def _ev(base):
    ev = base.evaluate()
    return _f(ev["obj_to_goal_dist"]), _f(ev["obj_to_goal_angle"]), bool(ev["success"].reshape(-1)[0])


def _grasp(base):
    try:
        return bool(base.agent.is_grasping(base.charger).reshape(-1)[0])
    except Exception:
        return None


def _make_raw():
    import gymnasium as gym
    return gym.make("PlugCharger-v1", num_envs=1, obs_mode="state_dict",
                    control_mode="pd_joint_pos", render_mode=None,
                    max_episode_steps=500, sim_backend="gpu",
                    robot_init_qpos_noise=0.02)


def _solve_with(env, seed, over_delta, refine_final, settle_n=25):
    """Grasp + align + insert(over_delta, refine_final) + closed settle. Returns
    (success, dist, angle, grasping)."""
    import sapien
    import trimesh
    from transforms3d.euler import euler2quat

    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )
    from mani_skill.examples.motionplanning.panda.utils import (
        compute_grasp_info_by_obb,
    )

    env.reset(seed=int(seed))
    base = env.unwrapped
    solver = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False, base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False,
        joint_vel_limits=0.5, joint_acc_limits=0.5)
    try:
        obb = trimesh.primitives.Box(
            extents=np.array(base._base_size) * 2,
            transform=base.charger_base_pose.sp.to_transformation_matrix())
        approaching = np.array([0.0, 0.0, -1.0])
        target_closing = base.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
        gi = compute_grasp_info_by_obb(obb, approaching=approaching,
                                       target_closing=target_closing, depth=0.025)
        gp = base.agent.build_grasp_pose(approaching, gi["closing"], gi["center"])
        gp = gp * sapien.Pose(q=euler2quat(0, np.deg2rad(15), 0))
        solver.move_to_pose_with_screw(gp * sapien.Pose([0, 0, -0.05]))
        solver.move_to_pose_with_screw(gp)
        solver.close_gripper()
        # align (pre-insert 5 cm back)
        pre = (base.goal_pose.sp * sapien.Pose([-0.05, 0, 0])
               * base.charger.pose.sp.inv() * base.agent.tcp.pose.sp)
        solver.move_to_pose_with_screw(pre, refine_steps=0)
        solver.move_to_pose_with_screw(pre, refine_steps=5)
        # insert with over-insertion (+x = deeper into the socket) + final refine
        ins = (base.goal_pose.sp * sapien.Pose([over_delta, 0, 0])
               * base.charger.pose.sp.inv() * base.agent.tcp.pose.sp)
        solver.move_to_pose_with_screw(ins, refine_steps=refine_final)
    finally:
        try:
            solver.close()
        except Exception:
            pass
    # closed-gripper settle (hold arm qpos, gripper closed)
    for _ in range(settle_n):
        qpos = base.agent.robot.get_qpos()[0, :ARM_NJ].cpu().numpy()
        env.step(np.hstack([qpos, -1.0]))
    d, a, s = _ev(base)
    return s, d, a, _grasp(base)


def main():
    env = _make_raw()
    N = 20
    DELTAS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
    REFINE = 15
    print(f"\n==== over-insertion sweep: refine_final={REFINE}, settle=25, N={N} seeds ====",
          flush=True)
    for delta in DELTAS:
        ok = 0
        dists, angles, grasps = [], [], []
        for seed in range(N):
            try:
                s, d, a, g = _solve_with(env, seed, delta, REFINE)
            except Exception as exc:
                print(f"    delta={delta} seed={seed} ERR {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            ok += int(bool(s))
            dists.append(d)
            angles.append(a)
            grasps.append(bool(g))
        if dists:
            print(f"  delta={delta:+.3f}m  SR={ok}/{len(dists)}  "
                  f"median dist={np.median(dists):.4f} angle={np.median(angles):.3f}  "
                  f"grasp_kept={sum(grasps)}/{len(grasps)}  "
                  f"<=5mm:{sum(x <= 5e-3 for x in dists)} <=1cm:{sum(x <= 1e-2 for x in dists)}",
                  flush=True)
    # probe refine sensitivity at the most promising delta band
    print("\n==== refine sweep at delta=+0.02 ====", flush=True)
    for refine in [5, 25, 40]:
        ok = 0
        dists = []
        for seed in range(N):
            try:
                s, d, a, g = _solve_with(env, seed, 0.02, refine)
            except Exception:
                continue
            ok += int(bool(s))
            dists.append(d)
        if dists:
            print(f"  refine={refine:>3d}  SR={ok}/{len(dists)}  "
                  f"median dist={np.median(dists):.4f}  "
                  f"<=5mm:{sum(x <= 5e-3 for x in dists)}", flush=True)
    env.close()
    print("\n[tune] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
