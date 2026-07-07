"""Strengthen the PlugCharger insert with progressive re-targeting + a hole search.

Diagnosis (from the SR gate + over-insert sweep): the grasp holds and the gross
align is OK, but the peg jams ~1.7 cm short with a ~0.17 rad (~9.6 deg) residual
angle → the tip (~2.7 mm off-axis) catches the 0.5 mm-clearance socket rim. A
single open-loop shove can't fix it, and over-insertion tears the charger out of
the gripper. The contact-rich recipe:
  * PROGRESSIVE insert: advance in small steps, RE-TARGETING from live poses each
    step (goal · charger.pose⁻¹ · tcp.pose) → closed-loop alignment that self-
    corrects the residual as it advances, instead of one shove.
  * SPIRAL search: small lateral (dy,dz) offsets at the socket mouth to find the
    hole (radius ~ a few mm), early-exit once the charger seats (dist < thresh).
  * TILT wiggle: small orientation perturbations to align the peg axis.

Sweeps a handful of named strategies on N seeds; reports SR / grasp-kept / median
dist. Bake the winner into experts._plan_plugcharger.

Run:  MODULE=...tools.diag_plugcharger_insert_search LOGTAG=plugcharger_insert_search \
        sbatch --partition=gpu-large --constraint=gpu-h100 --qos=batch-short --time=01:00:00 \
        tools/run_gen_normalizers.sh
"""
from __future__ import annotations

import math
import sys

import numpy as np

from ..envs import env_setup as E      # noqa: E402

E.register_envs()

ARM_NJ = 7
SEAT_THRESH = 0.006   # early-exit: charger within 6 mm of goal ⇒ effectively seated


def _f(x):
    return float(x.reshape(-1)[0]) if hasattr(x, "reshape") else float(x)


def _dist_angle(base):
    d, a = base._compute_distance()
    return _f(d), _f(a)


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
                    max_episode_steps=800, sim_backend="gpu",
                    robot_init_qpos_noise=0.02)


def _grasp_and_align(env, base, solver):
    """Grasp the charger + bring it to the socket mouth (goal - 5 cm)."""
    import sapien
    import trimesh
    from transforms3d.euler import euler2quat

    from mani_skill.examples.motionplanning.panda.utils import (
        compute_grasp_info_by_obb,
    )
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
    pre = (base.goal_pose.sp * sapien.Pose([-0.05, 0, 0])
           * base.charger.pose.sp.inv() * base.agent.tcp.pose.sp)
    solver.move_to_pose_with_screw(pre, refine_steps=0)
    solver.move_to_pose_with_screw(pre, refine_steps=5)


def _tcp_target(base, fwd, dy=0.0, dz=0.0, tilt=0.0):
    """tcp pose that brings the LIVE charger to goal·Pose([fwd,dy,dz], tilt)."""
    import sapien
    from transforms3d.euler import euler2quat
    if tilt:
        off = sapien.Pose([fwd, dy, dz], q=euler2quat(0.0, float(tilt), 0.0))
    else:
        off = sapien.Pose([fwd, dy, dz])
    charger_target = base.goal_pose.sp * off
    return charger_target * base.charger.pose.sp.inv() * base.agent.tcp.pose.sp


def _insert(env, base, solver, *, n_steps, fwd_target, spiral_r, n_spiral, tilt_amp):
    """Progressive insert with live re-targeting + optional spiral/tilt search."""
    start_fwd = -0.03
    for i in range(n_steps):
        fwd = start_fwd + (fwd_target - start_fwd) * (i + 1) / n_steps
        if spiral_r > 0 and n_spiral > 1:
            seated = False
            for j in range(n_spiral):
                ang = 2 * math.pi * j / n_spiral
                dy = spiral_r * math.cos(ang)
                dz = spiral_r * math.sin(ang)
                tilt = tilt_amp * math.sin(ang) if tilt_amp > 0 else 0.0
                solver.move_to_pose_with_screw(
                    _tcp_target(base, fwd, dy, dz, tilt), refine_steps=1)
                d, _ = _dist_angle(base)
                if d < SEAT_THRESH:
                    seated = True
                    break
            if seated:
                break
        else:
            tilt = tilt_amp if (tilt_amp > 0 and i % 2 == 0) else (-tilt_amp if tilt_amp > 0 else 0.0)
            solver.move_to_pose_with_screw(
                _tcp_target(base, fwd, 0.0, 0.0, tilt), refine_steps=2)
            d, _ = _dist_angle(base)
            if d < SEAT_THRESH:
                break
    # final seat at the goal, held
    solver.move_to_pose_with_screw(_tcp_target(base, fwd_target), refine_steps=15)


def _attempt(env, seed, cfg):
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )
    env.reset(seed=int(seed))
    base = env.unwrapped
    solver = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False, base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False,
        joint_vel_limits=0.5, joint_acc_limits=0.5)
    try:
        _grasp_and_align(env, base, solver)
        _insert(env, base, solver, **cfg)
    finally:
        try:
            solver.close()
        except Exception:
            pass
    # closed-gripper settle
    for _ in range(25):
        qpos = base.agent.robot.get_qpos()[0, :ARM_NJ].cpu().numpy()
        env.step(np.hstack([qpos, -1.0]))
    d, a, s = _ev(base)
    return s, d, a, _grasp(base)


STRATEGIES = {
    "prog8":            dict(n_steps=8,  fwd_target=0.000, spiral_r=0.000, n_spiral=0, tilt_amp=0.00),
    "prog8_over3":      dict(n_steps=8,  fwd_target=0.003, spiral_r=0.000, n_spiral=0, tilt_amp=0.00),
    "prog_tilt":        dict(n_steps=8,  fwd_target=0.000, spiral_r=0.000, n_spiral=0, tilt_amp=0.04),
    "spiral3_r2":       dict(n_steps=6,  fwd_target=0.002, spiral_r=0.002, n_spiral=6, tilt_amp=0.00),
    "spiral3_r3_tilt":  dict(n_steps=6,  fwd_target=0.002, spiral_r=0.003, n_spiral=6, tilt_amp=0.03),
}


def main():
    env = _make_raw()
    N = 15
    print(f"\n==== insert-search strategy sweep, N={N} seeds ====", flush=True)
    for name, cfg in STRATEGIES.items():
        ok = 0
        dists, angles, grasps = [], [], []
        for seed in range(N):
            try:
                s, d, a, g = _attempt(env, seed, cfg)
            except Exception as exc:
                print(f"    [{name}] seed={seed} ERR {type(exc).__name__}: {exc}", flush=True)
                continue
            ok += int(bool(s))
            dists.append(d)
            angles.append(a)
            grasps.append(bool(g))
        if dists:
            print(f"  {name:>16s}: SR={ok}/{len(dists)}  "
                  f"median dist={np.median(dists):.4f} angle={np.median(angles):.3f}  "
                  f"grasp_kept={sum(grasps)}/{len(grasps)}  "
                  f"<=5mm:{sum(x <= 5e-3 for x in dists)} <=1cm:{sum(x <= 1e-2 for x in dists)}",
                  flush=True)
    env.close()
    print("\n[search] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
