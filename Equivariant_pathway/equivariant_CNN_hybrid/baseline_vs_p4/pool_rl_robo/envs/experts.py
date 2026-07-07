"""Automated expert loader (no human in the loop).

Tier 1 (Stack/Pick/Plug): ManiSkill panda motion-planner. The motion planner
is RE-PLANNED from the LIVE mid-episode robot/scene state (no env.reset) and its
per-step ``[qpos, gripper]`` actions are captured into a demo TensorDict in the
same schema the PushT PPO expert produces, so the fork's dataset / training /
DAgger loop consume both experts identically.
Tier 2 (PushT): the fork's PPO ``MultipleExperts``, loaded from the Hydra cfg.

Interface the engine relies on:
  expert.reset(env)                 — bind the expert to a (wrapped) env instance
  expert.setup_task()               — per-episode setup hook (no-op for planners)
  expert.move_to_next_goal(ep_info) — run the expert to completion FROM THE
                                      CURRENT env state, returning the demo
                                      TensorDict trajectory (used by BOTH arms to
                                      collect the corrective demonstration and the
                                      shared initial demos)
  expert.generate_stationary_action() — hold the current pose (joint_pos[+gripper])

``get_action(obs)`` (per-state π_exp, for the action-discrepancy IIL rules) is
implemented ONLY by the PushT PPO expert. Motion-planner tasks do NOT implement
it: p4_select on those tasks detects via the policy's OWN diffusion loss
(decision 2026-06-08, see claude_context.md), so the per-state expert action is
never needed — only the episode-completing corrective demo (move_to_next_goal).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict

from . import env_setup as E

E.bootstrap_fork_path()


def load_expert(cfg, suite_env_id: str) -> Any:
    """Instantiate the task's expert from the fork Hydra cfg.

    PushT → `agents.RL_agent.MultipleExperts` (PPO), exactly as
    run_comparison.py does (`instantiate(cfg.expert, _recursive_=False)`).
    Stack/Pick/Plug → `MotionPlannerExpert` wrapping the panda solver.
    """
    spec = E.task_spec(suite_env_id)
    if spec.expert_kind == "ppo":
        from hydra.utils import instantiate
        return instantiate(cfg.expert, _recursive_=False)
    if spec.expert_kind == "motionplanner":
        return MotionPlannerExpert(suite_env_id)
    raise ValueError(f"unknown expert_kind {spec.expert_kind!r} for {suite_env_id}")


def expert_action(expert, obs_dict, *, expert_idx: int = 0):
    """π_exp(s): the expert's per-state action for the current observation,
    used by the action-discrepancy IIL rules (SafeDAgger / Ensemble / Dropout).
    PPO experts only. Deterministic. Returns the raw expert action tensor."""
    return expert.get_action(obs_dict, expert_idx=expert_idx, deterministic=True)


# ---- recording proxy ----------------------------------------------------
class _RecordingEnv:
    """Transparent proxy around the wrapped policy env that intercepts every
    ``step`` the motion planner makes and packs each transition into the PushT
    demo-TensorDict schema (obs-before + action_{joint_delta_pos,joint_pos,
    ee_delta_pose,ee_pose} + done + episode).

    The planner steps with absolute arm ``qpos`` (+ gripper command) while the
    env's effective action space is ``joint_pos`` — so the recorded
    ``action_joint_pos`` IS the executed action; the dataset later derives the
    rel_joint_pos training target as ``action_joint_pos[:7] - agent_qpos[:7]``
    (gripper command kept absolute), matching the PPO expert exactly.
    """

    def __init__(self, env, num_joints: int, seed: int):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_nj", int(num_joints))
        object.__setattr__(self, "_seed", int(seed))
        object.__setattr__(self, "_data", defaultdict(list))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_env"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_env"), name, value)

    def step(self, action):
        import copy

        import numpy as np
        import torch

        from util.maniskill_env import flatten_dict   # fork's flattener

        env = object.__getattribute__(self, "_env")
        nj = object.__getattribute__(self, "_nj")
        seed = object.__getattribute__(self, "_seed")
        data = object.__getattribute__(self, "_data")

        # obs at the state where this action is decided. base get_obs is NESTED for
        # obs_mode=state_dict ({agent:{qpos,..}, extra:{cubeA_pose,..}}); flatten it
        # to the SAME flat keys (agent_qpos, extra_cubeA_pose, …) the policy env's
        # FlattenObservationKeyWrapper produces, so the demo's obs keys match the
        # dataset's obs_keys / training tensors exactly. (StackCube-v1 extends
        # BaseEnv; only TableEnv-derived envs like PushT-v2 flatten get_obs.)
        obs = flatten_dict(env.get_obs(env.get_info()))
        td = dict(obs)
        prev_pose = copy.deepcopy(env.agent.tcp.pose)
        prev_qpos = env.agent.robot.get_qpos()[:, :nj].clone()

        result = env.step(action)
        _new_obs, _reward, terminated, _trunc, info = result

        a = action
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        a = a.to(prev_qpos.device).float()
        if a.ndim == 1:
            a = a[None]
        arm_target = a[:, :nj]
        has_grip = a.shape[1] > nj
        action_joint_pos = a[:, : nj + (1 if has_grip else 0)].clone()
        joint_delta = arm_target - prev_qpos
        if has_grip:
            action_joint_delta_pos = torch.cat([joint_delta, a[:, nj:nj + 1]], dim=-1)
        else:
            action_joint_delta_pos = joint_delta
        action_ee_delta_pose = env.compute_ee_delta_pose_from_qpos(prev_pose, arm_target)
        action_ee_pose = env.compute_ee_pose_from_qpos(arm_target)
        # The fork's compute_ee_pose_from_qpos returns a 7-wide (pos+quat) ARM pose
        # with NO gripper, so the absolute ee_pose_6d dataset path
        # (datasets.generate_action_sequence) drops the gripper → action_dim 9 and
        # the policy could never close the gripper. For a grasping task we append
        # the gripper COMMAND so these ee keys are 8-wide → the dataset's 8-wide
        # branch yields action_dim 10 (3 pos + 6 rot6d + 1 gripper) and
        # post_process_action / transform_action carry it through (use_gripper).
        # Gripperless tasks (PushT, panda_stick) keep the 7-wide path unchanged;
        # joint-space tasks (StackCube rel_joint_pos) never read these ee keys, so
        # this is backward-compatible.
        if has_grip:
            grip = a[:, nj:nj + 1]
            action_ee_delta_pose = torch.cat([action_ee_delta_pose, grip], dim=-1)
            action_ee_pose = torch.cat([action_ee_pose, grip], dim=-1)

        succ = info.get("success", terminated)
        if isinstance(succ, torch.Tensor):
            done_t = succ.reshape(-1)[:1].float().cpu()
        else:
            done_t = torch.tensor([float(bool(succ))])

        td.update({
            "action_joint_delta_pos": action_joint_delta_pos,
            "action_joint_pos": action_joint_pos,
            "action_ee_delta_pose": action_ee_delta_pose,
            "action_ee_pose": action_ee_pose,
            "done": done_t,
            "episode": torch.ones(1) * seed,
        })
        for k, v in td.items():
            data[k].append(v.detach().cpu() if hasattr(v, "detach") else v)
        return result

    def collected(self):
        import torch

        data = object.__getattribute__(self, "_data")
        if not data or len(data.get("episode", [])) == 0:
            return None
        return {k: torch.cat(v) for k, v in data.items()}


class MotionPlannerExpert:
    """Wrapper around the fork's panda motion-planner solutions
    (mani_skill.examples.motionplanning.panda.solutions). Used for Stack/Pick/
    Plug.

    The fork's ``solve(env, seed=...)`` is OPEN-LOOP and ``env.reset()``s first —
    wrong for a DAgger intervention, which must continue from where the policy
    left off. We instead build the ``PandaArmMotionPlanningSolver`` against the
    LIVE (already-reset / already-rolled-out) env and re-plan the whole task from
    the current robot qpos + object poses, recording every step into a demo
    TensorDict (``_RecordingEnv``). The planner has no per-state π_exp(s); the
    action-discrepancy IIL rules are not used on these tasks (diffusion-loss
    detection instead — see module docstring).
    """

    def __init__(self, suite_env_id: str):
        self.suite_env_id = suite_env_id
        if suite_env_id not in _PLAN_REGISTRY:
            raise NotImplementedError(
                f"no motion-planner plan registered for {suite_env_id!r}; "
                f"known: {sorted(_PLAN_REGISTRY)}")
        self._plan_fn = _PLAN_REGISTRY[suite_env_id]
        self.env = None
        # Post-plan settle gripper command. StackCube ends with the gripper OPEN
        # (cube released on cubeB) → settle open (+1). PlugCharger ends still
        # GRASPING the charger held in the socket → settle CLOSED (-1) so the
        # 25-step hold doesn't release the plug before the success flag
        # (charger.pose ≈ goal_pose within 5 mm / 0.2 rad) registers.
        self._settle_gripper = -1.0 if suite_env_id == "PlugCharger-v1" else 1.0
        # Used by the fork's collect_initial_demos_shared length-bookkeeping.
        self.max_episode_steps = 10 ** 9

    def reset(self, env):
        self.env = env

    def setup_task(self):
        pass

    def get_action(self, *a, **kw):
        raise NotImplementedError(
            "MotionPlannerExpert has no per-state get_action; motion-planner "
            "tasks use diffusion-loss detection, not action-discrepancy.")

    def generate_stationary_action(self):
        """Hold the current pose: absolute arm qpos (+ gripper command that
        holds the current finger width, if the robot has a gripper). Shape
        matches cfg.expert.action_space='joint_pos' (num_joints[+1])."""
        import torch

        env = self.env
        nj = int(env.num_joints)
        robot = env.unwrapped.agent.robot
        arm = robot.get_qpos()[:, :nj]
        if getattr(env, "use_gripper", False):
            qlim = robot.get_qlimits()           # (1, nq, 2)
            gmax = float(qlim[0, -1, 1])
            width = robot.get_qpos()[:, -2:].sum(dim=1, keepdim=True)
            grip = (width / (2.0 * max(gmax, 1e-6))) * 2.0 - 1.0   # → ~[-1, 1]
            return torch.cat([arm, grip.to(arm.device)], dim=-1)
        return arm

    def move_to_next_goal(self, ep_info: Dict[str, Any]):
        """Re-plan the whole task from the LIVE env state (NO reset) and return
        the recorded demo TensorDict. ``done[-1]`` reflects task success."""
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver,
        )

        env = self.env
        nj = int(env.num_joints)
        seed = int(ep_info.get("seed", 0))
        base = env.unwrapped

        rec = _RecordingEnv(env, nj, seed)
        solver = PandaArmMotionPlanningSolver(
            rec, debug=False, vis=False, base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False, print_env_info=False)
        try:
            self._plan_fn(solver, base, rec)
            # Settle: hold the final pose so the success condition registers. GPU
            # sim has noticeable residual angular velocity, so the static / pose
            # checks need a generous hold for the last recorded `done` to reflect
            # a stable end-state. The gripper command is per-task (open for a
            # released cube, closed for a still-grasped charger — see __init__).
            self._settle(rec, nj, n=25, gripper=self._settle_gripper)
        finally:
            try:
                solver.close()
            except Exception:
                pass

        data = rec.collected()
        if data is None:
            # Degenerate (planner produced no steps): emit a 1-step stationary
            # demo so the caller's bookkeeping (len(td["episode"])) never crashes;
            # done=False ⇒ the harness discards it as a failed demo.
            self._settle(rec, nj, n=1)
            data = rec.collected()
        return data

    @staticmethod
    def _settle(rec, nj: int, n: int = 8, gripper: float = 1.0):
        import numpy as np

        for _ in range(n):
            robot = rec.unwrapped.agent.robot
            qpos = robot.get_qpos()[0, :nj].cpu().numpy()
            if getattr(rec, "use_gripper", False):
                action = np.hstack([qpos, gripper])
            else:
                action = qpos
            rec.step(action)


# ---- per-task plans (ported from the fork solve()s, minus env.reset) ----
def _plan_stackcube(solver, env, rec) -> Any:
    """Re-plan StackCube from the LIVE state: grasp cubeA, lift, align over cubeB,
    release. Adapted from mani_skill .../panda/solutions/stack_cube.solve but built
    from the cube's TRUE (GPU) world pose.

    GPU-sim fix: the fork's ``get_actor_obb`` reads ``actor._objs[0]`` (the CPU-side
    physx component), whose pose is STALE under GPU sim — it returns the cube's LOCAL
    frame (center ≈ origin), so the obb-based grasp targets the world origin ~14 cm
    off the real cube and the gripper grabs air (verified via the diag). ``cubeA.pose``
    (the batched GPU pose) is correct, so we build the top-down grasp directly from it
    (a cube is graspable along either horizontal face axis with the wide panda
    gripper)."""
    import numpy as np
    import sapien
    from transforms3d.euler import euler2quat
    from transforms3d.quaternions import quat2mat

    approaching = np.array([0.0, 0.0, -1.0])
    cubeA_p = env.cubeA.pose.p.reshape(-1)[:3].cpu().numpy()
    cubeA_q = env.cubeA.pose.q.reshape(-1)[:4].cpu().numpy()           # wxyz
    Rc = quat2mat(cubeA_q)
    tcp_y = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    def _orth(v):
        v = v - approaching * float(v @ approaching)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else v

    # close along whichever of the cube's horizontal face axes best matches the
    # gripper's current closing (least wrist rotation).
    cands = [_orth(Rc[:, 0]), _orth(Rc[:, 1])]
    closing = max(cands, key=lambda c: abs(float(c @ tcp_y)))
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, cubeA_p)

    # search a reachable wrist orientation (dry-run plans, not recorded)
    angles = np.repeat(np.arange(0, np.pi * 2 / 3, np.pi / 2), 2)
    angles[1::2] *= -1
    for angle in angles:
        grasp_pose2 = grasp_pose * sapien.Pose(q=euler2quat(0, 0, angle))
        if solver.move_to_pose_with_screw(grasp_pose2, dry_run=True) == -1:
            continue
        grasp_pose = grasp_pose2
        break

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    solver.move_to_pose_with_screw(reach_pose)
    solver.move_to_pose_with_screw(grasp_pose)
    solver.close_gripper()
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    solver.move_to_pose_with_screw(lift_pose)
    goal_pose = env.cubeB.pose * sapien.Pose([0, 0, float(env.cube_half_size[2]) * 2])
    offset = (goal_pose.p - env.cubeA.pose.p).cpu().numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    solver.move_to_pose_with_screw(align_pose)
    return solver.open_gripper()


def _plan_plugcharger(solver, env, rec) -> Any:
    """Re-plan PlugCharger from the LIVE state: grasp the charger by its base,
    align the prongs to the socket, and insert. Ported from the fork solve
    ``mani_skill .../panda/solutions/plug_charger.py::solve``, MINUS ``env.reset()``
    and the solver construction (the solver/env/rec are passed in by
    ``move_to_next_goal`` so every step is recorded into the demo TensorDict).

    GPU-sim note: unlike StackCube, the fork plug solve never calls
    ``get_actor_obb`` (the stale CPU ``actor._objs[0]`` pose) — it builds the grasp
    obb from ``env.charger_base_pose`` and reads every pose via ``.sp``. We replace
    ``.sp`` with ``_sp0`` (read ``raw_pose[0]`` with an explicit ``.cpu().numpy()``):
    identical to ``.sp`` for the single-env collection path, but GPU-safe and
    robust to ``num_envs > 1`` (``.sp`` asserts batch==1). Poses are re-read at
    compute time (the align transform is captured AFTER ``close_gripper`` so it
    encodes the rigid charger→tcp offset of the actual grasp, exactly as the fork).

    The charger ends GRASPED (held in the socket); the post-plan settle holds the
    gripper CLOSED for this task (``_settle_gripper = -1``)."""
    import math

    import numpy as np
    import sapien
    import trimesh
    from transforms3d.euler import euler2quat

    from mani_skill.examples.motionplanning.panda.utils import (
        compute_grasp_info_by_obb,
    )

    def _sp0(pose) -> "sapien.Pose":
        """Batched ManiSkill Pose → sapien.Pose for env 0 (the ``.sp`` equivalent;
        explicit ``.cpu().numpy()`` so it is correct under GPU sim)."""
        raw = pose.raw_pose.reshape(-1, 7)[0].detach().cpu().numpy()
        return sapien.Pose(p=raw[:3], q=raw[3:])

    FINGER_LENGTH = 0.025

    # --- grasp the charger base (live GPU pose) ------------------------------
    charger_base0 = _sp0(env.charger_base_pose)
    tcp0 = _sp0(env.agent.tcp.pose)
    obb = trimesh.primitives.Box(
        extents=np.array(env._base_size) * 2,
        transform=charger_base0.to_transformation_matrix(),
    )
    approaching = np.array([0.0, 0.0, -1.0])
    target_closing = tcp0.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    grasp_pose = env.agent.build_grasp_pose(
        approaching, grasp_info["closing"], grasp_info["center"])
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, np.deg2rad(15), 0))

    # --- reach → grasp → close ----------------------------------------------
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    solver.move_to_pose_with_screw(reach_pose)
    solver.move_to_pose_with_screw(grasp_pose)
    solver.close_gripper()

    # --- gross pre-insert (bring the charger to the socket mouth, ~5 cm back) -
    # rigid-grasp transform captured at the grasp point: goal · charger⁻¹ · tcp
    # carries the charger from its current pose to (goal − 5 cm). Constant
    # charger→tcp offset, computed once for the gross approach.
    goal0 = _sp0(env.goal_pose)
    charger0 = _sp0(env.charger.pose)
    tcp0b = _sp0(env.agent.tcp.pose)
    pre_insert_pose = goal0 * sapien.Pose([-0.05, 0.0, 0.0]) * charger0.inv() * tcp0b
    solver.move_to_pose_with_screw(pre_insert_pose, refine_steps=0)
    solver.move_to_pose_with_screw(pre_insert_pose, refine_steps=5)

    # --- progressive spiral+tilt search insert (the strengthened planner) -----
    # A single open-loop shove jams the peg ~1.7 cm short (lateral + ~0.17 rad
    # angular misalignment, both > the 0.5 mm socket clearance), and pushing harder
    # tears the charger out of the gripper. Instead: advance in small forward steps
    # RE-TARGETING from LIVE poses each step (closed-loop alignment), and at each
    # step sweep a small lateral SPIRAL (r=3 mm) combined with a TILT wiggle
    # (0.03 rad) to find the hole; early-exit once the charger seats (<6 mm).
    # Empirically 13/15 (87%) vs ~20% open-loop; spiral-only (13%) and tilt-only
    # (33%) are far worse — the peg needs BOTH a lateral and an angular search.
    def _charger_to(fwd, dy=0.0, dz=0.0, tilt=0.0):
        if tilt:
            off = sapien.Pose([fwd, dy, dz], q=euler2quat(0.0, float(tilt), 0.0))
        else:
            off = sapien.Pose([fwd, dy, dz])
        return (_sp0(env.goal_pose) * off
                * _sp0(env.charger.pose).inv() * _sp0(env.agent.tcp.pose))

    def _seated(thresh=6e-3):
        d, _ = env._compute_distance()
        return float(d.reshape(-1)[0]) < thresh

    n_steps, n_spiral, spiral_r, tilt_amp = 6, 6, 3e-3, 0.03
    start_fwd, fwd_target = -0.03, 2e-3
    fwd = fwd_target
    for i in range(n_steps):
        fwd = start_fwd + (fwd_target - start_fwd) * (i + 1) / n_steps
        seated = False
        for j in range(n_spiral):
            ang = 2.0 * math.pi * j / n_spiral
            solver.move_to_pose_with_screw(
                _charger_to(fwd, spiral_r * math.cos(ang), spiral_r * math.sin(ang),
                            tilt_amp * math.sin(ang)), refine_steps=1)
            if _seated():
                seated = True
                break
        if seated:
            break
    return solver.move_to_pose_with_screw(_charger_to(fwd_target), refine_steps=15)


_PLAN_REGISTRY: Dict[str, Callable] = {
    "StackCube-v1": _plan_stackcube,
    "PlugCharger-v1": _plan_plugcharger,
}
