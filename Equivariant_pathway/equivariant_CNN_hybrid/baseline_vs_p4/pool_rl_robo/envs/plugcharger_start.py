"""PlugCharger-Start-v0 — a repositionable PlugCharger env for the p4 hybrid.

Mirrors envs/stackcube_start.py. The P4-LLM hybrid's BRIDGE branch prescribes a
corrective SCENE CONFIG (where to place the charger and the wall socket /
receptacle) and needs an env that HONOURS it on reset, so the motion-planner
expert can solve that exact configuration → one corrective demonstration. Stock
PlugCharger-v1 always randomises both poses in ``_initialize_episode`` (ignoring
any prescribed attributes), so we subclass it and override ``_initialize_episode``
to place the charger + receptacle at the prescribed poses when set, else fall back
to the parent's random spawn.

Two differences vs StackCube-Start-v0 (both load-bearing):
  * PlugChargerEnv uses ``self.scene_builder`` (NOT ``self.table_scene``) and also
    re-initialises the robot qpos / base pose inside ``_initialize_episode`` — we
    reproduce that verbatim (with ``self.robot_init_qpos_noise``, which the
    reposition env sets to 0.0).
  * ``goal_pose`` is DERIVED from the receptacle (``receptacle.pose · Rz(π)``) and is
    read by the motion-planner expert (pre_insert / insert poses) AND by
    ``evaluate()`` — we MUST recompute it AFTER placing the receptacle, or every
    corrective demo targets a stale goal.

Prescription attributes (set before reset, cleared after):
  _charger_init_xyz / _charger_init_zrot       : charger (dynamic) world pose  | None→random
  _receptacle_init_xyz / _receptacle_init_zrot : receptacle (kinematic) pose   | None→random

Registered as ``PlugCharger-Start-v0`` (suite-local; never edits the fork). The
main policy/eval envs keep using stock ``PlugCharger-v1``; only corrective-demo
collection uses this env.
"""
from __future__ import annotations

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


# Spawn/clamp bounds — mirror PlugChargerEnv._initialize_episode (and the KAG
# ws_charger / ws_receptacle nodes) so an out-of-distribution LLM prescription
# can't hand the motion planner an un-graspable / un-insertable pose. Both bodies
# stay inside the stock spawn region, where the planner is known to work (it is
# the same region the bootstrap demos are drawn from).
_CHARGER_X = (-0.10, -0.026)        # -0.01 - 2*_peg_size[0] = -0.026
_CHARGER_Y = (-0.20, 0.20)
_CHARGER_Z = 0.012                  # _base_size[2] (charger resting height)
_CHARGER_ZROT = (-np.pi / 3, np.pi / 3)
_RECEP_X = (0.01, 0.10)
_RECEP_Y = (-0.10, 0.10)
_RECEP_Z = 0.10
_RECEP_ZROT = (np.pi - np.pi / 8, np.pi + np.pi / 8)   # upright, hole faces the charger


def _zrot_quat(angle: torch.Tensor) -> torch.Tensor:
    """(b,) z-rotation angle → (b,4) wxyz quaternion (pure z-rotation; matches the
    parent's ``random_quaternions(lock_x=True, lock_y=True)`` convention)."""
    b = angle.shape[0]
    q = torch.zeros((b, 4), dtype=torch.float32, device=angle.device)
    q[:, 0] = torch.cos(angle / 2.0)
    q[:, 3] = torch.sin(angle / 2.0)
    return q


@register_env("PlugCharger-Start-v0", max_episode_steps=200)
class PlugChargerStartEnv(PlugChargerEnv):

    def __init__(self, *args,
                 charger_init_xyz=None, charger_init_zrot=None,
                 receptacle_init_xyz=None, receptacle_init_zrot=None, **kwargs):
        self._charger_init_xyz = charger_init_xyz
        self._charger_init_zrot = charger_init_zrot
        self._receptacle_init_xyz = receptacle_init_xyz
        self._receptacle_init_zrot = receptacle_init_zrot
        super().__init__(*args, **kwargs)

    # ── prescription API (set before reset, honoured by _initialize_episode) ──
    def set_prescription(self, charger_xyz=None, receptacle_xyz=None,
                         charger_zrot=None, receptacle_zrot=None) -> dict:
        """Set (clamped) prescribed poses; returns the applied/achievable config."""
        def _clamp_xy(xyz, xr, yr, z):
            if xyz is None:
                return None
            x = min(max(float(xyz[0]), xr[0]), xr[1])
            y = min(max(float(xyz[1]), yr[0]), yr[1])
            return [x, y, z]

        def _clamp_zrot(zr, lo, hi):
            return None if zr is None else min(max(float(zr), lo), hi)

        self._charger_init_xyz = _clamp_xy(charger_xyz, _CHARGER_X, _CHARGER_Y, _CHARGER_Z)
        self._receptacle_init_xyz = _clamp_xy(receptacle_xyz, _RECEP_X, _RECEP_Y, _RECEP_Z)
        self._charger_init_zrot = _clamp_zrot(charger_zrot, *_CHARGER_ZROT)
        self._receptacle_init_zrot = _clamp_zrot(receptacle_zrot, *_RECEP_ZROT)
        return {"charger_xyz": self._charger_init_xyz,
                "receptacle_xyz": self._receptacle_init_xyz,
                "charger_zrot": self._charger_init_zrot,
                "receptacle_zrot": self._receptacle_init_zrot}

    def clear_prescription(self) -> None:
        self._charger_init_xyz = self._receptacle_init_xyz = None
        self._charger_init_zrot = self._receptacle_init_zrot = None

    # ── reset honouring the prescription ─────────────────────────────────────
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # robot home pose + noise + base pose (verbatim from PlugChargerEnv)
            qpos = torch.tensor(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4,
                 0.04, 0.04])
            qpos = torch.normal(
                0, self.robot_init_qpos_noise, (b, len(qpos)), device=self.device) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # charger (dynamic): prescribed or parent-random
            self._place(self.charger, b, self._charger_init_xyz, self._charger_init_zrot,
                        _CHARGER_X, _CHARGER_Y, _CHARGER_Z, _CHARGER_ZROT)
            # receptacle (kinematic): prescribed or parent-random
            self._place(self.receptacle, b, self._receptacle_init_xyz,
                        self._receptacle_init_zrot, _RECEP_X, _RECEP_Y, _RECEP_Z, _RECEP_ZROT)

            # goal_pose is derived from the receptacle and read by the expert /
            # evaluate() — recompute it AFTER the receptacle is placed.
            self.goal_pose = self.receptacle.pose * sapien.Pose(q=euler2quat(0, 0, np.pi))

    def _place(self, actor, b, init_xyz, init_zrot, xr, yr, z, zrot_bounds):
        if init_xyz is not None:
            xyz = torch.tensor(init_xyz, dtype=torch.float32,
                               device=self.device).unsqueeze(0).repeat(b, 1)
        else:
            # parent random spawn box (uniform over the body's xy region)
            xy = randomization.uniform([xr[0], yr[0]], [xr[1], yr[1]], size=(b, 2))
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = xy
            xyz[:, 2] = z
        if init_zrot is not None:
            angle = torch.full((b,), float(init_zrot), device=self.device)
            q = _zrot_quat(angle)
        else:
            q = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=zrot_bounds)
        actor.set_pose(Pose.create_from_pq(p=xyz, q=q))
