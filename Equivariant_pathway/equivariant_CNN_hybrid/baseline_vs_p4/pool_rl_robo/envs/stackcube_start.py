"""StackCube-Start-v0 — a repositionable StackCube env for p4_top3.

The P4-LLM pipeline prescribes a corrective SCENE CONFIG (where to place cube A and
cube B) and needs an env that HONOURS it on reset, so the motion-planner expert can
solve that exact configuration → one corrective demonstration. Stock StackCube-v1
always randomises cube poses in ``_initialize_episode`` (ignoring any prescribed
attributes), exactly like PushT-v2 ignores the tee pose — so, mirroring the fork's
``PushT-Start-v0`` (env_restart/custom_envs/push_t_start.py), we subclass it and
override ``_initialize_episode`` to place the cubes at the prescribed positions when
set, else fall back to the parent's random spawn.

Prescription attributes (set by the suite pipeline before reset, then cleared):
  _cubeA_init_xyz : [x,y,z] world position of cube A (the cube to pick)   | None→random
  _cubeB_init_xyz : [x,y,z] world position of cube B (the base cube)       | None→random
  _cubeA_init_zrot, _cubeB_init_zrot : z-rotation (rad)                     | None→random

Registered as ``StackCube-Start-v0`` (suite-local; never edits the fork). The main
policy/eval envs keep using stock ``StackCube-v1``; only the corrective-demo collection
uses this env.
"""
from __future__ import annotations

import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


# Table workspace bounds for cube centres (matches StackCubeEnv's spawn region; the
# prescription is clamped here so an out-of-reach LLM config can't break the planner).
_X_RANGE = (-0.15, 0.15)
_Y_RANGE = (-0.25, 0.25)
_CUBE_Z = 0.02                       # cube half-size (rests on the table)
_MIN_SEP = 0.05                      # min cubeA–cubeB centre separation (graspable gap)


def _zrot_quat(angle: torch.Tensor) -> torch.Tensor:
    """(b,) z-rotation angle → (b,4) wxyz quaternion."""
    b = angle.shape[0]
    q = torch.zeros((b, 4), dtype=torch.float32, device=angle.device)
    q[:, 0] = torch.cos(angle / 2.0)
    q[:, 3] = torch.sin(angle / 2.0)
    return q


@register_env("StackCube-Start-v0", max_episode_steps=100)
class StackCubeStartEnv(StackCubeEnv):

    def __init__(self, *args,
                 cubeA_init_xyz=None, cubeA_init_zrot=None,
                 cubeB_init_xyz=None, cubeB_init_zrot=None, **kwargs):
        self._cubeA_init_xyz = cubeA_init_xyz
        self._cubeA_init_zrot = cubeA_init_zrot
        self._cubeB_init_xyz = cubeB_init_xyz
        self._cubeB_init_zrot = cubeB_init_zrot
        super().__init__(*args, **kwargs)

    # ── prescription API (set before reset, honoured by _initialize_episode) ──
    def set_prescription(self, cubeA_xyz=None, cubeB_xyz=None,
                         cubeA_zrot=None, cubeB_zrot=None) -> dict:
        """Set (clamped) prescribed cube poses; returns the applied/achievable config."""
        def _clamp_xy(xyz):
            if xyz is None:
                return None
            x = min(max(float(xyz[0]), _X_RANGE[0]), _X_RANGE[1])
            y = min(max(float(xyz[1]), _Y_RANGE[0]), _Y_RANGE[1])
            return [x, y, _CUBE_Z]

        a = _clamp_xy(cubeA_xyz)
        b = _clamp_xy(cubeB_xyz)
        # enforce a graspable separation: if A and B are too close, push A out along x
        if a is not None and b is not None:
            dx, dy = a[0] - b[0], a[1] - b[1]
            if (dx * dx + dy * dy) ** 0.5 < _MIN_SEP:
                a[0] = min(max(b[0] + _MIN_SEP, _X_RANGE[0]), _X_RANGE[1])
        self._cubeA_init_xyz = a
        self._cubeB_init_xyz = b
        self._cubeA_init_zrot = None if cubeA_zrot is None else float(cubeA_zrot)
        self._cubeB_init_zrot = None if cubeB_zrot is None else float(cubeB_zrot)
        return {"cubeA_xyz": a, "cubeB_xyz": b,
                "cubeA_zrot": self._cubeA_init_zrot, "cubeB_zrot": self._cubeB_init_zrot}

    def clear_prescription(self) -> None:
        self._cubeA_init_xyz = self._cubeB_init_xyz = None
        self._cubeA_init_zrot = self._cubeB_init_zrot = None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # cube B (base): prescribed or random (parent-style)
            self._place(self.cubeB, b, self._cubeB_init_xyz, self._cubeB_init_zrot)
            # cube A (movable): prescribed or random (parent-style)
            self._place(self.cubeA, b, self._cubeA_init_xyz, self._cubeA_init_zrot)

    def _place(self, cube, b, init_xyz, init_zrot):
        if init_xyz is not None:
            xyz = torch.tensor(init_xyz, dtype=torch.float32,
                               device=self.device).unsqueeze(0).repeat(b, 1)
        else:
            # parent random spawn box (StackCubeEnv._initialize_episode)
            xyz = torch.zeros((b, 3)); xyz[:, 2] = _CUBE_Z
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            sampler = randomization.UniformPlacementSampler(
                bounds=[[-0.1, -0.2], [0.1, 0.2]], batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            xyz[:, :2] = xy + sampler.sample(radius, 100, verbose=False)
        if init_zrot is not None:
            angle = torch.full((b,), float(init_zrot), device=self.device)
            q = _zrot_quat(angle)
        else:
            q = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
        cube.set_pose(Pose.create_from_pq(p=xyz, q=q))
