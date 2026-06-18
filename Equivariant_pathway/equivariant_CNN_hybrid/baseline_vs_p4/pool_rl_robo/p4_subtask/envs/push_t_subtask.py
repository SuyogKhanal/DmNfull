"""PushT-Subtask-v0 — PushT-Start-v0 + an OPTIONAL robot-qpos sub-task entry.

PushT-Start-v0 honours a prescribed tee pose but always sends the robot to the
canonical home keyframe (table_scene.initialize). For the sub-task entry we also
want to place the ARM at the failure-time configuration so the expert starts
mid-task near the contact (and demonstrates only the hard remaining push). This
subclass adds ``_agent_init_qpos``:

  * ``_agent_init_qpos is None``  → behaviour is byte-identical to PushT-Start-v0
    (so even if shared with p4_top3 it is a no-op).
  * otherwise → after the parent sets the tee + canonical robot, override the
    robot qpos with the recorded failure config.

The fork is never edited; this env lives suite-side and is registered via
``p4_subtask.envs`` import before the reposition env is built.
"""
from __future__ import annotations

import torch

from mani_skill.utils.registration import register_env

# env_restart is on sys.path via env_setup.bootstrap_fork_path()
from custom_envs.push_t_start import PushTStartEnv


@register_env("PushT-Subtask-v0", max_episode_steps=100)
class PushTSubtaskEnv(PushTStartEnv):

    def __init__(self, *args, agent_init_qpos=None, **kwargs):
        self._agent_init_qpos = agent_init_qpos
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 1. Parent: table + canonical robot + goal T + prescribed tee pose.
        super()._initialize_episode(env_idx, options)

        # 2. Optional sub-task entry: override the arm with the failure config.
        if self._agent_init_qpos is None:
            return
        try:
            with torch.device(self.device):
                b = len(env_idx)
                robot = self.agent.robot
                cur = robot.qpos                     # (b, dof)
                dof = int(cur.shape[-1])
                q = torch.as_tensor(self._agent_init_qpos, dtype=torch.float32,
                                    device=self.device).reshape(-1)
                if q.numel() >= dof:
                    q = q[:dof]
                else:
                    # pad with the canonical tail so length matches the robot dof
                    q = torch.cat([q, cur[0, q.numel():].detach()], dim=0)
                qb = q.unsqueeze(0).repeat(b, 1)
                robot.set_qpos(qb)
                try:
                    robot.set_qvel(torch.zeros((b, dof), device=self.device))
                except Exception:
                    pass
                # Hold the controller target at the new qpos so the arm does not
                # snap back toward the canonical keyframe before the expert acts.
                try:
                    self.agent.controller.reset()
                except Exception:
                    pass
        except Exception as e:  # never crash a reset — the tee lever still applies
            print(f"[PushT-Subtask-v0] agent_init_qpos skipped ({e})")
