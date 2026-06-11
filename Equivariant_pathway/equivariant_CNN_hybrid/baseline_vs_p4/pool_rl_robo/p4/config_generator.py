"""Component 3 — Configuration LLM (prescription → simulator reset config).

Translates the PrescriptionClass JSON into a concrete env.reset(options=...) dict
for the target task. Keys are task-specific but always include robot_init_qpos,
obj_init_pose, and goal_pose (when applicable). For PushT the suite engine maps
the prescription to the PushT-Start-v0 reposition keys (tee_init_xyz/tee_init_zrot)
in sim_bridge; this Config LLM is the generic translator for the other tasks.
"""
from __future__ import annotations

from typing import Dict, Optional

from . import prompts as P
from .runner import extract_json

# Per-task description of the reset keys the simulator accepts.
RESET_KEYS_DESC = {
    "PushT-v1": ("PushT honours tee_init_xyz=[x,y,z] (z=0.021) and "
                 "tee_init_zrot (radians); the goal T-pose is fixed. Map "
                 "object_pose→obj_init_pose and gripper_pose_at_start→the tcp "
                 "approach; goal_pose is fixed (omit)."),
    "StackCube-v1": ("StackCube reset: robot_init_qpos (7 joints, or null for "
                     "default), obj_init_pose = cube A pose, goal_pose = cube B "
                     "(stack target) pose."),
    "PickCube-v1": ("PickCube reset: robot_init_qpos, obj_init_pose = cube pose, "
                    "goal_pose = goal-site position."),
    "PlugCharger-v1": ("PlugCharger reset: robot_init_qpos, obj_init_pose = "
                       "charger pose, goal_pose = receptacle/socket pose."),
}


class ConfigGenerator:
    def __init__(self, client, suite_env_id: str, task_description: str):
        self.client = client
        self.env_id = suite_env_id
        self.task = task_description

    def generate(self, prescription_json: str) -> Optional[Dict]:
        """Returns an env.reset(options=...) dict, or None if the LLM cannot
        produce one (caller then falls back to a real sampled config)."""
        if self.client is None:
            return None
        desc = RESET_KEYS_DESC.get(self.env_id, "Provide the task's reset keys.")
        try:
            txt = self.client.structure(P.config_prompt(self.task, prescription_json, desc),
                                        system_prompt=P.CONFIG_SYSTEM)
            obj = extract_json(txt)
        except Exception:
            return None
        if not isinstance(obj, dict) or not obj:
            return None
        return obj
