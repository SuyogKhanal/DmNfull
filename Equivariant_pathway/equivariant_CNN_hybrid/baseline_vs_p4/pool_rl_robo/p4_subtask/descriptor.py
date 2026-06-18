"""Per-failure descriptor — reconstruct the t* failure state from meta.json.

Each failed episode writes a ``meta.json`` (HighLossImageSaver) whose
``high_loss_frames`` entries each carry the FULL env state at that timestep
(``_extract_env_state``): the tee pose (``obj_pose_xyz`` / ``obj_pose_quat_wxyz``),
the 7 arm joints (``arm_qpos_rad``), the 2 gripper-finger entries
(``gripper_finger_pos_m``), and the stick TCP (``tcp_pose_xyz``). We pick the
PEAK-loss frame (the moment the policy was most uncertain ≈ the failure point
t*) and turn it into a 6-d feature vector for clustering + the anchor state for
the sub-task entry.

The feature vector is quaternion-free (theta encoded as sin/cos), so it never
asks the LLM — or the clusterer — to reason about raw quaternions (Fix 3 is moot
for PushT, which uses a scalar z-rotation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def yaw_from_quat_wxyz(q) -> float:
    """Z-yaw (radians) from a [w, x, y, z] quaternion. PushT's tee only rotates
    about z, so this recovers the scalar `tee_zrot`."""
    if q is None or len(q) < 4:
        return 0.0
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _peak_frame(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The highest-loss frame that carries a tee pose; fall back to start/end."""
    best = None
    best_loss = -1.0
    for fr in (meta.get("high_loss_frames") or []):
        if fr.get("obj_pose_xyz") is None:
            continue
        dl = fr.get("diffusion_loss")
        dl = float(dl) if isinstance(dl, (int, float)) else -1.0
        if best is None or dl > best_loss:
            best, best_loss = fr, dl
    if best is not None:
        return best
    for k in ("end_frame_state", "start_frame_state"):
        st = meta.get(k)
        if st and st.get("obj_pose_xyz") is not None:
            return st
    return None


@dataclass
class FailureDescriptor:
    ep_dir: str
    episode_id: int
    seed: int                # env seed of the failed episode (scene identity —
                             # lets the v2 on-policy collector re-roll + correct it)
    peak_loss: float
    mean_loss: float
    t_star: int
    T: int
    tee_xyz: List[float]     # [x, y, z]
    tee_theta: float         # scalar yaw (radians)
    arm_qpos: List[float]    # 7 arm joints
    full_qpos: List[float]   # 9 (arm + gripper) — what robot.set_qpos wants
    tcp_xyz: List[float]     # [x, y, z]

    def feature(self) -> List[float]:
        """6-d clustering feature: tee position, tee orientation (sin/cos),
        task progress (t*/T), and tcp-to-tee distance (contact discriminator)."""
        progress = self.t_star / max(1, self.T)
        dx = self.tcp_xyz[0] - self.tee_xyz[0]
        dy = self.tcp_xyz[1] - self.tee_xyz[1]
        dz = self.tcp_xyz[2] - self.tee_xyz[2]
        tcp_to_tee = math.sqrt(dx * dx + dy * dy + dz * dz)
        return [self.tee_xyz[0], self.tee_xyz[1],
                math.sin(self.tee_theta), math.cos(self.tee_theta),
                float(progress), float(tcp_to_tee)]

    def digest(self) -> Dict[str, Any]:
        return {
            "ep_dir": self.ep_dir, "episode_id": self.episode_id, "seed": self.seed,
            "peak_loss": round(self.peak_loss, 6), "mean_loss": round(self.mean_loss, 6),
            "t_star": self.t_star, "T": self.T,
            "tee_xyz": [round(v, 4) for v in self.tee_xyz],
            "tee_theta": round(self.tee_theta, 4),
            "tcp_xyz": [round(v, 4) for v in self.tcp_xyz],
            "arm_qpos": [round(v, 4) for v in self.arm_qpos],
        }


def load_failure_descriptor(ep_dir: str, meta: Dict[str, Any]) -> Optional[FailureDescriptor]:
    """Build a FailureDescriptor from a failed episode's (already loss-augmented)
    meta dict. Returns None if no usable state frame exists."""
    fr = _peak_frame(meta)
    if fr is None:
        return None
    tee_xyz = [float(v) for v in fr["obj_pose_xyz"][:3]]
    theta = yaw_from_quat_wxyz(fr.get("obj_pose_quat_wxyz"))
    arm = [float(v) for v in (fr.get("arm_qpos_rad") or [])][:7]
    grip = [float(v) for v in (fr.get("gripper_finger_pos_m") or [])]
    full = arm + grip
    tcp = fr.get("tcp_pose_xyz")
    tcp_xyz = [float(v) for v in tcp[:3]] if tcp else list(tee_xyz)
    t_star = int(fr.get("timestep", 0) or 0)
    T = int(meta.get("total_steps") or meta.get("end_timestep") or t_star or 1)
    peak = meta.get("peak_loss")
    if not isinstance(peak, (int, float)):
        dl = fr.get("diffusion_loss")
        peak = float(dl) if isinstance(dl, (int, float)) else 0.0
    _eid = meta.get("episode_id")
    _sd = meta.get("seed")
    if _sd is None:                      # NOTE: no `or` — 0 is a valid seed/id
        _sd = _eid
    return FailureDescriptor(
        ep_dir=str(ep_dir),
        episode_id=int(_eid) if _eid is not None else -1,
        seed=int(_sd) if _sd is not None else -1,
        peak_loss=float(peak or 0.0),
        mean_loss=float(meta.get("mean_loss", 0.0) or 0.0),
        t_star=t_star, T=max(1, T),
        tee_xyz=tee_xyz, tee_theta=float(theta),
        arm_qpos=arm, full_qpos=full, tcp_xyz=tcp_xyz,
    )
