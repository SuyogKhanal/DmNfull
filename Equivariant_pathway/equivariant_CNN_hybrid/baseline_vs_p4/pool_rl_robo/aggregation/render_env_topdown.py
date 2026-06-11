"""Render ONE top-down RGB frame of the PushT scene + save the camera projection,
so the coverage heatmap can be overlaid EXACTLY aligned (tee world-coords → pixels
via OpenCV extrinsic/intrinsic). Needs a GPU (SAPIEN); run via srun, one frame only:

  srun -p gpu-large --gres=gpu:1 --constraint=gpu-h100 -t 00:08:00 \
    /home/s226137394/.conda/envs/diffdagger/bin/python -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.aggregation.render_env_topdown

Writes results/aggregate/astar/PushT-v1_topdown.npz {rgb, K(3x3), E(3x4 world->cam)}.
"""
from __future__ import annotations

from ..envs import env_setup as E

E.bootstrap_fork_path()

import numpy as np  # noqa: E402

from ..orchestrator.workspace import aggregate_dir  # noqa: E402


def _np(x):
    import torch
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def main(env_id_fork: str = "PushT-v2", out_name: str = "PushT-v1_topdown.npz",
         cx: float = -0.15, cy: float = -0.10, height: float = 0.95) -> None:
    import gymnasium as gym
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils

    E.register_envs()
    # top-down camera: straight down at the tee workspace center, up=+y. Override the
    # env's default render_camera fields (dict of name -> field overrides).
    pose = sapien_utils.look_at(eye=[cx, cy, height], target=[cx, cy, 0.0], up=[0, 1, 0])
    cam_override = {"render_camera": dict(pose=pose, width=640, height=640, fov=1.2)}
    env = gym.make(env_id_fork, num_envs=1, obs_mode="state_dict",
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   human_render_camera_configs=cam_override, sim_backend="gpu",
                   robot_init_qpos_noise=0.0, enable_shadow=True)
    env.reset(seed=0)
    rgb = _np(env.render())
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb.astype(np.uint8)
    params = env.unwrapped.scene.human_render_cameras["render_camera"].get_params()
    K = _np(params["intrinsic_cv"]); Ecv = _np(params["extrinsic_cv"])
    if K.ndim == 3:
        K = K[0]
    if Ecv.ndim == 3:
        Ecv = Ecv[0]
    out = aggregate_dir() / "astar" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rgb=rgb, K=K, E=Ecv, cam_eye=[cx, cy, height])
    print(f"[render] wrote {out}  rgb={rgb.shape} K={K.shape} E={Ecv.shape}")
    # quick sanity: project the known goal-tee COM (-0.156,-0.1, ~0.04) → pixel
    P = np.array([-0.156, -0.10, 0.04, 1.0])
    pc = Ecv @ P
    px = K @ pc[:3]
    print(f"[render] goal world(-0.156,-0.10) -> pixel ({px[0]/px[2]:.1f},{px[1]/px[2]:.1f}) "
          f"of {rgb.shape[1]}x{rgb.shape[0]} (should land on the red goal-T)")
    env.close()


if __name__ == "__main__":
    main()
