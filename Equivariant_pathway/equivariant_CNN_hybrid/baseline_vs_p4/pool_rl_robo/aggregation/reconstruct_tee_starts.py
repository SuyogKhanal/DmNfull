"""Reconstruct the tee's INITIAL (reset) pose per seed — the demo START positions.

env.reset(seed=s) deterministically places the tee, so this recovers exact start
poses without the original run. The bootstrap demos used seeds 0..19; DAgger episodes
draw from the same spawn distribution, so a wider sweep (0..N) shows the full
start-space all methods sample from. Needs a GPU (SAPIEN); run via srun:

  srun -p gpu-large --qos=interactive --gres=gpu:1 --constraint=gpu-h100 \
    --cpus-per-gpu=8 --mem=24G -t 00:10:00 \
    /home/s226137394/.conda/envs/diffdagger/bin/python -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.aggregation.reconstruct_tee_starts

Writes results/aggregate/astar/PushT-v1_tee_starts.npz {starts(N,3), seeds, n_bootstrap}.
"""
from __future__ import annotations

from ..envs import env_setup as E

E.bootstrap_fork_path()

import numpy as np  # noqa: E402

from ..orchestrator.workspace import aggregate_dir  # noqa: E402


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _tee_xyz(env):
    u = env.unwrapped
    for attr in ("tee", "obj", "object", "tee_block"):
        a = getattr(u, attr, None)
        if a is not None and hasattr(a, "pose"):
            return _np(a.pose.p).reshape(-1)[:3]
    obs = u.get_obs()                                  # fallback: from state obs
    for key in ("extra_obj_pose",):
        if key in obs:
            return _np(obs[key]).reshape(-1)[:3]
    ex = obs.get("extra", {})
    for key in ("obj_pose", "tee_pose"):
        if key in ex:
            return _np(ex[key]).reshape(-1)[:3]
    raise RuntimeError("could not locate the tee pose")


def main(env_id_fork: str = "PushT-v2", n_seeds: int = 256, n_bootstrap: int = 20,
         out_name: str = "PushT-v1_tee_starts.npz") -> None:
    import gymnasium as gym
    E.register_envs()
    env = gym.make(env_id_fork, num_envs=1, obs_mode="state_dict",
                   control_mode="pd_joint_pos", sim_backend="gpu",
                   robot_init_qpos_noise=0.0)
    starts = []
    for s in range(n_seeds):
        env.reset(seed=int(s))
        starts.append(_tee_xyz(env))
    starts = np.asarray(starts, dtype=np.float32)
    out = aggregate_dir() / "astar" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, starts=starts, seeds=np.arange(n_seeds), n_bootstrap=n_bootstrap)
    print(f"[starts] wrote {out}  starts={starts.shape} "
          f"x∈[{starts[:,0].min():.3f},{starts[:,0].max():.3f}] "
          f"y∈[{starts[:,1].min():.3f},{starts[:,1].max():.3f}]")
    env.close()


if __name__ == "__main__":
    main()
