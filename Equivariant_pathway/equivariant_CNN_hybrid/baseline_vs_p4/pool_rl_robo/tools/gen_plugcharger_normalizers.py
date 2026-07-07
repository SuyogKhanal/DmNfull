"""Generate PlugCharger-v1 normalizers + validate the ee_pose_6d data path.

Clone of gen_stackcube_normalizers.py, adapted for the paper-faithful Plugging
config (action_space=ee_pose_6d, action_dim=10). One-off GPU step (run AFTER the
planner-SR diag passes, BEFORE the bootstrap sweep / baseline). It:
  1. builds the PlugCharger env exactly as the pipeline does (MS.make_policy_env),
  2. runs MotionPlannerExpert.move_to_next_goal from a fresh reset AND a perturbed
     mid-episode state (the corrective-demo case — no env.reset),
  3. checks the demo schema; for ee_pose_6d it verifies action_ee_pose is 8-wide
     (pos3 + quat4 + gripper1 — the suite recorder's gripper append) so the dataset
     emits action_dim 10,
  4. collects N successful demos, fits a fresh TimeSeriesDataset's normalizers in
     the ee_pose_6d (6D-rotation) action space,
  5. asserts the DATASET action_seq dim == cfg.action_dim (the authoritative check
     for ee_pose_6d) and that a fresh policy forward pass accepts the obs
     (global_cond_dim / proprio_dim line up),
  6. saves the normalizers dict to assets/normalizers/PlugCharger-v1_normalizers.pth.

Run on a GPU node (1 GPU, no LLM):
  MODULE=Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.tools.gen_plugcharger_normalizers \
  LOGTAG=plugcharger_normgen  sbatch tools/run_gen_normalizers.sh
"""
from __future__ import annotations

import sys

import torch
from tensordict import TensorDict

PKG = "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"

from ..envs import env_setup as E      # noqa: E402
from ..envs import maniskill_env as MS  # noqa: E402
from ..envs import experts as X        # noqa: E402
from ..policies import factory as PF   # noqa: E402

ENV = "PlugCharger-v1"
N_TARGET = 22          # successful demos to fit normalizers on
N_TRIES = 80           # seed budget (tighter task → lower planner yield → more tries)


def _append(dataset, td):
    for key in list(td.keys()):
        if "rgb" in key:
            td[key] = td[key].permute(0, 3, 1, 2)
    dataset.rb.extend(TensorDict(td, batch_size=len(td["episode"])))


def _flags(env):
    """Final PlugCharger evaluate() flags (post-demo), for diagnosing failures."""
    try:
        ev = env.unwrapped.evaluate()
        out = {}
        for k, v in ev.items():
            if hasattr(v, "reshape"):
                rv = v.reshape(-1)[0]
                out[k] = bool(rv) if rv.dtype == torch.bool else round(float(rv), 4)
        return out
    except Exception as exc:
        return {"evaluate_error": str(exc)}


def _run_expert_episode(env, expert, seed):
    """Reset → run the planner to completion → (td, done, total_steps, any_success)."""
    obs, info = env.reset(seed=int(seed))
    env.set_action_space("joint_pos")
    expert.reset(env)
    expert.setup_task()
    td = expert.move_to_next_goal(dict(seed=int(seed)))
    if td is None:
        return None, False, 0, False
    td.update(dict(episode=torch.ones(len(td["episode"])) * int(seed)))
    done = bool(td["done"][-1].item()) if "done" in td else False
    any_succ = bool(td["done"].reshape(-1).max().item() > 0.5) if "done" in td else False
    total = len(td["episode"])
    return td, done, total, any_succ


def main() -> int:
    from omegaconf import open_dict
    cfg = MS.load_cfg(ENV)
    # gen doesn't render (state_dict obs); disable the render camera so this runs
    # render-free on the idle a100 `gpu` partition (no Vulkan contention).
    with open_dict(cfg.env):
        cfg.env.render_mode = None
    print(f"[gen] cfg loaded: action_space={cfg.action_space} action_dim={cfg.action_dim} "
          f"proprio_dim={cfg.proprio_dim} obs_keys={list(cfg.obs_keys)}")
    env = MS.make_policy_env(cfg)
    nj = int(env.num_joints)
    print(f"[gen] env built; num_joints={nj} use_gripper={getattr(env, 'use_gripper', None)}")
    expert = X.MotionPlannerExpert(ENV)
    is_ee = "joint" not in cfg.action_space

    # ── 1) fresh-reset demo (initial-demo path) ──────────────────────────
    td0, done0, n0, any0 = _run_expert_episode(env, expert, seed=0)
    assert td0 is not None and n0 > 0, "fresh-reset demo is EMPTY"
    obs_dims = {k: int(td0[k].shape[-1]) for k in cfg.obs_keys}
    proprio = sum(obs_dims[k] for k in cfg.obs_keys)
    aj = int(td0["action_joint_pos"].shape[-1])
    aep = int(td0["action_ee_pose"].shape[-1])
    print(f"[gen] fresh demo: steps={n0} success={done0} any_success_step={any0}")
    print(f"[gen] fresh demo final flags: {_flags(env)}")
    print(f"[gen] obs key dims: {obs_dims}  -> proprio_dim={proprio} (cfg={int(cfg.proprio_dim)})")
    print(f"[gen] action_joint_pos dim={aj}  action_ee_pose dim={aep}  "
          f"(ee path: expect action_ee_pose 8-wide = pos3+quat4+gripper1)")
    for need in ("action_joint_pos", "action_joint_delta_pos", "action_ee_pose",
                 "action_ee_delta_pose", "done", "episode"):
        assert need in td0, f"demo missing key {need!r}"
    assert proprio == int(cfg.proprio_dim), \
        f"FIXME: set proprio_dim={proprio} in plugcharger_state.yaml (cfg={int(cfg.proprio_dim)})"
    if is_ee:
        # ee_pose_6d with a gripper needs action_ee_pose 8-wide so the dataset emits
        # action_dim 10. The suite recorder (_RecordingEnv) appends the gripper.
        assert aep == 8, (f"action_ee_pose dim={aep}, expected 8 (pos3+quat4+gripper1). "
                          f"The recorder gripper-append for ee action spaces is missing.")
    else:
        assert aj == int(cfg.action_dim), \
            f"FIXME: set action_dim={aj} in plugcharger_state.yaml (cfg={int(cfg.action_dim)})"

    # ── 2) mid-episode demo (corrective path, NO reset before planning) ──
    obs, info = env.reset(seed=123)
    policy_obs = obs
    env.set_action_space("joint_pos")
    for _ in range(8):
        cur = env.unwrapped.agent.robot.get_qpos()[:, :nj]
        a = cur + (torch.rand_like(cur) - 0.5) * 0.05
        if getattr(env, "use_gripper", False):
            a = torch.cat([a, torch.ones(1, 1, device=a.device)], dim=-1)
        env.step({"action": a})
    expert.reset(env)
    expert.setup_task()
    td_mid = expert.move_to_next_goal(dict(seed=123))
    assert td_mid is not None and len(td_mid["episode"]) > 0, "MID-EPISODE demo is EMPTY"
    mid_done = bool(td_mid["done"][-1].item())
    print(f"[gen] MID-EPISODE demo: steps={len(td_mid['episode'])} success={mid_done} "
          f"(planned from a NON-reset perturbed state)")

    # ── 3) collect successful demos + fit normalizers ───────────────────
    dataset = PF.make_dataset_empty(cfg) if hasattr(PF, "make_dataset_empty") else _empty_dataset(cfg)
    collected = 0
    lengths = []
    for seed in range(N_TRIES):
        if collected >= N_TARGET:
            break
        td, done, total, anys = _run_expert_episode(env, expert, seed=seed)
        keep = (td is not None and done and 10 <= total <= int(cfg.expert.max_episode_steps))
        flg = _flags(env) if (not done) else None
        print(f"[gen] seed={seed}: done={done} any_success_step={anys} steps={total} "
              f"keep={keep}" + (f" flags={flg}" if flg else ""))
        if keep:
            _append(dataset, td)
            collected += 1
            lengths.append(total)
    assert collected >= 5, f"too few successful demos ({collected}); planner may be broken"
    print(f"[gen] collected {collected} demos; demo-length min/mean/max="
          f"{min(lengths)}/{sum(lengths)//len(lengths)}/{max(lengths)}")

    dataset.make_indices()
    dataset.generate_action_sequence()
    dataset.set_normalizers()
    action_dim = int(dataset.action_seq.shape[-1])
    print(f"[gen] dataset action_seq dim={action_dim} (cfg.action_dim={int(cfg.action_dim)})")
    assert action_dim == int(cfg.action_dim), \
        f"FIXME: dataset action_dim={action_dim} != cfg.action_dim={int(cfg.action_dim)}"
    print(f"[gen] normalizer keys: {sorted(dataset.normalizers.keys())}")

    # ── 4) fresh-policy forward pass: confirms proprio_dim/global_cond_dim ─
    policy = PF.build_policy(cfg)
    policy.set_normalizers(dataset.normalizers)
    policy.to(cfg.device)
    policy.reset()
    obs_seq = {k: torch.stack([policy_obs[k]] * cfg.obs_horizon).swapaxes(0, 1)
               for k in cfg.obs_keys}
    env.set_action_space(cfg.action_space)
    out = policy.get_action(obs_seq, dagger=False, return_dict=False)
    act = out["action"] if isinstance(out, dict) else out
    print(f"[gen] fresh-policy forward OK: action_seq shape={tuple(act.shape)} "
          f"(expect [...,{int(cfg.action_dim)}])")
    assert int(act.shape[-1]) == int(cfg.action_dim)

    # ── 5) save normalizers to the suite-local path ─────────────────────
    out_path = E.SUITE_ROOT / "assets" / "normalizers" / f"{ENV}_normalizers.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm = {k: v for k, v in dataset.normalizers.items()}
    for v in norm.values():
        try:
            v.to_device("cpu")
        except Exception:
            pass
    torch.save(norm, str(out_path))
    print(f"[gen] SAVED normalizers -> {out_path}")
    print("[gen] DONE — PlugCharger ee_pose_6d demo path validated; "
          "proprio_dim/action_dim(10) confirmed; normalizers written.")
    return 0


def _empty_dataset(cfg):
    from hydra.utils import instantiate
    return instantiate(cfg.dataset, _recursive_=False)


if __name__ == "__main__":
    sys.exit(main())
