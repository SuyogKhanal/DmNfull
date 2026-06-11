"""Generate StackCube-v1 normalizers + validate the motion-planner demo path.

One-off GPU step (run BEFORE the smoke). It:
  1. builds the StackCube env exactly as the pipeline does (MS.make_policy_env),
  2. validates MotionPlannerExpert.move_to_next_goal from a FRESH-reset state AND
     from a perturbed MID-EPISODE state (the corrective-demo case — no env.reset),
  3. checks the demo TensorDict schema (obs keys + action_joint_pos[8] + done),
  4. collects ~N successful demos, fits a fresh TimeSeriesDataset's normalizers,
  5. asserts proprio_dim / action_dim match the Hydra config (and a fresh policy's
     forward pass accepts the obs — i.e. global_cond_dim lines up),
  6. saves the normalizers dict to cfg.normalizers_path.

Run on a GPU node (1 GPU, no LLM):
  PYTHONPATH=/weka/s226137394/DmNfull \
  /home/s226137394/.conda/envs/diffdagger/bin/python -m \
    Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.tools.gen_stackcube_normalizers
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from tensordict import TensorDict

PKG = "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"

from ..envs import env_setup as E      # noqa: E402
from ..envs import maniskill_env as MS  # noqa: E402
from ..envs import experts as X        # noqa: E402
from ..policies import factory as PF   # noqa: E402

ENV = "StackCube-v1"
N_TARGET = 22          # successful demos to fit normalizers on
N_TRIES = 40           # seed budget (allow planner failures)


def _append(dataset, td):
    for key in list(td.keys()):
        if "rgb" in key:
            td[key] = td[key].permute(0, 3, 1, 2)
    dataset.rb.extend(TensorDict(td, batch_size=len(td["episode"])))


def _flags(env):
    """Final StackCube evaluate() flags (post-demo), for diagnosing failures."""
    try:
        ev = env.unwrapped.evaluate()
        return {k: bool(v.reshape(-1)[0]) for k, v in ev.items() if hasattr(v, "reshape")}
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
    cfg = MS.load_cfg(ENV)
    print(f"[gen] cfg loaded: action_space={cfg.action_space} action_dim={cfg.action_dim} "
          f"proprio_dim={cfg.proprio_dim} obs_keys={list(cfg.obs_keys)}")
    env = MS.make_policy_env(cfg)
    nj = int(env.num_joints)
    print(f"[gen] env built; num_joints={nj} use_gripper={getattr(env, 'use_gripper', None)}")
    expert = X.MotionPlannerExpert(ENV)

    # ── 1) fresh-reset demo (initial-demo path) ──────────────────────────
    td0, done0, n0, any0 = _run_expert_episode(env, expert, seed=0)
    assert td0 is not None and n0 > 0, "fresh-reset demo is EMPTY"
    obs_dims = {k: int(td0[k].shape[-1]) for k in cfg.obs_keys}
    proprio = sum(obs_dims[k] for k in cfg.obs_keys)
    aj = int(td0["action_joint_pos"].shape[-1])
    print(f"[gen] fresh demo: steps={n0} success={done0} any_success_step={any0}")
    print(f"[gen] fresh demo final flags: {_flags(env)}")
    print(f"[gen] obs key dims: {obs_dims}  -> proprio_dim={proprio} (cfg={int(cfg.proprio_dim)})")
    print(f"[gen] action_joint_pos dim={aj} (cfg.action_dim={int(cfg.action_dim)})")
    for need in ("action_joint_pos", "action_joint_delta_pos", "action_ee_pose",
                 "action_ee_delta_pose", "done", "episode"):
        assert need in td0, f"demo missing key {need!r}"
    assert proprio == int(cfg.proprio_dim), \
        f"FIXME: set proprio_dim={proprio} in stackcube_state.yaml (cfg={int(cfg.proprio_dim)})"
    assert aj == int(cfg.action_dim), \
        f"FIXME: set action_dim={aj} in stackcube_state.yaml (cfg={int(cfg.action_dim)})"

    # ── 2) mid-episode demo (corrective path, NO reset before planning) ──
    obs, info = env.reset(seed=123)
    policy_obs = obs
    # perturb into a realistic mid-episode state: small ABSOLUTE joint moves from
    # the current qpos (joint_pos action space), gripper held open.
    env.set_action_space("joint_pos")
    for _ in range(8):
        cur = env.unwrapped.agent.robot.get_qpos()[:, :nj]
        a = cur + (torch.rand_like(cur) - 0.5) * 0.05
        if int(cfg.action_dim) > nj:
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
    # detach to cpu for a portable checkpoint
    norm = {k: v for k, v in dataset.normalizers.items()}
    for v in norm.values():
        try:
            v.to_device("cpu")
        except Exception:
            pass
    torch.save(norm, str(out_path))
    print(f"[gen] SAVED normalizers -> {out_path}")
    print("[gen] DONE — StackCube motion-planner demo path validated; "
          "proprio_dim/action_dim confirmed; normalizers written.")
    return 0


def _empty_dataset(cfg):
    """A fresh TimeSeriesDataset with an empty replay buffer (load_count 0),
    independent of cfg.normalizers_path (which does not exist yet)."""
    from hydra.utils import instantiate
    return instantiate(cfg.dataset, _recursive_=False)


if __name__ == "__main__":
    sys.exit(main())
