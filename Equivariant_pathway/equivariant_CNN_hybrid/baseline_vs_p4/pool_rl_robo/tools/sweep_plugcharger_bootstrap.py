"""Bootstrap-size sweep for PlugCharger — THE sweep-gate for ee_pose_6d.

Clone of sweep_stackcube_bootstrap.py. Answers two questions:
  1. Does behaviour cloning learn AT ALL under the paper-faithful ee_pose_6d
     config (the single-step Jacobian-pinv IK is untested in this suite for a
     5 mm / 0.2 rad contact-rich insertion)? A held-out SR that climbs with more
     demos ⇒ learnable; flat ~0 across all N ⇒ the IK path can't track and we fall
     back to rel_joint_pos (the user's sweep-gate decision).
  2. What warm-start N_i gives a climbable common init SR to fork both arms from.

Collects expert demos from seeds 0..SEED_BUDGET-1 ONCE (keeps successes), then for
each N_i trains a fresh diffusion policy on demos from seeds < N_i and evaluates the
frozen held-out SR at seed_base 7777 (the SAME eval the real comparison uses).

Requires assets/normalizers/PlugCharger-v1_normalizers.pth (run
gen_plugcharger_normalizers first). Run on a GPU node (1 GPU, no LLM):
  MODULE=...tools.sweep_plugcharger_bootstrap LOGTAG=plugcharger_sweep \
    sbatch --time=10:00:00 tools/run_gen_normalizers.sh
"""
from __future__ import annotations

import gc
import os
import sys

import torch
from tensordict import TensorDict

from ..envs import env_setup as E       # noqa: E402
from ..envs import maniskill_env as MS   # noqa: E402
from ..envs import experts as X         # noqa: E402
from ..policies import factory as PF    # noqa: E402

E.bootstrap_fork_path()
from diffdagger.main_pipeline.sim_bridge import (  # noqa: E402
    train_policy, evaluate_heldout,
)

ENV = "PlugCharger-v1"
SEED_BUDGET = 840                       # collect demos from seeds 0..839 (~95% planner SR)
N_I_GRID = [400, 600, 800]             # CEILING probe: does rel_joint_pos keep climbing past 400
                                       # (0.31@100eps) toward ~0.4, or stall? (low-N curve already
                                       # measured: 50/100/200/400 = 0.19/0.24/0.28/0.31)
HELDOUT_N = 200                        # tighter read at the ceiling (small high-N gains vs noise)
EVAL_NUM_ENVS = 20                     # must divide HELDOUT_N (100 % 20 == 0)
INITIAL_EPOCHS = 300
MAX_TRAIN_STEPS = 30000


def main() -> int:
    cfg = MS.load_cfg(ENV)
    np_path = str(cfg.normalizers_path)
    if not os.path.isabs(np_path):
        cfg.normalizers_path = str(E.SUITE_ROOT / np_path)
    cfg.epoch = INITIAL_EPOCHS
    cfg.max_train_steps = MAX_TRAIN_STEPS
    from omegaconf import open_dict
    with open_dict(cfg.env):
        cfg.env.render_mode = None   # collection doesn't render → a100-friendly
    print(f"[sweep] cfg: action_space={cfg.action_space} action_dim={cfg.action_dim} "
          f"proprio_dim={cfg.proprio_dim} pred_horizon={cfg.pred_horizon} "
          f"epoch={cfg.epoch} max_train_steps={cfg.max_train_steps} "
          f"normalizers={cfg.normalizers_path}")

    env = MS.make_policy_env(cfg)
    eval_env = MS.make_eval_env(cfg, num_envs=EVAL_NUM_ENVS)
    expert = X.MotionPlannerExpert(ENV)
    make_policy = PF.policy_factory(cfg)
    make_dataset = PF.dataset_factory(cfg)
    ah, mes = cfg.action_horizon, cfg.env.max_episode_steps

    # ── collect demos seeds 0..SEED_BUDGET-1, keep successes (seed-tagged) ──
    demos = []
    for seed in range(SEED_BUDGET):
        env.reset(seed=int(seed))
        env.set_action_space("joint_pos")
        expert.reset(env)
        expert.setup_task()
        td = expert.move_to_next_goal(dict(seed=int(seed)))
        if td is None:
            print(f"[sweep] seed={seed} EMPTY demo", flush=True)
            continue
        td.update(dict(episode=torch.ones(len(td["episode"])) * int(seed)))
        done = bool(td["done"][-1].item())
        total = len(td["episode"])
        keep = done and 10 <= total <= int(cfg.expert.max_episode_steps)
        if keep:
            demos.append((int(seed), td))
        if seed % 10 == 0 or keep:
            print(f"[sweep] collect seed={seed} done={done} steps={total} "
                  f"kept={len(demos)}", flush=True)
    print(f"[sweep] collected {len(demos)} successful demos from {SEED_BUDGET} seeds "
          f"({100*len(demos)/SEED_BUDGET:.0f}% expert success)", flush=True)

    # ── for each N_i: train fresh policy on demos with seed < N_i, eval SR ──
    results = []
    for N in N_I_GRID:
        sub = [td for (s, td) in demos if s < N]
        if not sub:
            print(f"[sweep] N_i={N}: no demos, skip", flush=True)
            continue
        ds = make_dataset()
        for td in sub:
            for key in list(td.keys()):
                if "rgb" in key:
                    td[key] = td[key].permute(0, 3, 1, 2)
                elif key in list(ds.normalizers.keys()):
                    ds.normalizers[key].update(td[key])
            ds.rb.extend(TensorDict(td, batch_size=len(td["episode"])))
        pol = make_policy()
        res = train_policy(pol, ds, cfg,
                           checkpoint_dir=str(E.SUITE_ROOT / "assets" / "sweep_plug" / f"N{N}"),
                           log_dir=None, fine_tune=False)
        if res.get("status") != "ok":
            print(f"[sweep] N_i={N} TRAIN FAILED: {res.get('error')}", flush=True)
            continue
        pol = res["policy"]
        sr = float(evaluate_heldout(pol, eval_env, 7777, HELDOUT_N, ah, mes))
        results.append((N, len(sub), sr))
        print(f"[sweep] >>> N_i_seeds={N} demos={len(sub)} heldout_SR={sr:.3f} "
              f"(over {HELDOUT_N} eval eps)", flush=True)
        del pol, ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[sweep] ================= SUMMARY =================")
    print("[sweep] N_i_seeds | demos | held-out SR")
    for N, nd, sr in results:
        print(f"[sweep]   {N:>4d}     | {nd:>4d}  |   {sr:.3f}")
    print("[sweep] SWEEP-GATE: a curve that CLIMBS with N ⇒ ee_pose_6d is learnable, "
          "proceed. Flat ~0 at all N ⇒ the IK path can't track the insertion → fall "
          "back to rel_joint_pos (action_dim 8) and re-run this sweep.")
    print("[sweep] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
