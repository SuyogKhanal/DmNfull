"""Sweep v2 — locate the PlugCharger steep region + ceiling, and test the policy
strength knob (v_prediction vs epsilon), before committing the multi-day comparison.

Sweep v1 (ee_pose_6d, v_prediction) gave a CLIMBING, CONVEX curve:
  N=50→0.11, N=100→0.16, N=200→0.29 (accelerating, not saturating).
That passes the learnability gate, but (a) the absolute SR is low and the ceiling is
unknown — we must warm-start in the steep region with real headroom (the StackCube
mistake was warm-starting at a near-saturated N); and (b) StackCube needed `epsilon`
(V-obj was "too weak"), so v_prediction is the prime policy-strength suspect.

This collects ONE demo pool, then trains+evals a list of (prediction_type, N) configs
on the SAME frozen held-out eval (seed 7777). Read the result as:
  * vpred @ 300/400 → does the v_prediction curve keep climbing past 0.29? (ceiling/headroom)
  * eps   @ 200     → does epsilon lift SR vs v_prediction's 0.29 at the same N? (strength)
Pick (prediction_type, warm-start N) = a point on the steep part with clear headroom.

Run on a GPU node (1 GPU, no LLM, ~6 h):
  MODULE=...tools.sweep_plugcharger_v2 LOGTAG=plugcharger_sweep_v2 \
    sbatch --partition=gpu --constraint=gpu-a100 --qos=batch-long --time=12:00:00 \
    tools/run_gen_normalizers.sh
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
SEED_BUDGET = 420
# (prediction_type, N_i_seeds) — vpred extends v1's curve; epsilon tests the strength knob.
CONFIGS = [
    ("v_prediction", 300),
    ("v_prediction", 400),
    ("epsilon", 200),
    ("epsilon", 400),
]
HELDOUT_N = 100
EVAL_NUM_ENVS = 20
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
        cfg.env.render_mode = None
    print(f"[sweep2] cfg: action_space={cfg.action_space} action_dim={cfg.action_dim} "
          f"pred_horizon={cfg.pred_horizon} Td={cfg.policy.noise_scheduler.num_train_timesteps} "
          f"epoch={cfg.epoch}", flush=True)

    env = MS.make_policy_env(cfg)
    eval_env = MS.make_eval_env(cfg, num_envs=EVAL_NUM_ENVS)
    expert = X.MotionPlannerExpert(ENV)
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
            continue
        td.update(dict(episode=torch.ones(len(td["episode"])) * int(seed)))
        done = bool(td["done"][-1].item())
        total = len(td["episode"])
        if done and 10 <= total <= int(cfg.expert.max_episode_steps):
            demos.append((int(seed), td))
        if seed % 20 == 0:
            print(f"[sweep2] collect seed={seed} kept={len(demos)}", flush=True)
    print(f"[sweep2] collected {len(demos)} demos from {SEED_BUDGET} seeds "
          f"({100*len(demos)/SEED_BUDGET:.0f}% expert SR)", flush=True)

    # ── train+eval each (prediction_type, N) config ──
    results = []
    for ptype, N in CONFIGS:
        with open_dict(cfg):
            cfg.prediction_type = ptype                                  # ${...} resolves into the scheduler
            cfg.policy.noise_scheduler.prediction_type = ptype
        sub = [td for (s, td) in demos if s < N]
        if not sub:
            print(f"[sweep2] {ptype}@N{N}: no demos, skip", flush=True)
            continue
        make_policy = PF.policy_factory(cfg)    # fresh factory picks up prediction_type
        make_dataset = PF.dataset_factory(cfg)
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
                           checkpoint_dir=str(E.SUITE_ROOT / "assets" / "sweep_plug_v2" / f"{ptype}_N{N}"),
                           log_dir=None, fine_tune=False)
        if res.get("status") != "ok":
            print(f"[sweep2] {ptype}@N{N} TRAIN FAILED: {res.get('error')}", flush=True)
            continue
        sr = float(evaluate_heldout(res["policy"], eval_env, 7777, HELDOUT_N, ah, mes))
        results.append((ptype, N, len(sub), sr))
        print(f"[sweep2] >>> {ptype} N={N} demos={len(sub)} heldout_SR={sr:.3f}", flush=True)
        del pol, ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[sweep2] ============== SUMMARY (+ v1: vpred 50/100/200 = 0.11/0.16/0.29) ==============")
    print("[sweep2] pred_type    | N   | demos | held-out SR")
    for ptype, N, nd, sr in results:
        print(f"[sweep2]   {ptype:<12s}| {N:>3d} | {nd:>4d}  |  {sr:.3f}")
    print("[sweep2] Pick (prediction_type, warm-start N) on the steep part with clear headroom.")
    print("[sweep2] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
