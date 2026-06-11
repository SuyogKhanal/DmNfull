"""Bootstrap-size sweep for StackCube: how much does the shared init policy's
held-out SR improve with more initial demos? Informs the warm-start design
(what N_i to fork both arms from, and whether ~50% is even reachable).

Collects expert demos from seeds 0..MAX-1 ONCE (mirroring collect_initial_demos_shared,
which uses seeds 0..N_i-1 and keeps successes), then for each N_i trains a fresh
diffusion policy on the demos from seeds < N_i (same initial_epochs/max_train_steps
as the real bootstrap) and evaluates the frozen held-out SR.

Run on a GPU node (1 GPU, no LLM):
  MODULE=...tools.sweep_stackcube_bootstrap LOGTAG=stackcube_sweep sbatch tools/run_gen_normalizers.sh
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

ENV = "StackCube-v1"
SEED_BUDGET = 230                       # collect demos from seeds 0..229 (~210 successes)
N_I_GRID = [50, 100, 200]              # bootstrap seed counts to evaluate (canonical
                                       # low-dim DP uses ~200 demos for ~90% SR)
HELDOUT_N = 100
EVAL_NUM_ENVS = 10
INITIAL_EPOCHS = 300
MAX_TRAIN_STEPS = 30000


def main() -> int:
    cfg = MS.load_cfg(ENV)
    # absolutize the suite-local normalizers path (make_dataset torch.loads it)
    np_path = str(cfg.normalizers_path)
    if not os.path.isabs(np_path):
        cfg.normalizers_path = str(E.SUITE_ROOT / np_path)
    cfg.epoch = INITIAL_EPOCHS
    cfg.max_train_steps = MAX_TRAIN_STEPS
    print(f"[sweep] cfg: action_dim={cfg.action_dim} proprio_dim={cfg.proprio_dim} "
          f"epoch={cfg.epoch} max_train_steps={cfg.max_train_steps} "
          f"normalizers={cfg.normalizers_path}")

    env = MS.make_policy_env(cfg)
    eval_env = MS.make_eval_env(cfg, num_envs=EVAL_NUM_ENVS)
    expert = X.MotionPlannerExpert(ENV)
    make_policy = PF.policy_factory(cfg)
    make_dataset = PF.dataset_factory(cfg)
    ah, mes = cfg.action_horizon, cfg.env.max_episode_steps

    # ── collect demos seeds 0..SEED_BUDGET-1, keep successes (seed-tagged) ──
    demos = []   # list of (seed, td)
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
                           checkpoint_dir=str(E.SUITE_ROOT / "assets" / "sweep" / f"N{N}"),
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
    print("[sweep] Pick the smallest N_i giving a solid common SR to warm-start "
          "both arms from; if SR plateaus low, ~50% is not feasible for StackCube "
          "and a lower common target / bigger budget is the realistic design.")
    print("[sweep] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
