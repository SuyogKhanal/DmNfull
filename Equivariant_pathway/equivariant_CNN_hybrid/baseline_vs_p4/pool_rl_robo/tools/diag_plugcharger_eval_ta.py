"""Does lowering action_horizon (Ta) raise the PlugCharger ceiling above ~0.25?

The sweeps showed BC plateaus at ~0.25 heldout SR (ee_pose_6d, v_prediction AND
epsilon, N up to 400). Prime cheap suspect: Ta=8 means the policy commits to 8
OPEN-LOOP steps between re-plans — almost certainly too coarse for a contact-rich
SEARCH insertion (the peg needs near-closed-loop feedback to wiggle into the 0.5 mm
hole). Ta is a ROLLOUT parameter (parallel_evaluate(action_horizon=...)), so we can
RE-EVALUATE the already-trained sweep checkpoint at Ta=8/4/2/1 with NO retraining.

If a lower Ta lifts SR substantially (e.g. 0.24 → 0.5), that's the fix (Ta is shared
by both arms, so it doesn't bias the comparison). If not, the ~0.25 ceiling is more
fundamental (cloning the search / the IK path) and we escalate.

Run on a GPU node (1 GPU, no LLM):
  MODULE=...tools.diag_plugcharger_eval_ta LOGTAG=plugcharger_eval_ta \
    sbatch --partition=gpu --constraint=gpu-a100 --qos=batch-short --time=02:00:00 \
    tools/run_gen_normalizers.sh
"""
from __future__ import annotations

import os
import sys

from omegaconf import open_dict

from ..envs import env_setup as E       # noqa: E402
from ..envs import maniskill_env as MS   # noqa: E402
from ..policies import factory as PF    # noqa: E402

E.bootstrap_fork_path()
from diffdagger.main_pipeline.sim_bridge import evaluate_heldout  # noqa: E402

ENV = "PlugCharger-v1"
HELDOUT_N = 60                 # big-effect screen (looking for 0.24 → 0.5, not 0.24 → 0.27)
TA_GRID = [8, 4, 2, 1]
# (prediction_type, checkpoint dir under assets/sweep_plug_v2)
CKPTS = [("v_prediction", "v_prediction_N400")]


def main() -> int:
    cfg = MS.load_cfg(ENV)
    with open_dict(cfg.env):
        cfg.env.render_mode = None
    eval_env = MS.make_eval_env(cfg, num_envs=20)
    mes = cfg.env.max_episode_steps
    print(f"[ta] env ready; max_episode_steps={mes} HELDOUT_N={HELDOUT_N}", flush=True)

    for ptype, ckdir in CKPTS:
        with open_dict(cfg):
            cfg.prediction_type = ptype
            cfg.policy.noise_scheduler.prediction_type = ptype
        ckpt = str(E.SUITE_ROOT / "assets" / "sweep_plug_v2" / ckdir / "2.pth")
        if not os.path.isfile(ckpt):
            print(f"[ta] MISSING {ckpt}", flush=True)
            continue
        make_policy = PF.policy_factory(cfg)
        pol = make_policy()
        pol.load(ckpt)
        pol.to(cfg.device)
        pol.reset()
        print(f"\n[ta] checkpoint {ckdir} ({ptype}):", flush=True)
        for ta in TA_GRID:
            sr = float(evaluate_heldout(pol, eval_env, 7777, HELDOUT_N, ta, mes))
            print(f"[ta]   Ta={ta:>2d}  heldout_SR={sr:.3f}  (over {HELDOUT_N} eps)", flush=True)
    print("\n[ta] DONE — Ta=8 is the training/paper default; a large lift at Ta<=2 ⇒ "
          "lower Ta is the policy-strength fix (shared by both arms).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
