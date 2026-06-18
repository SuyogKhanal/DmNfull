"""Tighter re-evaluation of the near-final StackCube policies (Diff-DAgger vs V3
hybrid) to settle the noisy head-to-head: the per-round curves used 100 held-out eps
(~±0.10 95%-band), wider than the ~0.04 margins. Here we re-eval the saved
checkpoints for rounds {56,58,60} with N_EVAL=400 eps (band ~±0.05), on the SAME
frozen held-out seed (7777), and report each arm/seed's robust final-region SR + CI.

Eval-only (no rendering, no LLM) → any GPU. Run:
  python -m ...pool_rl_robo.tools.reeval_final
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from ..envs import env_setup as E       # noqa: E402
from ..envs import maniskill_env as MS   # noqa: E402
from ..policies import factory as PF    # noqa: E402

E.bootstrap_fork_path()
from diffdagger.main_pipeline.sim_bridge import evaluate_heldout  # noqa: E402

ENV = "StackCube-v1"
ROUNDS = [56, 58, 60]
N_EVAL = 400
EVAL_NUM_ENVS = 20
ARMS = ["diff_dagger", "p4_top3"]
SEEDS = [1, 2]


def ckpt(seed, arm, rnd):
    d = E.SUITE_ROOT / "results" / "StackCube-v1" / f"run_{seed}" / arm / "results" / "checkpoints" / f"round_{rnd:03d}"
    for name in ("2.pth", "1.pth", "0.pth"):   # 2.pth = 1.0-fraction (final) checkpoint
        if (d / name).is_file():
            return d / name
    return None


def ci95(p, n):
    return 1.96 * math.sqrt(max(p, 1e-9) * (1 - max(p, 1e-9)) / n)


def main() -> int:
    cfg = MS.load_cfg(ENV)
    np_path = str(cfg.normalizers_path)
    if not os.path.isabs(np_path):
        cfg.normalizers_path = str(E.SUITE_ROOT / np_path)
    E.register_envs()
    eval_env = MS.make_eval_env(cfg, num_envs=EVAL_NUM_ENVS)
    make_policy = PF.policy_factory(cfg)
    ah, mes = cfg.action_horizon, cfg.env.max_episode_steps
    print(f"[reeval] N_EVAL={N_EVAL} (95% band ±{ci95(0.6,N_EVAL):.3f} at SR~0.6) rounds={ROUNDS}", flush=True)

    results = {}   # (seed,arm) -> {round: sr}
    for s in SEEDS:
        for arm in ARMS:
            results[(s, arm)] = {}
            for rnd in ROUNDS:
                ck = ckpt(s, arm, rnd)
                if ck is None:
                    print(f"[reeval] s{s} {arm} r{rnd}: NO CHECKPOINT", flush=True); continue
                pol = make_policy(); pol.load(str(ck)); pol.to(cfg.device); pol.reset()
                sr = float(evaluate_heldout(pol, eval_env, 7777, N_EVAL, ah, mes))
                results[(s, arm)][rnd] = sr
                print(f"[reeval] s{s} {arm:<11} r{rnd}: SR={sr:.3f} ±{ci95(sr,N_EVAL):.3f}", flush=True)
                del pol
                import gc, torch
                gc.collect();  torch.cuda.is_available() and torch.cuda.empty_cache()

    print("\n[reeval] ===== ROBUST FINAL-REGION SR (mean of rounds " + str(ROUNDS) + ", 400 eps) =====", flush=True)
    agg = {a: [] for a in ARMS}
    for s in SEEDS:
        line = f"  seed {s}: "
        for arm in ARMS:
            vs = list(results[(s, arm)].values())
            m = sum(vs) / len(vs) if vs else float("nan")
            agg[arm].append(m)
            line += f"{arm}={m:.3f}  "
        d = (results.get((s, 'p4_top3'), {}) and results.get((s, 'diff_dagger'), {}))
        hy = sum(results[(s,'p4_top3')].values())/max(1,len(results[(s,'p4_top3')]))
        dd = sum(results[(s,'diff_dagger')].values())/max(1,len(results[(s,'diff_dagger')]))
        line += f" -> hybrid-dd = {hy-dd:+.3f}"
        print(line, flush=True)
    print("  MEAN across seeds: " + "  ".join(f"{a}={sum(agg[a])/len(agg[a]):.3f}" for a in ARMS)
          + f"   -> hybrid-dd = {sum(agg['p4_top3'])/len(agg['p4_top3'])-sum(agg['diff_dagger'])/len(agg['diff_dagger']):+.3f}", flush=True)
    print("[reeval] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
