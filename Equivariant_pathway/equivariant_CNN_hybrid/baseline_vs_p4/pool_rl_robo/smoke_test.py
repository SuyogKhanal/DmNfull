#!/usr/bin/env python
"""PHASE 1 — smoke test (the gate).

For each of the 5 environments: download the pretrained expert from HuggingFace,
load it, run 3 episodes, print mean reward + ✓. All 5 must print ✓ before Phase
2. Failures print ✗ + the error; every env is attempted; exit nonzero if any
fail.

    python -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.smoke_test
    python smoke_test.py                # from inside pool_rl_robo/
    python smoke_test.py --predownload  # just fetch the 5 zips into the HF cache
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Make the suite importable whether run as a module or directly.
_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.envs import (  # noqa: E402
    env_setup as E, experts as X,
)


def predownload() -> None:
    for name, spec in X.EXPERTS.items():
        try:
            print(f"downloaded {name}: {X._download(spec.repo, spec.filename)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED download {name}: {type(e).__name__}: {e}", flush=True)


def run(n_episodes: int = 3) -> bool:
    E.register_robotics()
    total = len(X.EXPERTS)
    ok = 0
    for name in X.EXPERTS:
        try:
            resolved, sub = E.resolve_env_id(name)
            note = f"  (resolved {name} -> {resolved})" if sub else ""
            model = X.load_expert(name, resolved_env_id=resolved, device="cpu")
            env = E.make_env(resolved)
            rewards, succ = [], []
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=1000 + ep)
                done = False
                total_r = 0.0
                term = trunc = False
                last_info = {}
                while not done:
                    obs, r, term, trunc, last_info = env.step(X.expert_action(model, obs))
                    total_r += float(r)
                    done = term or trunc
                rewards.append(total_r)
                succ.append(E.episode_success(name, last_info, term, trunc))
            env.close()
            sr = (sum(succ) / len(succ)) if E.is_goal_env(name) else None
            srtxt = f"  success_rate = {sr:.2f}" if sr is not None else ""
            print(f"{name}: mean reward = {sum(rewards) / len(rewards):.2f}{srtxt}{note}  ✓", flush=True)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAILED ✗  {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"\n{ok}/{total} experts loaded and played successfully.", flush=True)
    return ok == total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predownload", action="store_true")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()
    if args.predownload:
        predownload()
        sys.exit(0)
    sys.exit(0 if run(args.episodes) else 1)
