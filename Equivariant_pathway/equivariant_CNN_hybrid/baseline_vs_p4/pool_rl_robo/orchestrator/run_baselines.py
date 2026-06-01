"""Per-env IIL-baselines driver (continuous-control analogue of
pool_x_selector's ``orchestrator/run_baselines.py``). Runs the 5 DAgger-family
baselines (SafeDAgger*, DropoutDAgger, EnsembleDAgger, ThriftyDAgger, Stagger)
for ONE ``--env`` — NO vLLM needed — and writes
``results/{ENV}/run_{id}/run_summary_baselines.json`` (P4's run_summary.json is
never touched).

    python -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.orchestrator.run_baselines --env HalfCheetah-v4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import _common

SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SUITE_ROOT / "config_baselines.yaml"
DEFAULT_METHODS = ["safe_dagger", "dropout_dagger", "ensemble_dagger",
                   "thrifty_dagger", "stagger"]


def main() -> int:
    ap = argparse.ArgumentParser()
    _common.add_common_args(ap)
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    args.no_llm = True  # baselines never use the LLM
    return _common.run_main(args, DEFAULT_METHODS, "run_summary_baselines.json")


if __name__ == "__main__":
    sys.exit(main())
