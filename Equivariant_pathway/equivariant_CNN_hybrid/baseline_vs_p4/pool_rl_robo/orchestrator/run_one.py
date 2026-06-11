"""Per-env driver for the 7-method ManiSkill comparison.

Runs all 7 methods for ONE ``--env`` against a shared initial diffusion policy,
writing ``results/{ENV}/run_{id}/{method}/results/learning_curve.json`` and
``run_summary.json``. Needs a vLLM server reachable via OPENAI_BASE_URL for
p4_top3 (launched by run_pool_rl_robo.sh).

    python -m ...pool_rl_robo.orchestrator.run_one --env PushT-v1 --methods all

METHOD_SPEC is the dispatch table (lockstep §). No fixed/rotate axis; no
correction_n — every method adds EXACTLY ONE demo per round.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import _common

SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SUITE_ROOT / "config.yaml"

# (kind, selector_or_k). diff_dagger + the 5 IIL methods are all "baseline" kind;
# p4_top3 is the LLM method with top-k=3 failure analysis.
METHOD_SPEC = {
    "p4_top3":         ("p4",       3),
    "diff_dagger":     ("baseline", "diff_dagger"),
    "safe_dagger":     ("baseline", "safe_dagger"),
    "dropout_dagger":  ("baseline", "dropout_dagger"),
    "ensemble_dagger": ("baseline", "ensemble_dagger"),
    "thrifty_dagger":  ("baseline", "thrifty_dagger"),
    "stagger":         ("baseline", "stagger"),
    # Industry variant (not in the research config.yaml methods): P4-LLM selection
    # on top of SafeDAgger's detect + on-policy-correct. Run via config_p4select.yaml.
    "p4_select":       ("p4_select", "safe"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    _common.add_common_args(ap)
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    return _common.run_suite(args, METHOD_SPEC, Path(args.config), "run_summary.json")


if __name__ == "__main__":
    sys.exit(main())
