"""Per-env P4-LLM driver (continuous-control analogue of pool_x_selector's
``orchestrator/run_one.py``). Runs the P4-LLM method for ONE ``--env`` (needs a
vLLM server reachable via OPENAI_BASE_URL — launched by submit_one_qwen.sh) and
writes ``results/{ENV}/run_{id}/run_summary.json``.

    python -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.orchestrator.run_one --env HalfCheetah-v4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import _common

SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SUITE_ROOT / "config.yaml"
DEFAULT_METHODS = ["p4_llm"]


def main() -> int:
    ap = argparse.ArgumentParser()
    _common.add_common_args(ap)
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    return _common.run_main(args, DEFAULT_METHODS, "run_summary.json")


if __name__ == "__main__":
    sys.exit(main())
