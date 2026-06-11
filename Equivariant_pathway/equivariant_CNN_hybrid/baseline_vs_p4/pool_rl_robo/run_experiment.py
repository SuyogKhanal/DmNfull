#!/usr/bin/env python
"""Convenience single-env driver (STEP 8): run all 7 methods for ONE --env.

Thin wrapper around the package's orchestrator.run_one (which holds METHOD_SPEC
and the harness). Bootstraps DmNfull onto sys.path so the dotted package imports,
then delegates — so both forms work:
    python run_experiment.py --env PushT-v1 --methods all
    python -m ...pool_rl_robo.orchestrator.run_one --env PushT-v1 --methods all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # DmNfull

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.orchestrator \
    import run_one  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_one.main())
