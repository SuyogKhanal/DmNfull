"""P4 BATCH mode — LLM prescribes N layouts per round, capped to remaining
budget. Defers all round-loop logic to :mod:`_p4_common`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from ._p4_common import run_loop


def run(
    run_id: int,
    method_root: Path,
    shared_demo_dir: Path,
    shared_ckpt_dir: Path,
    correction_yaml: Path,
    heldout_yaml: Path,
    config: Dict,
) -> Dict:
    return run_loop(
        mode="batch",
        run_id=run_id,
        method_root=method_root,
        shared_demo_dir=shared_demo_dir,
        shared_ckpt_dir=shared_ckpt_dir,
        correction_yaml=correction_yaml,
        heldout_yaml=heldout_yaml,
        config=config,
    )
