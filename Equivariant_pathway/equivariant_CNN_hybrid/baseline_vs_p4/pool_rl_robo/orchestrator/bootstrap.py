"""Per-(env, run) bootstrap (continuous-control analogue of pool_x_selector's
``orchestrator/bootstrap.py``).

The maze bootstrap materialises shared init demos + an initial checkpoint via an
upstream subprocess. Here there is no upstream cache: every method's round-0
state is produced deterministically inside ``run_dagger`` (seed -> identical
seed-demo collection + initial BC across methods, so they start from the same
policy). Bootstrap therefore just (1) ensures the env's expert is present in the
HF cache (so offline compute-node jobs can load it) and (2) creates the per-run
workspace dirs. Idempotent.
"""
from __future__ import annotations

import time

from ..envs import experts as X
from .workspace import RunWorkspace, workspace_for


def _info(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [bootstrap] {msg}", flush=True)


def ensure_expert_cached(env_id: str) -> None:
    """Trigger the HF download (no-op if already cached). Run on a login node /
    before HF_HUB_OFFLINE jobs."""
    spec = X.EXPERTS[env_id]
    path = X._download(spec.repo, spec.filename)
    _info(f"{env_id} expert cached: {path}")


def ensure_bootstrap(env_id: str, run_id: int = 0,
                     download_expert: bool = False) -> RunWorkspace:
    if download_expert:
        ensure_expert_cached(env_id)
    ws = workspace_for(env_id, run_id)
    _info(f"workspace ready: {ws.root}")
    return ws
