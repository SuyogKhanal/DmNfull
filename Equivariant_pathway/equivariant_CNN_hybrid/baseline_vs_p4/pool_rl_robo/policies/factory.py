"""Diffusion-policy factory — the SHARED backbone for all 7 methods.

The diffusion policy itself comes from the fork (a V-objective CNN-UNet
DiffDAggerPolicy, instantiated from the task's Hydra config). Every method —
p4_top3, diff_dagger, and the 5 IIL baselines — uses this one architecture; only
the query / prescription rule differs. These two factories mirror exactly what
the fork's run_comparison.py uses (`make_policy` / `make_dataset`).
"""
from __future__ import annotations

from typing import Any

from ..envs import env_setup as E

E.bootstrap_fork_path()


def build_policy(cfg) -> Any:
    """A fresh, random-init diffusion policy on cfg.device (the per-arm novice)."""
    from hydra.utils import instantiate
    return instantiate(cfg.policy, _recursive_=False).float().to(cfg.device)


def make_dataset(cfg) -> Any:
    """A fresh TimeSeriesDataset with the task's normalizers loaded."""
    from hydra.utils import instantiate
    ds = instantiate(cfg.dataset, _recursive_=False)
    ds.set_normalizers_from_path(cfg.normalizers_path)
    return ds


def policy_factory(cfg):
    """Return a zero-arg callable producing fresh policies (for the harness)."""
    return lambda: build_policy(cfg)


def dataset_factory(cfg):
    return lambda: make_dataset(cfg)
