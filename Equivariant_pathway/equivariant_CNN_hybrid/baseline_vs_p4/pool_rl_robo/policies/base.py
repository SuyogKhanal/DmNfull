"""The unified novice-policy interface used by the DAgger loop + the 5 IIL rules.

The loop never sees backbone specifics — it works in terms of FEATURES:
  feat = policy.features(obs_raw, env)   # state: flatten; image: R3M(render)+goal
  a    = policy.act(feat)                # deterministic action (rollout / eval)
  S    = policy.samples(feat, k)         # (k, act_dim) stochastic samples -> uncertainty
  loss = policy.fit(feats, acts, ...)    # behavior-cloning update on aggregated D

``samples`` is the single uncertainty source for every backbone:
  MLP single -> MC-dropout; MLP ensemble -> the M members; Diffusion -> k denoise draws.
The expert always acts on the RAW state obs (it is a fixed state-based SB3 policy).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    obs_kind: str = "state"   # "state" | "image"
    act_dim: int = 0
    feature_dim: int = 0

    @abstractmethod
    def features(self, obs_raw, env) -> np.ndarray:
        """Map a raw env observation to the feature vector this backbone consumes."""

    @abstractmethod
    def act(self, feat: np.ndarray) -> np.ndarray:
        """Deterministic action for rollout/eval. Shape (act_dim,)."""

    @abstractmethod
    def samples(self, feat: np.ndarray, k: int) -> np.ndarray:
        """k stochastic action samples for the uncertainty rules. Shape (k, act_dim)."""

    @abstractmethod
    def fit(self, feats: np.ndarray, acts: np.ndarray, *, epochs: int,
            warmstart: bool = True) -> float:
        """Behavior-cloning fit on (feats -> acts). Returns final loss."""

    # convenience: batch-encode a list of raw obs (override for speed)
    def features_batch(self, obs_list, env) -> np.ndarray:
        return np.stack([self.features(o, env) for o in obs_list]) if obs_list else np.zeros((0, self.feature_dim), np.float32)
