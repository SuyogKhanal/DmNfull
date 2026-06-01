"""Gaussian-MLP backbone for the low-dim locomotion envs (policy_backbone_guide
§3/§10). Deterministic MSE-BC MLP (the guide's GaussianMLP using the mean at
inference). Single-policy variant (SafeDAgger/Stagger/DropoutDAgger/loco-P4) and
an ensemble variant (EnsembleDAgger/ThriftyDAgger). Reuses the validated
``model.MLPPolicy`` + ``selection.uncertainty`` machinery behind the Policy API.
"""
from __future__ import annotations

import numpy as np
import torch

from ..envs import env_setup as E
from ..model import MLPPolicy
from ..selection.uncertainty import Ensemble, mc_dropout_samples
from ..trainer.finetune_replay import train_bc
from .base import Policy


class MLPSingle(Policy):
    obs_kind = "state"

    def __init__(self, obs_dim, act_dim, act_low, act_high, *, hidden, dropout,
                 lr, batch_size, weight_decay, seed, device):
        torch.manual_seed(int(seed))
        self.net = MLPPolicy(obs_dim, act_dim, hidden=hidden, dropout=float(dropout),
                             act_low=act_low, act_high=act_high).to(device)
        self.device = device
        self.act_dim = int(act_dim)
        self.feature_dim = int(obs_dim)
        self._hp = dict(lr=lr, batch_size=batch_size, weight_decay=weight_decay, seed=seed)

    def features(self, obs_raw, env) -> np.ndarray:
        return E.flatten_obs(obs_raw)

    def act(self, feat) -> np.ndarray:
        return self.net.act(np.asarray(feat, np.float32), self.device)

    def samples(self, feat, k) -> np.ndarray:
        return mc_dropout_samples(self.net, np.asarray(feat, np.float32), int(k), self.device)

    def fit(self, feats, acts, *, epochs, warmstart=True) -> float:
        return train_bc(self.net, feats, acts, epochs=int(epochs), lr=self._hp["lr"],
                        batch_size=self._hp["batch_size"], weight_decay=self._hp["weight_decay"],
                        device=self.device, seed=self._hp["seed"])


class MLPEnsembleP(Policy):
    obs_kind = "state"

    def __init__(self, M, obs_dim, act_dim, act_low, act_high, *, hidden,
                 lr, batch_size, weight_decay, seed, device):
        self.ens = Ensemble(int(M), obs_dim, act_dim, hidden=hidden, act_low=act_low,
                            act_high=act_high, base_seed=int(seed), device=device)
        self.device = device
        self.act_dim = int(act_dim)
        self.feature_dim = int(obs_dim)
        self._hp = dict(lr=lr, batch_size=batch_size, weight_decay=weight_decay, seed=seed)

    def features(self, obs_raw, env) -> np.ndarray:
        return E.flatten_obs(obs_raw)

    def act(self, feat) -> np.ndarray:
        return self.ens.mean_action(np.asarray(feat, np.float32), self.device)

    def samples(self, feat, k) -> np.ndarray:
        return self.ens.member_actions(np.asarray(feat, np.float32), self.device)

    def fit(self, feats, acts, *, epochs, warmstart=True) -> float:
        losses = self.ens.train_all(feats, acts, base_seed=self._hp["seed"], epochs=int(epochs),
                                    lr=self._hp["lr"], batch_size=self._hp["batch_size"],
                                    weight_decay=self._hp["weight_decay"])
        return float(np.mean(losses)) if losses else 0.0
