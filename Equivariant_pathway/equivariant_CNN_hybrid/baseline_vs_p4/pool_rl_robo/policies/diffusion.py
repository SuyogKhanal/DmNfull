"""Diffusion-Policy backbone for the Fetch manipulation envs (policy_backbone_guide
§5/§6). Single-step DDPM (action chunking deferred) over the env's continuous
action, conditioned on a FROZEN-R3M image embedding + the goal. Implemented
without `diffusers` (the env's huggingface_hub pin breaks it), so the DDPM
schedule + conditional denoiser are written from scratch (standard).

Uncertainty for Dropout/Ensemble/Thrifty = variance across k independent denoise
samples (the user-chosen sample-variance route): ``samples(feat, k)`` returns k
draws; the IIL rules read their variance/ball just like MLP samples.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..envs import env_setup as E
from .base import Policy
from .r3m_encoder import R3MEncoder


def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(1, half - 1))
    a = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class _CondDenoiser(nn.Module):
    """Predicts the noise on a (noisy) action given the diffusion timestep and the
    conditioning features (R3M emb + goal)."""

    def __init__(self, act_dim: int, cond_dim: int, hidden: int = 512, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.cond = nn.Sequential(nn.Linear(cond_dim, 256), nn.SiLU(), nn.Linear(256, 256))
        self.net = nn.Sequential(
            nn.Linear(act_dim + 256 + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, noisy_act, t, cond_feat):
        c = self.cond(cond_feat)
        te = _sinusoidal(t, self.time_dim)
        return self.net(torch.cat([noisy_act, c, te], dim=-1))


class _DDPM:
    def __init__(self, T: int, device: torch.device):
        self.T = int(T)
        betas = torch.linspace(1e-4, 0.02, self.T, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.abar = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, noise, t):
        ab = self.abar[t].unsqueeze(-1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def step(self, pred_noise, t: int, xt):
        beta = self.betas[t]
        ab = self.abar[t]
        ab_prev = self.abar[t - 1] if t > 0 else torch.ones_like(ab)
        mean = (1.0 / torch.sqrt(self.alphas[t])) * (xt - (beta / torch.sqrt(1.0 - ab)) * pred_noise)
        if t > 0:
            var = beta * (1.0 - ab_prev) / (1.0 - ab)
            return mean + torch.sqrt(var) * torch.randn_like(xt)
        return mean


class DiffusionPolicy(Policy):
    obs_kind = "image"

    def __init__(self, act_dim, act_low, act_high, goal_dim, *, device, r3m_encoder: R3MEncoder,
                 hidden=512, T=100, lr=1e-4, batch_size=256, weight_decay=1e-6, seed=0):
        torch.manual_seed(int(seed))
        self.device = device
        self.act_dim = int(act_dim)
        self.r3m = r3m_encoder
        self.goal_dim = int(goal_dim)
        self.feature_dim = int(r3m_encoder.embed_dim + goal_dim)
        self.low = torch.tensor(np.asarray(act_low, np.float32), device=device)
        self.high = torch.tensor(np.asarray(act_high, np.float32), device=device)
        self.denoiser = _CondDenoiser(self.act_dim, self.feature_dim, hidden=hidden).to(device)
        self.ddpm = _DDPM(T, device)
        self._hp = dict(lr=lr, batch_size=batch_size, weight_decay=weight_decay, seed=seed)

    # --- features: R3M(image) ++ goal ---------------------------------------
    def _goal(self, obs_raw) -> np.ndarray:
        if isinstance(obs_raw, dict) and "desired_goal" in obs_raw:
            return np.asarray(obs_raw["desired_goal"], np.float32).ravel()
        return np.zeros(self.goal_dim, np.float32)

    def features(self, obs_raw, env) -> np.ndarray:
        img = env.render()
        emb = self.r3m.encode(img)
        return np.concatenate([emb, self._goal(obs_raw)]).astype(np.float32)

    def features_batch(self, obs_list, env):  # not used (rendering is per-step)
        return np.stack([self.features(o, env) for o in obs_list]) if obs_list else np.zeros((0, self.feature_dim), np.float32)

    # --- sampling / acting ---------------------------------------------------
    def _scale(self, x):  # tanh-free: clamp normalized action to bounds
        return torch.clamp(x, self.low, self.high)

    @torch.no_grad()
    def _denoise(self, feat, n: int) -> np.ndarray:
        cond = torch.from_numpy(np.asarray(feat, np.float32)).to(self.device).unsqueeze(0).repeat(n, 1)
        x = torch.randn(n, self.act_dim, device=self.device)
        for t in range(self.ddpm.T - 1, -1, -1):
            tt = torch.full((n,), t, device=self.device, dtype=torch.long)
            x = self.ddpm.step(self.denoiser(x, tt, cond), t, x)
        return self._scale(x).cpu().numpy().astype(np.float32)

    def act(self, feat) -> np.ndarray:
        return self._denoise(feat, 1)[0]

    def samples(self, feat, k) -> np.ndarray:
        return self._denoise(feat, int(k))

    # --- training (DDPM noise-prediction MSE) --------------------------------
    def fit(self, feats, acts, *, epochs, warmstart=True) -> float:
        X = torch.from_numpy(np.asarray(feats, np.float32)).to(self.device)
        Y = torch.from_numpy(np.clip(np.asarray(acts, np.float32),
                                     self.low.cpu().numpy(), self.high.cpu().numpy())).to(self.device)
        n = X.shape[0]
        if n == 0:
            return 0.0
        torch.manual_seed(self._hp["seed"])
        opt = torch.optim.AdamW(self.denoiser.parameters(), lr=self._hp["lr"],
                                weight_decay=self._hp["weight_decay"])
        bs = max(1, int(self._hp["batch_size"]))
        self.denoiser.train()
        last = 0.0
        for _ep in range(int(epochs)):
            perm = torch.randperm(n, device=self.device)
            ep = 0.0
            nb = 0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                a = Y[idx]
                t = torch.randint(0, self.ddpm.T, (a.shape[0],), device=self.device)
                noise = torch.randn_like(a)
                noisy = self.ddpm.add_noise(a, noise, t)
                loss = F.mse_loss(self.denoiser(noisy, t, X[idx]), noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep += float(loss.item())
                nb += 1
            last = ep / max(1, nb)
        self.denoiser.eval()
        return last
