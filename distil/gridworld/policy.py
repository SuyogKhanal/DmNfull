"""GridWorld classifier policy + self-uncertainty (Diff-DAgger analog for a classifier).

Policy: the p4m-equivariant CNN (EquivariantUNetPolicy) on the 5-channel semantic grid
encoding (state modality). Trained from scratch each round with the flagship multi-label
BCE objective over the BFS optimal-action mask (preserves the two-route multimodality;
pos-weight-balanced so "predict nothing" can't win) — matching the repo's
Equivariant_pathway/train.py.

Self-uncertainty l_t (paper Eq 2, "loss at the learner's own executed action"): for a
classifier we use the PREDICTIVE ENTROPY H(softmax(logits)) — label-free, so the SAME
functional is used at threshold-calibration (over training states) and at rollout
screening (Diff-DAgger's diffusion loss is likewise self-supervised / label-free). The
OOD threshold is the alpha-quantile of H over the training states; a rollout step is
"uncertain" when H exceeds it. K-patience query rule identical to the robot side.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import encode_state
from .encoder_rgb import encode_state as encode_rgb
from .equivariant import EquivariantUNetPolicy
from .expert import AStarExpert, build_grid, compute_distance_map, optimal_action_mask
from .rgb_policy import RGBCNNPolicy


class EmpiricalCDF:
    """alpha-quantile via nearest-rank over a stored sorted sample (== robot side)."""

    def __init__(self, values):
        self.sorted = np.sort(np.asarray(values, dtype=np.float64)) if len(values) else np.array([0.0])

    def quantile(self, q: float) -> float:
        n = len(self.sorted)
        return float(self.sorted[min(int(n * q), n - 1)])


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Predictive entropy H(softmax(logits)) per row, shape (B,)."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


class GridWorldPolicy:
    """Wraps the equivariant classifier with the DISTIL uncertainty/query contract."""

    def __init__(self, modality: str = "state", grid_size: int = 5,
                 alpha: float = 0.99, patience_window: int = 2, patience: int = 2,
                 channels: Sequence[int] = (16, 32, 64), device: str = "cpu"):
        assert modality in ("state", "image"), modality
        self.modality = modality
        self.grid_size = grid_size
        self.alpha = alpha
        self.patience_window = patience_window
        self.patience = patience
        self.device = device
        self.channels = tuple(channels)
        self.net = self._new_net()
        self.loss_threshold = float("inf")
        self._violations = deque(maxlen=patience_window)

    def _new_net(self) -> nn.Module:
        """state -> p4m-equivariant CNN over the 5-channel semantic grid.
        image -> plain RGB CNN over the 80x80 bird-eye raster (same 4-way logit head,
        so every downstream contract — entropy, threshold, BCE mask — is unchanged)."""
        if self.modality == "image":
            return RGBCNNPolicy(in_channels=3, num_actions=4).to(self.device)
        return EquivariantUNetPolicy(in_channels=5, channels=self.channels,
                                     num_actions=4).to(self.device)

    # ── scene -> policy input tensor ──────────────────────────────────────────
    def _encode(self, grid, agent, goal, fires) -> np.ndarray:
        enc = encode_rgb if self.modality == "image" else encode_state
        return enc(np.asarray(grid, dtype=np.int32), tuple(agent), tuple(goal),
                   [tuple(f) for f in fires])

    def _batch(self, scenes: List[Tuple]) -> torch.Tensor:
        arrs = [self._encode(*s) for s in scenes]
        return torch.from_numpy(np.stack(arrs)).float().to(self.device)

    @torch.no_grad()
    def logits(self, grid, agent, goal, fires) -> torch.Tensor:
        self.net.eval()
        x = self._batch([(grid, agent, goal, fires)])
        return self.net(x)[0]

    @torch.no_grad()
    def act(self, grid, agent, goal, fires) -> int:
        return int(self.logits(grid, agent, goal, fires).argmax().item())

    @torch.no_grad()
    def uncertainty(self, grid, agent, goal, fires) -> float:
        self.net.eval()
        x = self._batch([(grid, agent, goal, fires)])
        return float(_entropy(self.net(x))[0].item())

    # ── query rule (K-patience over threshold crossings) ─────────────────────
    def reset_query(self):
        self._violations = deque(maxlen=self.patience_window)

    def register_and_query(self, u: float) -> bool:
        self._violations.append(u > self.loss_threshold)
        return sum(self._violations) >= self.patience

    # ── threshold calibration over training states ───────────────────────────
    @torch.no_grad()
    def set_threshold_from_dataset(self, scenes: List[Tuple], max_points: int = 50000) -> np.ndarray:
        self.net.eval()
        if not scenes:
            self.loss_threshold = float("inf")
            return np.array([])
        scenes = scenes[:max_points]
        us = []
        for i in range(0, len(scenes), 256):
            x = self._batch(scenes[i:i + 256])
            us.append(_entropy(self.net(x)).cpu().numpy())
        us = np.concatenate(us)
        self.loss_threshold = EmpiricalCDF(us).quantile(self.alpha)
        return us

    # ── train-from-scratch on the aggregated demos (multi-label BCE / mask) ──
    def train_from_scratch(self, demos: List[Dict], *, steps: int, batch_size: int = 64,
                           lr: float = 3e-4, weight_decay: float = 1e-4, seed: int = 0,
                           log_fn=print):
        """Rebuild + train the classifier from scratch. Each demo = {grid,goal,fires,cells};
        one sample per path cell: input = encode(grid,cell,goal,fires), target = BFS mask."""
        rng = np.random.default_rng(seed)
        samples = self._build_samples(demos)            # list of (scene, mask)
        if not samples:
            log_fn("  [gw-train] no samples; skipping")
            return []
        self.net = self._new_net()
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
        self.net.train()
        n = len(samples)
        losses = []
        for step in range(steps):
            idx = rng.integers(0, n, size=min(batch_size, n))
            batch = [samples[i] for i in idx]
            x = self._batch([b[0] for b in batch])
            masks = torch.from_numpy(np.stack([b[1] for b in batch])).float().to(self.device)
            logits = self.net(x)
            # pos-weight balancing: (num_neg / num_pos) per row, applied to positives.
            n_pos = masks.sum(dim=1, keepdim=True).clamp(min=1.0)
            pos_w = (4.0 - n_pos) / n_pos
            loss = F.binary_cross_entropy_with_logits(
                logits, masks, weight=torch.where(masks > 0, pos_w, torch.ones_like(masks)))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(float(loss.item()))
        log_fn(f"  [gw-train] {n} samples, {steps} steps, final_loss={losses[-1]:.4f}")
        # calibrate the OOD threshold over the training scenes (== robot cadence)
        scenes = [s[0] for s in samples]
        us = self.set_threshold_from_dataset(scenes)
        log_fn(f"  [gw-calibrate] entropy threshold(alpha={self.alpha})={self.loss_threshold:.4f} "
               f"(median={np.median(us):.4f})" if len(us) else "  [gw-calibrate] no scenes")
        return losses

    @staticmethod
    def _build_samples(demos: List[Dict]) -> List[Tuple]:
        samples = []
        for d in demos:
            grid = build_grid(len(d["grid"]), d["fires"], d["goal"],
                              wall_template=np.asarray(d["grid"], dtype=np.int32))
            dmap = compute_distance_map(grid, tuple(d["goal"]))
            for cell in d["cells"]:
                cell = tuple(cell)
                if cell == tuple(d["goal"]):
                    continue
                mask = optimal_action_mask(dmap, cell, grid)
                if mask.sum() == 0:              # unreachable/goal cell — no supervision
                    continue
                samples.append(((d["grid"], cell, d["goal"], d["fires"]), mask))
        return samples

    def state_dict(self):
        return self.net.state_dict()
