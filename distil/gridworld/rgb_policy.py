"""RGB-image policy for the CNN_pathway pool-sweep stack.

Verbatim copy of the dmnpol/models/rgb_cnn_policy.py reference class —
takes a (B, 3, H, W) RGB tensor in and emits 4 action logits via a
CNN feature extractor + adaptive pool + MLP head. AdaptiveAvgPool2d(1)
makes the model spatial-size-invariant so a network trained on 5x5
(80x80 px) images runs unchanged on 6x6 / 7x7 inputs as long as
cell_size stays constant across grid sizes.

Lives in CNN_pathway/ as a separate module from the existing
CNN_pathway/model.py (CNNMLPPolicy, which is the *other* CNN baseline
that takes RGB + a 14-d state vector). The two coexist; this file is
strictly additive.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

DEFAULT_CELL_SIZE = 16


def _gn(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = num_groups
    while channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, channels)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            _gn(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            _gn(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RGBCNNPolicy(nn.Module):
    obs_type = "rgb"

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256),
        num_actions: int = 4,
        head_hidden: int = 128,
        dropout: float = 0.0,
        cell_size: int = DEFAULT_CELL_SIZE,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.cell_size = int(cell_size)
        self.feat_dim = int(channels[-1])

        blocks = []
        prev = in_channels
        for c in channels:
            blocks.append(_ConvBlock(prev, c))
            blocks.append(nn.MaxPool2d(kernel_size=2, ceil_mode=True))
            prev = c
        self.encoder = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(prev, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_actions),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Visual-only feature vector (B, channels[-1]). The hybrid model
        in Equivariant_pathway/equivariant_CNN_hybrid/model.py uses this
        same code path: encoder + pool, NO head."""
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encode(x)
        return self.head(feat)
