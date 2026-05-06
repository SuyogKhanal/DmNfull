"""p4m-equivariant policy (escnn-based, optional dependency).

This is a verbatim port of dmnpol/models/equivariant_policy.py — same
architecture, same hyperparameters, same gather operation. The point of
copying it (rather than depending on dmnpol as a package) is so the
DmNfull active loop can run end-to-end without the dmnpol checkout being
on PYTHONPATH.

Architecture
------------
Three stacked equivariant 3x3 conv blocks (no pooling, dilation=2 in the
last one to keep the receptive field large enough on 5x5 / 6x6 / 8x8 grids
without collapsing the feature map). The head is a 1x1 Conv2d projecting
to `num_actions` per-cell logits. The forward pass gathers the per-cell
logits at the agent's cell using the agent channel as a one-hot mask.

The model exposes both:
  - forward(x)         -> (B, 4) action logits at the agent cell
  - per_cell_logits(x) -> (B, 4, H, W) full per-cell logit map (for viz /
                         the rollout's Q-map saving in rollout_test.py).

Input shape: (B, 5, H, W) using the channel layout from encoder.py
(0=agent, 1=goal, 2=fire, 3=wall, 4=free).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _require_escnn():
    try:
        from escnn import gspaces, nn as enn
        return gspaces, enn
    except ImportError as e:
        raise ImportError(
            "escnn is required for EquivariantUNetPolicy. "
            "Install with `pip install escnn`."
        ) from e


class EquivariantUNetPolicy(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        channels: Sequence[int] = (16, 32, 64),
        num_actions: int = 4,
        agent_channel: int = 0,
    ):
        super().__init__()
        gspaces, enn = _require_escnn()
        self.gspace = gspaces.flipRot2dOnR2(N=4)
        self.in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])
        self.agent_channel = agent_channel
        self.num_actions = num_actions
        self.in_channels = in_channels
        self.channels = tuple(channels)

        c1, c2, c3 = channels
        t1 = enn.FieldType(self.gspace, c1 * [self.gspace.regular_repr])
        t2 = enn.FieldType(self.gspace, c2 * [self.gspace.regular_repr])
        t3 = enn.FieldType(self.gspace, c3 * [self.gspace.regular_repr])

        self.block1 = enn.SequentialModule(
            enn.R2Conv(self.in_type, t1, kernel_size=3, padding=1),
            enn.InnerBatchNorm(t1),
            enn.ReLU(t1, inplace=True),
        )
        self.block2 = enn.SequentialModule(
            enn.R2Conv(t1, t2, kernel_size=3, padding=1),
            enn.InnerBatchNorm(t2),
            enn.ReLU(t2, inplace=True),
        )
        self.block3 = enn.SequentialModule(
            enn.R2Conv(t2, t3, kernel_size=3, padding=2, dilation=2),
            enn.InnerBatchNorm(t3),
            enn.ReLU(t3, inplace=True),
        )

        d4_size = self.gspace.regular_repr.size
        self.head = nn.Conv2d(c3 * d4_size, num_actions, kernel_size=1)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        from escnn import nn as enn
        gx = enn.GeometricTensor(x, self.in_type)
        s1 = self.block1(gx)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        return self.head(s3.tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        per_cell_logits = self.forward_full(x)
        agent_mask = x[:, self.agent_channel : self.agent_channel + 1, :, :]
        return (per_cell_logits * agent_mask).sum(dim=(-1, -2))

    def per_cell_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_full(x)

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> int:
        self.eval()
        logits = self.forward(x)
        return int(logits.argmax(dim=-1).item())
