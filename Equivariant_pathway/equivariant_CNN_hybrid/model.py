"""EquivariantCNNHybridPolicy — RGB CNN encoder + p4m-equivariant policy.

Architecture
------------
Two parallel feature paths feed a small fusion head:

  RGB branch (NOT equivariant):
      rgb_obs (B, 3, H_px, W_px)
        -> RGBCNNPolicy.encoder      = stacked _ConvBlock+MaxPool
        -> RGBCNNPolicy.pool          = AdaptiveAvgPool2d(1)
        -> flatten                    -> (B, cnn_feat_dim)            # 256 by default

  Equivariant branch (p4m-equivariant):
      grid_obs (B, in_channels, H, W)
        -> EquivariantUNetPolicy.forward_full -> per-cell logits (B, num_actions, H, W)
        -> gather at agent cell (using the agent channel as a mask)
                                       -> (B, num_actions)

  Fusion:
      concat(cnn_feat, gathered_eq_logits)  -> (B, cnn_feat_dim + num_actions)
        -> MLP head (Linear -> GELU -> Dropout -> Linear)
        -> (B, num_actions)

Both sub-modules' weights are independently trainable. The
RGBCNNPolicy.head MLP is intentionally NOT instantiated — only the
visual encoder + pool feed the fusion head, exactly as the user prompt
specifies. This keeps the parameter count down and lets the equivariant
branch retain its 8-fold symmetry advantage on its own input while the
CNN branch contributes a non-equivariant visual context vector.

The equivariant gather uses the same trick as
EquivariantUNetPolicy.forward — multiply the per-cell logit map by the
agent channel and sum spatial — so the gathered logits are still
informed by the equivariant convolutions; we just route them through a
fusion head instead of returning them directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

# Make sure both CNN_pathway/ and Equivariant_pathway/ resolve when this
# module is imported by the hybrid sweep code (which lives a few levels
# deep). Adding REPO_ROOT to sys.path before the relative imports keeps
# the hybrid drop-in even when launched from arbitrary working dirs.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CNN_pathway.rgb_cnn_policy import RGBCNNPolicy
from Equivariant_pathway.model import EquivariantUNetPolicy


class EquivariantCNNHybridPolicy(nn.Module):
    """RGB CNN encoder + EquivariantUNetPolicy + concat fusion head.

    Parameters
    ----------
    rgb_in_channels : int
        Channel count of the RGB observation. 3 for standard RGB.
    rgb_channels : Sequence[int]
        CNN trunk channel widths fed into RGBCNNPolicy. Last entry is
        the visual feature width (default 256).
    grid_in_channels : int
        Channel count of the equivariant input encoding (5 for the
        DmNfull semantic encoder).
    eq_channels : Sequence[int]
        Per-block regular-rep channel counts inside the equivariant
        backbone (default (16, 32, 64) — same as
        EquivariantUNetPolicy).
    num_actions : int
        Final output dimensionality.
    head_hidden : int
        Hidden width of the fusion MLP.
    head_dropout : float
        Dropout between the fusion MLP layers.
    agent_channel : int
        Index of the agent one-hot channel in ``grid_obs`` — used as
        the gather mask, identical to ``EquivariantUNetPolicy``.
    """

    obs_type = "rgb+grid"

    def __init__(
        self,
        rgb_in_channels: int = 3,
        rgb_channels: Sequence[int] = (32, 64, 128, 256),
        grid_in_channels: int = 5,
        eq_channels: Sequence[int] = (16, 32, 64),
        num_actions: int = 4,
        head_hidden: int = 128,
        head_dropout: float = 0.0,
        agent_channel: int = 0,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.agent_channel = agent_channel

        # CNN visual encoder = encoder + adaptive pool from RGBCNNPolicy.
        # We instantiate the full RGBCNNPolicy and keep only its
        # encoder/pool — the head MLP is unused (the prompt explicitly
        # excludes it). RGBCNNPolicy.encode() exposes that exact slice
        # so we route through it directly.
        self._cnn_full = RGBCNNPolicy(
            in_channels=rgb_in_channels,
            channels=rgb_channels,
            num_actions=num_actions,        # unused; head is not called
            head_hidden=head_hidden,        # unused
            dropout=head_dropout,           # unused
        )
        self.cnn_feat_dim = int(self._cnn_full.feat_dim)

        # The equivariant branch is a vanilla EquivariantUNetPolicy. The
        # forward path uses its internals (forward_full + gather), not
        # its top-level forward, so the gathered logits remain available
        # to the fusion head without going through an extra layer.
        self.equivariant_policy = EquivariantUNetPolicy(
            in_channels=grid_in_channels,
            channels=eq_channels,
            num_actions=num_actions,
            agent_channel=agent_channel,
        )

        # Fusion MLP. Input dim is CNN feature width + equivariant
        # gathered logit count.
        fuse_in = self.cnn_feat_dim + num_actions
        self.fusion_head = nn.Sequential(
            nn.Linear(fuse_in, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_actions),
        )

    @property
    def cnn_encoder(self) -> nn.Module:
        """Public alias for the CNN visual stack used by the hybrid.

        Mirrors the attribute name promised in the user spec
        ('self.cnn_encoder'). Returns a ``nn.Sequential`` of the
        ``RGBCNNPolicy`` encoder followed by its adaptive pool.
        """
        return nn.Sequential(self._cnn_full.encoder, self._cnn_full.pool)

    # ---------------------------------------------------------------- #
    # Forward                                                            #
    # ---------------------------------------------------------------- #
    def _gather_eq_logits(self, grid_obs: torch.Tensor) -> torch.Tensor:
        """Run the equivariant branch and gather per-cell logits at the
        agent cell. Returns (B, num_actions). Mirrors the forward
        gather of EquivariantUNetPolicy verbatim so the contribution
        from this branch is identical in form to a standalone
        equivariant policy's output."""
        per_cell = self.equivariant_policy.forward_full(grid_obs)
        agent_mask = grid_obs[
            :, self.agent_channel : self.agent_channel + 1, :, :
        ]
        return (per_cell * agent_mask).sum(dim=(-1, -2))

    def forward(self, rgb_obs: torch.Tensor, grid_obs: torch.Tensor) -> torch.Tensor:
        cnn_feat = self._cnn_full.encode(rgb_obs)
        eq_logits = self._gather_eq_logits(grid_obs)
        fused = torch.cat([cnn_feat, eq_logits], dim=-1)
        return self.fusion_head(fused)

    # Convenience: separate accessors for diagnostic / training-mode
    # toggles. Both submodules are independently trainable by virtue of
    # being normal nn.Modules registered as attributes.
    def cnn_features(self, rgb_obs: torch.Tensor) -> torch.Tensor:
        return self._cnn_full.encode(rgb_obs)

    def equivariant_logits(self, grid_obs: torch.Tensor) -> torch.Tensor:
        return self._gather_eq_logits(grid_obs)
