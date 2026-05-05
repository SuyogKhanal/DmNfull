"""CNN + MLP policy for the 5x5 maze.

Input  : an 80x80x3 bird's-eye-view image of the grid (one frame, no history)
         + the 14-d state vector exposed by MazeNavEnv.
Output : action logits for the 4 discrete maze actions {UP, DOWN, LEFT, RIGHT}.

The CNN is cell-aligned (final stride collapses each 16x16 cell to one
spatial unit), so each spatial feature corresponds to exactly one grid cell.
Stack of 4 conv blocks → 5x5x256 features → flatten → MLP head fuses with
the state vector. The MLP is wide (512) and deep (4 hidden layers) with
BatchNorm + Dropout, sized to actually fit the full training distribution
(the previous, smaller, model topped out around 89% train accuracy).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class CNNMLPPolicy(nn.Module):

    def __init__(
        self,
        img_size: int = 80,
        grid_size: int = 5,
        cell_px: int = 16,
        state_dim: int = 14,
        action_dim: int = 4,
        cnn_channels: tuple = (64, 128, 192, 256),
        mlp_hidden: tuple = (512, 512, 256, 128),
        cnn_dropout: float = 0.05,
        mlp_dropout: float = 0.20,
    ):
        super().__init__()
        assert img_size == grid_size * cell_px, (
            f"img_size ({img_size}) must equal grid_size*cell_px "
            f"({grid_size}*{cell_px})."
        )
        self.grid_size = grid_size
        self.cell_px = cell_px
        self.state_dim = state_dim
        self.action_dim = action_dim

        # CNN: 4 stacked conv blocks on the full-resolution image, with one
        # cell-aligned avg-pool at the end so each spatial output is exactly
        # one grid cell. Channels grow 64 -> 128 -> 192 -> 256.
        c1, c2, c3, c4 = cnn_channels
        self.cnn = nn.Sequential(
            _ConvBlock(3,  c1, dropout=cnn_dropout),
            _ConvBlock(c1, c2, dropout=cnn_dropout),
            _ConvBlock(c2, c3, dropout=cnn_dropout),
            _ConvBlock(c3, c4, dropout=cnn_dropout),
            nn.AvgPool2d(kernel_size=cell_px, stride=cell_px),
        )
        cnn_feat_dim = grid_size * grid_size * c4

        # State pre-encoder: lift the 14-d state to a richer representation
        # before fusion with CNN features.
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # MLP head: deep + wide + BN + Dropout.
        h1, h2, h3, h4 = mlp_hidden
        in_dim = cnn_feat_dim + 64
        self.head = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),

            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),

            nn.Linear(h3, h4),
            nn.BatchNorm1d(h4),
            nn.ReLU(inplace=True),

            nn.Linear(h4, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        if image.ndim == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()
        feat = self.cnn(image).flatten(1)
        s = self.state_encoder(state)
        x = torch.cat([feat, s], dim=-1)
        return self.head(x)

    @torch.no_grad()
    def act(self, image: torch.Tensor, state: torch.Tensor) -> int:
        self.eval()
        logits = self.forward(image, state)
        return int(logits.argmax(dim=-1).item())
