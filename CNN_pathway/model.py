"""CNN + MLP policy for the 5x5 maze.

Input  : an 80x80x3 bird's-eye-view image of the grid (one frame, no history)
         + the 14-d state vector exposed by MazeNavEnv.
Output : action logits for the 4 discrete maze actions {UP, DOWN, LEFT, RIGHT}.

The CNN is cell-aligned (stride = cell_px) so each spatial feature corresponds
to exactly one grid cell; an MLP head fuses the flattened CNN features with
the state vector and predicts the action distribution.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNMLPPolicy(nn.Module):

    def __init__(
        self,
        img_size: int = 80,
        grid_size: int = 5,
        cell_px: int = 16,
        state_dim: int = 14,
        action_dim: int = 4,
        cnn_channels: int = 64,
        mlp_hidden: int = 256,
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

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=cell_px, stride=cell_px),
        )
        cnn_feat_dim = grid_size * grid_size * cnn_channels

        self.head = nn.Sequential(
            nn.Linear(cnn_feat_dim + state_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden // 2, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        if image.ndim == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2).contiguous()
        feat = self.cnn(image).flatten(1)
        x = torch.cat([feat, state], dim=-1)
        return self.head(x)

    @torch.no_grad()
    def act(self, image: torch.Tensor, state: torch.Tensor) -> int:
        self.eval()
        logits = self.forward(image, state)
        return int(logits.argmax(dim=-1).item())
