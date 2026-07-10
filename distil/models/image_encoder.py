"""Image encoder for the DISTIL image-modality diffusion policy (NO R3M).

A small conv trunk + spatial-softmax keypoints producing a compact per-frame
embedding, stacked over obs_horizon into the U-Net's global condition — structurally
identical to the state path, so the diffusion loss / uncertainty math is unchanged
(02_..md #4: only the POLICY encoder changes; the descriptor stays geometric).

`SpatialSoftArgmax` + `CoordinateUtils` are lifted verbatim from the fork's
diffdagger/model/vision/image_encoder.py (pure PyTorch, zero external deps).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateUtils:
    @staticmethod
    def get_image_coordinates(h, w, normalise):
        x_range = torch.arange(w, dtype=torch.float32)
        y_range = torch.arange(h, dtype=torch.float32)
        if normalise:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        return image_x, image_y


class SpatialSoftArgmax(nn.Module):
    """Spatial soft-argmax: (N, C, H, W) -> (N, C, 2) keypoints (one per channel)."""

    def __init__(self, temperature=None, normalise=True):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)) if temperature is None \
            else torch.tensor([float(temperature)])
        self.normalise = normalise

    def forward(self, x):
        n, c, h, w = x.size()
        sm = torch.nn.functional.softmax(x.reshape(n * c, h * w) / self.temperature, dim=1)
        sm = sm.view(n, c, h, w)
        ix, iy = CoordinateUtils.get_image_coordinates(h, w, normalise=self.normalise)
        coords = torch.cat((ix.unsqueeze(-1), iy.unsqueeze(-1)), dim=-1).to(x.device)
        out = torch.sum(sm.unsqueeze(-1) * coords.unsqueeze(0), dim=[2, 3])
        return out  # (N, C, 2)


def _gn(ch, groups=8):
    g = groups
    while ch % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, ch)


class SpatialSoftmaxEncoder(nn.Module):
    """RGB (N,3,H,W) in [0,1] -> keypoint embedding (N, 2*keypoints). Small, no pretrain."""

    def __init__(self, in_channels: int = 3, keypoints: int = 32,
                 channels=(32, 64, 64)):
        super().__init__()
        c1, c2, c3 = channels
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1), _gn(c1), nn.Mish(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), _gn(c2), nn.Mish(),
            nn.Conv2d(c2, c3, 3, stride=1, padding=1), _gn(c3), nn.Mish(),
            nn.Conv2d(c3, keypoints, 1),
        )
        self.ssm = SpatialSoftArgmax(normalise=True)
        self.out_dim = 2 * keypoints

    def forward(self, x):
        # x: (N, 3, H, W) float in [0,1]
        feat = self.trunk(x)             # (N, K, h', w')
        kp = self.ssm(feat)              # (N, K, 2)
        return kp.flatten(1)             # (N, 2K)
