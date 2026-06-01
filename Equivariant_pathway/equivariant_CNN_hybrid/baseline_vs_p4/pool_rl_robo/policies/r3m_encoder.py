"""Frozen R3M image encoder (facebookresearch/r3m).

Loads a pretrained R3M ResNet (resnet18 -> 512-d, resnet50 -> 2048-d), frozen,
and encodes RGB frames. R3M expects 0-255 RGB; we resize to 224x224. Weights are
cached under ~/.r3m (download once on a login node so offline jobs reuse it).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_EMBED = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


class R3MEncoder:
    def __init__(self, device: torch.device, model_name: str = "resnet18"):
        from r3m import load_r3m
        net = load_r3m(model_name)
        if isinstance(net, nn.DataParallel):
            net = net.module
        self.net = net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.embed_dim = _EMBED.get(model_name, 512)

    @torch.no_grad()
    def encode_batch(self, imgs) -> np.ndarray:
        """imgs: list of HxWx3 uint8 RGB -> (N, embed_dim) float32."""
        import torchvision.transforms.functional as TF
        if not imgs:
            return np.zeros((0, self.embed_dim), np.float32)
        x = torch.from_numpy(np.stack([np.asarray(i, dtype=np.uint8) for i in imgs])).permute(0, 3, 1, 2).float()
        x = TF.resize(x, [224, 224], antialias=True).to(self.device)  # R3M wants 0-255
        return self.net(x).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode(self, img) -> np.ndarray:
        return self.encode_batch([img])[0]
