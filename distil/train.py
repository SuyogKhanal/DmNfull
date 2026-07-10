"""Diffusion-policy training utilities: EMA, cosine-warmup LR, train loop.

Mirrors the recipe in the strong_diffusion_policies trainer (AdamW + Karras EMA
+ cosine-with-warmup), packaged as a single function so the Diff-DAgger loop can
retrain a fresh policy from scratch each round (as Algorithm 1 prescribes).
"""
import copy
import math

import torch
from torch.utils.data import DataLoader

from .dataset import collate


def cosine_warmup_lr(step, total, warmup, base_lr, min_lr_ratio=0.02):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


class EMA:
    """Karras EMA: decay = 1 - (1 + step/inv_gamma)^(-power), clamped to max."""

    def __init__(self, model, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999):
        self.inv_gamma, self.power = inv_gamma, power
        self.min_value, self.max_value = min_value, max_value
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _decay(self):
        s = max(0, self.step - 1)
        v = 1 - (1 + s / self.inv_gamma) ** (-self.power)
        return max(self.min_value, min(v, self.max_value))

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = self._decay()
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def train_diffusion_policy(
    policy,
    dataset,
    *,
    train_steps: int,
    batch_size: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    betas=(0.95, 0.999),
    warmup: int = None,
    ema_power: float = 0.75,
    device="cuda",
    num_workers: int = 0,
    log_fn=print,
    log_freq: int = 200,
):
    """Train `policy` in place; load EMA weights for inference; return policy."""
    eff_bs = min(batch_size, len(dataset))
    drop_last = len(dataset) >= 2 * eff_bs
    loader = DataLoader(
        dataset, batch_size=eff_bs, shuffle=True, drop_last=drop_last,
        collate_fn=collate, num_workers=num_workers,
        pin_memory=(str(device).startswith("cuda")),
        persistent_workers=num_workers > 0,
    )
    if warmup is None:
        warmup = min(max(1, train_steps // 10), 250)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay)
    ema = EMA(policy, power=ema_power)
    it = _cycle(loader)
    policy.train()
    running, n = 0.0, 0
    for step in range(train_steps):
        obs, act = next(it)
        obs = {k: v.to(device, non_blocking=True) for k, v in obs.items()}
        act = act.to(device, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = cosine_warmup_lr(step, train_steps, warmup, lr)
        loss = policy.compute_loss(obs, act)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        ema.update(policy)
        running += loss.item(); n += 1
        if (step + 1) % log_freq == 0:
            log_fn(f"    train step {step+1}/{train_steps} loss={running/max(1,n):.5f} "
                   f"lr={opt.param_groups[0]['lr']:.2e}")
            running, n = 0.0, 0

    policy.load_state_dict(ema.state_dict())  # use EMA weights for inference
    policy.eval()
    return policy
