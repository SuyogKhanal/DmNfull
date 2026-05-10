"""Behaviour-cloning trainer for EquivariantCNNHybridPolicy.

Multi-label BCE on the 4-action mask, exactly the same loss as the
equivariant trainer; the only differences are (a) the model takes
TWO inputs (rgb, grid) and (b) the checkpoint name is
best_hybrid_policy.pth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.dataset import (
    HybridDemoDataset, collate_fixed_size,
)
from Equivariant_pathway.equivariant_CNN_hybrid.model import EquivariantCNNHybridPolicy

NUM_ACTIONS = 4


class MultiLabelBCELoss(nn.Module):
    def __init__(self, balance_pos_weight=True):
        super().__init__()
        self.balance_pos_weight = balance_pos_weight

    def forward(self, logits, mask):
        if self.balance_pos_weight:
            num_optimal = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            num_total = torch.full_like(num_optimal, mask.shape[-1])
            pos_weight = ((num_total - num_optimal) / num_optimal).expand_as(mask)
            loss = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
            weights = torch.where(mask > 0.5, pos_weight, torch.ones_like(pos_weight))
            return (loss * weights).mean()
        return F.binary_cross_entropy_with_logits(logits, mask, reduction="mean")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "equivariant_CNN_hybrid" / "demos"))
    p.add_argument("--demo_paths", type=str, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_demos", type=int, default=None)
    p.add_argument("--checkpoint_dir", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "equivariant_CNN_hybrid" / "checkpoints"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cell_px", type=int, default=16)
    p.add_argument("--in_channels", type=int, default=5)
    return p.parse_args()


def _build_dataset(args):
    if args.demo_paths is None:
        return HybridDemoDataset(demo_dir=args.demo_dir, max_demos=args.max_demos,
                                 recursive=True, cell_px=args.cell_px)
    import glob as _glob, tempfile, shutil
    patterns = [s.strip() for s in args.demo_paths.split(",") if s.strip()]
    files = []
    for pat in patterns:
        files.extend(_glob.glob(pat, recursive=True))
    if not files:
        raise FileNotFoundError(f"No demos matched: {patterns}")
    files = sorted(set(files))
    if args.max_demos is not None and args.max_demos > 0:
        files = files[: args.max_demos]
    tmp = Path(tempfile.mkdtemp(prefix="hybrid_demos_union_"))
    for i, src in enumerate(files):
        dst = tmp / f"demo_{i:05d}.json"
        try:
            os.symlink(os.path.abspath(src), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
    return HybridDemoDataset(demo_dir=str(tmp), recursive=False, cell_px=args.cell_px)


def _epoch(model, loader, loss_fn, device, optim=None):
    is_train = optim is not None
    model.train(is_train)
    total_loss, total = 0.0, 0
    correct_top1 = 0
    multi_label_correct = 0
    multi_label_total = 0
    for batch in loader:
        grouped = batch["groups"] if "groups" in batch else [batch]
        for g in grouped:
            rgb  = g["rgb"].to(device)
            grid = g["grid"].to(device)
            mask = g["mask"].to(device)
            with torch.set_grad_enabled(is_train):
                logits = model(rgb, grid)
                loss = loss_fn(logits, mask)
                if is_train:
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            bsz = rgb.shape[0]
            total_loss += float(loss.item()) * bsz
            total += bsz
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                gathered = mask.gather(1, pred.unsqueeze(1)).squeeze(1)
                correct_top1 += int((gathered > 0.5).sum().item())
                any_optimal = (mask.sum(dim=-1) > 0).float()
                multi_label_correct += int(((gathered > 0.5).float() * any_optimal).sum().item())
                multi_label_total += int(any_optimal.sum().item())
    if total == 0:
        return None
    return {
        "loss": total_loss / total, "top1_acc": correct_top1 / total,
        "any_opt": (multi_label_correct / multi_label_total) if multi_label_total else float("nan"),
        "n": int(total),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hybrid-train] device={device} demo_dir={args.demo_dir}")

    dataset = _build_dataset(args)
    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)
    eff_batch = max(1, min(args.batch_size, len(train_set)))
    train_loader = DataLoader(train_set, batch_size=eff_batch, shuffle=True,
                              num_workers=0, drop_last=False, collate_fn=collate_fixed_size)
    val_loader   = DataLoader(val_set, batch_size=eff_batch, shuffle=False,
                              num_workers=0, drop_last=False, collate_fn=collate_fixed_size)

    model = EquivariantCNNHybridPolicy(
        rgb_in_channels=3, grid_in_channels=args.in_channels,
        num_actions=NUM_ACTIONS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[hybrid-train] trainable params: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 1))
    loss_fn = MultiLabelBCELoss(balance_pos_weight=True)

    last_path = ckpt_dir / "last_hybrid_policy.pth"
    best_path = ckpt_dir / "best_hybrid_policy.pth"
    best_val_loss = float("inf"); best_epoch = -1
    if args.resume:
        src = last_path if last_path.exists() else (best_path if best_path.exists() else None)
        if src is not None:
            ckpt = torch.load(str(src), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            print(f"[hybrid-train] WARM START from {src.name}")

    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_t0 = time.time()
        train_metrics = _epoch(model, train_loader, loss_fn, device, optim=optim)
        sched.step()
        val_metrics = _epoch(model, val_loader, loss_fn, device, optim=None)
        if train_metrics is None:
            continue
        rec = {
            "epoch": epoch + 1, "lr": float(optim.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - ep_t0),
            "elapsed_min": float((time.time() - t0) / 60.0),
            "train_loss": float(train_metrics["loss"]),
            "train_top1": float(train_metrics["top1_acc"]),
            "train_any_opt": float(train_metrics["any_opt"]),
            "val_loss": float(val_metrics["loss"]) if val_metrics else float("nan"),
            "val_top1": float(val_metrics["top1_acc"]) if val_metrics else float("nan"),
            "val_any_opt": float(val_metrics["any_opt"]) if val_metrics else float("nan"),
        }
        history.append(rec)
        improved = (val_metrics is not None) and (val_metrics["loss"] < best_val_loss)
        marker = "  *" if improved else ""
        print(f"[hybrid-train] ep {epoch+1:3d}/{args.epochs} | "
              f"train loss={rec['train_loss']:.4f} top1={rec['train_top1']:.3f} | "
              f"val loss={rec['val_loss']:.4f} | t={rec['epoch_time_sec']:.1f}s{marker}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "sched_state_dict": sched.state_dict(),
            "epoch": epoch + 1, "metrics": rec, "num_actions": NUM_ACTIONS,
            "in_channels": args.in_channels,
            "best_val_loss": best_val_loss, "best_epoch": best_epoch,
            "args": vars(args),
        }, last_path)
        if improved:
            best_val_loss = val_metrics["loss"]; best_epoch = epoch + 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1, "metrics": rec, "num_actions": NUM_ACTIONS,
                "in_channels": args.in_channels,
                "val_loss": best_val_loss, "args": vars(args),
            }, best_path)
        with open(ckpt_dir / "training_log.json", "w") as f:
            json.dump({
                "args": vars(args), "trainable_params": int(n_params),
                "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
                "best_epoch": int(best_epoch) if best_epoch > 0 else None,
                "history": history,
            }, f, indent=2)

    print(f"[hybrid-train] DONE best_val_loss={best_val_loss:.4f} @ epoch {best_epoch}")
    print(f"Loss: {history[-1]['train_loss'] if history else float('nan')}")


if __name__ == "__main__":
    main()
