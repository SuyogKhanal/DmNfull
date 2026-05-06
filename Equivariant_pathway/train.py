"""Behavior-cloning trainer for the equivariant maze policy.

Workflow
--------
    # 1. Collect 10 expert demos with the BFS expert (no human required).
    python -m Equivariant_pathway.collect_demos \
        --layouts Equivariant_pathway/training_layouts.yaml \
        --demo_dir Equivariant_pathway/demos \
        --num_demos 10

    # 2. Train the equivariant policy on those demos.
    python -m Equivariant_pathway.train \
        --demo_dir Equivariant_pathway/demos \
        --checkpoint_dir Equivariant_pathway/checkpoints

Loss is multi-label BCE (with positive-weight balancing) so multi-modal
targets — two equally good actions at the same state — are jointly
supervised. This matches the dmnpol training objective verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.dataset import EquivariantDemoDataset, collate_fixed_size
from Equivariant_pathway.model import EquivariantUNetPolicy

NUM_ACTIONS = 4
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


class MultiLabelBCELoss(nn.Module):
    """BCE-with-logits over the 4-action mask, with positive-weight balancing.

    Each sample's mask sums to (1, 2, 3, ...) — the count of optimal actions
    at that state. Without balancing the loss collapses to "predict zero
    everywhere" because non-optimal actions vastly outnumber optimal ones.
    """
    def __init__(self, balance_pos_weight: bool = True):
        super().__init__()
        self.balance_pos_weight = balance_pos_weight

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.balance_pos_weight:
            num_optimal = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            num_total   = torch.full_like(num_optimal, mask.shape[-1])
            pos_weight  = (num_total - num_optimal) / num_optimal
            pos_weight  = pos_weight.expand_as(mask)
            loss = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
            weights = torch.where(mask > 0.5, pos_weight, torch.ones_like(pos_weight))
            return (loss * weights).mean()
        return F.binary_cross_entropy_with_logits(logits, mask, reduction="mean")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "demos"),
                   help="Directory of demo JSONs (recursively searched).")
    p.add_argument("--demo_paths", type=str, default=None,
                   help="Comma-separated globs to UNION into the dataset. "
                        "Overrides --demo_dir when provided. Used by the "
                        "active loop to combine baseline + per-profile demos.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_demos", type=int, default=None,
                   help="Truncate dataset to first N demos (sorted by name). "
                        "Round 1 of every profile uses --max_demos 10 so all "
                        "profiles + baseline start from the same 10-demo set.")
    p.add_argument("--checkpoint_dir", type=str,
                   default=str(REPO_ROOT / "Equivariant_pathway" / "checkpoints"))
    p.add_argument("--resume", action="store_true",
                   help="If last_eq_policy.pth exists in checkpoint_dir, resume from it.")
    p.add_argument("--channels", type=str, default="16,32,64",
                   help="Equivariant block channel counts (c1,c2,c3).")
    p.add_argument("--in_channels", type=int, default=5)
    return p.parse_args()


def _build_dataset_from_paths(args) -> EquivariantDemoDataset:
    """Build a dataset that unions multiple glob patterns into a single
    in-memory list. We do this by writing one EquivariantDemoDataset per
    pattern and then concatenating their `samples` / cache into the first
    one — keeps the BFS distance cache shared across patterns."""
    if args.demo_paths is None:
        return EquivariantDemoDataset(demo_dir=args.demo_dir,
                                      max_demos=args.max_demos,
                                      recursive=True)
    import glob as _glob
    patterns = [s.strip() for s in args.demo_paths.split(",") if s.strip()]
    files = []
    for pat in patterns:
        files.extend(_glob.glob(pat, recursive=True))
    if not files:
        raise FileNotFoundError(f"No demos matched any of: {patterns}")
    files = sorted(set(files))
    if args.max_demos is not None and args.max_demos > 0:
        files = files[: args.max_demos]
    # Materialise into one dataset by giving it a parent dir + an exact
    # glob (so EquivariantDemoDataset's loader picks up exactly our list).
    # Easiest is to symlink them under a temp dir.
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="eq_demos_union_"))
    for i, src in enumerate(files):
        dst = tmp / f"demo_{i:05d}.json"
        try:
            os.symlink(os.path.abspath(src), dst)
        except (OSError, NotImplementedError):
            import shutil
            shutil.copy2(src, dst)
    return EquivariantDemoDataset(demo_dir=str(tmp), recursive=False)


def _epoch_pass(model, loader, loss_fn, device, optim=None):
    is_train = optim is not None
    model.train(is_train)
    total_loss, total = 0.0, 0
    correct_top1 = 0
    multi_label_correct = 0
    multi_label_total = 0
    for batch in loader:
        if "groups" in batch:
            grouped = batch["groups"]
        else:
            grouped = [batch]
        for g in grouped:
            inp  = g["input"].to(device)
            mask = g["mask"].to(device)
            with torch.set_grad_enabled(is_train):
                logits = model(inp)
                loss = loss_fn(logits, mask)
                if is_train:
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            bsz = inp.shape[0]
            total_loss += float(loss.item()) * bsz
            total += bsz
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                # Top-1 correct iff the predicted action is in the optimal set.
                gathered = mask.gather(1, pred.unsqueeze(1)).squeeze(1)
                correct_top1 += int((gathered > 0.5).sum().item())
                # Multi-label "any-optimal" hit-rate per sample.
                any_optimal = (mask.sum(dim=-1) > 0).float()
                multi_label_correct += int(((gathered > 0.5).float() * any_optimal).sum().item())
                multi_label_total += int(any_optimal.sum().item())
    if total == 0:
        return None
    return {
        "loss":      total_loss / total,
        "top1_acc":  correct_top1 / total,
        "any_opt":   (multi_label_correct / multi_label_total) if multi_label_total else float("nan"),
        "n":         int(total),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eq-train] device={device}")
    print(f"[eq-train] demo_dir={args.demo_dir}  demo_paths={args.demo_paths}")
    print(f"[eq-train] max_demos={args.max_demos}  epochs={args.epochs}")
    print(f"[eq-train] checkpoints -> {ckpt_dir}")

    dataset = _build_dataset_from_paths(args)
    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)

    eff_batch = max(1, min(args.batch_size, len(train_set)))
    train_loader = DataLoader(train_set, batch_size=eff_batch,
                              shuffle=True, num_workers=0, drop_last=False,
                              collate_fn=collate_fixed_size)
    val_loader   = DataLoader(val_set,   batch_size=eff_batch,
                              shuffle=False, num_workers=0, drop_last=False,
                              collate_fn=collate_fixed_size)
    print(f"[eq-train] train_size={train_size} val_size={val_size} "
          f"batches/epoch={len(train_loader)}")

    channels = tuple(int(x) for x in args.channels.split(","))
    model = EquivariantUNetPolicy(
        in_channels=args.in_channels,
        channels=channels,
        num_actions=NUM_ACTIONS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[eq-train] trainable params: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(args.epochs, 1))
    loss_fn = MultiLabelBCELoss(balance_pos_weight=True)

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = -1
    last_path = ckpt_dir / "last_eq_policy.pth"
    best_path = ckpt_dir / "best_eq_policy.pth"
    if args.resume and last_path.exists():
        # weights_only=False — checkpoint contains optimizer + scheduler state dicts
        # plus an args dict, none of which load under weights_only=True.
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optim.load_state_dict(ckpt["optim_state_dict"])
        if "sched_state_dict" in ckpt:
            sched.load_state_dict(ckpt["sched_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", -1))
        print(f"[eq-train] resumed from {last_path} @ epoch {start_epoch}")

    history = []
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        ep_t0 = time.time()
        train_metrics = _epoch_pass(model, train_loader, loss_fn, device, optim=optim)
        sched.step()
        val_metrics   = _epoch_pass(model, val_loader,   loss_fn, device, optim=None)
        if train_metrics is None:
            print(f"[eq-train] epoch {epoch+1}: empty loader, skipping.")
            continue

        rec = {
            "epoch":          epoch + 1,
            "lr":             float(optim.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - ep_t0),
            "elapsed_min":    float((time.time() - t0) / 60.0),
            "train_loss":     float(train_metrics["loss"]),
            "train_top1":     float(train_metrics["top1_acc"]),
            "train_any_opt":  float(train_metrics["any_opt"]),
            "val_loss":       float(val_metrics["loss"])    if val_metrics else float("nan"),
            "val_top1":       float(val_metrics["top1_acc"]) if val_metrics else float("nan"),
            "val_any_opt":    float(val_metrics["any_opt"])  if val_metrics else float("nan"),
        }
        history.append(rec)

        improved = (val_metrics is not None) and (val_metrics["loss"] < best_val_loss)
        marker = "  *" if improved else ""
        print(
            f"[eq-train] ep {epoch+1:3d}/{args.epochs} | "
            f"train loss={rec['train_loss']:.4f} top1={rec['train_top1']:.3f} "
            f"any_opt={rec['train_any_opt']:.3f} | "
            f"val loss={rec['val_loss']:.4f} any_opt={rec['val_any_opt']:.3f} | "
            f"lr={rec['lr']:.2e} t={rec['epoch_time_sec']:.1f}s{marker}"
        )

        torch.save({
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "sched_state_dict": sched.state_dict(),
            "epoch":            epoch + 1,
            "metrics":          rec,
            "in_channels":      args.in_channels,
            "channels":         channels,
            "num_actions":      NUM_ACTIONS,
            "best_val_loss":    best_val_loss,
            "best_epoch":       best_epoch,
            "args":             vars(args),
        }, last_path)

        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch + 1,
                "metrics":          rec,
                "in_channels":      args.in_channels,
                "channels":         channels,
                "num_actions":      NUM_ACTIONS,
                "val_loss":         best_val_loss,
                "args":             vars(args),
            }, best_path)

        with open(ckpt_dir / "training_log.json", "w") as f:
            json.dump({
                "args":             vars(args),
                "trainable_params": int(n_params),
                "best_val_loss":    float(best_val_loss) if best_val_loss != float("inf") else None,
                "best_epoch":       int(best_epoch) if best_epoch > 0 else None,
                "history":          history,
            }, f, indent=2)

    elapsed = time.time() - t0
    print(f"[eq-train] DONE in {elapsed/60:.1f}m | "
          f"best_val_loss={best_val_loss:.4f} @ epoch {best_epoch} | "
          f"checkpoints={ckpt_dir}")
    # Print the final loss prominently so active_loop's regex (Loss: <num>)
    # picks up the last training loss for its log.
    print(f"Loss: {history[-1]['train_loss'] if history else float('nan')}")


if __name__ == "__main__":
    main()
