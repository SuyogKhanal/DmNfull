"""Replay-buffer fine-tuner for the hybrid equivariant policy.

Wraps the same optimizer / scheduler / loss as
``Equivariant_pathway/equivariant_CNN_hybrid/train.py`` but trains on a
mixed batch drawn from two disjoint demo pools:

    * D_new  — demos added in the *current* round.
    * D_old  — every other demo currently in the cumulative pool.

A ``WeightedRandomSampler`` produces each minibatch in expectation as

    ~ replay_mix * |batch|  samples drawn uniformly from D_old
    ~ (1 - replay_mix) * |batch|  samples drawn uniformly from D_new

so the per-step loss reflects the prescribed replay/new mix regardless of
the relative pool sizes. ``replay_mix_floor`` clamps the *new* weight so
even with a single new demo the new samples don't completely dominate the
batch (and similarly when |D_old| ≫ |D_new|, we don't drown out the new
demo).

Usage (subprocess):
    python -m …baseline_vs_p4_sequential_batch.trainer.finetune_replay \\
        --demo_dir /path/to/cumulative \\
        --new_demos_dir /path/to/round_NNN \\
        --checkpoint_dir … \\
        --epochs 20 --lr 5e-5 --batch_size 64 \\
        --replay_mix 0.5 --replay_mix_floor 0.2 \\
        --resume

When ``--new_demos_dir`` is omitted or empty, the trainer falls back to
plain fine-tuning on ``--demo_dir`` (still warm-starting if ``--resume``
is set). This makes the bootstrap retrain path identical to upstream.

The checkpoint / training_log.json schema is byte-compatible with
upstream's so existing aggregation tooling keeps working.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, random_split

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.dataset import (  # noqa: E402
    HybridDemoDataset, collate_fixed_size,
)
from Equivariant_pathway.equivariant_CNN_hybrid.model import (  # noqa: E402
    EquivariantCNNHybridPolicy,
)

NUM_ACTIONS = 4


class MultiLabelBCELoss(nn.Module):
    """Verbatim copy of upstream's loss to keep numerics identical."""

    def __init__(self, balance_pos_weight: bool = True):
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


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir", type=str, required=True,
                   help="Directory holding the cumulative demo pool (recursive).")
    p.add_argument("--new_demos_dir", type=str, default=None,
                   help="Subset directory holding only THIS round's new demos. "
                        "If omitted or empty, plain fine-tuning on --demo_dir.")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--replay_mix", type=float, default=0.5,
                   help="Fraction of each batch (in expectation) drawn from "
                        "D_old. 0.5 = half/half; 1.0 = pure replay (no new).")
    p.add_argument("--replay_mix_floor", type=float, default=0.2,
                   help="Lower bound on min(new_weight_frac, old_weight_frac) so "
                        "no side gets fully drowned when one pool is small.")
    p.add_argument("--cell_px", type=int, default=16)
    p.add_argument("--in_channels", type=int, default=5)
    p.add_argument("--resume", action="store_true",
                   help="Warm-start model weights from last_/best_hybrid_policy.pth.")
    return p.parse_args()


def _list_demo_files(demo_dir: str) -> List[str]:
    return sorted(set(glob.glob(os.path.join(demo_dir, "**", "*.json"), recursive=True)))


def _make_symlink_dir(paths: List[str]) -> str:
    """Materialize a flat directory of symlinks (with .json extension) so
    HybridDemoDataset's glob picks them up. Falls back to copying on
    file-systems that block symlinks.
    """
    tmp = tempfile.mkdtemp(prefix="hybrid_replay_")
    for i, src in enumerate(paths):
        dst = os.path.join(tmp, f"demo_{i:05d}.json")
        try:
            os.symlink(os.path.abspath(src), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
    return tmp


def _build_split_datasets(
    demo_dir: str, new_demos_dir: Optional[str], cell_px: int,
) -> Tuple[HybridDemoDataset, Optional[HybridDemoDataset], List[str]]:
    """Build (D_old, D_new, scratch_dirs).

    When ``new_demos_dir`` is missing/empty, returns (D_full, None, [...]).
    """
    all_paths = _list_demo_files(demo_dir)
    if not all_paths:
        raise FileNotFoundError(f"No demos under {demo_dir}")

    if not new_demos_dir or not os.path.isdir(new_demos_dir):
        # Plain fine-tune — single dataset over the full pool.
        full = HybridDemoDataset(demo_dir=demo_dir, recursive=True, cell_px=cell_px)
        return full, None, []

    new_paths = _list_demo_files(new_demos_dir)
    new_set = {os.path.realpath(p) for p in new_paths}
    if not new_set:
        full = HybridDemoDataset(demo_dir=demo_dir, recursive=True, cell_px=cell_px)
        return full, None, []

    old_paths = [p for p in all_paths if os.path.realpath(p) not in new_set]
    if not old_paths:
        # Only new demos exist (round 1 might mean cumulative ⊆ new). Still
        # train, but warn — there is no old pool to replay from.
        print("[finetune_replay] WARN: no old demos found; degenerating to "
              "single-pool fine-tune over the new demos only.", flush=True)
        d_new = HybridDemoDataset(demo_dir=new_demos_dir, recursive=True, cell_px=cell_px)
        return d_new, None, []

    old_dir = _make_symlink_dir(old_paths)
    new_dir = _make_symlink_dir(new_paths)
    d_old = HybridDemoDataset(demo_dir=old_dir, recursive=True, cell_px=cell_px)
    d_new = HybridDemoDataset(demo_dir=new_dir, recursive=True, cell_px=cell_px)
    return d_old, d_new, [old_dir, new_dir]


def _build_sampler(
    d_old: HybridDemoDataset, d_new: Optional[HybridDemoDataset],
    replay_mix: float, replay_mix_floor: float,
) -> Tuple[ConcatDataset, WeightedRandomSampler, dict]:
    """Compose a (ConcatDataset, WeightedRandomSampler) pair.

    ``replay_mix`` is the *target* fraction of each batch coming from
    D_old. We clamp it to [replay_mix_floor, 1 - replay_mix_floor] so that
    neither side drops below the floor.
    """
    floor = float(max(0.0, min(0.49, replay_mix_floor)))
    target = float(min(1.0 - floor, max(floor, replay_mix)))

    if d_new is None or len(d_new) == 0:
        # No new pool — uniform sampling over D_old only. Equivalent to
        # plain fine-tuning at this round.
        weights = np.full(len(d_old), 1.0 / max(1, len(d_old)), dtype=np.float64)
        concat = ConcatDataset([d_old])
        sampler = WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(concat),
            replacement=True,
        )
        return concat, sampler, {
            "n_old_samples": len(d_old), "n_new_samples": 0,
            "replay_mix_target": 1.0, "replay_mix_effective": 1.0,
            "replay_mix_floor": floor,
        }

    n_old = len(d_old)
    n_new = len(d_new)
    w_old = (target / n_old) if n_old > 0 else 0.0
    w_new = ((1.0 - target) / n_new) if n_new > 0 else 0.0
    weights = np.concatenate([
        np.full(n_old, w_old, dtype=np.float64),
        np.full(n_new, w_new, dtype=np.float64),
    ])
    concat = ConcatDataset([d_old, d_new])
    # ``num_samples`` controls the size of the epoch; we keep it equal to
    # |concat| so an epoch traverses the same number of samples as a plain
    # DataLoader would (just with re-weighting).
    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(concat),
        replacement=True,
    )
    return concat, sampler, {
        "n_old_samples": n_old, "n_new_samples": n_new,
        "replay_mix_target": target,
        "replay_mix_effective": target,  # exact in expectation
        "replay_mix_floor": floor,
    }


def _epoch(model, loader, loss_fn, device, optim=None):
    """Verbatim copy of upstream's per-epoch metric loop."""
    is_train = optim is not None
    model.train(is_train)
    total_loss, total = 0.0, 0
    correct_top1 = 0
    multi_label_correct = 0
    multi_label_total = 0
    for batch in loader:
        grouped = batch["groups"] if "groups" in batch else [batch]
        for g in grouped:
            rgb = g["rgb"].to(device)
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
    args = _parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune_replay] device={device} demo_dir={args.demo_dir} "
          f"new_demos_dir={args.new_demos_dir}", flush=True)

    d_old, d_new, scratch_dirs = _build_split_datasets(
        args.demo_dir, args.new_demos_dir, cell_px=args.cell_px,
    )
    concat, sampler, sampler_audit = _build_sampler(
        d_old, d_new, replay_mix=args.replay_mix,
        replay_mix_floor=args.replay_mix_floor,
    )
    print(f"[finetune_replay] sampler audit: {sampler_audit}", flush=True)

    # Split into train/val using random_split — the sampler runs on the
    # train half only; val uses plain (unweighted) sampling on its slice.
    val_size = max(1, int(len(concat) * args.val_frac))
    train_size = len(concat) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(concat, [train_size, val_size], generator=gen)

    # WeightedRandomSampler iterates over concat indices; combine with the
    # train_set subset by mapping back to concat indices and re-weighting.
    train_indices = train_set.indices  # type: ignore[attr-defined]
    if isinstance(sampler, WeightedRandomSampler):
        full_weights = np.asarray(sampler.weights, dtype=np.float64)
        train_weights = full_weights[train_indices]
        # Re-normalize so the per-epoch num_samples == |train_set|.
        if train_weights.sum() > 0:
            train_weights = train_weights / train_weights.sum() * len(train_indices)
        sampler_train = WeightedRandomSampler(
            weights=train_weights.tolist(),
            num_samples=len(train_indices),
            replacement=True,
        )
    else:  # pragma: no cover
        sampler_train = None  # type: ignore[assignment]

    eff_batch = max(1, min(args.batch_size, len(train_set)))
    train_loader = DataLoader(
        train_set, batch_size=eff_batch, sampler=sampler_train,
        num_workers=0, drop_last=False, collate_fn=collate_fixed_size,
    )
    val_loader = DataLoader(
        val_set, batch_size=eff_batch, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate_fixed_size,
    )

    model = EquivariantCNNHybridPolicy(
        rgb_in_channels=3, grid_in_channels=args.in_channels,
        num_actions=NUM_ACTIONS,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[finetune_replay] trainable params: {n_params:,}", flush=True)

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
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
            print(f"[finetune_replay] WARM START from {src.name}", flush=True)

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
        print(
            f"[finetune_replay] ep {epoch+1:3d}/{args.epochs} | "
            f"train loss={rec['train_loss']:.4f} top1={rec['train_top1']:.3f} | "
            f"val loss={rec['val_loss']:.4f} | t={rec['epoch_time_sec']:.1f}s{marker}",
            flush=True,
        )
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
                "sampler": sampler_audit,
                "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
                "best_epoch": int(best_epoch) if best_epoch > 0 else None,
                "history": history,
            }, f, indent=2)

    # Clean up symlink scratch dirs.
    for d in scratch_dirs:
        try:
            shutil.rmtree(d)
        except OSError:
            pass

    print(
        f"[finetune_replay] DONE best_val_loss={best_val_loss:.4f} @ epoch {best_epoch}",
        flush=True,
    )


if __name__ == "__main__":
    main()
