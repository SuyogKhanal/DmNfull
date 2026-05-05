"""Train the CNN+MLP maze policy on demos collected via play_maze.

Workflow
--------
    # 1. record demos manually (auto-advances through every layout)
    python scripts/play_maze.py \
        --layouts-from CNN_pathway/training_layouts.yaml \
        --demo_dir CNN_pathway/demos

    # 2. train the CNN+MLP on the recorded demos
    python -m CNN_pathway.train \
        --demo_dir CNN_pathway/demos \
        --layouts CNN_pathway/training_layouts.yaml

Tracked metrics (per epoch, written every epoch to training_log.json so a
mid-run inspection is always possible):
    train_loss, train_acc, train_per_class_acc[4]
    val_loss,   val_acc,   val_per_class_acc[4]
    val_macro_precision, val_macro_recall, val_macro_f1
    lr, epoch_time_sec, elapsed_min

Checkpoints (under --checkpoint_dir):
    best_cnn_mlp.pth       — kept on highest val_acc seen so far
    last_cnn_mlp.pth       — overwritten every epoch
    best_meta.json         — quick-look snapshot of best metrics
    training_log.json      — full per-epoch history
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
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CNN_pathway.dataset import CNNMLPDemoDataset
from CNN_pathway.model import CNNMLPPolicy

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
NUM_ACTIONS = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "demos"),
                   help="Directory of play_maze demo JSONs (recursively searched).")
    p.add_argument("--layouts", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "training_layouts.yaml"),
                   help="Training layouts YAML — only used to read img_size / "
                        "grid_size / cell_px for model construction.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "checkpoints"))
    return p.parse_args()


def _per_class_accuracy(preds: np.ndarray, labels: np.ndarray) -> list:
    out = []
    for k in range(NUM_ACTIONS):
        mask = labels == k
        if mask.sum() == 0:
            out.append(float("nan"))
        else:
            out.append(float((preds[mask] == k).mean()))
    return out


def _macro_prf(preds: np.ndarray, labels: np.ndarray):
    precisions, recalls, f1s = [], [], []
    for k in range(NUM_ACTIONS):
        tp = int(((preds == k) & (labels == k)).sum())
        fp = int(((preds == k) & (labels != k)).sum())
        fn = int(((preds != k) & (labels == k)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = float("nan")
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    def _nm(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")
    return _nm(precisions), _nm(recalls), _nm(f1s), precisions, recalls, f1s


def _epoch_pass(model, loader, loss_fn, device, optim=None, sched=None):
    is_train = optim is not None
    model.train(is_train)
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []
    for img, state, action in loader:
        img = img.to(device)
        state = state.to(device)
        action = action.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(img, state)
            loss = loss_fn(logits, action)
            if is_train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
        bsz = img.shape[0]
        total_loss += float(loss.item()) * bsz
        total += bsz
        all_preds.append(logits.argmax(-1).detach().cpu().numpy())
        all_labels.append(action.detach().cpu().numpy())
    if total == 0:
        return None
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    avg_loss = total_loss / total
    acc = float((preds == labels).mean())
    per_class = _per_class_accuracy(preds, labels)
    return {
        "loss": avg_loss,
        "acc": acc,
        "per_class_acc": per_class,
        "preds": preds,
        "labels": labels,
        "n": int(total),
    }


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cnn-train] device={device}")
    print(f"[cnn-train] demo_dir={args.demo_dir}")
    print(f"[cnn-train] layouts ={args.layouts}")
    print(f"[cnn-train] epochs ={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"[cnn-train] checkpoints → {ckpt_dir}")

    dataset = CNNMLPDemoDataset(
        demo_dir=args.demo_dir,
        layouts_yaml=args.layouts,
        augment=True,
    )

    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, drop_last=False)
    print(f"[cnn-train] train_size={train_size} val_size={val_size} "
          f"batches/epoch={len(train_loader)}")

    cfg = dataset.layouts_cfg
    model = CNNMLPPolicy(
        img_size=int(cfg.get("img_size", 80)),
        grid_size=int(cfg.get("grid_size", 5)),
        cell_px=int(cfg.get("cell_px", 16)),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cnn-train] trainable params: {n_params:,}")

    best_val_acc = -float("inf")
    best_epoch = -1
    history: list = []
    t0 = time.time()

    for epoch in tqdm(range(args.epochs), desc="train", unit="epoch"):
        ep_t0 = time.time()
        train_metrics = _epoch_pass(model, train_loader, loss_fn, device, optim=optim)
        sched.step()
        val_metrics = _epoch_pass(model, val_loader, loss_fn, device, optim=None)
        if train_metrics is None or val_metrics is None:
            tqdm.write(f"[cnn-train] epoch {epoch+1}: empty loader, skipping.")
            continue

        macro_p, macro_r, macro_f1, prec_pc, rec_pc, f1_pc = _macro_prf(
            val_metrics["preds"], val_metrics["labels"],
        )
        epoch_time = time.time() - ep_t0
        elapsed_min = (time.time() - t0) / 60.0
        record = {
            "epoch":              epoch + 1,
            "lr":                 float(optim.param_groups[0]["lr"]),
            "epoch_time_sec":     float(epoch_time),
            "elapsed_min":        float(elapsed_min),
            "train_loss":         float(train_metrics["loss"]),
            "train_acc":          float(train_metrics["acc"]),
            "train_per_class_acc": {ACTION_NAMES[k]: train_metrics["per_class_acc"][k]
                                    for k in range(NUM_ACTIONS)},
            "train_n":            train_metrics["n"],
            "val_loss":           float(val_metrics["loss"]),
            "val_acc":            float(val_metrics["acc"]),
            "val_per_class_acc":  {ACTION_NAMES[k]: val_metrics["per_class_acc"][k]
                                   for k in range(NUM_ACTIONS)},
            "val_macro_precision":macro_p,
            "val_macro_recall":   macro_r,
            "val_macro_f1":       macro_f1,
            "val_per_class_precision": {ACTION_NAMES[k]: prec_pc[k] for k in range(NUM_ACTIONS)},
            "val_per_class_recall":    {ACTION_NAMES[k]: rec_pc[k]  for k in range(NUM_ACTIONS)},
            "val_per_class_f1":        {ACTION_NAMES[k]: f1_pc[k]   for k in range(NUM_ACTIONS)},
            "val_n":              val_metrics["n"],
        }
        history.append(record)

        # Persist the full history every epoch so the user can inspect mid-run.
        with open(ckpt_dir / "training_log.json", "w") as f:
            json.dump({
                "args": vars(args),
                "trainable_params": int(n_params),
                "best_val_acc": float(best_val_acc) if best_val_acc != -float("inf") else None,
                "best_epoch":   int(best_epoch) if best_epoch > 0 else None,
                "history":      history,
            }, f, indent=2)

        improved = val_metrics["acc"] > best_val_acc
        marker = "  *" if improved else ""
        if (epoch + 1) % 1 == 0:
            tqdm.write(
                f"[cnn-train] ep {epoch+1:3d}/{args.epochs} | "
                f"train loss={record['train_loss']:.4f} acc={record['train_acc']:.3f} | "
                f"val loss={record['val_loss']:.4f} acc={record['val_acc']:.3f} "
                f"f1={macro_f1:.3f} | "
                f"lr={record['lr']:.2e} t={epoch_time:.1f}s{marker}"
            )

        last_payload = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "sched_state_dict": sched.state_dict(),
            "epoch":     epoch + 1,
            "metrics":   record,
            "img_size":  int(cfg.get("img_size", 80)),
            "grid_size": int(cfg.get("grid_size", 5)),
            "cell_px":   int(cfg.get("cell_px", 16)),
            "args":      vars(args),
        }
        torch.save(last_payload, ckpt_dir / "last_cnn_mlp.pth")

        if improved:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch + 1
            best_payload = {
                "model_state_dict": model.state_dict(),
                "epoch":     epoch + 1,
                "metrics":   record,
                "img_size":  int(cfg.get("img_size", 80)),
                "grid_size": int(cfg.get("grid_size", 5)),
                "cell_px":   int(cfg.get("cell_px", 16)),
                "args":      vars(args),
            }
            torch.save(best_payload, ckpt_dir / "best_cnn_mlp.pth")
            with open(ckpt_dir / "best_meta.json", "w") as f:
                json.dump({
                    "best_epoch":  best_epoch,
                    "best_val_acc": float(best_val_acc),
                    "best_metrics": record,
                }, f, indent=2)

    elapsed = time.time() - t0
    print(
        f"[cnn-train] DONE in {elapsed/60:.1f}m | "
        f"best_val_acc={best_val_acc:.4f} @ epoch {best_epoch} | "
        f"checkpoints={ckpt_dir}"
    )


if __name__ == "__main__":
    main()
