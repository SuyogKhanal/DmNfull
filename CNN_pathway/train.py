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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "demos"),
                   help="Directory of play_maze demo JSONs (recursively searched).")
    p.add_argument("--layouts", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "training_layouts.yaml"),
                   help="Training layouts YAML — only used to read img_size / "
                        "grid_size / cell_px for model construction.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", type=str,
                   default=str(REPO_ROOT / "CNN_pathway" / "checkpoints"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cnn-train] device={device}")
    print(f"[cnn-train] demo_dir={args.demo_dir}")
    print(f"[cnn-train] layouts ={args.layouts}")

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

    best_val_loss = float("inf")
    history = []
    t0 = time.time()
    for epoch in tqdm(range(args.epochs), desc="train", unit="epoch"):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for img, state, action in train_loader:
            img = img.to(device)
            state = state.to(device)
            action = action.to(device)
            logits = model(img, state)
            loss = loss_fn(logits, action)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += float(loss.item()) * img.shape[0]
            train_correct += int((logits.argmax(-1) == action).sum().item())
            train_total += img.shape[0]
        sched.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for img, state, action in val_loader:
                img = img.to(device)
                state = state.to(device)
                action = action.to(device)
                logits = model(img, state)
                loss = loss_fn(logits, action)
                val_loss += float(loss.item()) * img.shape[0]
                val_correct += int((logits.argmax(-1) == action).sum().item())
                val_total += img.shape[0]

        train_loss /= max(1, train_total)
        val_loss /= max(1, val_total)
        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optim.param_groups[0]["lr"],
        })
        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(
                f"[cnn-train] epoch {epoch+1}/{args.epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
                f"lr={optim.param_groups[0]['lr']:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "img_size": int(cfg.get("img_size", 80)),
                    "grid_size": int(cfg.get("grid_size", 5)),
                    "cell_px": int(cfg.get("cell_px", 16)),
                },
                os.path.join(args.checkpoint_dir, "best_cnn_mlp.pth"),
            )

    log = {
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "history": history,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(args.checkpoint_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[cnn-train] DONE in {(time.time()-t0)/60:.1f}m | best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
