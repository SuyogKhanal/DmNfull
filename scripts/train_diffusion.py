import os
import sys
import json
import glob
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.diffusion_policy import MazeDiffusionPolicy

DEMO_DIR        = "demos"
CHECKPOINT_DIR  = "checkpoints"
EPOCHS          = 2000
BATCH_SIZE      = 32
LR              = 3e-4
OBS_HORIZON     = 4
PRED_HORIZON    = 3
OBS_DIM         = 14
ACTION_DIM      = 4
NUM_DIFF_STEPS  = 200
IMG_SIZE        = 80
GRID_SIZE       = 5
CELL_PX         = 16
DIM             = 128
DIM_MULTS       = (1, 2, 4, 8)
EMA_DECAY       = 0.995
WARMUP_EPOCHS   = 50

ROTATE_ACTION_90  = {0: 3, 1: 2, 2: 0, 3: 1}
ROTATE_ACTION_180 = {0: 1, 1: 0, 2: 3, 3: 2}
ROTATE_ACTION_270 = {0: 2, 1: 3, 2: 1, 3: 0}


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    k = {90: 3, 180: 2, 270: 1}[angle]
    return np.rot90(img, k=k, axes=(0, 1)).copy()


def rotate_state(state: np.ndarray, angle: int) -> np.ndarray:
    s = state.copy()
    r, c   = s[0], s[1]
    gr, gc = s[2], s[3]
    if angle == 90:
        s[0], s[1] = c,        1.0 - r
        s[2], s[3] = gc,       1.0 - gr
    elif angle == 180:
        s[0], s[1] = 1.0 - r,  1.0 - c
        s[2], s[3] = 1.0 - gr, 1.0 - gc
    elif angle == 270:
        s[0], s[1] = 1.0 - c,  r
        s[2], s[3] = 1.0 - gc, gr
    neigh = s[4:13].reshape(3, 3)
    k = {90: 3, 180: 2, 270: 1}[angle]
    neigh = np.rot90(neigh, k=k)
    s[4:13] = neigh.reshape(9)
    return s


def rotate_actions(acts: np.ndarray, angle: int) -> np.ndarray:
    table = {90: ROTATE_ACTION_90, 180: ROTATE_ACTION_180, 270: ROTATE_ACTION_270}[angle]
    return np.array([table[int(a)] for a in acts], dtype=np.int64)


def make_windows(obs_arr, img_arr, act_arr, obs_horizon, pred_horizon):
    """Sliding-window extraction from a single trajectory (original or rotated)."""
    samples = []
    T = len(obs_arr)
    if T < pred_horizon:
        return samples
    for t in range(T - pred_horizon + 1):
        obs_seq = np.zeros((obs_horizon, OBS_DIM),               dtype=np.float32)
        img_seq = np.zeros((obs_horizon, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for i in range(obs_horizon):
            src = t - obs_horizon + 1 + i
            if src >= 0:
                obs_seq[i] = obs_arr[src]
                img_seq[i] = img_arr[src]
        acts = act_arr[t:t + pred_horizon]
        action_seq = np.zeros((ACTION_DIM, pred_horizon), dtype=np.float32)
        for i, a in enumerate(acts):
            action_seq[int(a), i] = 1.0
        samples.append((obs_seq, img_seq, action_seq))
    return samples


def _resolve_demo_files(demo_dir, demo_paths):
    """Resolve the list of demo JSON files to use for training.

    Resolution order:
      1. If demo_paths is set (comma-separated globs), expand each glob
         (recursive when '**' is in the pattern) and union the results.
         This is what active_loop uses to combine baseline demos with the
         per-profile collected demos.
      2. Else, fall back to demo_dir/*.json (legacy behaviour).

    Files are de-duplicated and sorted for deterministic ordering.
    """
    files = []
    if demo_paths:
        patterns = [p.strip() for p in demo_paths.split(",") if p.strip()]
        for pat in patterns:
            recursive = "**" in pat
            matched = glob.glob(pat, recursive=recursive)
            if not matched:
                print(f"[train] WARNING: glob '{pat}' matched no files.")
            files.extend(matched)
    else:
        files = glob.glob(os.path.join(demo_dir, "*.json"))

    files = sorted({os.path.abspath(f) for f in files})
    return files


class MazeDemoDataset(Dataset):

    def __init__(self, demo_dir, obs_horizon, pred_horizon, demo_paths=None):
        self.obs_horizon  = obs_horizon
        self.pred_horizon = pred_horizon
        self.samples      = []

        demo_files = _resolve_demo_files(demo_dir, demo_paths)
        if not demo_files:
            src = demo_paths if demo_paths else demo_dir
            raise FileNotFoundError(f"No demo JSON files found for: {src}")
        print(f"[train] Demo files resolved: {len(demo_files)} (loading + 3x rotation augment...)")
        demo_iter = tqdm(demo_files, desc="demos", unit="file", dynamic_ncols=True, leave=False)

        for fpath in demo_iter:
            with open(fpath, "r") as f:
                demo = json.load(f)

            if "observations" not in demo or "actions" not in demo:
                continue

            obs_arr = np.array(demo.get("observations"), dtype=np.float32)
            act_arr = np.array(demo["actions"],      dtype=np.int64)

            if "images" in demo and demo["images"] is not None and len(demo["images"]) > 0:
                raw_imgs = np.array(demo["images"], dtype=np.uint8)
                if raw_imgs.shape[1] != IMG_SIZE or raw_imgs.shape[2] != IMG_SIZE:
                    import cv2
                    resized = [cv2.resize(f, (IMG_SIZE, IMG_SIZE),
                                          interpolation=cv2.INTER_AREA) for f in raw_imgs]
                    img_arr = np.stack(resized, axis=0)
                else:
                    img_arr = raw_imgs
            else:
                img_arr = np.zeros((len(obs_arr), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            self.samples.extend(make_windows(obs_arr, img_arr, act_arr, obs_horizon, pred_horizon))

            for angle in (90, 180, 270):
                rot_obs = np.stack([rotate_state(s, angle) for s in obs_arr], axis=0)
                rot_img = np.stack([rotate_image(img, angle) for img in img_arr], axis=0)
                rot_act = rotate_actions(act_arr, angle)
                self.samples.extend(make_windows(rot_obs, rot_img, rot_act, obs_horizon, pred_horizon))

        print(f"[train] Loaded {len(demo_files)} demos → {len(self.samples)} training windows "
              f"(1 original + 3 rotations × sliding windows per demo)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_seq, img_seq, action_seq = self.samples[idx]
        img_norm = img_seq.astype(np.float32) / 255.0
        brightness = np.random.uniform(0.85, 1.15)
        contrast   = np.random.uniform(0.90, 1.10)
        img_norm   = np.clip(img_norm * brightness * contrast, 0.0, 1.0)
        return (
            torch.from_numpy(obs_seq).float(),
            torch.from_numpy(img_norm).float(),
            torch.from_numpy(action_seq).float(),
        )


def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.data.copy_(b.data)


def warmup_cosine_lr(epoch: int, warmup: int, total: int, base_lr: float) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * progress))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_dir",        type=str,   default=DEMO_DIR)
    p.add_argument("--demo_paths",      type=str,   default=None,
                   help="Comma-separated glob patterns for demo JSON files. When set, "
                        "this OVERRIDES --demo_dir. Use '**' in a pattern to recurse "
                        "(e.g. 'demos/*.json,demos/active_loop/p3/**/*.json').")
    p.add_argument("--checkpoint_dir",  type=str,   default=CHECKPOINT_DIR)
    p.add_argument("--epochs",          type=int,   default=EPOCHS)
    p.add_argument("--batch_size",      type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",              type=float, default=LR)
    p.add_argument("--resume",          action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 72)
    print("[train] Starting diffusion-policy training")
    print("=" * 72)
    print(f"[train] checkpoint_dir : {args.checkpoint_dir}")
    print(f"[train] resume         : {args.resume}")
    print(f"[train] epochs         : {args.epochs}")
    print(f"[train] batch_size     : {args.batch_size}")
    print(f"[train] lr             : {args.lr}")
    if args.demo_paths:
        print(f"[train] demo_paths     : {args.demo_paths}")
    else:
        print(f"[train] demo_dir       : {args.demo_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device         : {device}")

    print("[train] Loading demo dataset...")
    t0 = time.time()
    dataset = MazeDemoDataset(
        demo_dir=args.demo_dir,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
        demo_paths=args.demo_paths,
    )
    print(f"[train] Dataset ready in {time.time()-t0:.1f}s | windows={len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    print(f"[train] Batches/epoch  : {len(loader)}")

    policy = MazeDiffusionPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
        num_diffusion_steps=NUM_DIFF_STEPS,
        dim=DIM,
        dim_mults=DIM_MULTS,
        use_vision=True,
        img_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        cell_px=CELL_PX,
        device=str(device),
    )
    policy.model.to(device)

    if args.resume:
        resume_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                policy.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            else:
                policy.model.load_state_dict(ckpt, strict=True)
            print(f"Resumed from checkpoint: {resume_path}")
        else:
            print(f"WARNING: --resume given but no checkpoint at {resume_path}; starting from scratch.")

    trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    ema_model = copy.deepcopy(policy.model)
    for p in ema_model.parameters():
        p.requires_grad = False

    print(f"[train] Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"[train] Dataset size: {len(dataset)} | Batches/epoch: {len(loader)}")
    print("[train] Starting epoch loop ↓ (best_loss tracked, EMA on, cosine LR with warmup)\n")

    best_loss  = float("inf")
    loss_curve = []
    train_t0 = time.time()

    epoch_pbar = tqdm(
        range(args.epochs),
        desc="train",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_pbar:
        lr_now = warmup_cosine_lr(epoch, WARMUP_EPOCHS, args.epochs, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        policy.model.train()
        epoch_loss = 0.0
        n_batches  = 0

        batch_pbar = tqdm(
            loader,
            desc=f"epoch {epoch+1}/{args.epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for obs_seq, img_seq, action_seq in batch_pbar:
            obs_seq    = obs_seq.to(device)
            img_seq    = img_seq.to(device)
            action_seq = action_seq.to(device)

            B     = action_seq.shape[0]
            t     = torch.randint(0, NUM_DIFF_STEPS, (B,), device=device).long()
            noise = torch.randn_like(action_seq)
            noisy_actions = policy.scheduler.add_noise(action_seq, noise, t)

            pred_noise = policy.model(
                noisy_actions, t, obs_seq, obs_images=img_seq,
            )

            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            ema_update(ema_model, policy.model, EMA_DECAY)

            epoch_loss += loss.item()
            n_batches  += 1
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

        avg_loss = epoch_loss / max(1, n_batches)
        loss_curve.append(avg_loss)

        improved = avg_loss < best_loss
        epoch_pbar.set_postfix(
            loss=f"{avg_loss:.6f}",
            best=f"{min(best_loss, avg_loss):.6f}",
            lr=f"{lr_now:.2e}",
            improved="*" if improved else " ",
        )
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - train_t0
            tqdm.write(
                f"[train] epoch {epoch+1}/{args.epochs} | "
                f"loss={avg_loss:.6f} | best={min(best_loss, avg_loss):.6f} | "
                f"lr={lr_now:.2e} | elapsed={elapsed/60:.1f}m"
            )

        if improved:
            best_loss = avg_loss
            policy.save(os.path.join(args.checkpoint_dir, "best_model.pth"))

            ema_ckpt = {
                "ema_policy":   ema_model.state_dict(),
                "obs_horizon":  OBS_HORIZON,
                "pred_horizon": PRED_HORIZON,
                "action_dim":   ACTION_DIM,
                "epoch":        epoch + 1,
                "loss":         avg_loss,
            }
            torch.save(ema_ckpt, os.path.join(args.checkpoint_dir, "best_model_ema.pth"))

            meta = {
                "obs_dim":        OBS_DIM,
                "action_dim":     ACTION_DIM,
                "obs_horizon":    OBS_HORIZON,
                "pred_horizon":   PRED_HORIZON,
                "num_diff_steps": NUM_DIFF_STEPS,
                "dim":            DIM,
                "dim_mults":      list(DIM_MULTS),
                "grid_size":      GRID_SIZE,
                "cell_px":        CELL_PX,
                "img_size":       IMG_SIZE,
                "use_vision":     True,
                "epoch":          epoch + 1,
                "loss":           avg_loss,
                "action_encoding": "one_hot",
            }
            with open(os.path.join(args.checkpoint_dir, "best_model_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

    log = {
        "epochs":     args.epochs,
        "final_loss": loss_curve[-1] if loss_curve else None,
        "best_loss":  best_loss,
        "loss_curve": loss_curve,
    }
    with open(os.path.join(args.checkpoint_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    elapsed = time.time() - train_t0
    print(
        f"\n[train] DONE in {elapsed/60:.1f}m | "
        f"best_loss={best_loss:.6f} | final_loss={loss_curve[-1] if loss_curve else float('nan'):.6f} | "
        f"checkpoints={args.checkpoint_dir}"
    )


if __name__ == "__main__":
    main()