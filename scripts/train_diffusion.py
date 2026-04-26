import os
import sys
import json
import glob
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.diffusion_policy import MazeDiffusionPolicy

DEMO_DIR        = "demos"
CHECKPOINT_DIR  = "checkpoints"
EPOCHS          = 5000
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


class MazeDemoDataset(Dataset):

    def __init__(self, demo_dir, obs_horizon, pred_horizon):
        self.obs_horizon  = obs_horizon
        self.pred_horizon = pred_horizon
        self.samples      = []

        demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.json")))
        if not demo_files:
            raise FileNotFoundError(f"No demo JSON files found in {demo_dir}")

        for fpath in demo_files:
            with open(fpath, "r") as f:
                demo = json.load(f)

            obs_arr = np.array(demo["observations"], dtype=np.float32)
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

        print(f"Loaded {len(demo_files)} demos → {len(self.samples)} training windows "
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
    p.add_argument("--checkpoint_dir",  type=str,   default=CHECKPOINT_DIR)
    p.add_argument("--epochs",          type=int,   default=EPOCHS)
    p.add_argument("--batch_size",      type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",              type=float, default=LR)
    p.add_argument("--resume",          action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MazeDemoDataset(
        demo_dir=args.demo_dir,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )

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

    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"Dataset size: {len(dataset)} | Batches/epoch: {len(loader)}")

    best_loss  = float("inf")
    loss_curve = []

    for epoch in range(args.epochs):
        lr_now = warmup_cosine_lr(epoch, WARMUP_EPOCHS, args.epochs, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        policy.model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for obs_seq, img_seq, action_seq in loader:
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

        avg_loss = epoch_loss / max(1, n_batches)
        loss_curve.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.6f} | lr={lr_now:.2e}")

        if avg_loss < best_loss:
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

    print(f"Training complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()