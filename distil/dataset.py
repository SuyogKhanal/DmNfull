"""Action-chunked dataset over aggregated expert trajectories.

A trajectory is a dict {"state": (L, state_dim), "action": (L, act_dim)} of raw
float32 arrays. The dataset yields Diffusion-Policy windows:

    obs_seq    : obs_horizon consecutive states  (conditioning)
    action_seq : pred_horizon consecutive actions (denoising target)

with the standard repeat-first / repeat-last boundary padding so every
transition contributes a window. Per-dim min/max normalization stats are
computed over the *current* set of trajectories (recomputed every DAgger round
as data is aggregated).
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    def __init__(self, trajectories: List[Dict[str, np.ndarray]], obs_horizon: int, pred_horizon: int):
        assert len(trajectories) > 0, "empty demonstration set"
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.states = [np.asarray(t["state"], dtype=np.float32) for t in trajectories]
        self.actions = [np.asarray(t["action"], dtype=np.float32) for t in trajectories]
        # optional per-step RGB frames (H,W,3 uint8) for the image modality.
        self.has_images = all("image" in t for t in trajectories)
        self.images = [np.asarray(t["image"], dtype=np.uint8) for t in trajectories] \
            if self.has_images else None
        self.num_traj = len(self.states)
        self.state_dim = self.states[0].shape[1]
        self.act_dim = self.actions[0].shape[1]

        pad_before = obs_horizon - 1
        pad_after = pred_horizon - obs_horizon
        self.slices: List[Tuple[int, int, int]] = []
        for ti in range(self.num_traj):
            L = self.actions[ti].shape[0]
            for start in range(-pad_before, L - pred_horizon + pad_after + 1):
                self.slices.append((ti, start, start + pred_horizon))
        if not self.slices:  # very short trajectories
            for ti in range(self.num_traj):
                self.slices.append((ti, -pad_before, -pad_before + pred_horizon))

        self._compute_norm()

    def _compute_norm(self):
        all_s = np.concatenate(self.states, axis=0)
        all_a = np.concatenate(self.actions, axis=0)
        self.normalization = dict(
            state_min=all_s.min(0).astype(np.float32),
            state_max=all_s.max(0).astype(np.float32),
            action_min=all_a.min(0).astype(np.float32),
            action_max=all_a.max(0).astype(np.float32),
        )

    def get_normalization(self):
        return self.normalization

    def __len__(self):
        return len(self.slices)

    def _pad_state(self, ti, start):
        oh = self.obs_horizon
        arr = self.states[ti]
        L = arr.shape[0]
        s, e = max(0, start), min(L, start + oh)
        win = arr[s:e]
        if start < 0:
            win = np.concatenate([np.repeat(arr[0:1], -start, axis=0), win], axis=0)
        if win.shape[0] < oh:
            win = np.concatenate([win, np.repeat(win[-1:], oh - win.shape[0], axis=0)], axis=0)
        return win

    def _pad_image(self, ti, start):
        """obs_horizon window of frames as (oh, C, H, W) float in [0,1] (HWC->CHW/255)."""
        oh = self.obs_horizon
        arr = self.images[ti]                              # (L, H, W, 3) uint8
        L = arr.shape[0]
        s, e = max(0, start), min(L, start + oh)
        win = arr[s:e]
        if start < 0:
            win = np.concatenate([np.repeat(arr[0:1], -start, axis=0), win], axis=0)
        if win.shape[0] < oh:
            win = np.concatenate([win, np.repeat(win[-1:], oh - win.shape[0], axis=0)], axis=0)
        win = win.astype(np.float32) / 255.0
        return np.transpose(win, (0, 3, 1, 2))             # (oh, C, H, W)

    def _obs_window(self, index):
        ti, start, end = self.slices[index]
        obs = {"state": self._pad_state(ti, start)}
        if self.has_images:
            obs["image"] = self._pad_image(ti, start)
        return obs

    def _window(self, index):
        ti, start, end = self.slices[index]
        arr = self.actions[ti]
        L = arr.shape[0]
        s, e = max(0, start), min(L, end)
        act = arr[s:e]
        if start < 0:
            act = np.concatenate([np.repeat(arr[0:1], -start, axis=0), act], axis=0)
        if end > L:
            act = np.concatenate([act, np.repeat(arr[-1:], end - L, axis=0)], axis=0)
        return self._obs_window(index), act

    def __getitem__(self, index):
        obs_win, act = self._window(index)
        obs = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in obs_win.items()}
        return obs, torch.from_numpy(np.ascontiguousarray(act)).float()

    def iter_windows(self, max_windows: int = None, shuffle: bool = True, seed: int = 0):
        """Yield (obs_dict, action_seq) pairs for threshold calibration (obs_dict has
        'state' and, for image runs, 'image'). Shuffled before capping so the sample
        represents the WHOLE aggregated dataset (incl. new intervention windows)."""
        idx = np.arange(len(self.slices))
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        if max_windows is not None:
            idx = idx[:max_windows]
        for i in idx:
            yield self._window(int(i))


def collate(batch):
    keys = batch[0][0].keys()
    obs = {k: torch.stack([b[0][k] for b in batch], dim=0) for k in keys}
    act = torch.stack([b[1] for b in batch], dim=0)
    return obs, act
