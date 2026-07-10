"""State-based Diffusion Policy + Diff-DAgger uncertainty head.

This is a self-contained port of the diffusion policy used in Diff-DAgger
(Lee et al., 2024) adapted to the robosuite UR5e setup. It is a *pure*
conditional action-denoising imitation learner (Chi et al., 2023):

  - obs_horizon  past observations are flattened into a global condition vector
  - pred_horizon actions are denoised by a 1D conditional U-Net (FiLM cond.)
  - act_horizon  of the predicted actions are executed before re-planning

Normalization (canonical DP / DP3): per-dim min/max -> [-1, 1] for both the
state observation and the action, stored as buffers so a checkpoint is fully
self-contained and train/eval use identical transforms.

Diff-DAgger uncertainty (the contribution of the paper):
  The *training* loss of the diffusion model is reused at deployment as an
  out-of-distribution score. For the current observation and the policy's own
  predicted action, we estimate the expected diffusion loss

      L(o, a, pi) = E_{eps~N(0,I), t~U(1,T)} || target - f_theta(o, a_t, t) ||^2   (Eqn 2)

  by averaging the MSE over *all* T denoising timesteps (repeated
  `batch_multiplier` times) and `num_per_batch` fresh noise draws -- exactly the
  `get_avg_diffusion_loss_ndata` estimator from the official DiffDAgger code.

  A baseline distribution of this score is built over the training set; its
  empirical CDF gives a quantile threshold (alpha). The robot queries the expert
  when the score exceeds the threshold for `patience` of the last
  `patience_window` decisions (the K-patience rule from the paper).
"""
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .conditional_unet1d import ConditionalUnet1D


def _to_buf(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class EmpiricalCDF:
    """Empirical CDF over a 1D loss sample, with quantile inversion.

    `quantile(alpha)` reproduces the nearest-rank threshold of the official
    DiffDAgger `util/cdf.py` (`sorted[int(n*alpha)]`), which is what actually
    gates queries. `__call__` returns a plain right-continuous ECDF value
    (fraction of training losses <= value); it is informational only (reported
    as cdf_value) and does not affect the query decision, which compares the raw
    loss against `loss_threshold`.
    """

    def __init__(self, data: List[float]):
        self.sorted = np.sort(np.asarray(data, dtype=np.float64))
        n = len(self.sorted)
        self.cdf = np.arange(1, n + 1, dtype=np.float64) / n

    def __call__(self, value: float) -> float:
        idx = np.searchsorted(self.sorted, value, side="right")
        return float(idx) / len(self.sorted)

    def quantile(self, q: float) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        idx = min(int(len(self.sorted) * q), len(self.sorted) - 1)
        return float(self.sorted[idx])


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        act_dim: int,
        state_dim: int,
        normalization: Dict[str, np.ndarray],
        # horizons
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        # diffusion
        num_diffusion_iters: int = 16,
        prediction_type: str = "v_prediction",   # "epsilon" | "sample" | "v_prediction"
        scheduler_type: str = "ddim",             # "ddpm" | "ddim"
        # unet
        diffusion_step_embed_dim: int = 64,
        down_dims=(64, 128, 256),
        # Diff-DAgger uncertainty / query hyperparameters
        alpha: float = 0.99,            # quantile threshold for OOD
        patience_window: int = 2,       # K-window of recent decisions
        patience: int = 2,              # #violations within the window to query
        batch_multiplier: int = 1,      # repeats of the full timestep sweep
        num_per_batch: int = 1,         # fresh-noise averaging passes
        # image modality (02_..md #4): only the encoder changes; loss/uncertainty unchanged
        modality: str = "state",        # "state" | "image" | "hybrid"
        image_channels: int = 3,
        keypoints: int = 32,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.modality = modality
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.num_diffusion_iters = num_diffusion_iters
        self.prediction_type = prediction_type
        self.scheduler_type = scheduler_type

        # uncertainty hyperparameters
        self.alpha = alpha
        self.patience_window = patience_window
        self.patience = patience
        self.batch_multiplier = batch_multiplier
        self.num_per_batch = num_per_batch

        # ---- normalization buffers (min/max -> [-1, 1]) ----
        self.register_buffer("state_min", _to_buf(normalization["state_min"]))
        self.register_buffer("state_max", _to_buf(normalization["state_max"]))
        self.register_buffer("action_min", _to_buf(normalization["action_min"]))
        self.register_buffer("action_max", _to_buf(normalization["action_max"]))

        # image/hybrid conditioning: a spatial-softmax keypoint embedding per frame,
        # stacked over obs_horizon (structurally identical to the flattened state).
        self.image_encoder = None
        if modality in ("image", "hybrid"):
            from .image_encoder import SpatialSoftmaxEncoder
            self.image_encoder = SpatialSoftmaxEncoder(in_channels=image_channels,
                                                       keypoints=keypoints)
            img_dim = self.image_encoder.out_dim
            per_frame = img_dim + (state_dim if modality == "hybrid" else 0)
        else:
            per_frame = state_dim
        global_cond_dim = per_frame * obs_horizon
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=tuple(down_dims),
        )

        # Diff-DAgger's recipe: DDIM + v_prediction with few steps converges
        # faster and performs better than DDPM-100 + epsilon (paper Sec. IV-E).
        sched_cls = DDIMScheduler if scheduler_type == "ddim" else DDPMScheduler
        self.noise_scheduler = sched_cls(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type=prediction_type,
        )

        # OOD threshold state (set by `set_threshold_from_dataset`)
        self.loss_cdf: Optional[EmpiricalCDF] = None
        self.loss_threshold: float = float("inf")
        self._violations = deque(maxlen=patience_window)

    # ---------------- normalization ----------------
    def normalize_state(self, s):
        rng = (self.state_max - self.state_min).clamp(min=1e-6)
        return 2.0 * (s - self.state_min) / rng - 1.0

    def normalize_action(self, a):
        rng = (self.action_max - self.action_min).clamp(min=1e-6)
        return 2.0 * (a - self.action_min) / rng - 1.0

    def unnormalize_action(self, a):
        rng = (self.action_max - self.action_min).clamp(min=1e-6)
        return (a + 1.0) / 2.0 * rng + self.action_min

    # ---------------- conditioning ----------------
    def _encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs (B, obs_horizon, C, H, W) in [0,1] -> (B, obs_horizon, img_dim)."""
        B, oh = imgs.shape[0], imgs.shape[1]
        flat = imgs.reshape(B * oh, *imgs.shape[2:]).float()
        emb = self.image_encoder(flat)             # (B*oh, img_dim)
        return emb.reshape(B, oh, -1)

    def encode_obs(self, obs_seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        """-> (B, global_cond_dim). state: normalized flat state; image: keypoint
        embedding per frame; hybrid: [image_emb, normalized_state] per frame."""
        if self.modality == "state":
            return self.normalize_state(obs_seq["state"].float()).flatten(start_dim=1)
        img_emb = self._encode_images(obs_seq["image"])            # (B, oh, img_dim)
        if self.modality == "image":
            return img_emb.flatten(start_dim=1)
        state = self.normalize_state(obs_seq["state"].float())     # hybrid
        return torch.cat([img_emb, state], dim=-1).flatten(start_dim=1)

    # ---------------- core diffusion MSE ----------------
    def _diffusion_mse(self, obs_cond, naction, timesteps, noise):
        """MSE between the model prediction and the prediction target for the
        given (already-normalized) actions, conditioning, timesteps and noise."""
        noisy = self.noise_scheduler.add_noise(naction, noise, timesteps)
        pred = self.noise_pred_net(noisy, timesteps, global_cond=obs_cond)
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = naction
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(naction, noise, timesteps)
        else:
            raise ValueError(self.prediction_type)
        return F.mse_loss(pred, target)

    # ---------------- training ----------------
    def compute_loss(self, obs_seq, action_seq):
        """Standard DP training loss (random timestep + noise)."""
        obs_cond = self.encode_obs(obs_seq)
        naction = self.normalize_action(action_seq.float())
        B = naction.shape[0]
        noise = torch.randn_like(naction)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,),
            device=naction.device,
        ).long()
        return self._diffusion_mse(obs_cond, naction, timesteps, noise)

    # ---------------- inference ----------------
    @torch.no_grad()
    def _denoise(self, obs_cond):
        B = obs_cond.shape[0]
        device = obs_cond.device
        naction = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(naction, k, global_cond=obs_cond)
            naction = self.noise_scheduler.step(noise_pred, k, naction).prev_sample
        return naction  # (B, pred_horizon, act_dim), normalized

    @torch.no_grad()
    def get_action(self, obs_seq):
        """Returns act_horizon executable actions (B, act_horizon, act_dim)."""
        was_training = self.training
        self.eval()
        obs_cond = self.encode_obs(obs_seq)
        naction = self._denoise(obs_cond)
        start = self.obs_horizon - 1
        action = naction[:, start:start + self.act_horizon]
        if was_training:
            self.train()
        return self.unnormalize_action(action)

    # ---------------- Diff-DAgger uncertainty ----------------
    @torch.no_grad()
    def _avg_diffusion_loss(self, obs_cond, naction):
        """Eqn (2) estimator: average diffusion MSE over the full timestep sweep
        (repeated `batch_multiplier` times) and `num_per_batch` noise draws.

        obs_cond: (1, gcd)   naction: (1, pred_horizon, act_dim), normalized.
        """
        T = self.noise_scheduler.config.num_train_timesteps
        n = T * self.batch_multiplier
        obs_rep = obs_cond.repeat(n, 1)
        act_rep = naction.repeat(n, 1, 1)
        timesteps = (torch.arange(n, device=naction.device).long() % T)
        total = 0.0
        for _ in range(self.num_per_batch):
            noise = torch.randn_like(act_rep)
            total += self._diffusion_mse(obs_rep, act_rep, timesteps, noise).item()
        return total / self.num_per_batch

    @torch.no_grad()
    def plan(self, obs_seq):
        """Denoise once and return (executable_action_chunk, naction_norm).

        `naction_norm` is the full normalized predicted sequence, cached so that
        per-step uncertainty can be evaluated against the current observation
        between re-plans (receding-horizon execution).
        """
        was_training = self.training
        self.eval()
        obs_cond = self.encode_obs(obs_seq)            # (1, gcd)
        naction = self._denoise(obs_cond)              # (1, pred_h, act_dim)
        start = self.obs_horizon - 1
        action = self.unnormalize_action(naction[:, start:start + self.act_horizon])
        if was_training:
            self.train()
        return action, naction

    @torch.no_grad()
    def uncertainty(self, obs_seq, naction_norm):
        """Diff-DAgger score for the current obs against a (cached) normalized
        predicted action sequence."""
        obs_cond = self.encode_obs(obs_seq)
        return self._avg_diffusion_loss(obs_cond, naction_norm)

    @torch.no_grad()
    def get_action_and_uncertainty(self, obs_seq):
        """Convenience: (executable_action, diffusion_loss) for one observation."""
        action, naction = self.plan(obs_seq)
        return action, self._avg_diffusion_loss(self.encode_obs(obs_seq), naction)

    @torch.no_grad()
    def set_threshold_from_dataset(self, windows, num_iter: int = 4, max_points: int = 50000):
        """Build the OOD baseline: expected diffusion loss for each training
        window (using the *ground-truth* expert action), then the empirical CDF
        and the alpha-quantile threshold (paper Sec. III-C / `get_stats_from_dataset`).

        `windows` is an iterable of (state_seq, action_seq) numpy/tensor pairs
        in *raw* units (obs_horizon x state_dim, pred_horizon x act_dim).
        """
        was_training = self.training
        self.eval()
        device = self.state_min.device
        losses: List[float] = []
        for _ in range(num_iter):
            for obs_win, action_seq in windows:
                # obs_win is either a raw state_seq (state modality) or an obs dict
                # {"state":..,"image":..} (image/hybrid) — build the tensor obs.
                if isinstance(obs_win, dict):
                    obs = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)[None]
                           for k, v in obs_win.items()}
                else:
                    obs = {"state": torch.as_tensor(np.asarray(obs_win), dtype=torch.float32, device=device)[None]}
                a = torch.as_tensor(np.asarray(action_seq), dtype=torch.float32, device=device)[None]
                obs_cond = self.encode_obs(obs)
                naction = self.normalize_action(a)
                losses.append(self._avg_diffusion_loss(obs_cond, naction))
                if len(losses) >= max_points:
                    break
            if len(losses) >= max_points:
                break
        self.loss_cdf = EmpiricalCDF(losses)
        self.loss_threshold = self.loss_cdf.quantile(self.alpha)
        if was_training:
            self.train()
        return losses

    # ---- per-rollout query state ----
    def reset_query(self):
        self._violations = deque(maxlen=self.patience_window)

    def cdf_value(self, diffusion_loss: float) -> float:
        return self.loss_cdf(diffusion_loss) if self.loss_cdf is not None else 0.0

    def register_and_query(self, diffusion_loss: float) -> bool:
        """Append a violation flag and return whether to query the expert now."""
        self._violations.append(diffusion_loss > self.loss_threshold)
        return sum(self._violations) >= self.patience
