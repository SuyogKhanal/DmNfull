import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel),
            Conv1dBlock(out_ch, out_ch, kernel),
        ])
        self.cond_proj = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_ch * 2))
        self.residual_conv = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        scale, bias = self.cond_proj(cond).chunk(2, dim=-1)
        out = out * (scale.unsqueeze(-1) + 1) + bias.unsqueeze(-1)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class CellAlignedEncoder(nn.Module):
    """Grid-cell-aware encoder: 1x1 convs + cell-aligned AvgPool, zero cross-cell bleed."""

    IMG_FEAT_DIM = 128

    def __init__(self, grid_size: int = 5, cell_px: int = 16,
                 mid_ch: int = 32, embed_dim: int = 128):
        super().__init__()
        self.cell_px = cell_px
        self.pixel_embed = nn.Sequential(
            nn.Conv2d(3, mid_ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1),
            nn.ReLU(),
        )
        self.cell_pool = nn.AvgPool2d(kernel_size=cell_px, stride=cell_px)
        self.proj = nn.Linear(grid_size * grid_size * mid_ch, embed_dim)
        self.IMG_FEAT_DIM = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes (B, C, H, W) or (B, T, C, H, W) float32 [0,1] images."""
        assert x.dtype == torch.float32, (
            f"CellAlignedEncoder expects float32 input, got {x.dtype}. "
            "Normalise images to [0,1] before calling."
        )
        assert x.max() <= 1.0 + 1e-5, (
            f"CellAlignedEncoder expects images in [0,1], got max={x.max():.4f}."
        )
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            out = self._encode(x.flatten(0, 1))    # (B*T, embed_dim)
            return out.unflatten(0, (B, T))         # (B, T, embed_dim)
        return self._encode(x)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_embed(x)   # (N, mid_ch, H, W)
        x = self.cell_pool(x)     # (N, mid_ch, grid_size, grid_size)
        x = x.flatten(1)          # (N, grid_size^2 * mid_ch)
        return self.proj(x)       # (N, embed_dim)


class UNet1D(nn.Module):

    def __init__(
        self,
        action_dim: int = 4,      # FIX: was 1 — now 4 for one-hot
        obs_dim: int = 14,
        obs_horizon: int = 4,
        pred_horizon: int = 3,
        dim: int = 64,
        dim_mults: tuple = (1, 2, 4),
        use_vision: bool = False,
        grid_size: int = 5,
        cell_px: int = 16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_vision = use_vision

        time_dim = dim * 4
        obs_flat_dim = obs_dim * obs_horizon

        if use_vision:
            self.cell_encoder = CellAlignedEncoder(
                grid_size=grid_size, cell_px=cell_px,
            )
            self.img_feat_dim = self.cell_encoder.IMG_FEAT_DIM
            obs_flat_dim += self.img_feat_dim * obs_horizon
        else:
            self.img_feat_dim = 0

        cond_dim = time_dim + obs_flat_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim), nn.Mish(),
            nn.Linear(time_dim, time_dim),
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_flat_dim, cond_dim), nn.Mish(),
            nn.Linear(cond_dim, obs_flat_dim),
        )

        dims = [action_dim] + [dim * m for m in dim_mults]
        self.encoder = nn.ModuleList([
            ResidualBlock1D(dims[i], dims[i + 1], cond_dim) for i in range(len(dims) - 1)
        ])
        self.decoder = nn.ModuleList([
            ResidualBlock1D(dims[i + 1] * 2, dims[i], cond_dim)
            for i in reversed(range(len(dims) - 1))
        ])
        self.final_conv = nn.Conv1d(dims[0], action_dim, 1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_state: torch.Tensor,
        obs_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predicts the noise added to noisy_actions given obs conditioning."""
        B = noisy_actions.shape[0]

        assert noisy_actions.shape == (B, self.action_dim, self.pred_horizon), (
            f"noisy_actions shape mismatch: expected ({B},{self.action_dim},{self.pred_horizon}), "
            f"got {tuple(noisy_actions.shape)}."
        )

        t_emb = self.time_mlp(timestep)
        obs_flat = obs_state.reshape(B, -1)

        if self.use_vision and obs_images is not None:
            if obs_images.ndim == 5 and obs_images.shape[-1] in (1, 3):
                obs_images = obs_images.permute(0, 1, 4, 2, 3).contiguous()
            img_feat = self.cell_encoder(obs_images)    # (B, T, embed_dim)
            obs_flat = torch.cat([obs_flat, img_feat.reshape(B, -1)], dim=-1)
        elif self.use_vision:
            obs_flat = torch.cat([
                obs_flat,
                torch.zeros(B, self.img_feat_dim * self.obs_horizon, device=obs_state.device),
            ], dim=-1)

        cond = torch.cat([t_emb, self.obs_encoder(obs_flat)], dim=-1)
        x = noisy_actions
        skips = []
        for block in self.encoder:
            x = block(x, cond); skips.append(x)
        for block in self.decoder:
            x = torch.cat([x, skips.pop()], dim=1); x = block(x, cond)
        return self.final_conv(x)


class DDPMScheduler:

    def __init__(self, num_steps: int = 200):    # FIX: was 100, now 200
        self.num_diffusion_steps = num_steps      # FIX: added alias used by run_policy.py
        self.num_steps = num_steps
        s = 0.008
        t = torch.linspace(0, num_steps, num_steps + 1) / num_steps
        ab = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        ab = ab / ab[0]
        betas = torch.clamp(1 - ab[1:] / ab[:-1], max=0.999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        for name, val in [
            ("betas", betas), ("alphas", alphas),
            ("alphas_cumprod", alphas_cumprod),
            ("sqrt_alphas_cumprod", alphas_cumprod.sqrt()),
            ("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt()),
        ]:
            setattr(self, name, val)

    def add_noise(self, x0, noise, t):
        """Adds DDPM noise at timestep t to x0."""
        s = self.sqrt_alphas_cumprod[t.cpu()].to(x0.device)
        sm = self.sqrt_one_minus_alphas_cumprod[t.cpu()].to(x0.device)
        while s.dim() < x0.dim():
            s = s.unsqueeze(-1); sm = sm.unsqueeze(-1)
        return s * x0 + sm * noise

    @torch.no_grad()
    def denoise(self, model, obs_state, device, obs_images=None):
        """Runs the full reverse diffusion chain; returns denoised action sequence."""
        B = obs_state.shape[0]
        x = torch.randn(B, model.action_dim, model.pred_horizon, device=device)

        for t in reversed(range(self.num_steps)):
            ts = torch.full((B,), t, dtype=torch.long, device=device)
            eps_pred = model(x, ts, obs_state, obs_images=obs_images)

            alpha     = self.alphas[t].to(device)
            alpha_bar = self.alphas_cumprod[t].to(device)
            beta      = self.betas[t].to(device)

            x0_pred = ((x - (1 - alpha_bar).sqrt() * eps_pred) / alpha_bar.sqrt()).clamp(-1, 1)

            if t > 0:
                x = (
                    alpha.sqrt() * (1 - self.alphas_cumprod[t-1].to(device))
                    / (1 - alpha_bar) * x
                    + self.alphas_cumprod[t-1].to(device).sqrt() * beta
                    / (1 - alpha_bar) * x0_pred
                    + beta.sqrt() * torch.randn_like(x)
                )
            else:
                x = x0_pred

        return x


class MazeDiffusionPolicy:

    def __init__(
        self,
        obs_dim: int = 14,
        action_dim: int = 4,          # FIX: was 1
        obs_horizon: int = 4,
        pred_horizon: int = 3,
        num_diffusion_steps: int = 200,   # FIX: was 100
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_vision: bool = False,
        dim: int = 64,
        dim_mults: tuple = (1, 2, 4),
        grid_size: int = 5,
        cell_px: int = 16,
        img_size: int = 80,           # accepted but unused — cell_px derived from grid
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.device = torch.device(device)
        self.use_vision = use_vision

        self.model = UNet1D(
            action_dim, obs_dim, obs_horizon, pred_horizon,
            dim=dim, dim_mults=dim_mults, use_vision=use_vision,
            grid_size=grid_size, cell_px=cell_px,
        ).to(self.device)
        self.scheduler = DDPMScheduler(num_diffusion_steps)

    def get_action(self, obs_seq) -> int:
        """Denoises an action sequence and argmax-decodes the first one-hot action."""
        self.model.eval()

        if isinstance(obs_seq, dict):
            obs_t = torch.FloatTensor(obs_seq["state"]).unsqueeze(0).to(self.device)
            obs_images = None
            if self.use_vision and "image" in obs_seq:
                img_arr = np.array(obs_seq["image"])
                assert img_arr.dtype == np.float32, "Images must be float32 [0,1]."
                obs_images = torch.FloatTensor(img_arr).unsqueeze(0).to(self.device)
            elif self.use_vision:
                from envs.maze_env import IMG_SIZE
                obs_images = torch.zeros(
                    1, self.obs_horizon, 3, IMG_SIZE, IMG_SIZE, device=self.device
                )
        else:
            obs_t = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)
            obs_images = None

        action_seq = self.scheduler.denoise(
            self.model, obs_t, self.device, obs_images=obs_images
        )

        # FIX: one-hot argmax decode — was scalar round()
        raw = action_seq[0, :, 0]           # (4,) — first pred step, all channels
        action_int = int(raw.argmax().item())
        return action_int

    def save(self, path: str):
        """Saves model state dict to path."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved: {path}")

    def load(self, path: str):
        """Loads model state dict from path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device), strict=True)
        print(f"Model loaded: {path}")