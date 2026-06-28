"""Latent VISUAL embeddings of failures — the image-based clustering feature.

For the IMAGE-based PushT comparison (apples-to-apples with Diff-DAgger's
image-based diffusion policy) we cluster failures in a LATENT VISUAL space
instead of the 6-D geometric descriptor: each failure is summarised by R3M
(ResNet18) features over its rendered key-frames, then the round's failures are
dimensionality-reduced (numpy PCA) before clustering.

Only the *clustering feature* becomes visual — the prescription/reset geometry
(anchor t* state, on-policy re-roll seed, coverage centroid) is unchanged
(see ``clustering.cluster_failures`` / ``diversity`` / ``subtask_entry``).

Per-failure "trajectory embedding":
    start_frame.png  ─┐
    high_loss/t*.png  ├─ R3M(resnet18) → 512-d each → POOL → per-failure vector
    end_frame.png    ─┘   (pool="concat" → 1536-d, "mean" → 512-d)
Across the round's N failures: standardize + PCA (SVD) → compact dim (≈16),
clustered by the existing silhouette-swept agglomerative machinery.

Offline-safe: R3M weights are cached at ``~/.r3m/r3m_18/model.pt`` (verified) and
PCA is pure numpy (no sklearn/umap on the cluster). Robustness: every failure
path here is best-effort — a missing frame or an encoder import error makes
``embed_failure`` return ``None``, and the planner falls back to the geometric
feature for that round, so a bad embed can never crash the run.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

# R3M / torchvision are imported lazily inside FrameEmbedder so a geometric run
# never pays the torch-vision import cost, and an embed failure degrades cleanly.


class FrameEmbedder:
    """Embed rendered failure key-frames with a frozen visual encoder (R3M).

    Lazily loads the encoder once; caches per-frame embeddings by absolute path
    (the same start/end frame can recur across a round). Thread-unsafe by design
    (the planner runs single-threaded in the orchestrator process)."""

    def __init__(self, model_name: str = "r3m", pool: str = "concat",
                 device: Optional[str] = None, image_size: int = 224):
        self.model_name = str(model_name)
        self.pool = str(pool)
        self.image_size = int(image_size)
        self._device = device
        self._model = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._cache: Dict[str, np.ndarray] = {}

    # ── encoder ───────────────────────────────────────────────────────────
    def _ensure(self) -> bool:
        """Load the encoder once. Returns True iff usable. Never raises."""
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            import torch  # noqa: F401
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.model_name == "r3m":
                import r3m
                m = r3m.load_r3m("resnet18")     # cached weights → offline
            elif self.model_name in ("resnet18", "imagenet"):
                import torchvision
                from torchvision.models import ResNet18_Weights
                net = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                net.fc = torch.nn.Identity()
                m = net
            else:
                raise ValueError(f"unknown embed model {self.model_name!r}")
            m = m.eval().to(self._device)
            for p in m.parameters():
                p.requires_grad_(False)
            self._model = m
        except Exception as exc:  # pragma: no cover - depends on cluster env
            self._model = None
            self._load_error = f"{type(exc).__name__}: {exc}"
            print(f"[embed] WARNING: encoder '{self.model_name}' unavailable "
                  f"({self._load_error}); clustering falls back to geometric features.")
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ── per-frame ─────────────────────────────────────────────────────────
    def _embed_frames(self, paths: Sequence[str]) -> Optional[np.ndarray]:
        """Return (len(paths), 512) R3M features, or None on any failure."""
        import torch
        from PIL import Image

        want, todo = [], []
        for p in paths:
            ap = os.path.abspath(str(p))
            want.append(ap)
            if ap not in self._cache:
                todo.append(ap)
        # encode the not-yet-cached frames in one batch
        if todo:
            try:
                batch = []
                for ap in todo:
                    im = Image.open(ap).convert("RGB").resize(
                        (self.image_size, self.image_size))
                    a = np.asarray(im).astype(np.float32)            # H,W,C [0,255]
                    batch.append(torch.from_numpy(a).permute(2, 0, 1))  # C,H,W
                x = torch.stack(batch).to(self._device)               # R3M wants [0,255]
                with torch.no_grad():
                    feats = self._model(x).float().cpu().numpy()      # (n, 512)
                if not np.isfinite(feats).all():
                    return None
                for ap, f in zip(todo, feats):
                    self._cache[ap] = f
            except Exception as exc:  # pragma: no cover
                print(f"[embed] frame encode failed ({type(exc).__name__}: {exc})")
                return None
        try:
            return np.stack([self._cache[ap] for ap in want])         # (k, 512)
        except KeyError:
            return None

    # ── per-failure ───────────────────────────────────────────────────────
    def embed_failure(self, frame_paths: Sequence[str]) -> Optional[np.ndarray]:
        """Pool a failure's key-frame embeddings into one vector, or None.

        Missing files are dropped; a failure with no readable frame returns
        None (planner uses the geometric feature for it / the round)."""
        if not self._ensure():
            return None
        existing = [str(p) for p in frame_paths if p and os.path.exists(str(p))]
        if not existing:
            return None
        per_frame = self._embed_frames(existing)
        if per_frame is None or per_frame.size == 0:
            return None
        if self.pool == "mean":
            return per_frame.mean(axis=0)
        # "concat": keep the start→t*→end ordering as a coarse trajectory shape.
        # Pad/truncate to 3 keyframes so every failure has the same length.
        k = per_frame.shape[1]
        slots = [per_frame[0], per_frame[min(1, len(per_frame) - 1)], per_frame[-1]]
        return np.concatenate(slots[:3]) if len(per_frame) else np.zeros(3 * k)


def pca_reduce(M: np.ndarray, k: int = 16, *, standardize: bool = True) -> np.ndarray:
    """Numpy PCA (centered SVD) to ≤k components — no sklearn needed.

    Returns the n×k' score matrix (k' = min(k, n-1, d)). With n≤2 (clustering
    treats those as singletons anyway) the input is returned unchanged."""
    X = np.asarray(M, dtype=float)
    n, d = X.shape
    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xs = (X - mu) / sd
    else:
        Xs = X - X.mean(axis=0)
    if n <= 2:
        # too few points to PCA; return STANDARDIZED features (consistent scale —
        # clustering treats n<=3 as singletons anyway, so the value is moot there).
        return Xs
    kk = int(max(1, min(int(k), n - 1, d)))
    # economy SVD; columns of Vt are the principal axes
    try:
        _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover
        return Xs
    return Xs @ Vt[:kk].T
