"""Per-env policy factory: maps each environment to its backbone and builds the
right Policy for a given method. This registry IS the per-env policy config
(policy_backbone_guide §2/§10):

  HalfCheetah-v4 / Hopper-v4 / Walker2d-v4  -> "mlp"        (Gaussian MLP, state)
  FetchReach-v4 / FetchPickAndPlace-v4      -> "diffusion"  (R3M-image DDPM)

All 6 methods on an env share the backbone; only EnsembleDAgger/ThriftyDAgger get
the ensemble MLP variant and DropoutDAgger gets dropout (locomotion only — on the
diffusion envs every method uses the one diffusion policy with sample-variance
uncertainty).
"""
from __future__ import annotations

from ..envs import env_setup as E

ENV_BACKBONE = {
    "HalfCheetah": "mlp", "Hopper": "mlp", "Walker2d": "mlp",
    "FetchReach": "diffusion", "FetchPickAndPlace": "diffusion",
}

_R3M_CACHE = {}


def backbone_for(env_id: str) -> str:
    base = env_id.split("-")[0]
    return ENV_BACKBONE.get(base, "mlp")


def needs_render(env_id: str) -> bool:
    return backbone_for(env_id) == "diffusion"


def _get_r3m(device, model_name):
    key = (str(device), model_name)
    if key not in _R3M_CACHE:
        from .r3m_encoder import R3MEncoder
        _R3M_CACHE[key] = R3MEncoder(device, model_name)
    return _R3M_CACHE[key]


def build_policy(env_id: str, method_kind: str, cfg: dict, device, env):
    bk = backbone_for(env_id)
    a_dim = E.act_dim(env)
    low, high = E.act_bounds(env)
    seed = int(cfg.get("seed", 42))

    if bk == "diffusion":
        from .diffusion import DiffusionPolicy
        d = cfg.get("diffusion", {}) or {}
        r3m = _get_r3m(device, d.get("r3m_model", "resnet18"))
        return DiffusionPolicy(
            a_dim, low, high, goal_dim=3, device=device, r3m_encoder=r3m,
            hidden=int(d.get("hidden", 512)), T=int(d.get("T", 100)),
            lr=float(d.get("lr", 1e-4)), batch_size=int(d.get("batch_size", 256)),
            weight_decay=float(d.get("weight_decay", 1e-6)), seed=seed)

    from .mlp import MLPEnsembleP, MLPSingle
    o_dim = E.obs_dim(env)
    hidden = int(cfg.get("hidden", 256))
    lr = float(cfg.get("lr", 3e-4))
    bs = int(cfg.get("batch_size", 256))
    wd = float(cfg.get("weight_decay", 1e-4))
    bl = cfg.get("baselines", {}) or {}
    if method_kind in ("ensemble", "thrifty"):
        M = int((bl.get(method_kind, {}) or {}).get("M", 5))
        return MLPEnsembleP(M, o_dim, a_dim, low, high, hidden=hidden, lr=lr,
                            batch_size=bs, weight_decay=wd, seed=seed, device=device)
    drop = float((bl.get("dropout", {}) or {}).get("dropout_d", 0.1)) if method_kind == "dropout" else 0.0
    return MLPSingle(o_dim, a_dim, low, high, hidden=hidden, dropout=drop, lr=lr,
                     batch_size=bs, weight_decay=wd, seed=seed, device=device)
