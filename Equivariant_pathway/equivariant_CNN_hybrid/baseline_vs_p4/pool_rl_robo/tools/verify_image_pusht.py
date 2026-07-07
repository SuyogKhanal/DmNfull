#!/usr/bin/env python
"""verify_image_pusht.py — fast GPU sanity for the IMAGE-based PushT bring-up.

Validates, in minutes (not a learning run), the load-bearing NEW bits before any
pipeline smoke is submitted:

  1. R3M(resnet18) loads offline (cached weights).
  2. PushT-v2 obs_mode=rgb yields a camera image → DISCOVER the exact flattened
     obs-key name + shape (so configs/hydra/pusht_image.yaml is correct).
  3. The suite composes pusht_image.yaml (via $SUITE_HYDRA_CFG) and the proprio
     keys + rgb key are all present on the wrapped policy env.
  4. The fork DiffDAggerPolicy builds with vision_model=r3m (frozen) — obs_dim /
     UNet global_cond_dim are consistent — and the R3M encoder forwards a REAL
     rgb observation to the expected 512-d feature.
  5. The p4_subtask visual-embedding path (FrameEmbedder + pca_reduce + descriptor
     frame_paths + cluster_feature) runs on a real saved failure episode.

Run on a GPU node (Vulkan needed for rgb): tools/verify_image_pusht.sh.
Prints [PASS]/[FAIL] per check and exits non-zero if any hard check fails.
"""
from __future__ import annotations

import glob
import os
import sys
import traceback
from pathlib import Path

# Use the image Hydra config for the suite compose path (same as the launchers).
SUITE = Path(__file__).resolve().parents[1]
os.environ.setdefault("SUITE_HYDRA_CFG", str(SUITE / "configs/hydra/pusht_image.yaml"))

PKG = "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo"
sys.path.insert(0, str(SUITE.parents[3]))  # DmNfull root for the dotted package

import importlib

_fails = []


def check(name):
    def deco(fn):
        print(f"\n=== {name} ===", flush=True)
        try:
            out = fn()
            print(f"[PASS] {name}", flush=True)
            return out
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            _fails.append(name)
            return None
    return deco


def main():
    import numpy as np
    import torch
    print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
          "dev", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    E = importlib.import_module(PKG + ".envs.env_setup")
    E.bootstrap_fork_path()
    E.register_envs()

    state = {}

    @check("1. R3M loads offline")
    def _():
        import r3m
        m = r3m.load_r3m("resnet18").eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = m.to(dev)
        x = torch.zeros(2, 3, 224, 224, device=dev)  # R3M wants [0,255]
        with torch.no_grad():
            f = m(x)
        assert f.shape == (2, 512), f"R3M out {tuple(f.shape)} != (2,512)"
        print("  R3M ok, feature dim 512")

    @check("2. PushT-v2 rgb obs key + shape (discover)")
    def _():
        import gymnasium as gym
        from diffdagger.util.maniskill_env import FlattenObservationKeyWrapper
        env = gym.make("PushT-v2", obs_mode="rgb", control_mode="pd_joint_pos",
                       render_mode="rgb_array", num_envs=1, sim_backend="gpu",
                       enable_shadow=True, robot_init_qpos_noise=0.0)
        env = FlattenObservationKeyWrapper(env)
        obs, _ = env.reset(seed=0)
        rgb_keys, proprio_keys = [], []
        for k, v in obs.items():
            shp = tuple(v.shape) if hasattr(v, "shape") else None
            tag = "RGB" if ("rgb" in k or "image" in k) else ""
            print(f"   {k:42s} {str(shp):24s} {tag}")
            (rgb_keys if tag else proprio_keys).append(k)
        assert rgb_keys, "no rgb/image obs key found in obs_mode=rgb!"
        state["rgb_key"] = rgb_keys[0]
        for need in ("agent_qpos", "extra_tcp_pose"):
            assert need in obs, f"proprio key {need!r} absent in rgb mode"
        print(f"  DISCOVERED rgb key = {rgb_keys[0]!r}; proprio present.")
        env.close()

    @check("3. suite composes pusht_image.yaml; obs_keys match")
    def _():
        MS = importlib.import_module(PKG + ".envs.maniskill_env")
        cfg = MS.load_cfg("PushT-v1")
        print("  obs_mode =", cfg.obs_mode, "| obs_keys =", list(cfg.obs_keys),
              "| proprio_dim =", cfg.proprio_dim, "| frozen =", cfg.policy.frozen_encoder)
        assert str(cfg.obs_mode) == "rgb", f"obs_mode={cfg.obs_mode} (SUITE_HYDRA_CFG not honored?)"
        ck = [k for k in cfg.obs_keys if "rgb" in k or "image" in k]
        assert ck, "no rgb key in cfg.obs_keys"
        if state.get("rgb_key") and ck[0] != state["rgb_key"]:
            raise AssertionError(
                f"cfg rgb key {ck[0]!r} != discovered {state['rgb_key']!r} — "
                f"fix obs_keys in configs/hydra/pusht_image.yaml")
        state["cfg"] = cfg

    @check("4. image policy builds + R3M encoder forwards a real rgb obs")
    def _():
        from hydra.utils import instantiate
        MS = importlib.import_module(PKG + ".envs.maniskill_env")
        cfg = state["cfg"]
        env = MS.make_policy_env(cfg)
        obs, _ = env.reset(seed=1)
        rk = [k for k in obs.keys() if "rgb" in k or "image" in k][0]
        print("  wrapped policy obs keys:", list(obs.keys()))
        print("  rgb obs shape:", tuple(obs[rk].shape))
        policy = instantiate(cfg.policy)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        policy = policy.to(dev)
        exp_obs_dim = 512 * (1 + int(cfg.policy.spatial_softmax)) * 1 + int(cfg.proprio_dim)
        print(f"  policy.obs_dim = {policy.obs_dim} (expected {exp_obs_dim})")
        assert int(policy.obs_dim) == exp_obs_dim, "obs_dim mismatch"
        gcd = policy.model["model"]["policy_head"].global_cond_dim if hasattr(
            policy.model["model"]["policy_head"], "global_cond_dim") else None
        print("  UNet global_cond_dim =", gcd)
        # forward a REAL rgb obs through the frozen encoder (the risky novel path)
        enc = policy.model["model"]["image_encoder"]
        img = obs[rk].to(dev).float()                       # (B,H,W,3) channel-last
        if img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2).contiguous()      # (B,3,H,W)
        img = policy.eval_transform(img)
        with torch.no_grad():
            feat = enc(img, rk)
        print("  encoder feature shape:", tuple(feat.shape))
        assert feat.shape[-1] == 512, f"encoder out {tuple(feat.shape)} last-dim != 512"
        env.close()

    @check("5. p4_subtask visual-embedding path on a real failure")
    def _():
        import json
        emb = importlib.import_module(PKG + ".p4_subtask.embedding")
        desc = importlib.import_module(PKG + ".p4_subtask.descriptor")
        metas = sorted(glob.glob(str(SUITE / "results/PushT-v1/**/high_loss_images/episode_*/meta.json"),
                                 recursive=True))[:6]
        if not metas:
            print("  (no saved PushT failure frames found — skipping live-frame embed; "
                  "smoke will generate them)")
            return
        descs = []
        for mp in metas:
            ep = os.path.dirname(mp)
            d = desc.load_failure_descriptor(ep, json.load(open(mp)))
            if d is not None:
                descs.append(d)
        print(f"  built {len(descs)} descriptors; frame_paths[0]={descs[0].frame_paths}")
        fe = emb.FrameEmbedder(model_name="r3m", pool="concat")
        vecs = []
        for d in descs:
            paths = [d.frame_paths[k] for k in ("start", "peak", "end") if d.frame_paths.get(k)]
            v = fe.embed_failure(paths)
            assert v is not None, f"embed_failure None for {d.episode_id}"
            vecs.append(v)
        M = np.stack(vecs)
        red = emb.pca_reduce(M, k=16)
        print(f"  raw embed {M.shape} → pca {red.shape}")
        for i, d in enumerate(descs):
            d.visual_embedding = [float(x) for x in red[i]]
        cf = descs[0].cluster_feature()
        assert len(cf) == red.shape[1], "cluster_feature length != reduced dim"
        print(f"  cluster_feature dim = {len(cf)} (visual); geometric would be {len(descs[0].feature())}")

    main_done(state)


def main_done(state):
    print("\n" + "=" * 60)
    if _fails:
        print(f"RESULT: {len(_fails)} FAILED → {_fails}")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED — image-based PushT bring-up is wired.")
    sys.exit(0)


if __name__ == "__main__":
    main()
