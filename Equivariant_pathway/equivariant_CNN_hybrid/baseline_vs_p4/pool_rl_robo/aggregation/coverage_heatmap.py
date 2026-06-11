"""Tee-object COVERAGE heatmaps from saved checkpoints (no sim re-run), averaged
across seeds and (optionally) overlaid on a rendered environment image.

Each from-scratch retrain re-fits the diffusion-policy normalizers on the CURRENT
dataset, and `SafeLimitsNormalizer.X` keeps EVERY observed value — so
`normalizers['extra_obj_pose'].X[:, :2]` is the (x, y) of the tee at every timestep of
every demo in the dataset at that checkpoint. We bin those into a workspace grid and
AVERAGE the per-seed histograms: GREEN = covered, RED = not covered yet.

  * INITIAL — the shared bootstrap (identical 20 demos across seeds).
  * FINAL   — each method's last-round checkpoint, averaged over the given seeds.

    python -m ...pool_rl_robo.aggregation.coverage_heatmap --env PushT-v1 --runs 1,2,3,4,5
    # overlay on a rendered env frame (see render_env_topdown.py for bg + extent):
    #   ... --bg results/aggregate/astar/PushT-v1_topdown.png --bg-extent x0 x1 y0 y1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..envs import env_setup as E

E.bootstrap_fork_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..orchestrator.workspace import aggregate_dir  # noqa: E402

METHODS = ["p4_select", "diff_dagger"]
LABELS = {"p4_select": "P4-LLM-select", "diff_dagger": "Diff-DAgger"}


def _field_xy(ckpt: Path, field: str, ndim: int = 2):
    import gc
    d = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    nz = d.get("normalizers", {}).get(field)
    if nz is None or not hasattr(nz, "X"):
        del d; gc.collect(); return None
    X = nz.X.detach().cpu().numpy()[:, :ndim]
    del d; gc.collect()
    return X


def _project(P_xyz, K, Ecv):
    """World (N,3) -> pixel (N,2) via OpenCV extrinsic (3x4 world->cam) + intrinsic."""
    N = P_xyz.shape[0]
    Ph = np.concatenate([P_xyz, np.ones((N, 1))], axis=1)        # (N,4)
    cam = Ph @ Ecv.T                                             # (N,3)
    z = np.clip(cam[:, 2:3], 1e-6, None)
    pix = (cam @ K.T)                                            # (N,3)
    return pix[:, :2] / pix[:, 2:3], cam[:, 2]                   # (u,v), depth


def _smooth(Hh, sigma):
    """Gaussian-smooth a 2D density for a continuous gradient (vs blocky bins).
    Uses scipy if present, else a separable numpy Gaussian convolution."""
    if sigma <= 0:
        return Hh
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(Hh, sigma=sigma, mode="constant")
    except Exception:
        r = max(1, int(round(3 * sigma)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, Hh)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
        return out


def _final_ckpt(method_dir: Path):
    cks = method_dir / "results" / "checkpoints"
    rounds = sorted([p for p in cks.glob("round_*") if (p / "0.pth").exists()],
                    key=lambda p: int(p.name.split("_")[-1])) if cks.exists() else []
    if not rounds:
        return None, None
    return rounds[-1] / "0.pth", int(rounds[-1].name.split("_")[-1])


def _gather(env: str, runs, field: str, ndim: int = 2):
    """Return {'init': [X per run], 'p4_select': [...], 'diff_dagger': [...]} plus
    per-method final demo counts."""
    out = {"init": [], "p4_select": [], "diff_dagger": []}
    rounds = {"p4_select": [], "diff_dagger": []}
    for r in runs:
        run_dir = E.RESULTS_ROOT / env / f"run_{r}"
        xi = _field_xy(run_dir / "shared_baselines" / "init_ckpt.pth", field, ndim)
        if xi is not None:
            out["init"].append(xi)
        for m in METHODS:
            ck, rnd = _final_ckpt(run_dir / m)
            if ck is None:
                continue
            xm = _field_xy(ck, field, ndim)
            if xm is not None:
                out[m].append(xm); rounds[m].append(rnd)
    return out, rounds


def starts_on_render(env: str, render_npz: str, starts_npz: str, bins: int) -> None:
    """Project the per-seed tee START positions onto the rendered env and plot them as a
    coverage map. The bootstrap demos (seeds 0..n_bootstrap-1) are highlighted; the wider
    sweep shows the fixed spawn region ALL methods draw starts from (no method changes it)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[cov] matplotlib unavailable ({e})"); return
    z = np.load(render_npz); rgb, K, Ecv = z["rgb"], z["K"], z["E"]
    H, W = rgb.shape[:2]
    s = np.load(starts_npz); starts = s["starts"]; nb = int(s["n_bootstrap"])
    if starts.shape[1] < 3:                                  # need z for projection
        starts = np.concatenate([starts, np.full((len(starts), 1), 0.04)], axis=1)
    uv, depth = _project(starts[:, :3], K, Ecv)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(rgb, zorder=0)
    # density of ALL sampled starts → the spawn region
    xe = np.linspace(0, W, bins + 1); ye = np.linspace(0, H, bins + 1)
    Hh, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=[xe, ye])
    vmax = Hh.max() or 1.0
    ax.imshow(np.log1p(Hh.T), origin="upper", extent=[0, W, H, 0], aspect="auto",
              cmap="RdYlGn", vmin=0, vmax=np.log1p(vmax),
              alpha=0.30 + 0.5 * np.clip(np.log1p(Hh.T) / np.log1p(vmax), 0, 1), zorder=1)
    # the exact 20 bootstrap demo starts
    ax.scatter(uv[:nb, 0], uv[:nb, 1], s=70, facecolors="none", edgecolors="black",
               linewidths=1.6, zorder=3, label=f"initial demos (seeds 0–{nb-1})")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    spawn_cov = float((Hh > 0).sum()) / Hh.size
    ax.set_title(f"{env}: tee START positions on the env (n={len(starts)} seeds)\n"
                 f"spawn region = {spawn_cov:.0%} of frame · ALL methods draw starts here "
                 f"(env-fixed, not method-driven)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = aggregate_dir() / "astar" / f"astar_{env}_tee_START_coverage_onenv.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"[cov] wrote {out}")
    print(f"[cov] START spawn region covers {spawn_cov:.0%} of the frame; "
          f"{nb} bootstrap-demo starts highlighted")


def coverage_on_render(env: str, runs, field: str, bins: int, render_npz: str,
                       smooth: float = 2.0) -> None:
    """Project tee world-coords -> pixels of a rendered top-down frame and overlay the
    per-seed-averaged coverage heatmap as a SMOOTH GRADIENT (Gaussian-smoothed density +
    bilinear interpolation), EXACTLY aligned (uses the saved K, E)."""
    z = np.load(render_npz)
    rgb, K, Ecv = z["rgb"], z["K"], z["E"]
    H, W = rgb.shape[:2]
    data, rounds = _gather(env, runs, field, ndim=3)
    if not data["init"]:
        print(f"[cov] no bootstrap checkpoints for runs {runs}"); return

    xe = np.linspace(0, W, bins + 1); ye = np.linspace(0, H, bins + 1)

    def per_run_mean(xs):
        Hs, covs = [], []
        for x in xs:
            uv, depth = _project(x, K, Ecv)
            m = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            uv = uv[m]
            Hh, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=[xe, ye])
            Hs.append(Hh); covs.append(float((Hh > 0).sum()) / Hh.size)
        return np.mean(Hs, axis=0), float(np.mean(covs)), float(np.std(covs)), len(xs)

    stages = {"init": per_run_mean(data["init"])}
    for m in METHODS:
        if data[m]:
            stages[m] = per_run_mean(data[m])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[cov] matplotlib unavailable ({e})"); return
    methods = [m for m in METHODS if m in stages]
    # smooth each stage's density → continuous gradient; consistent color scale
    smoothed = {k: _smooth(v[0], smooth) for k, v in stages.items()}
    vmax = max(s.max() for s in smoothed.values()) or 1.0
    fname = {"extra_obj_pose": "tee", "extra_goal_pos": "goal"}.get(field, field)

    def draw(ax, Hh, title):
        ax.imshow(rgb, zorder=0)                                  # env render (origin upper)
        dens = np.log1p(Hh.T)
        norm = np.clip(dens / np.log1p(vmax), 0, 1)
        ax.imshow(dens, origin="upper", extent=[0, W, H, 0], aspect="auto",
                  cmap="RdYlGn", vmin=0, vmax=np.log1p(vmax),
                  alpha=0.18 + 0.62 * norm, interpolation="bilinear",
                  zorder=1)                                       # green=covered,red=to-cover
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, W); ax.set_ylim(H, 0)

    _, c0, s0, n0 = stages["init"]
    fig, axes = plt.subplots(len(methods), 2, figsize=(9.8, 4.9 * len(methods)), squeeze=False)
    for r, m in enumerate(methods):
        draw(axes[r][0], smoothed["init"], f"INITIAL — shared bootstrap (20 demos)\ncoverage={c0:.0%}±{s0:.0%}")
        _, c, s, n = stages[m]; dm = int(np.mean(rounds[m])) if rounds[m] else 0
        draw(axes[r][1], smoothed[m], f"{LABELS[m]} FINAL (~{dm} demos)\ncoverage={c:.0%}±{s:.0%}")
    fig.suptitle(f"{env}: {fname} coverage on the env (avg over {len(runs)} seeds)\n"
                 f"green=covered, red=to-cover · both rows share the SAME bootstrap", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = aggregate_dir() / "astar"; out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"astar_{env}_{fname}_coverage_{len(runs)}seed_onenv.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"[cov] wrote {p}")
    msg = f"[cov] {fname} cover% over render (mean±std, {len(runs)} seeds): INIT={c0:.0%}±{s0:.0%}"
    for m in methods:
        _, c, s, _ = stages[m]; msg += f" | {LABELS[m]}={c:.0%}±{s:.0%}"
    print(msg)


def coverage(env: str, runs, field: str, bins: int, bg=None, bg_extent=None) -> None:
    data, rounds = _gather(env, runs, field)
    if not data["init"]:
        print(f"[cov] no bootstrap checkpoints for runs {runs}"); return

    allpts = np.concatenate([x for v in data.values() for x in v], axis=0)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.05 * (hi - lo + 1e-6); lo -= pad; hi += pad
    xe = np.linspace(lo[0], hi[0], bins + 1)
    ye = np.linspace(lo[1], hi[1], bins + 1)

    def per_run_mean(xs):
        """Mean of per-seed 2D histograms + per-seed coverage% (mean,std)."""
        Hs, covs = [], []
        for x in xs:
            H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[xe, ye])
            Hs.append(H); covs.append(float((H > 0).sum()) / H.size)
        Hm = np.mean(Hs, axis=0)
        return Hm, float(np.mean(covs)), float(np.std(covs)), len(xs)

    stages = {"init": per_run_mean(data["init"])}
    for m in METHODS:
        if data[m]:
            stages[m] = per_run_mean(data[m])

    _plot(env, runs, field, xe, ye, stages, rounds, bg, bg_extent)


def _plot(env, runs, field, xe, ye, stages, rounds, bg, bg_extent):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except Exception as e:  # pragma: no cover
        print(f"[cov] matplotlib unavailable ({e})"); return

    methods = [m for m in METHODS if m in stages]
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    vmax = max(s[0].max() for s in stages.values()) or 1.0
    bg_img = mpimg.imread(bg) if bg and Path(bg).exists() else None
    be = bg_extent if bg_extent else extent
    fname = {"extra_obj_pose": "tee", "extra_goal_pos": "goal"}.get(field, field)

    def draw(ax, H, title):
        if bg_img is not None:
            ax.imshow(bg_img, extent=be, aspect="auto", zorder=0)
            alpha = np.clip(np.log1p(H.T) / np.log1p(vmax), 0.0, 1.0)
            ax.imshow(np.log1p(H.T), origin="lower", extent=extent, aspect="auto",
                      cmap="RdYlGn", vmin=0.0, vmax=np.log1p(vmax),
                      alpha=0.45 + 0.5 * alpha, zorder=1)
        else:
            ax.imshow(np.log1p(H.T), origin="lower", extent=extent, aspect="auto",
                      cmap="RdYlGn", vmin=0.0, vmax=np.log1p(vmax), zorder=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("tee x (m)"); ax.set_ylabel("tee y (m)")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    Hm0, c0, s0, n0 = stages["init"]
    fig, axes = plt.subplots(len(methods), 2, figsize=(10.5, 4.8 * len(methods)),
                             squeeze=False)
    for r, m in enumerate(methods):
        draw(axes[r][0], Hm0,
             f"INITIAL — shared bootstrap (20 demos)\ncoverage={c0:.0%}±{s0:.0%} ({n0} seeds)")
        Hm, c, s, n = stages[m]
        dm = int(np.mean(rounds[m])) if rounds[m] else 0
        draw(axes[r][1], Hm,
             f"{LABELS[m]} FINAL (~{dm} demos)\ncoverage={c:.0%}±{s:.0%} ({n} seeds)")
    fig.suptitle(f"{env}: {fname}-object coverage averaged over {len(runs)} seeds "
                 f"(green=covered, red=to-cover)\nboth rows share the SAME bootstrap; "
                 f"right = each method's expansion", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = aggregate_dir() / "astar"; out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_onenv" if bg_img is not None else ""
    p = out_dir / f"astar_{env}_{fname}_coverage_{len(runs)}seed{tag}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[cov] wrote {p}")
    msg = f"[cov] {fname} cover% (mean±std over {len(runs)} seeds): INITIAL={c0:.0%}±{s0:.0%}"
    for m in methods:
        _, c, s, _ = stages[m]; msg += f" | {LABELS[m]}={c:.0%}±{s:.0%}"
    print(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PushT-v1")
    ap.add_argument("--runs", default="1,2,3,4,5")
    ap.add_argument("--field", default="extra_obj_pose")
    ap.add_argument("--bins", type=int, default=36)
    ap.add_argument("--bg", default=None, help="background env image to overlay on")
    ap.add_argument("--bg-extent", nargs=4, type=float, default=None,
                    metavar=("X0", "X1", "Y0", "Y1"),
                    help="world rect the bg image covers (x0 x1 y0 y1)")
    ap.add_argument("--render", default=None,
                    help="npz from render_env_topdown.py {rgb,K,E} → exact projected overlay")
    ap.add_argument("--starts", default=None,
                    help="npz from reconstruct_tee_starts.py → START-position map (needs --render)")
    ap.add_argument("--smooth", type=float, default=2.0,
                    help="Gaussian sigma (in bins) for the gradient heatmap; 0 = blocky bins")
    a = ap.parse_args()
    runs = [int(x) for x in str(a.runs).split(",") if x.strip()]
    if a.starts and a.render:
        starts_on_render(a.env, a.render, a.starts, a.bins)
    elif a.render:
        coverage_on_render(a.env, runs, a.field, a.bins, a.render, smooth=a.smooth)
    else:
        coverage(a.env, runs, a.field, a.bins, bg=a.bg, bg_extent=a.bg_extent)


if __name__ == "__main__":
    main()
