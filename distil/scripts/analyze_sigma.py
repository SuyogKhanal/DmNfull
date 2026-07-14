"""Calibrate the Gaussian memory-kernel width sigma from cluster centroids that ALREADY
exist in completed runs. Pure CPU analysis: no rollouts, no LLM, no code defaults changed.

The kernel under study (distil/p4/memory.py:56-70):

    recency_penalty(c) = SUM_i gamma^(now-round_i) * exp(-||c_xy - e_i_xy||^2 / (2 sigma^2))
    score(cluster)     = cluster.mean_peak_loss - lambda * recency_penalty(centroid)

Only the planar x,y of the RAW, UNSTANDARDIZED centroid enters. Units are per-task: metres
for the robot tasks, GRID-CELL INDICES for GridWorld. One global sigma=0.06
(distil/config.py:71) is applied to both.

Design notes (each fixes a defect found by adversarial review of the first version):
  * The penalty distribution is reported SEPARATELY over the two distance sets. They answer
    different questions and MUST NOT be pooled: sigma is calibrated on the NN set, so the
    headline penalty stats are over the NN set.
  * n_nn is reported distinctly from n_pairwise (they differ for any k != 3).
  * GridWorld's length scale is the SPAWN EXTENT of the grid (GRID_SIZE-1 = 4 cell-widths),
    which is the true analogue of the robots' randomisation extents -- NOT the grid pitch.
  * The DECISION-LEVEL test: we replay select_target's argmax with and without the penalty,
    using the TRUE gamma-weighted SUM over the actual memory entries, and count how often the
    penalty flips the choice -- at the current sigma and at the recommended sigma. This is the
    only statistic that tests whether re-scaling sigma would change anything.
  * _smoke runs are EXCLUDED (they are plumbing tests, not experiments).

Outputs -> paper_aaai2027/COC_REPORT/build/sigma_calibration/.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("DISTIL_ROOT", "/weka/s226137394/DmNfull"))
OUT = ROOT / "paper_aaai2027/COC_REPORT/build/sigma_calibration"
FIG = OUT / "figures"

SIGMA_CURRENT = 0.06          # distil/config.py:71  (p4.memory_sigma)
GAMMA = 0.6                   # distil/config.py:70  (p4.memory_gamma)
LAMBDA = 1.0                  # distil/config.py:72  (p4.memory_lambda)

MODALITIES = ("state", "image")

# Operating point: map the median inter-centroid distance to a penalty of 0.5.
#   exp(-d^2/(2 s^2)) = 1/2  <=>  s = d / sqrt(2 ln 2)
ALPHA_HALF = 1.0 / math.sqrt(2.0 * math.log(2.0))     # 0.8493218


def kernel(d, sigma):
    """Single-entry kernel value at planar distance d."""
    d = np.asarray(d, dtype=float)
    return np.exp(-(d ** 2) / (2.0 * sigma * sigma))


def recency_penalty(xy, entries, now, sigma, gamma=GAMMA):
    """The TRUE kernel as deployed: gamma-weighted SUM over all memory entries with
    round < now (distil/p4/memory.py:56-70). Bounded by sum gamma^k = 1/(1-gamma) = 2.5."""
    pen = 0.0
    two_s2 = 2.0 * sigma * sigma
    for e in entries:
        if e["round"] >= now:
            continue
        w = gamma ** max(0, now - e["round"])
        d2 = (xy[0] - e["xy"][0]) ** 2 + (xy[1] - e["xy"][1]) ** 2
        pen += w * math.exp(-d2 / two_s2)
    return pen


# ---------------------------------------------------------------------------
# 1. HARVEST
# ---------------------------------------------------------------------------

def _id_from_distil_path(p: Path):
    parts = p.parts
    task = modality = arm = seed = None
    for i, x in enumerate(parts):
        if x in ("Lift", "Wipe", "Door", "GridWorld") and i + 2 < len(parts):
            task = x
            if parts[i + 1] in MODALITIES:
                modality = parts[i + 1]
                arm = parts[i + 2]
                m = re.match(r"seed(\d+)", parts[i + 3]) if i + 3 < len(parts) else None
                seed = int(m.group(1)) if m else None
            break
    tree = "_compute" if "_compute" in parts else ("_smoke" if "_smoke" in parts else "production")
    return task, modality, arm, seed, tree


def _clusters_from_event(e):
    """Handles BOTH schemas:
         distil : per-cluster 'centroid_xy' (robots) / 'centroid_rc' (GridWorld), 2-D,
                  and mean_peak_loss NOT persisted (reconstructed from descriptors).
         fork   : per-cluster 'centroid_xyz' (3-D) + 'mean_peak_loss' persisted directly.
       Dropping either key silently loses a whole task family."""
    ploss = {d.get("episode_id"): d.get("peak_loss")
             for d in e.get("descriptors", []) or []}
    out = []
    for c in e.get("clusters", []) or []:
        xy = c.get("centroid_xy") or c.get("centroid_rc") or c.get("centroid_xyz")
        if xy is None or len(xy) < 2:
            continue
        mpl = c.get("mean_peak_loss")            # fork persists it
        if mpl is None:                          # distil: reconstruct via members -> peak_loss
            mem = [ploss.get(m) for m in (c.get("members") or [])]
            mem = [v for v in mem if isinstance(v, (int, float))]
            mpl = float(np.mean(mem)) if mem else None
        out.append({"label": c.get("label"), "size": c.get("size"),
                    "xy": [float(xy[0]), float(xy[1])],
                    "is_target": bool(c.get("is_target")),
                    "mean_peak_loss": (float(mpl) if mpl is not None else None)})
    return out


def _memory_entries(path: Path):
    try:
        d = json.loads(path.read_text())
    except Exception:
        return [], None, None
    ent = [{"round": int(x.get("round", 0)),
            "xy": [float(x["centroid_xyz"][0]), float(x["centroid_xyz"][1])]}
           for x in d.get("entries", []) if x.get("centroid_xyz")]
    return sorted(ent, key=lambda z: z["round"]), d.get("sigma"), d.get("gamma")


def harvest():
    """-> runs: {run_dir: {task, modality, arm, seed, tree, rounds:[...], memory:[...]}}"""
    runs = {}

    def _add_round(run_dir, meta, rnd, clusters):
        r = runs.setdefault(str(run_dir), {**meta, "rounds": [], "memory": [],
                                           "files": set()})
        r["rounds"].append({"round": rnd, "clusters": clusters})

    # ---- distil ----
    for f in sorted((ROOT / "distil/results").rglob("telemetry/round_*.jsonl")):
        task, modality, arm, seed, tree = _id_from_distil_path(f)
        if task is None or tree == "_smoke":       # smoke = plumbing test, not an experiment
            continue
        run_dir = f.parent.parent
        meta = dict(task=task, modality=modality or "state", arm=arm, seed=seed,
                    tree=tree, source="distil")
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") != "round_setup":
                continue
            cl = _clusters_from_event(e)
            if cl:
                _add_round(run_dir, meta, e.get("round"), cl)
                runs[str(run_dir)]["files"].add(str(f.relative_to(ROOT)))

    for f in sorted((ROOT / "distil/results").rglob("telemetry/centroid_memory.json")):
        task, modality, arm, seed, tree = _id_from_distil_path(f)
        if task is None or tree == "_smoke":
            continue
        run_dir = f.parent.parent
        ent, sg, gm = _memory_entries(f)
        r = runs.setdefault(str(run_dir), dict(task=task, modality=modality or "state", arm=arm,
                                               seed=seed, tree=tree, source="distil",
                                               rounds=[], memory=[], files=set()))
        r["memory"] = ent
        r["sigma_used"], r["gamma_used"] = sg, gm
        r["files"].add(str(f.relative_to(ROOT)))

    # ---- fork (pool_rl_robo). pool_x_selector is NOT walked: ~835k episode files and
    # ZERO centroid artifacts (verified: 0 centroid_memory.json, 0 telemetry dirs). Its
    # "clusters" are LLM prompt prose; it prescribes discrete grid layouts, not centroids.
    fork = ROOT / "Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/pool_rl_robo"
    if fork.is_dir():
        print(f"[harvest] walking {fork} ...", flush=True)
        for f in sorted(fork.rglob("telemetry/round_*.jsonl")):
            s = str(f)
            task = next((t for t in ("PushT", "StackCube", "PlugCharger")
                         if t.lower() in s.lower()), "PushT")
            run_dir = f.parent.parent
            meta = dict(task=task, modality="state", arm="p4_fork", seed=None,
                        tree="fork", source="pool_rl_robo")
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("kind") != "round_setup":
                    continue
                cl = _clusters_from_event(e)
                if cl:                     # p4_top3 (StackCube/PlugCharger) emits no clusters[]
                    _add_round(run_dir, meta, e.get("round"), cl)
                    runs[str(run_dir)]["files"].add(str(f.relative_to(ROOT)))
        for f in sorted(fork.rglob("telemetry/centroid_memory.json")):
            s = str(f)
            task = next((t for t in ("PushT", "StackCube", "PlugCharger")
                         if t.lower() in s.lower()), "PushT")
            run_dir = f.parent.parent
            ent, sg, gm = _memory_entries(f)
            r = runs.setdefault(str(run_dir), dict(task=task, modality="state", arm="p4_fork",
                                                   seed=None, tree="fork", source="pool_rl_robo",
                                                   rounds=[], memory=[], files=set()))
            r["memory"] = ent
            r["sigma_used"], r["gamma_used"] = sg, gm
            r["files"].add(str(f.relative_to(ROOT)))
    return runs


# ---------------------------------------------------------------------------
# 2. DISTANCES + 3. DECISION-LEVEL REPLAY
# ---------------------------------------------------------------------------

def nn_and_pairwise(clusters):
    xy = np.array([c["xy"] for c in clusters], dtype=float)
    if len(xy) < 2:
        return [], []
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    pw = [float(d[i, j]) for i in range(len(xy)) for j in range(i + 1, len(xy))]
    dd = d.copy()
    np.fill_diagonal(dd, np.inf)
    return [float(v) for v in dd.min(axis=1)], pw


def candidates(clusters):
    """select_target's candidate set (memory.py:80): size >= dominant.size - 1."""
    cl = [c for c in clusters if c.get("size") is not None]
    if not cl:
        return []
    top = max(c["size"] for c in cl)
    return [c for c in cl if c["size"] >= top - 1]


def replay_argmax(clusters, memory, now, sigma):
    """Replay select_target. Returns (flipped, pen_spread) or (None, None) if the round
    cannot decide anything (singleton candidate set / missing mean_peak_loss / no memory).

    flipped = does adding -lambda*penalty change the argmax vs mean_peak_loss alone?"""
    cands = candidates(clusters)
    if len(cands) < 2:
        return None, None                    # sigma is INERT here at ANY value
    if any(c["mean_peak_loss"] is None for c in cands):
        return None, None
    prior = [e for e in memory if e["round"] < now]
    if not prior:
        return None, None                    # empty memory -> penalty is 0 for everyone
    pens = [recency_penalty(c["xy"], prior, now, sigma) for c in cands]
    mpl = [c["mean_peak_loss"] for c in cands]
    i_no = int(np.argmax(mpl))
    i_yes = int(np.argmax([m - LAMBDA * p for m, p in zip(mpl, pens)]))
    return (i_no != i_yes), float(max(pens) - min(pens))


def stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(n=0, min=None, p25=None, median=None, p75=None, max=None, iqr=None)
    return dict(n=int(v.size), min=float(v.min()), p25=float(np.percentile(v, 25)),
                median=float(np.median(v)), p75=float(np.percentile(v, 75)),
                max=float(v.max()),
                iqr=float(np.percentile(v, 75) - np.percentile(v, 25)))


# ---------------------------------------------------------------------------
# LENGTH SCALES
# ---------------------------------------------------------------------------
# The randomisation EXTENT of each task, in that task's own centroid units.
# Robots: from distil/p4/bounds.py TASK_BOUNDS + paper_aaai2027/context/kag_ur5_bounds.md.
# GridWorld: the SPAWN EXTENT of the 5x5 lattice = GRID_SIZE-1 = 4 cell-widths (the index
#   span an agent can occupy). This -- NOT the grid pitch of 1 cell -- is the true analogue
#   of the robots' randomisation extents.
# Wipe: NO reset bounds exist. It is SELECT-only and absent from TASK_BOUNDS entirely; the
#   randomised quantity is a ~100-marker dirt path, not an object pose. This is not a lookup
#   failure -- the quantity does not exist.
SPAWN_EXTENT = {
    "Lift": 0.06,        # cube xy in [-0.03, 0.03]  -> 0.06 m span
    "Door": 0.027,       # frame x in [-0.135,-0.108] -> 0.027 m span
    "Wipe": None,        # DOES NOT EXIST
    "GridWorld": 4.0,    # 5x5 lattice -> index span 0..4 = 4 cell-widths
    "PushT": 0.40,       # ManiSkill PushT-v1 tee x in [-0.20, 0.20]
}
UNITS = {"Lift": "m", "Door": "m", "Wipe": "m", "PushT": "m",
         "StackCube": "m", "PlugCharger": "m", "GridWorld": "grid cells"}

MIN_CLUSTERS = 20


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("[harvest] walking distil/results ...", flush=True)
    runs = harvest()
    nr = sum(len(r["rounds"]) for r in runs.values())
    nm = sum(1 for r in runs.values() if r["memory"])
    print(f"[harvest] {len(runs)} run dirs, {nr} cluster-carrying rounds, "
          f"{nm} centroid_memory files (_smoke EXCLUDED; pool_x_selector not walked: "
          f"0 centroids by construction)", flush=True)

    cells = defaultdict(lambda: {
        "nn": [], "pairwise": [], "tgt_mem": [], "n_rounds": 0, "n_clusters": 0,
        "runs": set(), "files": set(), "arms": set(), "trees": set(),
        "n_rounds_scored": 0, "n_cand_ge2": 0,
        "replay": [],           # (flipped@cur, spread@cur) for decidable rounds
        "pen_applied_cur": [],  # TRUE gamma-weighted penalty actually applied, sigma=0.06
    })

    for rd, r in runs.items():
        key = (r["task"], r.get("modality") or "state")
        c = cells[key]
        c["runs"].add(rd)
        c["files"] |= r.get("files", set())
        c["arms"].add(r.get("arm"))
        c["trees"].add(r.get("tree"))
        mem = r.get("memory", [])
        for rnd in r["rounds"]:
            cl = rnd["clusters"]
            c["n_rounds"] += 1
            c["n_clusters"] += len(cl)
            nn, pw = nn_and_pairwise(cl)
            c["nn"] += nn
            c["pairwise"] += pw
            cands = candidates(cl)
            if cands:
                c["n_rounds_scored"] += 1
                if len(cands) >= 2:
                    c["n_cand_ge2"] += 1
            now = rnd["round"]
            if isinstance(now, int):
                prior = [e for e in mem if e["round"] < now]
                for cc in cl:
                    if prior:
                        c["pen_applied_cur"].append(
                            recency_penalty(cc["xy"], prior, now, SIGMA_CURRENT))
        # target <-> memory distances (what the kernel sees at prescription time)
        ent = mem
        for i in range(1, len(ent)):
            for j in range(i):
                c["tgt_mem"].append(float(np.linalg.norm(
                    np.array(ent[i]["xy"]) - np.array(ent[j]["xy"]))))

    # ---- rows ----
    rows, detail = [], {}
    for (task, modality), c in sorted(cells.items()):
        unit = UNITS.get(task, "m")
        nn_s, pw_s, tm_s = stats(c["nn"]), stats(c["pairwise"]), stats(c["tgt_mem"])
        d_med = nn_s["median"]                       # L_task, from the NN set ONLY
        sigma_rec = (ALPHA_HALF * d_med) if d_med else None
        L_spawn = SPAWN_EXTENT.get(task)
        insufficient = (c["n_clusters"] < MIN_CLUSTERS) or (d_med is None)

        # Penalty stats, reported SEPARATELY per distance set (never pooled).
        pen_nn_cur, pen_nn_new = stats(kernel(c["nn"], SIGMA_CURRENT)), (
            stats(kernel(c["nn"], sigma_rec)) if sigma_rec else stats([]))
        pen_tm_cur, pen_tm_new = stats(kernel(c["tgt_mem"], SIGMA_CURRENT)), (
            stats(kernel(c["tgt_mem"], sigma_rec)) if sigma_rec else stats([]))

        # DECISION-LEVEL replay at current vs recommended sigma.
        flips_cur = flips_new = decidable = 0
        for rd, r in runs.items():
            if (r["task"], r.get("modality") or "state") != (task, modality):
                continue
            for rnd in r["rounds"]:
                if not isinstance(rnd["round"], int):
                    continue
                f1, _ = replay_argmax(rnd["clusters"], r.get("memory", []),
                                      rnd["round"], SIGMA_CURRENT)
                if f1 is None:
                    continue
                decidable += 1
                flips_cur += int(f1)
                if sigma_rec:
                    f2, _ = replay_argmax(rnd["clusters"], r.get("memory", []),
                                          rnd["round"], sigma_rec)
                    flips_new += int(bool(f2))

        rows.append(dict(
            task=task, modality=modality, unit=unit,
            n_runs=len(c["runs"]), n_rounds=c["n_rounds"], n_clusters=c["n_clusters"],
            trees=";".join(sorted(t for t in c["trees"] if t)),
            arms=";".join(sorted(a for a in c["arms"] if a)),
            # distances: NN set (what sigma is calibrated on)
            n_nn=nn_s["n"], d_nn_min=nn_s["min"], d_nn_p25=nn_s["p25"],
            d_median=d_med, d_nn_p75=nn_s["p75"], d_nn_max=nn_s["max"],
            # distances: all-pairs and target<->memory, reported separately
            n_pairwise=pw_s["n"], d_pairwise_median=pw_s["median"],
            n_tgt_mem=tm_s["n"], d_tgt_mem_median=tm_s["median"],
            # length scales
            L_task=d_med, L_spawn_extent=L_spawn,
            ratio_d_over_spawn=((d_med / L_spawn) if (d_med and L_spawn) else None),
            alpha=ALPHA_HALF,
            sigma_current=SIGMA_CURRENT, sigma_recommended=sigma_rec,
            # penalty over the NN set (headline: the set sigma is calibrated on)
            pen_nn_median_cur=pen_nn_cur["median"], pen_nn_iqr_cur=pen_nn_cur["iqr"],
            pen_nn_median_new=pen_nn_new["median"], pen_nn_iqr_new=pen_nn_new["iqr"],
            # penalty over the target<->memory set
            pen_tgtmem_median_cur=pen_tm_cur["median"], pen_tgtmem_iqr_cur=pen_tm_cur["iqr"],
            pen_tgtmem_median_new=pen_tm_new["median"], pen_tgtmem_iqr_new=pen_tm_new["iqr"],
            # decision-level
            rounds_scored=c["n_rounds_scored"], rounds_cand_ge2=c["n_cand_ge2"],
            frac_rounds_kernel_can_act=(c["n_cand_ge2"] / c["n_rounds_scored"]
                                        if c["n_rounds_scored"] else None),
            rounds_decidable=decidable,
            argmax_flips_at_sigma_current=flips_cur,
            argmax_flips_at_sigma_recommended=flips_new,
            verdict=("INSUFFICIENT DATA" if insufficient else "OK"),
        ))
        detail[f"{task}/{modality}"] = dict(nn=nn_s, pairwise=pw_s, tgt_mem=tm_s,
                                            runs=sorted(c["runs"]), files=sorted(c["files"]))

    with (OUT / "sigma_per_task.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] sigma_per_task.csv ({len(rows)} cells)")

    json.dump(rows, (OUT / "sigma_rows.json").open("w"), indent=1, default=str)
    (OUT / "harvest_raw.json").write_text(json.dumps(
        {"n_run_dirs": len(runs), "n_rounds": nr, "n_memory_files": nm,
         "sigma_current": SIGMA_CURRENT, "gamma": GAMMA, "lambda": LAMBDA,
         "alpha": ALPHA_HALF,
         "alpha_criterion": "penalty(d_median)=0.5 => sigma = d_median/sqrt(2 ln 2)",
         "smoke_excluded": True, "per_cell": detail}, indent=1, default=str))

    # ---- figures: NN distances (the calibration set) + kernel curves ----
    for r in rows:
        task, modality = r["task"], r["modality"]
        c = cells[(task, modality)]
        if not c["nn"]:
            continue
        unit = r["unit"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(c["nn"], bins=40, color="#4C78A8", edgecolor="white")
        ax[0].axvline(r["d_median"], color="#E45756", ls="--",
                      label=f"median NN = {r['d_median']:.4g} {unit}")
        ax[0].legend(fontsize=8)
        ax[0].set_xlabel(f"nearest-neighbour inter-centroid distance [{unit}]")
        ax[0].set_ylabel("count")
        ax[0].set_title(f"{task}/{modality}: distances the kernel must resolve\n"
                        f"n={r['n_nn']} ({r['n_clusters']} clusters, {r['n_rounds']} rounds)")

        grid = np.linspace(0, max(c["nn"]) * 1.2, 400)
        ax[1].plot(grid, kernel(grid, SIGMA_CURRENT), color="#E45756",
                   label=f"current $\\sigma$ = {SIGMA_CURRENT}")
        if r["sigma_recommended"]:
            ax[1].plot(grid, kernel(grid, r["sigma_recommended"]), color="#4C78A8",
                       label=f"recommended $\\sigma$ = {r['sigma_recommended']:.4g}")
        ax[1].axvline(r["d_median"], color="grey", ls=":", lw=1)
        ax[1].axhline(0.5, color="grey", ls=":", lw=1)
        ax[1].set_ylim(-0.02, 1.02)
        ax[1].set_xlabel(f"distance [{unit}]")
        ax[1].set_ylabel("single-entry kernel value")
        ax[1].set_title("kernel vs distance (dotted: median NN, penalty 0.5)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"{task}_{modality}.{ext}", dpi=150)
        plt.close(fig)
    print(f"[write] figures -> {FIG}")
    print("[done]")


if __name__ == "__main__":
    sys.exit(main())
