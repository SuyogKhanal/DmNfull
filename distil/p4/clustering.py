"""Code-level clustering of the round's failures (Fix 1).

The original p4_top3 ranks failures by peak diffusion loss only — three
near-identical failures can dominate the top-3. Here we cluster the round's
failures in a standardized 6-d descriptor space, pick the DOMINANT cluster (most
members; tie-break highest mean peak-loss), and compute a coverage centroid for
it. The dominant cluster's representative (member nearest its centroid) becomes
the sub-task anchor.

Deterministic: prefers sklearn Agglomerative + silhouette-chosen k when present,
else a deterministic numpy single-linkage fallback. The chosen path is recorded
so a run is reproducible/auditable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# Task-agnostic (robosuite port): descriptors are DUCK-TYPED. A failure descriptor
# must expose: cluster_feature() -> List[float] (vector clustering operates on),
# centroid_pos() -> [x,y,z] and centroid_theta() -> float (geometric centroid, task
# frame), peak_loss: float, episode_id: int (dominant tie-break + representative).
from typing import Any as FailureDescriptor  # duck-typed alias for type hints


@dataclass
class Cluster:
    label: int
    member_idxs: List[int]
    centroid_xyz: List[float]      # env coords, clamped to KAG tee bounds
    centroid_theta: float          # circular mean yaw
    mean_peak_loss: float
    size: int
    representative_idx: int        # global index of the member nearest the centroid


@dataclass
class ClusterResult:
    clusters: List[Cluster]
    labels: List[int]
    method: str
    dominant: Cluster = field(default=None)


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


def _numpy_silhouette(Xs: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score over points (numpy fallback when sklearn is absent)."""
    n = len(Xs)
    uniq = sorted(set(int(x) for x in labels.tolist()))
    if len(uniq) < 2 or len(uniq) >= n:
        return -2.0
    D = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=-1)
    sil = []
    for i in range(n):
        same = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if not same:
            sil.append(0.0)
            continue
        a = float(np.mean([D[i, j] for j in same]))
        b = min(float(np.mean([D[i, j] for j in range(n) if labels[j] == c]))
                for c in uniq if c != labels[i])
        denom = max(a, b)
        sil.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(sil)) if sil else -2.0


def _best_k_clustering(Xs: np.ndarray, kmax: int) -> Tuple[np.ndarray, str]:
    """Sweep k in [2, kmax], pick the labelling with the best silhouette. Uses
    sklearn Agglomerative+silhouette when importable, else a deterministic numpy
    single-linkage + numpy silhouette. Identical sweep logic on both paths."""
    use_sklearn = True
    try:
        from sklearn.cluster import AgglomerativeClustering  # noqa: F401
        from sklearn.metrics import silhouette_score          # noqa: F401
    except Exception:
        use_sklearn = False

    best_labels, best_s = None, -3.0
    for k in range(2, kmax + 1):
        if use_sklearn:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            lab = np.asarray(AgglomerativeClustering(n_clusters=k).fit_predict(Xs), dtype=int)
            try:
                s = float(silhouette_score(Xs, lab))
            except Exception:
                s = -2.0
        else:
            lab = _numpy_single_linkage(Xs, k)
            s = _numpy_silhouette(Xs, lab)
        if s > best_s:
            best_labels, best_s = lab, s
    if best_labels is None:
        best_labels = np.zeros(len(Xs), dtype=int)
    k_star = len(set(int(x) for x in best_labels.tolist()))
    return best_labels, f"{'sklearn' if use_sklearn else 'numpy'}-silhouette(k*={k_star})"


def _numpy_single_linkage(Xs: np.ndarray, k: int) -> np.ndarray:
    """Deterministic agglomerative single-linkage down to k clusters."""
    n = len(Xs)
    clusters: List[List[int]] = [[i] for i in range(n)]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.linalg.norm(Xs[i] - Xs[j]))
    while len(clusters) > k:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = min(D[i, j] for i in clusters[a] for j in clusters[b])
                if best is None or d < best[0] - 1e-12:
                    best = (d, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    labels = np.zeros(n, dtype=int)
    for ci, c in enumerate(clusters):
        for idx in c:
            labels[idx] = ci
    return labels


def _choose_k(n: int, max_k: int) -> int:
    return max(2, min(int(max_k), n - 1))


def cluster_failures(descs: List[FailureDescriptor], *, max_k: int = 6) -> ClusterResult:
    """Cluster failures; return clusters + the dominant pick."""
    n = len(descs)
    # cluster_feature() == the R3M visual embedding for image-based runs (when one
    # was attached this round), else the 6-D geometric feature. The cluster
    # GEOMETRY below (centroid_xyz/theta from raw tee pose) is unchanged either
    # way, so prescription/reset stay geometric regardless of the feature space.
    feats = np.asarray([d.cluster_feature() for d in descs], dtype=float)

    if n == 0:
        return ClusterResult([], [], "empty", None)
    if n <= 3:
        labels = np.arange(n, dtype=int)
        method = "singletons"
    else:
        Xs = _standardize(feats)
        kmax = _choose_k(n, max_k)
        labels, method = _best_k_clustering(Xs, kmax)

    Xs_for_rep = _standardize(feats) if n > 1 else feats
    clusters: List[Cluster] = []
    for lab in sorted(set(int(x) for x in labels.tolist())):
        idxs = [i for i in range(n) if int(labels[i]) == lab]
        cp = [descs[i].centroid_pos() for i in idxs]
        cx = float(np.mean([p[0] for p in cp]))
        cy = float(np.mean([p[1] for p in cp]))
        cz = float(np.mean([p[2] for p in cp]))
        cs = float(np.mean([math.sin(descs[i].centroid_theta()) for i in idxs]))
        cc = float(np.mean([math.cos(descs[i].centroid_theta()) for i in idxs]))
        ctheta = math.atan2(cs, cc)
        centroid_xyz = [cx, cy, cz]
        mpl = float(np.mean([descs[i].peak_loss for i in idxs]))
        # representative = member nearest the cluster centroid in feature space
        cfeat = np.mean(Xs_for_rep[idxs], axis=0)
        rep = min(idxs, key=lambda i: float(np.linalg.norm(Xs_for_rep[i] - cfeat)))
        clusters.append(Cluster(
            label=lab, member_idxs=idxs, centroid_xyz=centroid_xyz,
            centroid_theta=float(ctheta), mean_peak_loss=mpl,
            size=len(idxs), representative_idx=rep))

    dominant = pick_dominant(clusters, descs)
    return ClusterResult(clusters=clusters, labels=[int(x) for x in labels.tolist()],
                         method=method, dominant=dominant)


def pick_dominant(clusters: List[Cluster], descs: List[FailureDescriptor]) -> Cluster:
    """Dominant = most members; tie-break highest mean peak-loss; final tie-break
    lowest min episode_id (determinism)."""
    def key(c: Cluster):
        min_ep = min((descs[i].episode_id for i in c.member_idxs), default=0)
        return (c.size, c.mean_peak_loss, -min_ep)
    return max(clusters, key=key)
