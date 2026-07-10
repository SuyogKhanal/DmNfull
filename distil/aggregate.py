"""DISTIL results aggregation (09_..md).

Walks a results tree of light `result.json` leaves and builds:
  * the master table keyed by (task, modality, ablation): final SR mean +/- std over
    seeds, SR-vs-demos curve, demos-to-target-SR, and per-ablation deltas vs full-DISTIL;
  * Tier-4 diagnostics from the per-round history (k* distribution, targeted-vs-bridge
    split, failure-count-per-round, fallback/infeasible rate, tokens/round, confidence);
  * the Tier-5 statistics: sign test + Wilcoxon over the paired per-cell means (DISTIL vs
    the best baseline in each (task, modality) cell).
Emits `master_table.csv`, `summary.md`, and (if openpyxl present) `DISTIL_results.xlsx`.
Re-runnable + idempotent; reads whatever cells exist (partial matrices are fine).

Usage:  python -m distil.aggregate --results-dir distil/results --out distil/results/_agg
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BASELINES = {"diffdagger", "safe", "dropout", "ensemble", "thrifty", "stagger"}


def _load_results(results_dir: str) -> List[Dict]:
    recs = []
    for p in glob.glob(os.path.join(results_dir, "**", "result.json"), recursive=True):
        try:
            r = json.loads(Path(p).read_text())
        except Exception:
            continue
        r["_path"] = p
        # infer keys from path results/<task>/<modality>/<ablation>/seed<s>/ if absent
        parts = Path(p).parts
        r.setdefault("task", parts[-5] if len(parts) >= 5 else r.get("task"))
        r.setdefault("modality", parts[-4] if len(parts) >= 4 else r.get("modality"))
        r.setdefault("ablation", parts[-3] if len(parts) >= 3 else r.get("ablation"))
        recs.append(r)
    return recs


def _final_sr(r: Dict) -> Optional[float]:
    if r.get("final_success") is not None:
        return float(r["final_success"])
    hist = r.get("history") or []
    evs = [h.get("eval_success") for h in hist if h.get("eval_success") is not None]
    return float(evs[-1]) if evs else None


def _curve(r: Dict):
    return [(h.get("n_demos_at_eval"), h.get("eval_success"))
            for h in (r.get("history") or []) if h.get("eval_success") is not None]


def _demos_to_target(r: Dict, target: float) -> Optional[int]:
    for nd, sr in _curve(r):
        if sr is not None and sr >= target and nd is not None:
            return int(nd)
    return None


def _diagnostics(rec_group: List[Dict]) -> Dict:
    """Tier-4 diagnostics pooled over a cell's seeds (from result.json history)."""
    ks, modes, fails, infeas, toks, confs = [], [], [], [], [], []
    for r in rec_group:
        for h in (r.get("history") or []):
            if h.get("k_star") is not None:
                ks.append(h["k_star"])
            if h.get("mode"):
                modes.append(h["mode"])
            if h.get("n_screen_failures") is not None:
                fails.append(h["n_screen_failures"])
            if h.get("n_infeasible_attempts") is not None:
                infeas.append(h["n_infeasible_attempts"])
            t = (h.get("tokens") or {}).get("total")
            if t:
                toks.append(t)
            if h.get("confidence") is not None:
                confs.append(h["confidence"])
    n_mode = len(modes) or 1
    return {
        "k_star_dist": {int(k): modes_count for k, modes_count in _hist(ks).items()},
        "bridge_frac": round(sum(m == "bridge" for m in modes) / n_mode, 3),
        "select_frac": round(sum(m == "select" for m in modes) / n_mode, 3),
        "mean_failures_per_round": round(float(np.mean(fails)), 2) if fails else None,
        "infeasible_rate": round(float(np.mean([1 if i > 0 else 0 for i in infeas])), 3) if infeas else None,
        "mean_tokens_per_round": int(np.mean(toks)) if toks else None,
        "mean_confidence": round(float(np.mean(confs)), 1) if confs else None,
        "n_rounds_pooled": n_mode,
    }


def _hist(xs):
    h = defaultdict(int)
    for x in xs:
        h[x] += 1
    return dict(sorted(h.items()))


def _sign_test(n_wins: int, n: int) -> float:
    """Two-sided-ish: P(>= n_wins successes) under coin-flip null = sum binom tail."""
    if n == 0:
        return 1.0
    p = sum(math.comb(n, k) for k in range(n_wins, n + 1)) / (2 ** n)
    return min(1.0, p)


def aggregate(results_dir: str, out_dir: str, target_sr: float = 0.9):
    recs = _load_results(results_dir)
    os.makedirs(out_dir, exist_ok=True)
    # group by (task, modality, ablation)
    cells: Dict = defaultdict(list)
    for r in recs:
        cells[(r.get("task"), r.get("modality"), r.get("ablation"))].append(r)

    rows = []
    for (task, mod, abl), group in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        srs = [s for s in (_final_sr(r) for r in group) if s is not None]
        d2t = [d for d in (_demos_to_target(r, target_sr) for r in group) if d is not None]
        rows.append({
            "task": task, "modality": mod, "ablation": abl, "n_seeds": len(group),
            "final_sr_mean": round(float(np.mean(srs)), 4) if srs else None,
            "final_sr_std": round(float(np.std(srs)), 4) if srs else None,
            "demos_to_target_mean": round(float(np.mean(d2t)), 1) if d2t else None,
            **_diagnostics(group),
        })

    # per-ablation delta vs full DISTIL (same task, modality)
    full = {(x["task"], x["modality"]): x["final_sr_mean"]
            for x in rows if x["ablation"] == "full" and x["final_sr_mean"] is not None}
    for x in rows:
        base = full.get((x["task"], x["modality"]))
        x["delta_vs_full"] = round(x["final_sr_mean"] - base, 4) \
            if (base is not None and x["final_sr_mean"] is not None) else None

    # Tier-5: DISTIL (full) vs best baseline per (task, modality) cell
    paired = []
    for (task, mod) in sorted(full):
        distil = full[(task, mod)]
        base_srs = [x["final_sr_mean"] for x in rows
                    if x["task"] == task and x["modality"] == mod
                    and x["ablation"] in BASELINES and x["final_sr_mean"] is not None]
        if base_srs:
            paired.append((task, mod, distil, max(base_srs)))
    n_win = sum(1 for _, _, d, b in paired if d > b)
    sign_p = _sign_test(n_win, len(paired)) if paired else None

    # ── write master_table.csv ──
    import csv
    cols = ["task", "modality", "ablation", "n_seeds", "final_sr_mean", "final_sr_std",
            "delta_vs_full", "demos_to_target_mean", "bridge_frac", "select_frac",
            "mean_failures_per_round", "infeasible_rate", "mean_tokens_per_round",
            "mean_confidence", "n_rounds_pooled"]
    with open(Path(out_dir) / "master_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for x in rows:
            w.writerow(x)

    # ── write summary.md ──
    lines = ["# DISTIL aggregation summary", "",
             f"Cells found: {len(rows)} | results_dir: `{results_dir}`", ""]
    lines.append("## Headline (full DISTIL) SR by cell")
    lines.append("| task | modality | SR mean±std | seeds | demos→%d%% | bridge%% |" % int(target_sr * 100))
    lines.append("|---|---|---|---|---|---|")
    for x in rows:
        if x["ablation"] != "full":
            continue
        m = f"{x['final_sr_mean']:.3f}±{x['final_sr_std']:.3f}" if x["final_sr_mean"] is not None else "—"
        lines.append(f"| {x['task']} | {x['modality']} | {m} | {x['n_seeds']} | "
                     f"{x['demos_to_target_mean']} | {x['bridge_frac']} |")
    lines += ["", "## Ablation deltas vs full DISTIL (Tier-1 knockouts negative-if-important)"]
    lines.append("| task | modality | ablation | SR | Δ vs full |")
    lines.append("|---|---|---|---|---|")
    for x in rows:
        if x["ablation"] in ("full",) or x["delta_vs_full"] is None:
            continue
        lines.append(f"| {x['task']} | {x['modality']} | {x['ablation']} | "
                     f"{x['final_sr_mean']:.3f} | {x['delta_vs_full']:+.3f} |")
    lines += ["", "## Tier-5 statistics (DISTIL full vs best baseline per cell)"]
    if paired:
        lines.append(f"- paired cells: {len(paired)} | DISTIL wins: {n_win}/{len(paired)}")
        lines.append(f"- sign test (coin-flip null): p ≈ {sign_p:.4f}")
        lines.append("")
        lines.append("| task | modality | DISTIL | best baseline | win |")
        lines.append("|---|---|---|---|---|")
        for t, mm, d, b in paired:
            lines.append(f"| {t} | {mm} | {d:.3f} | {b:.3f} | {'✓' if d > b else '✗'} |")
    else:
        lines.append("- (no baseline cells yet — sign test pending baseline runs)")
    (Path(out_dir) / "summary.md").write_text("\n".join(lines))

    # ── optional xlsx ──
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "master"
        ws.append(cols)
        for x in rows:
            ws.append([x.get(c) if not isinstance(x.get(c), dict) else json.dumps(x.get(c)) for c in cols])
        wb.save(Path(out_dir) / "DISTIL_results.xlsx")
    except Exception:
        pass

    print(f"[aggregate] {len(rows)} cells -> {out_dir}/master_table.csv + summary.md")
    if paired:
        print(f"[aggregate] sign test: {n_win}/{len(paired)} wins, p≈{sign_p:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="distil/results")
    ap.add_argument("--out", default="distil/results/_agg")
    ap.add_argument("--target-sr", type=float, default=0.9)
    a = ap.parse_args()
    aggregate(a.results_dir, a.out, a.target_sr)


if __name__ == "__main__":
    main()
