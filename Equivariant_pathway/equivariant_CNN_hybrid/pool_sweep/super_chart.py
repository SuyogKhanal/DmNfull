"""Hybrid pool-sweep super-plot. Same shape as the equivariant + CNN
super charts but reads from the hybrid sweep tree.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def _curve(data) -> Tuple[List[int], List[float]]:
    if not data:
        return [], []
    h = data.get("history", []) or []
    return ([int(r.get("cum_demos", 0) or 0) for r in h],
            [float(r.get("heldout_sr", 0.0) or 0.0) for r in h])


def _first_cross(xs, ys, target):
    for x, y in zip(xs, ys):
        if y >= target:
            return x
    return None


def _peak(ys):
    return max(ys) if ys else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True)
    p.add_argument("--pool_sizes", nargs="+", type=int, default=[20, 30, 40, 50])
    p.add_argument("--target_sr", type=float, default=0.90)
    p.add_argument("--out_path", type=str, default=None)
    args = p.parse_args()

    sweep = Path(args.sweep_dir).resolve()
    out_path = Path(args.out_path) if args.out_path else sweep / "super_baseline_vs_p4_hybrid.png"

    runs: Dict[int, Dict[str, Dict]] = {}
    for pool in args.pool_sizes:
        bo = _load(sweep / f"pool_{pool}" / "baseline_only" / "results" / "learning_curve.json")
        p4 = _load(sweep / f"pool_{pool}" / "p4_only"       / "results" / "learning_curve.json")
        runs[pool] = {"baseline_only": bo, "p4_only": p4}

    n = len(args.pool_sizes)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.5*cols, 4.6*rows),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    summary_rows = []

    for i, pool in enumerate(args.pool_sizes):
        ax = axes_flat[i]
        bo_xs, bo_ys = _curve(runs[pool]["baseline_only"])
        p4_xs, p4_ys = _curve(runs[pool]["p4_only"])
        if bo_xs:
            ax.plot(bo_xs, bo_ys, color="tab:orange", marker="s",
                    linestyle="--", alpha=0.85, label="baseline_only_hybrid")
        if p4_xs:
            ax.plot(p4_xs, p4_ys, color="tab:purple", marker="o",
                    linestyle="-", label="p4_only_hybrid")
        ax.axhline(args.target_sr, ls="--", color="tab:red", alpha=0.6,
                   label=f"target = {args.target_sr:.2f}")
        ax.set_title(f"correction pool size = {pool}")
        ax.set_xlabel("cumulative demos")
        ax.set_ylabel("heldout success rate")
        ax.set_ylim(0, 1.05)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)

        bo_cross = _first_cross(bo_xs, bo_ys, args.target_sr)
        p4_cross = _first_cross(p4_xs, p4_ys, args.target_sr)
        bo_init = bo_xs[0] if bo_xs else 20
        p4_init = p4_xs[0] if p4_xs else 20
        summary_rows.append({
            "pool": pool,
            "baseline_extra_to_target": (bo_cross - bo_init) if bo_cross is not None else None,
            "p4_extra_to_target":       (p4_cross - p4_init) if p4_cross is not None else None,
            "baseline_peak": _peak(bo_ys),
            "p4_peak":       _peak(p4_ys),
        })

    for j in range(len(args.pool_sizes), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Hybrid pathway: P4 LLM-compression vs baseline DAgger across pool sizes\n"
                 "(EquivariantCNNHybridPolicy)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[super_chart] grid -> {out_path}")

    lines = ["pool | baseline_extra | p4_extra | baseline_peak | p4_peak | verdict",
             "-----+----------------+----------+---------------+---------+---------"]
    for r in summary_rows:
        b_e, p_e = r["baseline_extra_to_target"], r["p4_extra_to_target"]
        b_p, p_p = r["baseline_peak"], r["p4_peak"]
        if b_e is not None and p_e is not None:
            verdict = (f"P4 wins by {b_e - p_e} demos" if p_e < b_e
                       else f"baseline wins by {p_e - b_e} demos" if p_e > b_e
                       else "tied")
        elif b_e is not None:
            verdict = "baseline crossed target; P4 did not"
        elif p_e is not None:
            verdict = "P4 crossed target; baseline did not"
        else:
            d = (p_p or 0.0) - (b_p or 0.0)
            verdict = f"neither hit target (peak gap P4-baseline = {d:+.3f})"
        def _fmt(x):
            if x is None: return "  -  "
            if isinstance(x, float): return f"{x:.3f}"
            return f"{x:>5}"
        lines.append(f"{r['pool']:>4} | {_fmt(b_e):>14} | {_fmt(p_e):>8} | "
                     f"{_fmt(b_p):>13} | {_fmt(p_p):>7} | {verdict}")
    txt = "\n".join(lines)
    print(txt)
    out_path.with_name(out_path.stem + "_summary.txt").write_text(txt + "\n")
    out_path.with_name(out_path.stem + "_summary.json").write_text(
        json.dumps({"target_sr": args.target_sr, "pool_sizes": args.pool_sizes,
                    "rows": summary_rows}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
