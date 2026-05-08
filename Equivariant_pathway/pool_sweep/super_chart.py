"""Super-plot for the correction-pool-size sweep.

Reads, for each pool size N in --pool_sizes:

    <sweep_dir>/pool_<N>/baseline_only/results/learning_curve.json
    <sweep_dir>/pool_<N>/p4_only/results/learning_curve.json

and produces

    <sweep_dir>/super_baseline_vs_p4.png

A 2x2 figure (one subplot per pool size) overlaying the
baseline-DAgger curve against the P4-LLM-compressed curve. Two
additional summary panels next to the grid show:

  - extra demos collected by each method to first cross target_sr
    (or "—" if neither curve reaches it);
  - peak heldout SR achieved by each method.

Both panels use the demo-budget gap as the headline number — that is
what the supervisor's question reduces to: at the same demo budget,
does P4 close the held-out gap faster than the baseline?

Missing curves are skipped silently so a partial sweep still renders.
"""
from __future__ import annotations

import argparse
import json
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
    except Exception as e:
        print(f"[super_chart] failed to read {path}: {e}")
        return None


def _curve(data: Optional[Dict]) -> Tuple[List[int], List[float]]:
    if not data:
        return [], []
    h = data.get("history", []) or []
    xs = [int(r.get("cum_demos", 0) or 0) for r in h]
    ys = [float(r.get("heldout_sr", 0.0) or 0.0) for r in h]
    return xs, ys


def _first_cross(xs: List[int], ys: List[float], target: float) -> Optional[int]:
    """Smallest cum_demos at which sr >= target."""
    for x, y in zip(xs, ys):
        if y >= target:
            return x
    return None


def _peak(ys: List[float]) -> Optional[float]:
    return max(ys) if ys else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True,
                   help="Directory containing pool_<N>/{baseline_only,p4_only}/results/...")
    p.add_argument("--pool_sizes", nargs="+", type=int, default=[20, 30, 40, 50])
    p.add_argument("--target_sr", type=float, default=0.90)
    p.add_argument("--out_path", type=str, default=None)
    args = p.parse_args()

    sweep = Path(args.sweep_dir).resolve()
    out_path = Path(args.out_path) if args.out_path else sweep / "super_baseline_vs_p4.png"

    # ------------------------------------------------------------------
    # Load all curves up front so we can both plot and summarise.
    # ------------------------------------------------------------------
    runs: Dict[int, Dict[str, Dict]] = {}
    for pool in args.pool_sizes:
        bo = _load(sweep / f"pool_{pool}" / "baseline_only" / "results" / "learning_curve.json")
        p4 = _load(sweep / f"pool_{pool}" / "p4_only"       / "results" / "learning_curve.json")
        runs[pool] = {"baseline_only": bo, "p4_only": p4}

    # ------------------------------------------------------------------
    # 2x2 grid for the four pool sizes.
    # ------------------------------------------------------------------
    n = len(args.pool_sizes)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 4.6 * rows),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    summary_rows: List[Dict] = []
    for i, pool in enumerate(args.pool_sizes):
        ax = axes_flat[i]
        bo_data = runs[pool]["baseline_only"]
        p4_data = runs[pool]["p4_only"]
        bo_xs, bo_ys = _curve(bo_data)
        p4_xs, p4_ys = _curve(p4_data)

        if bo_xs:
            ax.plot(bo_xs, bo_ys, color="tab:blue", marker="s",
                    linestyle="--", alpha=0.85,
                    label="baseline_only")
            for r, x, y in zip(bo_data.get("history", []), bo_xs, bo_ys):
                ax.annotate(f"r{r.get('round', '?')}", (x, y),
                            textcoords="offset points", xytext=(4, -10),
                            fontsize=7, color="tab:blue", alpha=0.7)
        if p4_xs:
            ax.plot(p4_xs, p4_ys, color="tab:green", marker="o",
                    linestyle="-", label="p4_only")
            for r, x, y in zip(p4_data.get("history", []), p4_xs, p4_ys):
                ax.annotate(f"r{r.get('round', '?')}", (x, y),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color="tab:green", alpha=0.7)

        ax.axhline(args.target_sr, ls="--", color="tab:red", alpha=0.6,
                   label=f"target = {args.target_sr:.2f}")
        ax.set_title(f"correction pool size = {pool}")
        ax.set_xlabel("cumulative demos (initial 20 + extras)")
        ax.set_ylabel("heldout success rate")
        ax.set_ylim(0, 1.05)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)

        bo_cross = _first_cross(bo_xs, bo_ys, args.target_sr)
        p4_cross = _first_cross(p4_xs, p4_ys, args.target_sr)
        bo_peak = _peak(bo_ys)
        p4_peak = _peak(p4_ys)
        # extras = cum_demos beyond the initial 20
        bo_initial = bo_xs[0] if bo_xs else 20
        p4_initial = p4_xs[0] if p4_xs else 20
        summary_rows.append({
            "pool":              pool,
            "baseline_initial":  bo_initial,
            "p4_initial":        p4_initial,
            "baseline_cross":    bo_cross,
            "p4_cross":          p4_cross,
            "baseline_peak":     bo_peak,
            "p4_peak":           p4_peak,
            "baseline_extra_to_target": (bo_cross - bo_initial) if bo_cross is not None else None,
            "p4_extra_to_target":       (p4_cross - p4_initial) if p4_cross is not None else None,
        })

    # Hide unused axes if any
    for j in range(len(args.pool_sizes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "P4 LLM-compression vs. baseline DAgger across correction-pool sizes\n"
        "(per subplot: heldout success rate vs. cumulative demos collected)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[super_chart] grid     -> {out_path}")

    # ------------------------------------------------------------------
    # Summary table dump (text + json) — easier for the supervisor than
    # squinting at the chart. The "verdict" column is the headline:
    #   - if both methods crossed target_sr, the one with smaller
    #     extra_to_target wins (lower is better);
    #   - if only one crossed, that one wins;
    #   - if neither crossed, we report peak SR difference and explicitly
    #     mark it as "neither hit target".
    # ------------------------------------------------------------------
    summary_path_json = out_path.with_name(out_path.stem + "_summary.json")
    summary_path_txt  = out_path.with_name(out_path.stem + "_summary.txt")

    lines: List[str] = []
    lines.append("pool | baseline_extra | p4_extra | baseline_peak | p4_peak | verdict")
    lines.append("-----+----------------+----------+---------------+---------+---------")
    for r in summary_rows:
        b_extra = r["baseline_extra_to_target"]
        p_extra = r["p4_extra_to_target"]
        b_peak  = r["baseline_peak"]
        p_peak  = r["p4_peak"]

        if b_extra is not None and p_extra is not None:
            if p_extra < b_extra:
                verdict = f"P4 wins by {b_extra - p_extra} demos"
            elif p_extra > b_extra:
                verdict = f"baseline wins by {p_extra - b_extra} demos"
            else:
                verdict = "tied"
        elif b_extra is not None and p_extra is None:
            verdict = "baseline crossed target; P4 did not"
        elif p_extra is not None and b_extra is None:
            verdict = "P4 crossed target; baseline did not"
        else:
            d = (p_peak or 0.0) - (b_peak or 0.0)
            verdict = f"neither hit target (peak gap P4-baseline = {d:+.3f})"

        def _fmt(x):
            if x is None:
                return "  -  "
            if isinstance(x, float):
                return f"{x:.3f}"
            return f"{x:>5}"
        lines.append(
            f"{r['pool']:>4} | "
            f"{_fmt(b_extra):>14} | "
            f"{_fmt(p_extra):>8} | "
            f"{_fmt(b_peak):>13} | "
            f"{_fmt(p_peak):>7} | "
            f"{verdict}"
        )
    txt = "\n".join(lines)
    print(txt)
    summary_path_txt.write_text(txt + "\n")
    summary_path_json.write_text(json.dumps({
        "target_sr": args.target_sr,
        "pool_sizes": args.pool_sizes,
        "rows": summary_rows,
    }, indent=2, default=str))
    print(f"[super_chart] summary  -> {summary_path_json}")
    print(f"[super_chart] summary  -> {summary_path_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
