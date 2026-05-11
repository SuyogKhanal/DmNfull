"""Hybrid 4-way comparison super-plot.

Reads:
    <sweep_dir>/baseline_only/results/learning_curve.json
    <sweep_dir>/p4_only/results/learning_curve.json
    <sweep_dir>/p5_only/results/learning_curve.json
    <sweep_dir>/p6_only/results/learning_curve.json

Produces:
    <sweep_dir>/baseline_vs_p4.png
    <sweep_dir>/baseline_vs_p5.png
    <sweep_dir>/baseline_vs_p6.png
    <sweep_dir>/super_4way.png
    <sweep_dir>/summary.{txt,json}
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ["baseline_only", "p4_only", "p5_only", "p6_only"]
COLORS = {
    "baseline_only": "tab:orange",
    "p4_only":       "tab:purple",
    "p5_only":       "tab:cyan",
    "p6_only":       "tab:brown",
}
MARKERS = {
    "baseline_only": "s",
    "p4_only":       "o",
    "p5_only":       "^",
    "p6_only":       "D",
}
LINESTYLES = {
    "baseline_only": "--",
    "p4_only":       "-",
    "p5_only":       "-",
    "p6_only":       "-",
}
PRETTY = {
    "baseline_only": "baseline_only_hybrid",
    "p4_only":       "p4_only_hybrid",
    "p5_only":       "p5_only_hybrid",
    "p6_only":       "p6_only_hybrid",
}


def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except Exception as e:
        print(f"[super_chart] failed to read {path}: {e}")
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


def _plot_pair(curves, baseline_name, other_name, target, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bxs, bys = _curve(curves.get(baseline_name))
    oxs, oys = _curve(curves.get(other_name))
    if bxs:
        ax.plot(bxs, bys, color=COLORS[baseline_name],
                marker=MARKERS[baseline_name], linestyle=LINESTYLES[baseline_name],
                alpha=0.85, label=PRETTY[baseline_name])
    if oxs:
        ax.plot(oxs, oys, color=COLORS[other_name],
                marker=MARKERS[other_name], linestyle=LINESTYLES[other_name],
                label=PRETTY[other_name])
        for x, y in zip(oxs, oys):
            ax.annotate("", (x, y))
    ax.axhline(target, ls="--", color="tab:red", alpha=0.6,
               label=f"target = {target:.2f}")
    ax.set_xlabel("cumulative demos")
    ax.set_ylabel("heldout success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Hybrid: {PRETTY[baseline_name]} vs {PRETTY[other_name]}")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[super_chart] pair  -> {out_path}")


def _plot_super(curves, target, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for m in METHODS:
        xs, ys = _curve(curves.get(m))
        if not xs:
            continue
        ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m],
                linestyle=LINESTYLES[m], label=PRETTY[m], alpha=0.9)
    ax.axhline(target, ls="--", color="tab:red", alpha=0.6,
               label=f"target = {target:.2f}")
    ax.set_xlabel("cumulative demos (initial 20 + extras)")
    ax.set_ylabel("heldout success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Hybrid pathway 4-way: baseline vs P4 vs P5 vs P6")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[super_chart] super -> {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True)
    p.add_argument("--target_sr", type=float, default=0.90)
    args = p.parse_args()

    sweep = Path(args.sweep_dir).resolve()
    curves: Dict[str, Optional[Dict]] = {}
    for m in METHODS:
        curves[m] = _load(sweep / m / "results" / "learning_curve.json")

    _plot_pair(curves, "baseline_only", "p4_only", args.target_sr,
               sweep / "baseline_vs_p4.png")
    _plot_pair(curves, "baseline_only", "p5_only", args.target_sr,
               sweep / "baseline_vs_p5.png")
    _plot_pair(curves, "baseline_only", "p6_only", args.target_sr,
               sweep / "baseline_vs_p6.png")
    _plot_super(curves, args.target_sr, sweep / "super_4way.png")

    # Summary table
    rows = []
    for m in METHODS:
        xs, ys = _curve(curves.get(m))
        cross = _first_cross(xs, ys, args.target_sr)
        peak = _peak(ys)
        initial = xs[0] if xs else 20
        rows.append({
            "method": m,
            "extra_to_target": (cross - initial) if cross is not None else None,
            "peak_sr": peak,
            "rounds_run": len(xs) - 1 if xs else 0,
            "final_cum_demos": xs[-1] if xs else None,
            "final_sr": ys[-1] if ys else None,
        })

    lines = ["method        | extra_to_target | peak_sr | rounds | final_demos | final_sr",
             "--------------+-----------------+---------+--------+-------------+---------"]
    def _fmt(x):
        if x is None: return "  -  "
        if isinstance(x, float): return f"{x:.3f}"
        return f"{x:>5}"
    for r in rows:
        lines.append(
            f"{r['method']:<13} | {_fmt(r['extra_to_target']):>15} | "
            f"{_fmt(r['peak_sr']):>7} | {_fmt(r['rounds_run']):>6} | "
            f"{_fmt(r['final_cum_demos']):>11} | {_fmt(r['final_sr']):>7}"
        )

    # Add a head-to-head verdict line per method vs baseline
    bo_extra = next((r["extra_to_target"] for r in rows if r["method"] == "baseline_only"), None)
    bo_peak  = next((r["peak_sr"] for r in rows if r["method"] == "baseline_only"), None)
    lines.append("")
    lines.append("verdict vs baseline (lower extra_to_target = better, peak diff if neither hit target):")
    for r in rows:
        if r["method"] == "baseline_only":
            continue
        m_extra, m_peak = r["extra_to_target"], r["peak_sr"]
        if bo_extra is not None and m_extra is not None:
            if m_extra < bo_extra:
                v = f"{r['method']} wins by {bo_extra - m_extra} demos"
            elif m_extra > bo_extra:
                v = f"baseline wins by {m_extra - bo_extra} demos"
            else:
                v = "tied"
        elif bo_extra is not None:
            v = f"{r['method']} did not hit target; baseline did"
        elif m_extra is not None:
            v = f"{r['method']} hit target; baseline did not"
        else:
            d = (m_peak or 0.0) - (bo_peak or 0.0)
            v = f"neither hit target (peak gap {r['method']}-baseline = {d:+.3f})"
        lines.append(f"  {r['method']:<13} : {v}")

    txt = "\n".join(lines)
    print(txt)
    (sweep / "summary.txt").write_text(txt + "\n")
    (sweep / "summary.json").write_text(json.dumps(
        {"target_sr": args.target_sr, "rows": rows}, indent=2, default=str))
    print(f"[super_chart] summary -> {sweep / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
