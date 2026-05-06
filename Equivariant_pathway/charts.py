"""End-of-cycle line charts: held-out success rate vs cumulative
expert-demo budget, one line per learner.

Reads the per-method `learning_curve.json` that run_full_cycle.py writes
under `results/equivariant_pathway/cycle_<timestamp>/<method>/learning_curve.json`,
and emits four PNG charts under `<cycle_dir>/charts/`:

  baseline_vs_p4.png
  baseline_vs_p5.png
  baseline_vs_p6.png
  baseline_vs_all.png

Each curve is plotted as `additional_demos_collected` (cumulative, on top
of the initial 10 demos) on the x-axis vs `heldout_success_rate` on the
y-axis. The horizontal dashed line at 0.90 is the termination threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHODS = ["baseline_dagger", "p4", "p5", "p6"]
TARGET_SR = 0.90


def _load_curve(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"[charts] missing {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def _extract_xy(curve: Dict) -> Dict[str, list]:
    rounds = curve.get("rounds", []) or []
    xs = [int(r.get("additional_demos_collected", 0)) for r in rounds]
    ys = [float(r.get("heldout_success_rate", 0.0)) for r in rounds]
    return {"x": xs, "y": ys, "label": curve.get("method", "?")}


def _plot_pair(baseline: Dict, other: Dict, out_path: Path,
               title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    if baseline:
        b = _extract_xy(baseline)
        ax.plot(b["x"], b["y"], marker="o", linewidth=2, color="#1f77b4",
                label=f"Baseline DAgger ({b['label']})")
    if other:
        o = _extract_xy(other)
        ax.plot(o["x"], o["y"], marker="s", linewidth=2, color="#d62728",
                label=f"{o['label'].upper()}")

    ax.axhline(y=TARGET_SR, linestyle="--", color="gray", alpha=0.6,
               label=f"Target SR = {TARGET_SR:.2f}")
    ax.set_xlabel("Additional expert demos collected (on top of initial 10)")
    ax.set_ylabel("Held-out random-50 success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[charts] wrote {out_path}")


def _plot_all(curves: Dict[str, Dict], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "baseline_dagger": "#1f77b4",
        "p4":              "#2ca02c",
        "p5":              "#ff7f0e",
        "p6":              "#d62728",
    }
    markers = {"baseline_dagger": "o", "p4": "s", "p5": "^", "p6": "D"}
    pretty  = {"baseline_dagger": "Baseline DAgger",
               "p4": "P4", "p5": "P5", "p6": "P6"}

    for method, curve in curves.items():
        if not curve:
            continue
        xy = _extract_xy(curve)
        ax.plot(
            xy["x"], xy["y"],
            marker=markers.get(method, "x"),
            linewidth=2,
            color=colors.get(method, "#000000"),
            label=pretty.get(method, method),
        )

    ax.axhline(y=TARGET_SR, linestyle="--", color="gray", alpha=0.6,
               label=f"Target SR = {TARGET_SR:.2f}")
    ax.set_xlabel("Additional expert demos collected (on top of initial 10)")
    ax.set_ylabel("Held-out random-50 success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Equivariant policy: demo-budget vs held-out success rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[charts] wrote {out_path}")


def make_charts(cycle_dir: Path, out_dir: Optional[Path] = None) -> Dict:
    out_dir = out_dir or (cycle_dir / "charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    curves: Dict[str, Dict] = {}
    for m in METHODS:
        curves[m] = _load_curve(cycle_dir / m / "learning_curve.json")

    if curves.get("baseline_dagger") and curves.get("p4"):
        _plot_pair(curves["baseline_dagger"], curves["p4"],
                   out_dir / "baseline_vs_p4.png",
                   title="Baseline DAgger vs P4 (KAG)")
    if curves.get("baseline_dagger") and curves.get("p5"):
        _plot_pair(curves["baseline_dagger"], curves["p5"],
                   out_dir / "baseline_vs_p5.png",
                   title="Baseline DAgger vs P5 (KAG + RAG)")
    if curves.get("baseline_dagger") and curves.get("p6"):
        _plot_pair(curves["baseline_dagger"], curves["p6"],
                   out_dir / "baseline_vs_p6.png",
                   title="Baseline DAgger vs P6 (KAG + RAG + TKF)")

    _plot_all(curves, out_dir / "baseline_vs_all.png")

    summary = {
        "cycle_dir": str(cycle_dir),
        "out_dir":   str(out_dir),
        "demos_to_target": {
            m: _demos_to_target(c) for m, c in curves.items()
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def _demos_to_target(curve: Optional[Dict]) -> Optional[int]:
    """Return the smallest `additional_demos_collected` at which the curve
    reaches >= TARGET_SR. None if the curve never reaches it.
    """
    if not curve:
        return None
    rounds = curve.get("rounds", []) or []
    for r in rounds:
        if float(r.get("heldout_success_rate", 0.0)) >= TARGET_SR:
            return int(r.get("additional_demos_collected", 0))
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cycle_dir", type=str, required=True,
                   help="results/equivariant_pathway/cycle_<timestamp>/ "
                        "produced by run_full_cycle.py")
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = make_charts(Path(args.cycle_dir),
                      Path(args.out_dir) if args.out_dir else None)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
