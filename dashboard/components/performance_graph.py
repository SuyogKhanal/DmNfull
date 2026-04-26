import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METHOD_ORDER = [
    "Naive DAgger",
    "Diff-DAgger",
    "Plain LLM",
    "Plain+VLM (end frame only)",
    "Plain+VLM (end+high-loss frames)",
    "Plain+VLM+RAG",
    "Plain+VLM+RAG+KAG",
    "Plain+VLM+RAG+KAG+TKF (Full System)",
]


def _ablation_to_method(flags: Dict) -> Optional[str]:
    use_vlm       = bool(flags.get("use_vlm", False))
    use_kag       = bool(flags.get("use_kag", False))
    use_rag       = bool(flags.get("use_rag", False))
    use_reasoning = bool(flags.get("use_reasoning", False))
    use_tkf       = bool(flags.get("use_tkf", False))

    if use_vlm and use_kag and use_rag and use_reasoning and use_tkf:
        return "Plain+VLM+RAG+KAG+TKF (Full System)"
    if use_vlm and use_kag and use_rag and use_reasoning and not use_tkf:
        return "Plain+VLM+RAG+KAG"
    if use_vlm and use_rag and use_reasoning and not use_kag and not use_tkf:
        return "Plain+VLM+RAG"
    if use_vlm and use_reasoning and not use_rag and not use_kag and not use_tkf:
        return "Plain+VLM (end+high-loss frames)"
    if use_reasoning and not use_vlm and not use_rag and not use_kag and not use_tkf:
        return "Plain LLM"
    return None


def _scan_run(run_path: Path) -> Optional[Dict]:
    fp = run_path / "full_output.json"
    if not fp.exists():
        return None
    try:
        with open(fp, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    meta = data.get("metadata", {})
    flags = meta.get("pipeline_flags") or data.get("config", {}).get("pipeline", {})
    n_ep = int(meta.get("n_episodes", 0) or 0)
    n_succ = int(meta.get("n_successes", 0) or 0)
    n_fail = int(meta.get("n_failures", n_ep - n_succ) or 0)
    success_rate = (n_succ / n_ep) if n_ep > 0 else 0.0
    parsed = data.get("phase_c", {}).get("parsed_prescription", {})
    total_demos = 0
    if isinstance(parsed, dict):
        total_demos = int(parsed.get("total_demonstrations_needed", 0) or 0)
    if total_demos == 0:
        total_demos = n_fail
    method = _ablation_to_method(flags)
    return {
        "run_dir":       str(run_path),
        "pipeline_flags":flags,
        "method":        method,
        "n_episodes":    n_ep,
        "n_successes":   n_succ,
        "n_failures":    n_fail,
        "success_rate":  success_rate,
        "total_demos":   total_demos,
    }


def _scan_baseline(run_path: Path) -> Optional[Dict]:
    fp = run_path / "baseline_summary.json"
    if not fp.exists():
        return None
    try:
        with open(fp, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    name = data.get("baseline", "")
    method = {"naive_dagger": "Naive DAgger", "diff_dagger": "Diff-DAgger"}.get(name)
    return {
        "run_dir":       str(run_path),
        "method":        method,
        "n_episodes":    int(data.get("n_episodes", 0) or 0),
        "n_successes":   int(data.get("n_successes", 0) or 0),
        "n_failures":    int(data.get("n_failures", 0) or 0),
        "success_rate":  float(data.get("success_rate", 0.0) or 0.0),
        "total_demos":   int(data.get("total_demonstrations_needed", 0) or 0),
    }


def collect_method_stats(ablations_dir: str = "results/ablations", baselines_dir: str = "results/baselines", runs_dir: str = "results/runs") -> Dict[str, List[Dict]]:
    stats: Dict[str, List[Dict]] = {m: [] for m in METHOD_ORDER}
    ab = Path(ablations_dir)
    if ab.exists():
        for p in sorted(ab.iterdir()):
            if p.is_dir():
                s = _scan_run(p)
                if s and s.get("method") in stats:
                    stats[s["method"]].append(s)
    rn = Path(runs_dir)
    if rn.exists():
        for p in sorted(rn.iterdir()):
            if p.is_dir():
                s = _scan_run(p)
                if s and s.get("method") in stats:
                    stats[s["method"]].append(s)
    bl = Path(baselines_dir)
    if bl.exists():
        for p in sorted(bl.iterdir()):
            if p.is_dir():
                s = _scan_baseline(p)
                if s and s.get("method") in stats:
                    stats[s["method"]].append(s)
    return stats


def build_performance_figure(ablations_dir: str = "results/ablations", baselines_dir: str = "results/baselines", runs_dir: str = "results/runs"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = collect_method_stats(ablations_dir, baselines_dir, runs_dir)

    x_labels = METHOD_ORDER
    mean_success = []
    mean_demos = []
    for m in x_labels:
        runs = stats.get(m, [])
        if runs:
            mean_success.append(sum(r["success_rate"] for r in runs) / len(runs))
            mean_demos.append(sum(r["total_demos"] for r in runs) / len(runs))
        else:
            mean_success.append(0.0)
            mean_demos.append(0.0)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    xs = list(range(len(x_labels)))
    ax1.plot(xs, mean_success, marker="o", linewidth=2, label="Policy success rate", color="#1f77b4")
    ax1.set_ylabel("Policy success rate", color="#1f77b4")
    ax1.set_ylim(0.0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(x_labels, rotation=30, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Method variants — success rate and demos prescribed")

    ax2 = ax1.twinx()
    ax2.bar(xs, mean_demos, alpha=0.25, color="#d62728", label="Total demos prescribed")
    ax2.set_ylabel("Total demonstrations prescribed", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    return fig, stats