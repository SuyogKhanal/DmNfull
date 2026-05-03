"""Paired t-test on prescription quality across active-loop rounds.

Complements mcnemar_analysis.py. Where the McNemar test asks "did P_a's
trained policy outperform P_b's on the same held-out episodes?", this script
asks "did P_a's LLM prescribe higher-quality layouts than P_b's during the
active-loop rounds themselves?".

For each round of each profile we extract two metrics:

  save_rate   = n_saved / n_prescribed
                — fraction of LLM-recommended layouts the BFS / human expert
                  was able to actually demonstrate. Captures whether the
                  prescription was solvable (no impossible start/goal pairs,
                  no fire-locked goals, etc.).

  mean_steps  = mean(len(actions) for each saved demo this round)
                — prescription richness. Longer-trajectory demos indicate the
                  prescribed layout actually challenged the agent rather
                  than being trivially short.

Pairing: round_i across profiles is one paired observation. Different
profiles produce a different schedule of layouts at round_i, but the round
index is the unit of analysis (the LLM had the same training-loop history
length when it produced that prescription). We require the profiles share at
least one common rounds prefix to pair against. If P_a ran longer than P_b,
the comparison is truncated to min(len(P_a.rounds), len(P_b.rounds)).

n_saved discovery (auto-detected, per round demo dir):
  1. <demo_dir>/bfs_collection_summary.json              # original location
  2. <demo_dir>.parent/bfs_collection_summary.json       # post-fix location
  3. count of demo_*.json files in <demo_dir>            # human / fallback

Bonferroni: 2 comparisons (p4_vs_p5, p5_vs_p6) per metric, so corrected
alpha = 0.025 within each metric.

Usage:
  python scripts/prescription_quality_analysis.py \
    --p4-loop-log results/active_loop/p4_<.../>loop_<id>/loop_log.json \
    --p5-loop-log results/active_loop/p5_<...>/loop_<id>/loop_log.json \
    --p6-loop-log results/active_loop/p6_<...>/loop_<id>/loop_log.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.stats import ttest_rel


PAIRS = [("p4", "p5"), ("p5", "p6")]
METRICS = ["save_rate", "mean_steps"]
ALPHA = 0.05
N_COMPARISONS = len(PAIRS)
ALPHA_CORRECTED = ALPHA / N_COMPARISONS


def parse_args():
    p = argparse.ArgumentParser(
        description="Paired t-test on prescription quality across active-loop rounds."
    )
    p.add_argument("--p4-loop-log", type=str, required=True, dest="p4_loop_log")
    p.add_argument("--p5-loop-log", type=str, required=True, dest="p5_loop_log")
    p.add_argument("--p6-loop-log", type=str, required=True, dest="p6_loop_log")
    p.add_argument("--out", type=str, default=None,
                   help="Optional output JSON path. Default: alongside p4 loop_log.")
    return p.parse_args()


def _read_demo_actions_len(demo_path: Path) -> Optional[int]:
    """Return len(actions) for a demo JSON, or None if it's not a demo file."""
    try:
        with open(demo_path, "r") as f:
            payload = json.load(f)
    except Exception:
        return None
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return None
    return len(actions)


def _count_saved_and_steps(demo_dir: Path) -> Tuple[int, List[int]]:
    """Count demos in `demo_dir` and return their action-sequence lengths.

    Counts only files matching `demo_*.json` so summary/metadata sidecars
    can never inflate the saved count. Recurses one level (demo_dir layouts
    are typically flat, but some collectors group by layout_id subdir).
    """
    if not demo_dir.exists():
        return 0, []
    saved = sorted(demo_dir.rglob("demo_*.json"))
    steps: List[int] = []
    n = 0
    for p in saved:
        n_actions = _read_demo_actions_len(p)
        if n_actions is None:
            continue
        steps.append(n_actions)
        n += 1
    return n, steps


def _read_n_saved_from_bfs_summary(demo_dir: Path) -> Optional[int]:
    """Try both legacy (round-level) and post-fix (loop-level) summary paths."""
    for candidate in (demo_dir / "bfs_collection_summary.json",
                      demo_dir.parent / "bfs_collection_summary.json"):
        if candidate.exists():
            try:
                with open(candidate, "r") as f:
                    s = json.load(f)
                # Loop-level summary covers all rounds; only use the
                # round-level summary if it points at THIS demo_dir.
                if str(s.get("demo_dir", "")) and Path(str(s["demo_dir"])).resolve() != demo_dir.resolve():
                    continue
                v = s.get("n_saved")
                if isinstance(v, int):
                    return v
            except Exception:
                continue
    return None


def _load_round_record(round_record: Dict) -> Dict:
    """Pull n_prescribed / n_saved / mean_steps for one round.

    n_prescribed comes from the round's recommended_layouts.json (authoritative
    for what the LLM asked for, post-validation). n_saved is the count of
    valid demo_*.json files in the round demo dir, with the bfs summary used
    only as a sanity-check / cross-reference.
    """
    round_dir = Path(round_record["round_dir"])
    rec_path = round_dir / "recommended_layouts.json"
    if not rec_path.exists():
        raise FileNotFoundError(f"missing {rec_path}")
    with open(rec_path, "r") as f:
        rec = json.load(f)
    n_prescribed = int(rec.get("n_layouts", len(rec.get("layouts", []) or [])))
    demo_dir = Path(rec.get("demo_dir", ""))

    n_saved_counted, step_lens = _count_saved_and_steps(demo_dir)
    n_saved_summary = _read_n_saved_from_bfs_summary(demo_dir)

    # Trust the file count. Note any disagreement so the user can investigate
    # without the script silently swallowing it.
    n_saved = n_saved_counted
    summary_disagreement = (
        n_saved_summary is not None and n_saved_summary != n_saved_counted
    )

    save_rate = (n_saved / n_prescribed) if n_prescribed > 0 else 0.0
    mean_steps = mean(step_lens) if step_lens else 0.0

    return {
        "round":              int(round_record["round"]),
        "round_dir":          str(round_dir),
        "demo_dir":           str(demo_dir),
        "n_prescribed":       n_prescribed,
        "n_saved":            int(n_saved),
        "n_saved_summary":    n_saved_summary,
        "summary_disagreement": summary_disagreement,
        "save_rate":          float(save_rate),
        "mean_steps":         float(mean_steps),
    }


def _load_profile_rounds(loop_log_path: Path) -> List[Dict]:
    if not loop_log_path.exists():
        raise FileNotFoundError(f"missing {loop_log_path}")
    with open(loop_log_path, "r") as f:
        log = json.load(f)
    if not isinstance(log, list):
        raise ValueError(f"{loop_log_path}: expected list of round records")
    rounds = [_load_round_record(r) for r in log]
    rounds.sort(key=lambda r: r["round"])
    return rounds


def _paired_t(name_a: str, name_b: str, vec_a: List[float], vec_b: List[float], metric: str) -> Dict:
    if len(vec_a) != len(vec_b) or len(vec_a) < 2:
        return {
            "comparison": f"{name_a}_vs_{name_b}",
            "metric":     metric,
            "n_pairs":    len(vec_a),
            "mean_a":     mean(vec_a) if vec_a else 0.0,
            "mean_b":     mean(vec_b) if vec_b else 0.0,
            "std_a":      pstdev(vec_a) if len(vec_a) > 1 else 0.0,
            "std_b":      pstdev(vec_b) if len(vec_b) > 1 else 0.0,
            "t_statistic": None,
            "p_value":     None,
            "alpha_uncorrected": ALPHA,
            "alpha_bonferroni":  ALPHA_CORRECTED,
            "significant_uncorrected": False,
            "significant_bonferroni":  False,
            "note": "fewer than 2 paired observations — t-test undefined",
        }
    res = ttest_rel(vec_a, vec_b)
    p = float(res.pvalue)
    return {
        "comparison": f"{name_a}_vs_{name_b}",
        "metric":     metric,
        "n_pairs":    len(vec_a),
        "mean_a":     mean(vec_a),
        "mean_b":     mean(vec_b),
        "std_a":      pstdev(vec_a) if len(vec_a) > 1 else 0.0,
        "std_b":      pstdev(vec_b) if len(vec_b) > 1 else 0.0,
        "delta":      mean(vec_a) - mean(vec_b),
        "t_statistic": float(res.statistic),
        "p_value":     p,
        "alpha_uncorrected": ALPHA,
        "alpha_bonferroni":  ALPHA_CORRECTED,
        "significant_uncorrected": p < ALPHA,
        "significant_bonferroni":  p < ALPHA_CORRECTED,
    }


def _print_pair(r: Dict):
    print(f"\n--- {r['comparison']} | metric = {r['metric']} ---")
    print(f"  n_pairs = {r['n_pairs']}")
    print(f"  A: mean={r['mean_a']:.4f}  std={r['std_a']:.4f}")
    print(f"  B: mean={r['mean_b']:.4f}  std={r['std_b']:.4f}")
    if r["t_statistic"] is None:
        print(f"  {r.get('note','')}")
        return
    print(f"  Δ(A-B) = {r['delta']:+.4f}")
    print(f"  t = {r['t_statistic']:+.4f}   p = {r['p_value']:.6f}")
    print(f"  uncorrected α=0.05  -> {'SIGNIFICANT' if r['significant_uncorrected'] else 'ns'}")
    print(f"  Bonferroni  α={ALPHA_CORRECTED}  -> {'SIGNIFICANT' if r['significant_bonferroni'] else 'ns'}")


def main():
    args = parse_args()
    log_paths = {
        "p4": Path(args.p4_loop_log),
        "p5": Path(args.p5_loop_log),
        "p6": Path(args.p6_loop_log),
    }
    rounds = {k: _load_profile_rounds(p) for k, p in log_paths.items()}
    n_min = min(len(rounds[k]) for k in rounds)
    if n_min == 0:
        raise SystemExit("at least one profile has zero rounds in its loop_log.json")
    if any(len(rounds[k]) != n_min for k in rounds):
        print(f"[prescription_quality] WARNING: round counts differ "
              f"(p4={len(rounds['p4'])}, p5={len(rounds['p5'])}, p6={len(rounds['p6'])}); "
              f"truncating each profile to its first {n_min} rounds.")
    for k in rounds:
        rounds[k] = rounds[k][:n_min]

    # Surface any per-round summary/file disagreements so the user can
    # investigate before drawing conclusions.
    for k, rs in rounds.items():
        bad = [r for r in rs if r["summary_disagreement"]]
        if bad:
            print(f"[prescription_quality] WARNING: {k} has {len(bad)} round(s) where "
                  f"bfs_collection_summary.json disagrees with the file count. "
                  f"Using the file count. Affected rounds: {[r['round'] for r in bad]}")

    print(f"\n{'='*72}\n[prescription_quality] {n_min} paired rounds per profile\n{'='*72}")
    for k in ("p4", "p5", "p6"):
        for r in rounds[k]:
            print(f"  {k} round {r['round']:>3d}: "
                  f"n_prescribed={r['n_prescribed']:>3d}  n_saved={r['n_saved']:>3d}  "
                  f"save_rate={r['save_rate']:.3f}  mean_steps={r['mean_steps']:.2f}")

    pair_results: List[Dict] = []
    for metric in METRICS:
        for a, b in PAIRS:
            vec_a = [r[metric] for r in rounds[a]]
            vec_b = [r[metric] for r in rounds[b]]
            res = _paired_t(a, b, vec_a, vec_b, metric)
            pair_results.append(res)

    for r in pair_results:
        _print_pair(r)

    out = {
        "n_paired_rounds":  n_min,
        "alpha":            ALPHA,
        "alpha_bonferroni": ALPHA_CORRECTED,
        "loop_logs":        {k: str(p) for k, p in log_paths.items()},
        "rounds_per_profile": {k: rs for k, rs in rounds.items()},
        "comparisons":      pair_results,
    }
    out_path = Path(args.out) if args.out else (log_paths["p4"].parent / "prescription_quality_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[prescription_quality] wrote {out_path}")


if __name__ == "__main__":
    main()
