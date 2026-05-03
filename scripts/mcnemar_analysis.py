"""McNemar paired-binary analysis for the P4-vs-P5 and P5-vs-P6 comparisons.

Consumes the per-episode success vectors written by run_mcnemar_eval.py and
runs statsmodels.stats.contingency_tables.mcnemar on each pair. Bonferroni-
corrects across the two comparisons within a single mode (corrected alpha =
0.05 / 2 = 0.025).

Why McNemar (vs. McNemar-fishing for direction): the outcome is paired binary
(same episode_id, same env seed, same layout, run under two profiles). The
discordant pairs (b, c) are exactly the per-episode disagreements; concordant
pairs are uninformative for the test. statsmodels picks `exact` (binomial)
when b+c is small (< 25), continuity-corrected chi-square otherwise — we
follow the standard rule below.

Usage:
  python scripts/mcnemar_analysis.py --results-dir results/mcnemar/run_<ts> \
      --mode policy
  python scripts/mcnemar_analysis.py --results-dir results/mcnemar/run_<ts> \
      --mode llm_actor
  python scripts/mcnemar_analysis.py --results-dir results/mcnemar/run_<ts> \
      --mode both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statsmodels.stats.contingency_tables import mcnemar


PROFILE_DIR = {
    "p4": "p4_vlm_reasoning_kag_cross_plain_llm",
    "p5": "p5_vlm_reasoning_kag_rag_cross_plain_llm",
    "p6": "p6_vlm_reasoning_kag_rag_tkf_cross_plain_llm",
}
PAIRS = [("p4", "p5"), ("p5", "p6")]
ALPHA = 0.05
N_COMPARISONS = len(PAIRS)
ALPHA_CORRECTED = ALPHA / N_COMPARISONS


def parse_args():
    p = argparse.ArgumentParser(description="McNemar test on paired per-episode success vectors.")
    p.add_argument("--results-dir", type=str, required=True, dest="results_dir",
                   help="Directory written by scripts/run_mcnemar_eval.py")
    p.add_argument("--mode", type=str, choices=["policy", "llm_actor", "both"], required=True)
    return p.parse_args()


def _load_success_vec(profile_dir: Path, mode: str) -> Tuple[List[int], Dict]:
    """Return (success_vec_indexed_by_episode_id, raw_payload)."""
    fname = profile_dir / f"per_episode_success_{mode}.json"
    if not fname.exists():
        raise FileNotFoundError(f"missing {fname}")
    with open(fname, "r") as f:
        payload = json.load(f)
    eps = payload["episodes"]
    eps_sorted = sorted(eps, key=lambda e: e["episode_id"])
    ids = [e["episode_id"] for e in eps_sorted]
    if ids != list(range(len(ids))):
        raise ValueError(f"{fname}: episode_ids must be 0..N-1 contiguous, got {ids[:5]}...")
    return [int(e["success"]) for e in eps_sorted], payload


def _build_2x2(a: List[int], b: List[int]) -> Dict:
    """Standard McNemar 2x2 in the layout statsmodels expects:

                     B=success     B=fail
        A=success      n_ss          n_sf
        A=fail         n_fs          n_ff

    Discordant pairs are b=n_sf (A succeeds, B fails) and c=n_fs (A fails,
    B succeeds). McNemar tests H0: b == c.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n_ss = n_sf = n_fs = n_ff = 0
    for ai, bi in zip(a, b):
        if ai == 1 and bi == 1: n_ss += 1
        elif ai == 1 and bi == 0: n_sf += 1
        elif ai == 0 and bi == 1: n_fs += 1
        else: n_ff += 1
    return {
        "table":   [[n_ss, n_sf], [n_fs, n_ff]],
        "n_ss":    n_ss, "n_sf": n_sf, "n_fs": n_fs, "n_ff": n_ff,
        "n_total": n_ss + n_sf + n_fs + n_ff,
    }


def _run_pair(name_a: str, name_b: str, vec_a: List[int], vec_b: List[int]) -> Dict:
    table_info = _build_2x2(vec_a, vec_b)
    discordant = table_info["n_sf"] + table_info["n_fs"]
    use_exact = discordant < 25
    correction = not use_exact

    res = mcnemar(table_info["table"], exact=use_exact, correction=correction)

    sr_a = sum(vec_a) / len(vec_a) if vec_a else 0.0
    sr_b = sum(vec_b) / len(vec_b) if vec_b else 0.0

    return {
        "comparison": f"{name_a}_vs_{name_b}",
        "n_episodes": len(vec_a),
        "sr_a":       sr_a,
        "sr_b":       sr_b,
        "delta_sr":   sr_a - sr_b,
        "table":      table_info["table"],
        "n_ss":       table_info["n_ss"],
        "n_sf":       table_info["n_sf"],
        "n_fs":       table_info["n_fs"],
        "n_ff":       table_info["n_ff"],
        "discordant": discordant,
        "test":       "exact_binomial" if use_exact else "chi2_with_continuity_correction",
        "statistic":  float(res.statistic) if res.statistic is not None else None,
        "p_value":    float(res.pvalue),
        "alpha_uncorrected": ALPHA,
        "alpha_bonferroni":  ALPHA_CORRECTED,
        "significant_uncorrected": float(res.pvalue) < ALPHA,
        "significant_bonferroni":  float(res.pvalue) < ALPHA_CORRECTED,
    }


def _print_pair(r: Dict):
    print(f"\n--- {r['comparison']} ---")
    print(f"  n_episodes={r['n_episodes']}  SR(A)={r['sr_a']:.3f}  SR(B)={r['sr_b']:.3f}  Δ={r['delta_sr']:+.3f}")
    print(f"            B=success  B=fail")
    print(f"  A=succ      {r['n_ss']:>5d}    {r['n_sf']:>5d}")
    print(f"  A=fail      {r['n_fs']:>5d}    {r['n_ff']:>5d}")
    print(f"  discordant b+c = {r['discordant']}  test = {r['test']}")
    stat_s = f"{r['statistic']:.4f}" if r["statistic"] is not None else "n/a"
    print(f"  statistic = {stat_s}   p = {r['p_value']:.6f}")
    print(f"  uncorrected α=0.05  -> {'SIGNIFICANT' if r['significant_uncorrected'] else 'ns'}")
    print(f"  Bonferroni  α={ALPHA_CORRECTED}  -> {'SIGNIFICANT' if r['significant_bonferroni'] else 'ns'}")


def _run_mode(results_dir: Path, mode: str) -> Dict:
    print(f"\n{'='*72}\n[mcnemar] mode = {mode}\n{'='*72}")
    vecs: Dict[str, List[int]] = {}
    sources: Dict[str, str] = {}
    for short, profile in PROFILE_DIR.items():
        prof_dir = results_dir / profile
        vec, payload = _load_success_vec(prof_dir, mode)
        vecs[short] = vec
        sources[short] = str(prof_dir / f"per_episode_success_{mode}.json")
        print(f"[mcnemar/{mode}] {short}: SR={sum(vec)}/{len(vec)} ({sum(vec)/len(vec):.2%})")

    # Confirm seed schedules match across profiles — the McNemar pairing
    # depends on episode_id i meaning the same env layout in each profile.
    # If anyone drifted, bail out loudly rather than report a paired test
    # that isn't actually paired.
    expected_n = len(vecs["p4"])
    if not (len(vecs["p5"]) == len(vecs["p6"]) == expected_n):
        raise ValueError(f"profile lengths differ: "
                         f"p4={len(vecs['p4'])} p5={len(vecs['p5'])} p6={len(vecs['p6'])}")

    pair_results = []
    for a, b in PAIRS:
        pair_results.append(_run_pair(a, b, vecs[a], vecs[b]))
    for r in pair_results:
        _print_pair(r)

    return {
        "mode":          mode,
        "n_comparisons": N_COMPARISONS,
        "alpha":         ALPHA,
        "alpha_bonferroni": ALPHA_CORRECTED,
        "sources":       sources,
        "comparisons":   pair_results,
    }


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"results dir not found: {results_dir}")

    modes = ["policy", "llm_actor"] if args.mode == "both" else [args.mode]
    out: Dict = {"results_dir": str(results_dir), "modes": {}}
    for m in modes:
        out["modes"][m] = _run_mode(results_dir, m)

    out_path = results_dir / f"mcnemar_results_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[mcnemar] wrote {out_path}")


if __name__ == "__main__":
    main()
