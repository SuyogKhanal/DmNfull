"""Cross-run contamination guard.

Upstream's ``layout_setup.assert_no_contamination`` already enforces
``correction ⊥ training ⊥ heldout`` *within* a single run. This module
adds the missing across-run check: for every pair of completed runs, the
correction-pool signatures and per-method demo identities must be
disjoint. Surfaces a single PASS / VIOLATION line + writes a structured
report to ``results/aggregate/contamination_report.json``.

The check is best-effort by design: a violation is reported but does not
abort downstream plotting, so the user can see both the curves and the
issue.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

from .layout_setup import RESULTS_ROOT, _signatures_from_yaml


# Methods we know to scan demos for. Anything else in the per-run dir
# (e.g. ``shared/``) is ignored.
KNOWN_METHODS = ("baseline", "p4_sequential", "p4_batch")


def _run_dirs(results_root: Path) -> List[Path]:
    return sorted(
        d for d in results_root.glob("run_*") if d.is_dir() and (d / "shared").is_dir()
    )


def _per_round_correction_signatures(run_dir: Path) -> Dict[int, Set[Tuple]]:
    """Return ``{round_idx: signature_set}`` for every per-round
    correction pool yaml under ``run_dir/shared/round_*/``.

    Falls back to the legacy single-file
    ``run_dir/shared/correction_layouts.yaml`` (stored under the
    sentinel round_idx ``0``) so reports stay sensible if the user has a
    mix of pre-rotation and post-rotation runs on disk.
    """
    out: Dict[int, Set[Tuple]] = {}
    shared = run_dir / "shared"
    for d in sorted(shared.glob("round_*")):
        if not d.is_dir():
            continue
        try:
            idx = int(d.name.split("_")[-1])
        except ValueError:
            continue
        sigs = _signatures_from_yaml(d / "correction_layouts.yaml")
        if sigs:
            out[idx] = sigs
    if not out:
        legacy = _signatures_from_yaml(shared / "correction_layouts.yaml")
        if legacy:
            out[0] = legacy
    return out


def _demo_signature(demo_json: dict) -> Tuple:
    """Identify a demo by its (layout_id, start, goal, fires, actions hash).

    Two demos that collect the same trajectory on the same layout will
    hash identically; two demos for the same layout that differ even by
    one action will not. This is the cross-run identity we want — if any
    run-pair collected byte-identical demos for the same layout that
    would indicate the correction pools overlapped.
    """
    start = tuple(demo_json.get("start_pos") or ())
    goal = tuple(demo_json.get("goal_pos") or ())
    fires = tuple(sorted(tuple(p) for p in (demo_json.get("fire_positions") or [])))
    actions = demo_json.get("actions") or []
    h = hashlib.sha256(json.dumps(actions, sort_keys=True).encode()).hexdigest()[:16]
    return (demo_json.get("layout_id"), start, goal, fires, h)


def _collect_demo_sigs(method_root: Path) -> Set[Tuple]:
    out: Set[Tuple] = set()
    if not method_root.exists():
        return out
    demos_dir = method_root / "demos"
    if not demos_dir.exists():
        return out
    for p in demos_dir.rglob("*.json"):
        try:
            with open(p, "r") as f:
                obj = json.load(f)
        except Exception:
            continue
        out.add(_demo_signature(obj))
    return out


def cross_run_check(
    results_root: Path = RESULTS_ROOT,
    methods: Tuple[str, ...] = KNOWN_METHODS,
    sample_overlap_limit: int = 5,
) -> Dict:
    """Run the across-run contamination check and return a structured
    report.

    Per-round rotation means each run owns N pools (one per round). We
    pair-check every ``(run_a, round_a) × (run_b, round_b)`` for layout
    overlap and split the findings:

    * ``correction_pool_intra_run_overlap`` — same run, different rounds.
      Should always be empty if
      :func:`layout_setup.ensure_correction_layouts_for_round`'s blocked
      set worked. Surfacing it is a correctness check on the rotation.
    * ``correction_pool_cross_run_overlap`` — different runs. Overlaps
      here are statistically possible (rounds at the same index in two
      different runs may happen to draw the same layout); we just want
      visibility.

    Report shape::

        {
          "status": "ok" | "violation",
          "n_runs_checked": int,
          "n_pools_checked": int,
          "correction_pool_intra_run_overlap": [
              {"pool_a": [run_i, round_a], "pool_b": [run_i, round_b],
               "n_overlap": int, "samples": [...]}, ...
          ],
          "correction_pool_cross_run_overlap": [
              {"pool_a": [run_i, round_a], "pool_b": [run_j, round_b],
               "n_overlap": int, "samples": [...]}, ...
          ],
          "demo_cross_run_overlap": {
              "<method>": [
                  {"runs": [i, j], "n_overlap": int, "samples": [...]},
                  ...
              ],
              ...
          }
        }
    """
    run_dirs = _run_dirs(results_root)
    run_ids = [int(d.name.split("_")[-1]) for d in run_dirs]

    # ---- correction pool overlaps -----------------------------------------
    # pools[(run_id, round_idx)] = signature_set
    pools: Dict[Tuple[int, int], Set[Tuple]] = {}
    for rid, rdir in zip(run_ids, run_dirs):
        for round_idx, sigs in _per_round_correction_signatures(rdir).items():
            pools[(rid, round_idx)] = sigs

    intra_overlaps: List[Dict] = []
    cross_overlaps: List[Dict] = []
    for (key_a, sigs_a), (key_b, sigs_b) in combinations(pools.items(), 2):
        inter = sigs_a & sigs_b
        if not inter:
            continue
        samples = [list(s) for s in list(inter)[:sample_overlap_limit]]
        entry = {
            "pool_a": [key_a[0], key_a[1]],
            "pool_b": [key_b[0], key_b[1]],
            "n_overlap": len(inter),
            "samples": samples,
        }
        if key_a[0] == key_b[0]:
            intra_overlaps.append(entry)
        else:
            cross_overlaps.append(entry)

    # ---- per-method demo overlaps (unchanged: aggregate per run) ----------
    demo_overlaps: Dict[str, List[Dict]] = {m: [] for m in methods}
    for m in methods:
        demo_sigs: Dict[int, Set[Tuple]] = {
            rid: _collect_demo_sigs(d / m) for rid, d in zip(run_ids, run_dirs)
        }
        for (rid_a, sigs_a), (rid_b, sigs_b) in combinations(demo_sigs.items(), 2):
            inter = sigs_a & sigs_b
            if inter:
                samples = [list(s) for s in list(inter)[:sample_overlap_limit]]
                demo_overlaps[m].append({
                    "runs": [rid_a, rid_b],
                    "n_overlap": len(inter),
                    "samples": samples,
                })

    any_violation = (
        bool(intra_overlaps)
        or bool(cross_overlaps)
        or any(demo_overlaps[m] for m in methods)
    )
    return {
        "status": "violation" if any_violation else "ok",
        "n_runs_checked": len(run_dirs),
        "n_pools_checked": len(pools),
        "correction_pool_intra_run_overlap": intra_overlaps,
        "correction_pool_cross_run_overlap": cross_overlaps,
        "demo_cross_run_overlap": demo_overlaps,
    }


def write_report(report: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def print_summary(report: Dict) -> None:
    n_runs = report.get("n_runs_checked", 0)
    n_pools = report.get("n_pools_checked", 0)
    if report.get("status") == "ok":
        print(
            f"[contamination] OK  "
            f"(n_runs_checked={n_runs}, n_pools_checked={n_pools})"
        )
        return
    print(
        f"[contamination] VIOLATION  "
        f"(n_runs_checked={n_runs}, n_pools_checked={n_pools})"
    )
    for overlap in report.get("correction_pool_intra_run_overlap", []) or []:
        print(
            f"  INTRA-run pool overlap: {overlap['pool_a']} vs {overlap['pool_b']} "
            f"n_overlap={overlap['n_overlap']}  (this should never happen)"
        )
    for overlap in report.get("correction_pool_cross_run_overlap", []) or []:
        print(
            f"  cross-run pool overlap: {overlap['pool_a']} vs {overlap['pool_b']} "
            f"n_overlap={overlap['n_overlap']}"
        )
    for m, lst in (report.get("demo_cross_run_overlap") or {}).items():
        for overlap in lst:
            print(
                f"  demo overlap [{m}]: runs={overlap['runs']} "
                f"n_overlap={overlap['n_overlap']}"
            )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_root", type=str, default=str(RESULTS_ROOT),
        help="Root holding run_*/ subdirs (default: this suite's results/)",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="Write the report JSON here (default: <results_root>/aggregate/contamination_report.json)",
    )
    args = p.parse_args()

    root = Path(args.results_root).resolve()
    out = (
        Path(args.out).resolve()
        if args.out
        else root / "aggregate" / "contamination_report.json"
    )
    report = cross_run_check(results_root=root)
    write_report(report, out)
    print_summary(report)
    print(f"  report -> {out}")
