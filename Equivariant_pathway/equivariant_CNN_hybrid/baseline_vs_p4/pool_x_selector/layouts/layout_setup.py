"""Per-run layout sourcing for the pool_x_selector suite.

Adapted from
``Equivariant_pathway/equivariant_CNN_hybrid/baseline_vs_p4/layout_setup.py``.

Two changes from upstream:

1. Outputs live under this suite's ``results/run_{id}/shared/`` so the
   suite never touches upstream's ``runs/`` tree.
2. ``correction_n`` defaults are driven by config (50 in this suite vs 40
   upstream); the function accepts it as a parameter so the orchestrator
   can pass it through.

We *reuse* upstream's training_layouts.yaml + heldout_layouts.yaml verbatim
(they are stable, seed-deterministic, and shared across the whole repo).
If those files do not exist, we materialize them via the same seed scheme
upstream uses so the initial-20 BFS demos + initial checkpoint stay
byte-identical to upstream's `shared/init_*/` cache.
"""
from __future__ import annotations

import fcntl
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

# Resolve the repo root so we can import the upstream shared utilities.
# This file lives at: …/baseline_vs_p4/pool_x_selector/layouts/
# Parents:           [4] = pool_x_selector, [5] = baseline_vs_p4,
#                    [6] = equivariant_CNN_hybrid, [7] = Equivariant_pathway,
#                    [8] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.layout_sampler import (  # noqa: E402
    _load_blocked_signatures,
    _signature,
    sample_layouts,
    write_yaml,
)

# Upstream shared dir — read-only source for training_layouts.yaml +
# heldout_layouts.yaml. We never write here.
UPSTREAM_COMPARE_ROOT = (
    REPO_ROOT / "Equivariant_pathway" / "equivariant_CNN_hybrid" / "baseline_vs_p4"
)
UPSTREAM_SHARED_DIR = UPSTREAM_COMPARE_ROOT / "shared"

# This suite's own root: …/baseline_vs_p4/pool_x_selector/
SUITE_ROOT = UPSTREAM_COMPARE_ROOT / "pool_x_selector"
RESULTS_ROOT = SUITE_ROOT / "results"

# Seeds for the GLOBAL layout sets — kept identical to upstream so that the
# initial-20 demos + initial checkpoint cached at
# ``shared/init_demos/`` and ``shared/init_checkpoints/`` remain valid.
GLOBAL_TRAIN_SEED = 11
GLOBAL_HELDOUT_SEED = 22

# Per-run correction-pool seed base. Kept identical to upstream so the
# in-run contamination guard semantics are unchanged.
CORRECTION_SEED_BASE = 91_000_000


def _write_training_layouts(layouts: List[Dict], path: Path) -> None:
    payload = {
        "img_size": 80, "grid_size": 5, "cell_px": 16,
        "num_fires": 3, "min_manhattan": 4, "n_repetitions": 1,
        "training_layouts": [
            {**L, "n_repetitions": int(L.get("n_repetitions", 1) or 1)}
            for L in layouts
        ],
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def ensure_shared_layouts(initial_demos: int, heldout_n: int) -> Dict[str, Path]:
    """Return paths to training_layouts.yaml and heldout_layouts.yaml.

    These live in *upstream's* shared dir. They are seed-deterministic and
    shared across the entire repo; we do not duplicate them.

    If they are missing (fresh checkout), we materialize them in-place
    using the upstream seed scheme so the byte content matches upstream's.
    """
    UPSTREAM_SHARED_DIR.mkdir(parents=True, exist_ok=True)
    train_yaml = UPSTREAM_SHARED_DIR / "training_layouts.yaml"
    heldout_yaml = UPSTREAM_SHARED_DIR / "heldout_layouts.yaml"

    if not heldout_yaml.exists():
        heldout = sample_layouts(
            n=heldout_n, grid_size=5, num_fires=3, min_manhattan=4,
            seed=GLOBAL_HELDOUT_SEED, blocked_signatures=set(),
        )
        write_yaml(heldout, heldout_yaml, grid_size=5)

    if not train_yaml.exists():
        blocked = _load_blocked_signatures([str(heldout_yaml)])
        train = sample_layouts(
            n=initial_demos, grid_size=5, num_fires=3, min_manhattan=4,
            seed=GLOBAL_TRAIN_SEED, blocked_signatures=blocked,
        )
        _write_training_layouts(train, train_yaml)

    return {"train": train_yaml, "heldout": heldout_yaml}


def _signatures_from_yaml(path: Path) -> Set[Tuple]:
    if not path.exists():
        return set()
    with open(path, "r") as f:
        spec = yaml.safe_load(f) or {}
    out: Set[Tuple] = set()
    for key in ("test_layouts", "training_layouts", "heldout_test_layouts", "layouts"):
        for L in spec.get(key, []) or []:
            out.add(_signature(L["start_pos"], L["goal_pos"], L["fire_positions"]))
    return out


def assert_no_contamination(
    correction_yaml: Path, train_yaml: Path, heldout_yaml: Path,
) -> Dict[str, int]:
    """Verify correction ⊥ training ⊥ heldout. Raises SystemExit on leak.

    Verbatim semantics from upstream — same signature scheme, same error
    surface so existing log readers keep working.
    """
    corr_sigs = _signatures_from_yaml(correction_yaml)
    train_sigs = _signatures_from_yaml(train_yaml)
    heldout_sigs = _signatures_from_yaml(heldout_yaml)

    leak_train = corr_sigs & train_sigs
    leak_heldout = corr_sigs & heldout_sigs
    if leak_train or leak_heldout:
        lines = ["[layout-setup] CONTAMINATION DETECTED in correction pool:"]
        for sig in sorted(leak_train):
            lines.append(f"  * shares (start,goal,fires) with TRAINING set: {sig}")
        for sig in sorted(leak_heldout):
            lines.append(f"  * shares (start,goal,fires) with HELDOUT  set: {sig}")
        raise SystemExit("\n".join(lines))

    return {
        "n_correction": len(corr_sigs),
        "n_train": len(train_sigs),
        "n_heldout": len(heldout_sigs),
        "overlap_train": 0,
        "overlap_heldout": 0,
    }


def ensure_correction_layouts_for_run(
    run_id: int,
    correction_n: int,
    train_yaml: Path,
    heldout_yaml: Path,
    out_dir: Path,
) -> Path:
    """Sample correction layouts for a single run and write
    ``out_dir/correction_layouts.yaml``.

    Per-run seed = CORRECTION_SEED_BASE + run_id so each of the N runs
    produces a deterministically distinct pool.

    LEGACY: this function pre-dates per-round rotation. The orchestrator
    no longer calls it; new code should use
    :func:`ensure_correction_layouts_for_round`. Kept for back-compat with
    any external caller that still expects a single per-run pool yaml.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "correction_layouts.yaml"
    if out_path.exists():
        report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
        _save_report(out_dir, run_id, report)
        return out_path

    blocked = _load_blocked_signatures([str(train_yaml), str(heldout_yaml)])
    layouts = sample_layouts(
        n=correction_n, grid_size=5, num_fires=3, min_manhattan=4,
        seed=CORRECTION_SEED_BASE + int(run_id),
        blocked_signatures=blocked,
    )
    for i, L in enumerate(layouts):
        L["name"] = f"corr_r{int(run_id):02d}_{i + 1:03d}"
    write_yaml(layouts, out_path, grid_size=5)

    report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
    _save_report(out_dir, run_id, report)
    return out_path


def _prior_round_yamls(run_shared_dir: Path, current_round: int) -> List[Path]:
    """Return every ``round_NNN/correction_layouts.yaml`` under
    ``run_shared_dir`` with round index strictly less than
    ``current_round``. Used to feed the sampler's blocked-signature set
    so each round's pool is disjoint from every earlier round in the
    same run.
    """
    out: List[Path] = []
    if not run_shared_dir.exists():
        return out
    for d in sorted(run_shared_dir.glob("round_*")):
        if not d.is_dir():
            continue
        try:
            idx = int(d.name.split("_")[-1])
        except ValueError:
            continue
        if idx >= int(current_round):
            continue
        p = d / "correction_layouts.yaml"
        if p.exists():
            out.append(p)
    return out


def ensure_correction_layouts_for_round(
    run_id: int,
    round_idx: int,
    correction_n: int,
    train_yaml: Path,
    heldout_yaml: Path,
    run_shared_dir: Path,
) -> Path:
    """Sample a fresh correction pool for a single (run, round) pair.

    Writes ``run_shared_dir/round_{round_idx:03d}/correction_layouts.yaml``.
    The seed combines run_id and round_idx so different rounds in the same
    run produce different layouts, and different runs at the same round
    index also differ. The blocked-signature set is the union of:

    * the global training layouts,
    * the global heldout layouts,
    * every earlier round's correction pool *for this run* (so rounds
      within one run are mutually disjoint).

    Idempotent — if the round's yaml already exists on disk we just
    re-verify non-contamination against training + heldout and return the
    path. This means baseline / p4_sequential / p4_batch can all call
    this function with the same (run_id, round_idx) and get the same
    yaml back; whoever runs first does the sampling, the others read.
    """
    round_dir = run_shared_dir / f"round_{int(round_idx):03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    out_path = round_dir / "correction_layouts.yaml"
    if out_path.exists():
        report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
        _save_report(round_dir, run_id, report, round_idx=round_idx)
        return out_path

    prior = _prior_round_yamls(run_shared_dir, round_idx)
    blocked_inputs = [str(train_yaml), str(heldout_yaml)] + [str(p) for p in prior]
    blocked = _load_blocked_signatures(blocked_inputs)
    layouts = sample_layouts(
        n=correction_n, grid_size=5, num_fires=3, min_manhattan=4,
        seed=CORRECTION_SEED_BASE + int(run_id) * 1000 + int(round_idx),
        blocked_signatures=blocked,
    )
    for i, L in enumerate(layouts):
        L["name"] = f"corr_r{int(run_id):02d}_rnd{int(round_idx):03d}_{i + 1:03d}"
    write_yaml(layouts, out_path, grid_size=5)

    report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
    _save_report(round_dir, run_id, report, round_idx=round_idx)
    return out_path


def _layouts_in_yaml(path: Path) -> List[Dict]:
    """Return the raw layout list from a correction-pool yaml (probe the
    usual keys). Unlike ``_signatures_from_yaml`` this preserves the count
    (no de-dup), which is what the seed-bump fill check needs.
    """
    if not path.exists():
        return []
    with open(path, "r") as f:
        spec = yaml.safe_load(f) or {}
    if not isinstance(spec, dict):
        return []
    for key in ("heldout_test_layouts", "test_layouts",
                "training_layouts", "layouts"):
        v = spec.get(key)
        if isinstance(v, list):
            return v
    return []


def ensure_correction_layouts_for_round_filled(
    run_id: int,
    round_idx: int,
    correction_n: int,
    train_yaml: Path,
    heldout_yaml: Path,
    run_shared_dir: Path,
    max_seed_bumps: int = 200,
) -> Path:
    """Rotated-only correction pool with seed-bump fill (used by the IIL
    baselines).

    Two-step contract:

    1. **REUSE** — if ``run_shared_dir/round_{round_idx:03d}/correction_layouts.yaml``
       already exists with at least ``correction_n`` layouts (e.g. the pool
       P4 wrote for this (run, round)), it is returned untouched. This makes
       the baselines evaluate on the *exact* layouts P4 saw and keeps the
       call idempotent across methods / reruns.
    2. **FILL** — otherwise sample a fresh pool. ``sample_layouts`` raises
       when it cannot produce ``correction_n`` unique unblocked layouts (the
       documented behaviour — there is no auto seed-bump upstream). We catch
       that, increment the sampler seed by +1, and retry until a full pool
       of ``correction_n`` layouts is obtained, then persist it. This is the
       "increase one seed at a time" requirement for rounds the baselines
       reach beyond P4's count (or any round whose base seed can't fill the
       pool against the growing blocked set).

    The blocked-signature set is identical to
    :func:`ensure_correction_layouts_for_round` (train + heldout + every
    earlier round of this run), so the intra-run disjointness invariant
    (``correction_pool_intra_run_overlap == []``) is preserved. The chosen
    ``seed_bump`` is recorded in ``layout_setup_report.json`` for audit.
    """
    round_dir = run_shared_dir / f"round_{int(round_idx):03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    out_path = round_dir / "correction_layouts.yaml"

    # Serialize concurrent callers for THIS round with an exclusive file lock
    # (flock is honoured on the Weka FS). When the 5 baselines run in parallel
    # and two reach the same un-generated round simultaneously, the first fills
    # the pool and the rest reuse it — no torn writes, no divergent pools. The
    # common case (pool pre-generated by the driver) just acquires the lock,
    # reads, and releases instantly.
    lockf = open(round_dir / ".pool.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX)

        # 1. REUSE a full pool already on disk (idempotent; shares P4's layouts).
        #    The report is (re)written only if missing — pure read otherwise.
        if out_path.exists():
            existing = _layouts_in_yaml(out_path)
            if len(existing) >= int(correction_n):
                rep_path = round_dir / "layout_setup_report.json"
                if not rep_path.exists():
                    report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
                    report["seed_bump"] = 0
                    report["reused_existing"] = True
                    _save_report(round_dir, run_id, report, round_idx=round_idx)
                return out_path
            # Short pool on disk — fall through and overwrite with a full one.

        # 2. FILL: block against train + heldout + every earlier round, then bump
        #    the seed until sample_layouts yields a full pool.
        prior = _prior_round_yamls(run_shared_dir, round_idx)
        blocked_inputs = [str(train_yaml), str(heldout_yaml)] + [str(p) for p in prior]
        blocked = _load_blocked_signatures(blocked_inputs)
        base_seed = CORRECTION_SEED_BASE + int(run_id) * 1000 + int(round_idx)
        for bump in range(int(max_seed_bumps) + 1):
            try:
                layouts = sample_layouts(
                    n=correction_n, grid_size=5, num_fires=3, min_manhattan=4,
                    seed=base_seed + bump, blocked_signatures=blocked,
                )
            except RuntimeError:
                continue  # couldn't fill the pool at this seed — bump and retry
            if len(layouts) >= int(correction_n):
                for i, L in enumerate(layouts):
                    L["name"] = f"corr_r{int(run_id):02d}_rnd{int(round_idx):03d}_{i + 1:03d}"
                write_yaml(layouts, out_path, grid_size=5)
                report = assert_no_contamination(out_path, train_yaml, heldout_yaml)
                report["seed_bump"] = bump
                report["reused_existing"] = False
                _save_report(round_dir, run_id, report, round_idx=round_idx)
                return out_path
        raise RuntimeError(
            f"seed-bump fill exhausted {max_seed_bumps} bumps for run {run_id} "
            f"round {round_idx} (could not sample {correction_n} unique layouts)"
        )
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


def pregenerate_round_pools(
    run_id: int,
    n_rounds: int,
    correction_n: int,
    train_yaml: Path,
    heldout_yaml: Path,
    run_shared_dir: Path,
) -> List[Path]:
    """Sequentially materialize round_001..round_{n_rounds} correction pools
    for a run (reusing P4's where present, seed-bump-filling the rest).

    Called ONCE by the driver BEFORE spawning the parallel method processes so
    every method only ever hits the read-only REUSE branch — no concurrent
    sampling/writing race, and all methods share byte-identical per-round
    pools. Each round blocks against train + heldout + every earlier round, so
    intra-run disjointness (the load-bearing contamination invariant) holds.
    """
    paths: List[Path] = []
    for r in range(1, int(n_rounds) + 1):
        try:
            paths.append(ensure_correction_layouts_for_round_filled(
                run_id=run_id, round_idx=r, correction_n=correction_n,
                train_yaml=train_yaml, heldout_yaml=heldout_yaml,
                run_shared_dir=run_shared_dir,
            ))
        except RuntimeError as exc:
            # Layout space exhausted at this round — stop pre-generating. The
            # methods will not reach it in practice (budget caps demos far
            # below this); if one ever does, the flock-guarded helper samples
            # it on demand. Don't crash the whole run over an unused round.
            print(f"[layout-setup] pre-gen stopped at round {r} for run "
                  f"{run_id}: {exc}", flush=True)
            break
    return paths


def _save_report(
    out_dir: Path,
    run_id: int,
    report: Dict[str, int],
    round_idx: Optional[int] = None,
) -> None:
    if round_idx is None:
        seed = CORRECTION_SEED_BASE + int(run_id)
    else:
        seed = CORRECTION_SEED_BASE + int(run_id) * 1000 + int(round_idx)
    out = {
        "run_id": int(run_id),
        "round_idx": int(round_idx) if round_idx is not None else None,
        "correction_seed": seed,
        "global_train_seed": GLOBAL_TRAIN_SEED,
        "global_heldout_seed": GLOBAL_HELDOUT_SEED,
        **report,
    }
    with open(out_dir / "layout_setup_report.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def run_dir_for(run_id: int) -> Path:
    return RESULTS_ROOT / f"run_{int(run_id)}"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run_id", type=int, required=True)
    p.add_argument("--initial_demos", type=int, default=20)
    p.add_argument("--heldout_n", type=int, default=200)
    p.add_argument("--correction_n", type=int, default=20)
    args = p.parse_args()

    shared = ensure_shared_layouts(args.initial_demos, args.heldout_n)
    out_dir = run_dir_for(args.run_id) / "shared"
    corr = ensure_correction_layouts_for_run(
        args.run_id, args.correction_n,
        train_yaml=shared["train"], heldout_yaml=shared["heldout"],
        out_dir=out_dir,
    )
    print(f"[layout-setup] OK run_id={args.run_id}")
    print(f"  train   -> {shared['train']}")
    print(f"  heldout -> {shared['heldout']}")
    print(f"  corr    -> {corr}")
