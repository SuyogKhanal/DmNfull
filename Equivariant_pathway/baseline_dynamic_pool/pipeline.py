"""Random-expansion baseline mirroring p6_dynamic_pool.

End-to-end flow
---------------
The 50 held-out layouts in baseline_only/heldout_layouts.yaml stay
read-only; nothing they expose ever ends up in the demo set. The only
heldout signal that flows back into the loop is the FAILURE COUNT,
which sizes the next round's random-expansion budget.

  1. Bootstrap: copy baseline_only's initial 20 BFS demos
     (filtered by source) and the immutable initial_*_eq_policy.pth
     snapshot (or train fresh on the 20 demos if the snapshot is
     missing). Seed dynamic_pool.yaml from
     baseline_only/correction_layouts.yaml.

  2. Round 0: heldout eval on the shared starting model. Record
     n_failures_prior = (heldout episodes - successes).

  3. Round N (N >= 1):
       a. Pool rollout for diagnostics (pool_sr).
       b. Sample n_failures_prior (capped by max_pool_expansion_per_round)
          NEW layouts uniformly at random with signatures blocked
          against heldout + training + the current dynamic pool.
       c. Append the sampled layouts to dynamic_pool.yaml.
       d. BFS-collect demos for those layouts.
       e. Retrain (random init when train_from_scratch).
       f. Heldout eval; update n_failures_prior for next round.
       g. Stop when heldout_sr >= target_sr or no random layouts can be
          sampled (pool space saturated).

Compared to p6_dynamic_pool, this baseline:
  - Uses NO LLM. Random sampling replaces the VLM/Reasoning/KAG/RAG/TKF
    stack and the cross-episode aggregator.
  - Spends a much LARGER expert-query budget (often 10-20+ layouts per
    round vs P6's 1-3) because it scales with heldout failure count.
  - Picks layouts uniformly at random within the same family, so it has
    no notion of "covering a corridor" or "exploiting TKF gaps".

The comparison is the headline ablation for the dynamic-pool claim.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PATHWAY_ROOT = REPO_ROOT / "Equivariant_pathway"
ROOT = PATHWAY_ROOT / "baseline_dynamic_pool"
DEMO_DIR = ROOT / "demos"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
DYNAMIC_POOL_YAML = ROOT / "dynamic_pool.yaml"

BO_ROOT = PATHWAY_ROOT / "baseline_only"
BO_TRAIN_YAML = BO_ROOT / "training_layouts.yaml"
BO_HELDOUT_YAML = BO_ROOT / "heldout_layouts.yaml"
BO_CORRECTION_YAML = BO_ROOT / "correction_layouts.yaml"
BO_DEMOS = BO_ROOT / "demos"
BO_CKPT = BO_ROOT / "checkpoints"

TARGET_SR = 0.90
DEFAULT_ROUND_EPOCHS = 100
DEFAULT_MAX_EXPANSION = 0  # 0 = no cap; budget = full heldout failure count


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [bdp] {msg}", flush=True)


def _section(title: str, char: str = "=") -> None:
    print(f"\n{char * 80}\n{title}\n{char * 80}", flush=True)


def _run(cmd: List[str]) -> int:
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _count_demos() -> int:
    if not DEMO_DIR.exists():
        return 0
    return sum(1 for _ in DEMO_DIR.rglob("*.json"))


def _check_baseline_artifacts() -> None:
    missing = []
    for path, label in [
        (BO_TRAIN_YAML,      "training_layouts.yaml"),
        (BO_HELDOUT_YAML,    "heldout_layouts.yaml"),
        (BO_CORRECTION_YAML, "correction_layouts.yaml"),
        (BO_DEMOS,           "demos/"),
    ]:
        if not path.exists():
            missing.append((label, str(path)))
    n_init_demos = sum(1 for _ in BO_DEMOS.glob("*.json")) if BO_DEMOS.exists() else 0
    if n_init_demos == 0:
        missing.append(("baseline_only top-level *.json demos", str(BO_DEMOS)))
    if missing:
        msg = ["baseline_only/ artefacts missing — run baseline_only first."]
        for label, p in missing:
            msg.append(f"  - {label}: {p}")
        raise SystemExit("\n".join(msg))


def _load_pool(yaml_path: Path) -> List[Dict]:
    with open(yaml_path, "r") as f:
        spec = yaml.safe_load(f) or {}
    for k in ("heldout_test_layouts", "test_layouts", "training_layouts", "layouts"):
        v = spec.get(k)
        if isinstance(v, list):
            return v
    return []


def _write_pool(yaml_path: Path, layouts: List[Dict]) -> None:
    spec: Dict = {}
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            spec = yaml.safe_load(f) or {}
    spec["img_size"]              = spec.get("img_size", 80)
    spec["grid_size"]             = spec.get("grid_size", 5)
    spec["cell_px"]               = spec.get("cell_px", 16)
    spec["heldout_test_layouts"]  = layouts
    for k in ("layouts", "test_layouts", "training_layouts"):
        spec.pop(k, None)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


def _layout_signature(layout: Dict) -> Tuple:
    sp = tuple(int(x) for x in (layout.get("start_pos") or []))
    gp = tuple(int(x) for x in (layout.get("goal_pos") or []))
    fires = tuple(sorted(
        tuple(int(x) for x in fp)
        for fp in (layout.get("fire_positions") or [])
    ))
    return (sp, gp, fires)


def _bootstrap(seed: int, initial_epochs: int, initial_demos: int) -> None:
    """Same as p6_dynamic_pool's bootstrap — copy initial BFS demos
    filtered by source, copy initial_*_eq_policy.pth snapshot if
    present (else train fresh on the 20 demos), seed dynamic pool from
    baseline_only's correction pool."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_skipped_dagger = 0
    n_skipped_unknown = 0
    for src in sorted(BO_DEMOS.glob("*.json")):
        try:
            with open(src, "r") as f:
                payload = json.load(f)
        except Exception as e:
            _info(f"demos: skipping unreadable {src.name} ({e!r})")
            n_skipped_unknown += 1
            continue
        source = (payload.get("source") or "").strip()
        if source == "baseline_dagger_correction":
            n_skipped_dagger += 1
            continue
        if source != "equivariant_pathway_bfs":
            _info(f"demos: skipping {src.name} (unrecognised source={source!r})")
            n_skipped_unknown += 1
            continue
        dst = DEMO_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            n_copied += 1
    n_now = sum(1 for _ in DEMO_DIR.glob("*.json"))
    _info(
        f"demos: copied {n_copied} initial BFS demos from {BO_DEMOS}; "
        f"skipped {n_skipped_dagger} baseline-DAgger corrections, "
        f"{n_skipped_unknown} unknown. Total now in {DEMO_DIR}: {n_now}."
    )
    if n_now == 0:
        raise SystemExit(
            f"After filtering, {DEMO_DIR} is empty. Re-run baseline_only/run.sh."
        )

    snap_best = BO_CKPT / "initial_best_eq_policy.pth"
    snap_last = BO_CKPT / "initial_last_eq_policy.pth"
    dst_best = CKPT_DIR / "best_eq_policy.pth"
    dst_last = CKPT_DIR / "last_eq_policy.pth"

    if snap_best.exists():
        shutil.copy2(snap_best, dst_best)
        _info(f"initial checkpoint: copied {snap_best.name} ({dst_best.stat().st_size} bytes)")
        if snap_last.exists():
            shutil.copy2(snap_last, dst_last)
        else:
            shutil.copy2(snap_best, dst_last)
    else:
        _info(
            f"initial checkpoint: snapshot not at {BO_CKPT}; training own initial "
            f"model on {n_now} demos for {initial_epochs} epochs (seed={seed})."
        )
        rc = _run([
            sys.executable, "-u", "-m", "Equivariant_pathway.train",
            "--demo_dir",       str(DEMO_DIR),
            "--checkpoint_dir", str(CKPT_DIR),
            "--epochs",         str(initial_epochs),
            "--seed",           str(seed),
            "--max_demos",      str(initial_demos),
        ])
        if rc != 0:
            raise RuntimeError(f"initial training failed (rc={rc})")
        if not dst_best.exists():
            raise RuntimeError(f"initial training rc=0 but {dst_best} missing")

    if not DYNAMIC_POOL_YAML.exists():
        shutil.copy2(BO_CORRECTION_YAML, DYNAMIC_POOL_YAML)
        n_pool = len(_load_pool(DYNAMIC_POOL_YAML))
        _info(f"dynamic pool: seeded from {BO_CORRECTION_YAML} ({n_pool} layouts).")
    else:
        n_pool = len(_load_pool(DYNAMIC_POOL_YAML))
        _info(f"dynamic pool: using existing {DYNAMIC_POOL_YAML} ({n_pool} layouts).")


def _rollout(seed: int, layouts_yaml: Path, rollout_dir: Path,
             max_steps: int = 60) -> None:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.rollout_test",
        "--checkpoint", str(CKPT_DIR / "best_eq_policy.pth"),
        "--layouts",    str(layouts_yaml),
        "--out_dir",    str(rollout_dir),
        "--seed",       str(seed),
        "--max_steps",  str(max_steps),
    ])
    if rc != 0:
        raise RuntimeError(f"rollout on {layouts_yaml.name} failed (rc={rc})")


def _read_sr(rollout_dir: Path) -> Dict:
    full = json.load(open(rollout_dir / "full_output.json"))
    md = full.get("metadata", {}) or {}
    n = int(md.get("n_episodes", 0) or 0)
    ns = int(md.get("n_successes", 0) or 0)
    return {
        "n_episodes": n,
        "n_successes": ns,
        "success_rate": ns / n if n else 0.0,
    }


def _gather_blocked_signatures() -> Set[Tuple]:
    """Block heldout + training + current dynamic pool signatures so a
    random sample never lands inside any of them."""
    from Equivariant_pathway.layout_sampler import _load_blocked_signatures
    blocked = _load_blocked_signatures([
        str(BO_HELDOUT_YAML),
        str(BO_TRAIN_YAML),
    ])
    if DYNAMIC_POOL_YAML.exists():
        for L in _load_pool(DYNAMIC_POOL_YAML):
            blocked.add(_layout_signature(L))
    return blocked


def _sample_random_layouts(n: int, seed: int) -> List[Dict]:
    """Sample n random 5x5 / 3-fire layouts disjoint from heldout +
    training + current pool. Returns a list of dicts shaped the same
    way layout_sampler.sample_layouts returns them."""
    if n <= 0:
        return []
    from Equivariant_pathway.layout_sampler import sample_layouts
    blocked = _gather_blocked_signatures()
    return sample_layouts(
        n=n, grid_size=5, num_fires=3,
        min_manhattan=4, seed=seed, blocked_signatures=blocked,
    )


def _expand_pool(new_layouts: List[Dict], rnd: int) -> int:
    if not new_layouts:
        return 0
    pool = _load_pool(DYNAMIC_POOL_YAML)
    seen = {_layout_signature(L) for L in pool}
    added: List[Dict] = []
    for li, L in enumerate(new_layouts):
        sig = _layout_signature(L)
        if sig in seen:
            continue
        entry = {
            "name":           f"random_round{rnd:03d}_idx{li:03d}",
            "description":    f"Random pool extension (round {rnd})",
            "grid":           [[0]*5 for _ in range(5)],
            "start_pos":      list(L.get("start_pos", [])),
            "goal_pos":       list(L.get("goal_pos", [])),
            "fire_positions": [list(p) for p in L.get("fire_positions", []) or []],
            "n_repetitions":  1,
            "source":         "random_dynamic_pool_extension",
            "round":          rnd,
        }
        pool.append(entry)
        seen.add(sig)
        added.append(entry)
    if added:
        _write_pool(DYNAMIC_POOL_YAML, pool)
    _info(f"dynamic pool: appended {len(added)} new random layout(s); "
          f"pool size now {len(pool)} (was {len(pool) - len(added)}).")
    return len(added)


def _write_recommended(layouts: List[Dict], out_path: Path) -> None:
    """Build a recommended_layouts.json that collect_demos.py
    --layouts_from accepts."""
    flat = []
    for li, L in enumerate(layouts):
        flat.append({
            "parent_demo_id": "random",
            "layout_index":   li,
            "repetition":     1,
            "n_repetitions":  1,
            "start_pos":      list(L.get("start_pos", [])),
            "goal_pos":       list(L.get("goal_pos", [])),
            "fire_positions": [list(p) for p in L.get("fire_positions", []) or []],
            "rationale":      "uniform random sample (signature-disjoint from heldout/train/pool)",
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"layouts": flat, "n_layouts": len(flat)}, f, indent=2, default=str)


def _collect_prescribed(rec_path: Path, round_demo_dir: Path, seed: int) -> int:
    pre = sum(1 for _ in round_demo_dir.rglob("*.json")) if round_demo_dir.exists() else 0
    round_demo_dir.mkdir(parents=True, exist_ok=True)
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.collect_demos",
        "--layouts_from", str(rec_path),
        "--demo_dir",     str(round_demo_dir),
        "--seed",         str(seed),
    ])
    if rc != 0:
        _info(f"collect_demos exited rc={rc}; round may add 0 demos")
    post = sum(1 for _ in round_demo_dir.rglob("*.json"))
    return post - pre


def _retrain(seed: int, epochs: int, train_from_scratch: bool) -> None:
    cmd = [
        sys.executable, "-u", "-m", "Equivariant_pathway.train",
        "--demo_dir",       str(DEMO_DIR),
        "--checkpoint_dir", str(CKPT_DIR),
        "--epochs",         str(epochs),
        "--seed",           str(seed),
    ]
    if not train_from_scratch:
        cmd.append("--resume")
    rc = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"retrain failed (rc={rc})")


CONFIG_PATH = ROOT / "config.yml"


def _load_config() -> Dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "on")


def main():
    global TARGET_SR
    cfg = _load_config()
    p = argparse.ArgumentParser(description="Random-expansion baseline (mirrors p6_dynamic_pool).")
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 0)))
    p.add_argument("--round_epochs", type=int,
                   default=int(cfg.get("round_epochs", DEFAULT_ROUND_EPOCHS)))
    p.add_argument("--initial_epochs", type=int,
                   default=int(cfg.get("initial_epochs", 500)))
    p.add_argument("--initial_demos", type=int,
                   default=int(cfg.get("initial_demos", 20)))
    p.add_argument("--max_rounds", type=int,
                   default=int(cfg.get("max_rounds", 50)))
    p.add_argument("--max_steps", type=int,
                   default=int(cfg.get("max_steps", 60)))
    p.add_argument("--target_sr", type=float,
                   default=float(cfg.get("target_sr", TARGET_SR)))
    p.add_argument("--train_from_scratch", type=_to_bool,
                   default=_to_bool(cfg.get("train_from_scratch", True)))
    p.add_argument("--max_pool_expansion_per_round", type=int,
                   default=int(cfg.get("max_pool_expansion_per_round",
                                       DEFAULT_MAX_EXPANSION)),
                   help="Cap on layouts added per round. 0 (default) = no cap, "
                        "use the full heldout failure count as the budget. Set "
                        "to e.g. 5 to artificially constrain to roughly p6's "
                        "prescription scale and isolate the LLM-vs-random axis.")
    p.add_argument("--force_restart", action="store_true")
    args = p.parse_args()
    TARGET_SR = args.target_sr

    _section("RANDOM-EXPANSION BASELINE (mirror of p6_dynamic_pool)")
    _info(f"baseline_dynamic_pool root: {ROOT}")
    _info(f"baseline_only root (read-only): {BO_ROOT}")
    _info(f"seed={args.seed}  round_epochs={args.round_epochs}  "
          f"max_rounds={args.max_rounds}  target_sr={args.target_sr}  "
          f"max_pool_expansion_per_round={args.max_pool_expansion_per_round}  "
          f"train_from_scratch={args.train_from_scratch}")

    _check_baseline_artifacts()

    if args.force_restart:
        _info("--force_restart: wiping baseline_dynamic_pool/{demos,checkpoints,results,dynamic_pool.yaml}.")
        for d in (DEMO_DIR, CKPT_DIR, RESULTS_DIR):
            if d.exists():
                shutil.rmtree(d)
        if DYNAMIC_POOL_YAML.exists():
            DYNAMIC_POOL_YAML.unlink()

    _section("PHASE 1: BOOTSTRAP", char="-")
    _bootstrap(args.seed, args.initial_epochs, args.initial_demos)

    _section("PHASE 2: ROUND 0 — HELDOUT EVAL ON SHARED STARTING MODEL", char="-")
    round0_eval_dir = RESULTS_DIR / "round_000_heldout_eval"
    _rollout(args.seed, BO_HELDOUT_YAML, round0_eval_dir, max_steps=args.max_steps)
    init_metrics = _read_sr(round0_eval_dir)
    cum_demos = _count_demos()
    pool_size = len(_load_pool(DYNAMIC_POOL_YAML))
    n_failures_prior = init_metrics["n_episodes"] - init_metrics["n_successes"]
    _info(f"INITIAL: cum_demos={cum_demos}  pool_size={pool_size}  "
          f"heldout_sr={init_metrics['success_rate']:.3f} "
          f"({init_metrics['n_successes']}/{init_metrics['n_episodes']})  "
          f"n_failures_prior={n_failures_prior}")
    history: List[Dict] = [{
        "round":               0,
        "cum_demos":           cum_demos,
        "pool_size":           pool_size,
        "pool_sr":             None,
        "heldout_sr":          init_metrics["success_rate"],
        "heldout_n_successes": init_metrics["n_successes"],
        "heldout_n_episodes":  init_metrics["n_episodes"],
        "expansion_budget":    None,
        "n_appended":          0,
        "n_new_demos":         0,
    }]
    _persist_curve(history)

    if init_metrics["success_rate"] >= TARGET_SR:
        _info(f"target {TARGET_SR} already reached at round 0; done.")
        _make_chart()
        return

    _section("PHASE 3: RANDOM-EXPANSION LOOP", char="-")
    for rnd in range(1, args.max_rounds + 1):
        _section(f"ROUND {rnd}", char="-")
        round_dir = RESULTS_DIR / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pool rollout for diagnostics only (does NOT drive sampling).
        corr_rollout_dir = round_dir / "pool_rollout"
        _rollout(
            args.seed + rnd * 1000,
            DYNAMIC_POOL_YAML,
            corr_rollout_dir,
            max_steps=args.max_steps,
        )
        corr_metrics = _read_sr(corr_rollout_dir)
        pool_sr = corr_metrics["success_rate"]
        _info(f"round {rnd} pool rollout sr={pool_sr:.3f} "
              f"({corr_metrics['n_successes']}/{corr_metrics['n_episodes']}) "
              f"on pool_size={len(_load_pool(DYNAMIC_POOL_YAML))}")

        # 2. Determine expansion budget = prior round's heldout failure
        # count (capped if user set a cap). The cap exists so a user can
        # constrain this baseline to roughly p6's prescription scale and
        # isolate the LLM-vs-random axis from the budget axis.
        budget = n_failures_prior
        if args.max_pool_expansion_per_round > 0:
            budget = min(budget, args.max_pool_expansion_per_round)
        _info(f"round {rnd} expansion budget: {budget} "
              f"(n_failures_prior={n_failures_prior}, "
              f"cap={args.max_pool_expansion_per_round or 'none'})")

        # 3. Sample budget random layouts, signature-disjoint from
        # heldout + training + current pool. If sampling can't find
        # enough fresh layouts (pool space exhausted), it returns
        # whatever it could find and we proceed with that.
        n_appended = 0
        n_new_demos = 0
        if budget > 0:
            try:
                sampled = _sample_random_layouts(budget, seed=args.seed + rnd * 1000)
            except RuntimeError as e:
                _info(f"random sampling exhausted available space: {e!r}; "
                      f"stopping with what we have.")
                sampled = []

            # 4. Append sampled layouts to the live dynamic pool YAML.
            n_appended = _expand_pool(sampled, rnd)

            if n_appended > 0:
                # 5. BFS-collect demos for those layouts.
                rec_path = round_dir / "recommended_layouts.json"
                _write_recommended(sampled, rec_path)
                round_demo_dir = DEMO_DIR / f"round_{rnd:03d}"
                n_new_demos = _collect_prescribed(
                    rec_path, round_demo_dir, seed=args.seed + 1000 + rnd,
                )

            # 6. Retrain if we actually got new demos.
            if n_new_demos > 0:
                _retrain(args.seed, args.round_epochs, args.train_from_scratch)
            else:
                _info(f"round {rnd}: no new demos (sampling produced 0 valid); "
                      f"skipping retrain.")
        else:
            _info(f"round {rnd}: budget=0 (no heldout failures last round); "
                  f"skipping expansion + retrain.")

        # 7. Heldout eval — drives next round's budget.
        post_dir = round_dir / "heldout_eval"
        _rollout(args.seed + rnd, BO_HELDOUT_YAML, post_dir, max_steps=args.max_steps)
        post_metrics = _read_sr(post_dir)
        n_failures_prior = post_metrics["n_episodes"] - post_metrics["n_successes"]
        cum_demos = _count_demos()
        pool_size = len(_load_pool(DYNAMIC_POOL_YAML))
        history.append({
            "round":               rnd,
            "cum_demos":           cum_demos,
            "pool_size":           pool_size,
            "pool_sr":             pool_sr,
            "heldout_sr":          post_metrics["success_rate"],
            "heldout_n_successes": post_metrics["n_successes"],
            "heldout_n_episodes":  post_metrics["n_episodes"],
            "expansion_budget":    budget,
            "n_appended":          n_appended,
            "n_new_demos":         n_new_demos,
        })
        _persist_curve(history)
        _info(f"round {rnd}: cum_demos={cum_demos}  pool_size={pool_size}  "
              f"pool_sr={pool_sr:.3f}  heldout_sr={post_metrics['success_rate']:.3f}  "
              f"budget={budget}  appended={n_appended}  new_demos={n_new_demos}  "
              f"n_failures_for_next_round={n_failures_prior}")

        if post_metrics["success_rate"] >= TARGET_SR:
            _info(f"target {TARGET_SR} reached at round {rnd}; stopping.")
            break
        if budget > 0 and n_appended == 0:
            _info("sampler returned 0 valid layouts (pool space saturated); stopping.")
            break

    _section("PHASE 4: CHART", char="-")
    _make_chart()
    _section("DONE")


def _persist_curve(history: List[Dict]) -> None:
    out = {
        "method":             "baseline_dynamic_pool",
        "target_sr":          TARGET_SR,
        "demo_dir":           str(DEMO_DIR),
        "checkpoint_dir":     str(CKPT_DIR),
        "heldout_yaml":       str(BO_HELDOUT_YAML),
        "dynamic_pool_yaml":  str(DYNAMIC_POOL_YAML),
        "training_yaml":      str(BO_TRAIN_YAML),
        "history":            history,
    }
    with open(RESULTS_DIR / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def _make_chart() -> None:
    rc = _run([sys.executable, "-u", "-m",
               "Equivariant_pathway.baseline_dynamic_pool.chart"])
    if rc != 0:
        _info(f"chart generation exited rc={rc}")


if __name__ == "__main__":
    main()
