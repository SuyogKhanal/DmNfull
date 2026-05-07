"""LLM-guided dynamic-pool-expansion equivariant policy pipeline.

End-to-end flow (single invocation = full cycle)
================================================
The 50 held-out layouts in baseline_only/heldout_layouts.yaml are NEVER
written to by any demo collector. Zero feedback flows from them; they
are touched only by the heldout eval that produces the published SR
curve. The correction pool, on the other hand, is GROWING: every round
the LLM prescribes new layouts that get APPENDED to a local
`dynamic_pool.yaml`, so the next round rolls the policy out on a larger,
LLM-extended distribution.

Pre-requisite: baseline_only/ has been run with the new correction-pool
architecture so its correction_layouts.yaml + heldout_layouts.yaml +
training_layouts.yaml + demos/ exist.

Bootstrap
---------
1. Verify baseline_only/ artefacts.
2. Copy initial BFS demos (filtered by source) and the
   initial_*_eq_policy.pth snapshot (or train from scratch on the demos
   if the snapshot is absent).
3. Copy baseline_only/correction_layouts.yaml into
   p6_dynamic_pool/dynamic_pool.yaml. This file is the LIVE correction
   pool — every round's rollout reads from it, every round's LLM
   prescription appends to it.

Round 0
-------
Pure heldout eval on the shared starting model. No LLM, no pool change.

Round N (N >= 1)
----------------
  a. Rollout policy on the CURRENT dynamic_pool.yaml -> pool_sr.
  b. (Optional gating) If `stagnation_threshold` > 0 and pool_sr is
     below it, skip the LLM this round — the existing pool is still
     teaching the policy. The DAgger-style baseline implicit here is:
     no inline BFS correction in this method (we are testing whether
     LLM-only pool expansion can replace inline correction).
  c. Run the P6 LLM analyzer on the rollout. It produces a structured
     prescription of NEW layouts.
  d. APPEND those layouts to dynamic_pool.yaml so the next round will
     roll out on the expanded pool.
  e. BFS-collect demos for those layouts (these are the new training
     samples).
  f. Retrain on the cumulative demo set (random init when
     train_from_scratch is true).
  g. Heldout eval -> heldout_sr. This is the only number the public
     learning curve plots.
  h. Stop when heldout_sr >= target_sr.

Claim defended by this layout: "LLM-guided dynamic pool expansion
overcomes correction-pool saturation in DAgger without test-set leakage."
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PATHWAY_ROOT = REPO_ROOT / "Equivariant_pathway"
ROOT = PATHWAY_ROOT / "p6_dynamic_pool"
DEMO_DIR = ROOT / "demos"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
RAG_BANK_ROOT = ROOT / "rag_bank"
DYNAMIC_POOL_YAML = ROOT / "dynamic_pool.yaml"

BO_ROOT = PATHWAY_ROOT / "baseline_only"
BO_TRAIN_YAML = BO_ROOT / "training_layouts.yaml"
BO_HELDOUT_YAML = BO_ROOT / "heldout_layouts.yaml"
BO_CORRECTION_YAML = BO_ROOT / "correction_layouts.yaml"
BO_DEMOS = BO_ROOT / "demos"
BO_CKPT = BO_ROOT / "checkpoints"

TARGET_SR = 0.90
DEFAULT_ROUND_EPOCHS = 100
DEFAULT_STAGNATION_THRESHOLD = 0.0  # 0.0 = always fire LLM; raise to gate


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [pdp] {msg}", flush=True)


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
    """Return the layout list from a correction-pool YAML. write_yaml
    stores it under `heldout_test_layouts` (legacy key), but we accept
    any of the keys rollout_test.py recognises so a hand-edited YAML
    with `layouts:` works too."""
    with open(yaml_path, "r") as f:
        spec = yaml.safe_load(f) or {}
    for k in ("heldout_test_layouts", "test_layouts", "training_layouts", "layouts"):
        v = spec.get(k)
        if isinstance(v, list):
            return v
    return []


def _write_pool(yaml_path: Path, layouts: List[Dict]) -> None:
    """Write the dynamic pool back to disk under the same key
    rollout_test.py accepts. Keeps the file standalone (preserves the
    img_size / grid_size / cell_px metadata block when present)."""
    spec: Dict = {}
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            spec = yaml.safe_load(f) or {}
    spec["img_size"]              = spec.get("img_size", 80)
    spec["grid_size"]             = spec.get("grid_size", 5)
    spec["cell_px"]               = spec.get("cell_px", 16)
    spec["heldout_test_layouts"]  = layouts
    # Drop other layout keys so there's exactly one source of truth.
    for k in ("layouts", "test_layouts", "training_layouts"):
        spec.pop(k, None)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


def _layout_signature(layout: Dict) -> tuple:
    """Same signature scheme as Equivariant_pathway/layout_sampler.py."""
    sp = tuple(int(x) for x in (layout.get("start_pos") or []))
    gp = tuple(int(x) for x in (layout.get("goal_pos") or []))
    fires = tuple(sorted(
        tuple(int(x) for x in fp)
        for fp in (layout.get("fire_positions") or [])
    ))
    return (sp, gp, fires)


def _bootstrap(seed: int, initial_epochs: int, initial_demos: int) -> None:
    """Same bootstrap semantics as p6_only — copy initial BFS demos,
    copy the initial_*_eq_policy.pth snapshot if present, otherwise
    train p6_dynamic_pool's own initial model from scratch on the same
    20 demos. Then seed the dynamic_pool.yaml from baseline_only's
    correction_layouts.yaml so round 1 starts on the canonical pool."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAG_BANK_ROOT.mkdir(parents=True, exist_ok=True)

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

    # Seed the dynamic correction pool from baseline_only.
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


def _analyze(rollout_dir: Path, analysis_dir: Path, rag_bank: Path) -> None:
    from Equivariant_pathway.p6_dynamic_pool.analyze import run as run_p6
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rag_bank.mkdir(parents=True, exist_ok=True)
    run_p6(
        rollout_dir=str(rollout_dir),
        out_dir=str(analysis_dir),
        demo_dir=str(DEMO_DIR),
        rag_bank=str(rag_bank),
    )


def _flatten_recommended_layouts(analysis_dir: Path) -> Optional[Path]:
    rec_path = analysis_dir / "recommended_layouts.json"
    if rec_path.exists():
        return rec_path
    report_path = analysis_dir / "p6_dynamic_pool_prescription_report.json"
    if not report_path.exists():
        _info(f"no prescription report at {report_path}; skipping demo collection")
        return None
    report = json.load(open(report_path))
    flat: List[Dict] = []
    for pres in report.get("prescription", {}).get("demonstration_prescriptions", []) or []:
        cluster = pres.get("cluster", "?")
        for li, lay in enumerate(pres.get("recommended_layouts", []) or []):
            flat.append({
                "parent_demo_id": cluster,
                "layout_index":   li,
                "repetition":     1,
                "n_repetitions":  int(lay.get("n_repetitions", 1) or 1),
                "start_pos":      list(lay.get("start_pos", [])),
                "goal_pos":       list(lay.get("goal_pos", [])),
                "fire_positions": [list(p) for p in lay.get("fire_positions", []) or []],
                "rationale":      lay.get("rationale", ""),
            })
    with open(rec_path, "w") as f:
        json.dump({"layouts": flat, "n_layouts": len(flat)}, f, indent=2, default=str)
    _info(f"wrote {len(flat)} prescribed layouts -> {rec_path}")
    return rec_path


def _expand_pool(rec_path: Path, rnd: int) -> int:
    """Append prescribed layouts to dynamic_pool.yaml, deduplicated by
    signature so a layout the LLM accidentally re-proposes doesn't
    inflate next round's rollout count. Returns the number of layouts
    actually added (after dedup)."""
    with open(rec_path, "r") as f:
        rec = json.load(f) or {}
    new_layouts = rec.get("layouts", []) or []
    if not new_layouts:
        return 0

    pool = _load_pool(DYNAMIC_POOL_YAML)
    seen = {_layout_signature(L) for L in pool}
    added: List[Dict] = []
    for li, L in enumerate(new_layouts):
        sig = _layout_signature(L)
        if sig in seen:
            continue
        # Mark the source so a downstream consumer can tell hand-sampled
        # layouts apart from LLM-prescribed extensions.
        entry = {
            "name":           f"llm_round{rnd:03d}_idx{li:03d}",
            "description":    f"LLM-prescribed pool extension (round {rnd})",
            "grid":           [[0]*5 for _ in range(5)],
            "start_pos":      list(L.get("start_pos", [])),
            "goal_pos":       list(L.get("goal_pos", [])),
            "fire_positions": [list(p) for p in L.get("fire_positions", []) or []],
            "n_repetitions":  1,
            "source":         "llm_dynamic_pool_extension",
            "round":          rnd,
        }
        pool.append(entry)
        seen.add(sig)
        added.append(entry)
    if added:
        _write_pool(DYNAMIC_POOL_YAML, pool)
    _info(f"dynamic pool: appended {len(added)} new layout(s); "
          f"pool size now {len(pool)} (was {len(pool) - len(added)}).")
    return len(added)


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
    p = argparse.ArgumentParser(description="P6 dynamic-pool-expansion equivariant pipeline.")
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
    p.add_argument("--stagnation_threshold", type=float,
                   default=float(cfg.get("stagnation_threshold",
                                         DEFAULT_STAGNATION_THRESHOLD)),
                   help="Pool SR threshold above which the LLM fires. 0.0 (default) "
                        "= always fire. Set e.g. 0.85 to skip LLM analysis on rounds "
                        "where pool_sr is still below 85%% — the rationale being "
                        "that the existing pool still has things to teach.")
    p.add_argument("--force_restart", action="store_true")
    args = p.parse_args()
    TARGET_SR = args.target_sr

    _section("P6 DYNAMIC-POOL-EXPANSION EQUIVARIANT PIPELINE")
    _info(f"p6_dynamic_pool root: {ROOT}")
    _info(f"baseline_only root (read-only): {BO_ROOT}")
    _info(f"seed={args.seed}  round_epochs={args.round_epochs}  "
          f"max_rounds={args.max_rounds}  target_sr={args.target_sr}  "
          f"stagnation_threshold={args.stagnation_threshold}")

    _check_baseline_artifacts()

    if args.force_restart:
        _info("--force_restart: wiping p6_dynamic_pool/{demos,checkpoints,results,rag_bank,dynamic_pool.yaml}.")
        for d in (DEMO_DIR, CKPT_DIR, RESULTS_DIR, RAG_BANK_ROOT):
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
    _info(f"INITIAL: cum_demos={cum_demos}  pool_size={pool_size}  "
          f"heldout_sr={init_metrics['success_rate']:.3f} "
          f"({init_metrics['n_successes']}/{init_metrics['n_episodes']})")
    history: List[Dict] = [{
        "round":               0,
        "cum_demos":           cum_demos,
        "pool_size":           pool_size,
        "pool_sr":             None,
        "heldout_sr":          init_metrics["success_rate"],
        "heldout_n_successes": init_metrics["n_successes"],
        "heldout_n_episodes":  init_metrics["n_episodes"],
        "llm_fired":           False,
        "n_prescribed_layouts": 0,
        "n_pool_appended":     0,
        "n_new_demos":         0,
    }]
    _persist_curve(history)

    if init_metrics["success_rate"] >= TARGET_SR:
        _info(f"target {TARGET_SR} already reached at round 0; done.")
        _make_chart()
        return

    _section("PHASE 3: DYNAMIC-POOL LOOP", char="-")
    for rnd in range(1, args.max_rounds + 1):
        _section(f"ROUND {rnd}", char="-")
        round_dir = RESULTS_DIR / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 1. Rollout on the live dynamic pool.
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

        # 2. Stagnation gate: skip the LLM when pool_sr is below the
        # threshold (= the existing pool still has things to teach the
        # policy). Default threshold 0.0 -> fire every round.
        llm_fired = pool_sr >= args.stagnation_threshold
        n_prescribed = 0
        n_appended = 0
        n_new_demos = 0

        if llm_fired:
            # 3. P6 analysis with the dynamic-pool prompts.
            analysis_dir = round_dir / "p6_analysis"
            rag_bank = RAG_BANK_ROOT / f"round_{rnd:03d}"
            _analyze(corr_rollout_dir, analysis_dir, rag_bank)

            # 4. Flatten recommended_layouts.json.
            rec_path = _flatten_recommended_layouts(analysis_dir)
            if rec_path is not None:
                try:
                    n_prescribed = int(json.load(open(rec_path)).get("n_layouts", 0) or 0)
                except Exception:
                    n_prescribed = 0
                # 5. APPEND prescribed layouts to the dynamic pool so
                # next round rolls out on the expanded set.
                n_appended = _expand_pool(rec_path, rnd)
                # 6. BFS-collect demos for the prescribed layouts.
                round_demo_dir = DEMO_DIR / f"round_{rnd:03d}"
                n_new_demos = _collect_prescribed(
                    rec_path, round_demo_dir, seed=args.seed + 1000 + rnd,
                )
                # 7. Retrain on the cumulative demo set.
                if n_new_demos > 0:
                    _retrain(args.seed, args.round_epochs, args.train_from_scratch)
        else:
            _info(f"round {rnd}: pool_sr={pool_sr:.3f} < stagnation_threshold="
                  f"{args.stagnation_threshold:.3f}; skipping LLM expansion. "
                  f"The pool already has things left to teach; rerun with a "
                  f"lower threshold to force LLM-driven expansion every round.")

        # 8. Heldout eval (always — it's the only published curve point).
        post_dir = round_dir / "heldout_eval"
        _rollout(args.seed + rnd, BO_HELDOUT_YAML, post_dir, max_steps=args.max_steps)
        post_metrics = _read_sr(post_dir)
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
            "llm_fired":           llm_fired,
            "n_prescribed_layouts": n_prescribed,
            "n_pool_appended":     n_appended,
            "n_new_demos":         n_new_demos,
        })
        _persist_curve(history)
        _info(f"round {rnd}: cum_demos={cum_demos}  pool_size={pool_size}  "
              f"pool_sr={pool_sr:.3f}  heldout_sr={post_metrics['success_rate']:.3f}  "
              f"llm_fired={llm_fired}  appended={n_appended}  new_demos={n_new_demos}")

        if post_metrics["success_rate"] >= TARGET_SR:
            _info(f"target {TARGET_SR} reached at round {rnd}; stopping.")
            break
        if llm_fired and n_new_demos == 0:
            _info("LLM fired but produced 0 new demos; loop has stalled, stopping.")
            break

    _section("PHASE 4: CHART", char="-")
    _make_chart()
    _section("DONE")


def _persist_curve(history: List[Dict]) -> None:
    out = {
        "method":             "p6_dynamic_pool",
        "target_sr":          TARGET_SR,
        "demo_dir":           str(DEMO_DIR),
        "checkpoint_dir":     str(CKPT_DIR),
        "heldout_yaml":       str(BO_HELDOUT_YAML),
        "dynamic_pool_yaml":  str(DYNAMIC_POOL_YAML),
        "training_yaml":      str(BO_TRAIN_YAML),
        "rag_bank_root":      str(RAG_BANK_ROOT),
        "prompts_dir":        str(ROOT / "prompts"),
        "history":            history,
    }
    with open(RESULTS_DIR / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def _make_chart() -> None:
    rc = _run([sys.executable, "-u", "-m", "Equivariant_pathway.p6_dynamic_pool.chart"])
    if rc != 0:
        _info(f"chart generation exited rc={rc}")


if __name__ == "__main__":
    main()
