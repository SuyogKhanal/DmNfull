"""Minimal P4-only equivariant policy pipeline.

End-to-end flow (single invocation = full cycle)
================================================
Pre-requisite: Equivariant_pathway/baseline_only/ has already been run
once, so its training_layouts.yaml, heldout_layouts.yaml, the initial
20 expert demos, and the 500-epoch initial checkpoint exist on disk.
This pipeline reuses those artefacts so P4 starts in lockstep with
baseline DAgger and any divergence is attributable to the LLM
prescriptions (not different starting weights).

  1. Verify baseline_only/ has the four required artefacts. Abort
     otherwise with a clear message.

  2. Bootstrap p4_only/:
       - Copy the initial 20 *.json demos from baseline_only/demos/
         (top-level only — round_NNN/ subfolders, which are baseline's
         corrective demos, are NOT copied).
       - Copy best_eq_policy.pth and last_eq_policy.pth from
         baseline_only/checkpoints/.
       - heldout_layouts.yaml is read directly from baseline_only/
         (read-only reference) so the held-out evaluation is on the
         exact same 50 layouts as baseline_only.

  3. Heldout eval on the shared starting model (round 0). Record
     (cum_demos, success_rate) — this is the same point baseline_only
     starts from, by construction.

  4. Until heldout success rate reaches 0.90:
       a. Rollout on the 50 heldout layouts (rollout_test.py) so the
          analyzer has a full_output.json + episodes/ tree.
       b. Run the P4 analyzer with the p4_only prompt addenda
          (Equivariant_pathway.p4_only.analyze) — the addenda
          strengthen the existing minimum-demos guidance with
          explicit holistic / sample-efficiency framing.
       c. Flatten the analyzer's per-cluster recommended_layouts into
          recommended_layouts.json (the format collect_demos.py
          --layouts_from accepts).
       d. BFS-collect those prescribed layouts into demos/round_NNN/.
       e. If at least one new demo was collected, retrain the policy
          on (initial 20 UNION every prescribed demo so far).
       f. Re-eval on heldout, record (round, cum_demos, success_rate).

  5. Persist learning_curve.json + a chart of success_rate vs
     cumulative demos under p4_only/results/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PATHWAY_ROOT = REPO_ROOT / "Equivariant_pathway"

# ROOT and BO_ROOT are overridable via P4_ONLY_ROOT and BASELINE_ONLY_ROOT
# so the pool-size sweep can run isolated copies in parallel. Without
# overrides, behaviour is unchanged.
_P4_ROOT_OVERRIDE = os.environ.get("P4_ONLY_ROOT")
_BO_ROOT_OVERRIDE = os.environ.get("BASELINE_ONLY_ROOT")
ROOT = Path(_P4_ROOT_OVERRIDE).resolve() if _P4_ROOT_OVERRIDE else PATHWAY_ROOT / "p4_only"
BO_ROOT = Path(_BO_ROOT_OVERRIDE).resolve() if _BO_ROOT_OVERRIDE else PATHWAY_ROOT / "baseline_only"

DEMO_DIR = ROOT / "demos"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

BO_TRAIN_YAML = BO_ROOT / "training_layouts.yaml"
BO_HELDOUT_YAML = BO_ROOT / "heldout_layouts.yaml"
BO_CORRECTION_YAML = BO_ROOT / "correction_layouts.yaml"
BO_DEMOS = BO_ROOT / "demos"
BO_CKPT = BO_ROOT / "checkpoints"

TARGET_SR = 0.90
DEFAULT_ROUND_EPOCHS = 100


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [p4o] {msg}", flush=True)


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
    """Fail fast if baseline_only/ is missing the artefacts we MUST share
    (heldout YAML, training YAML, top-level initial demos). The initial
    checkpoint is NOT required here — if no `initial_best_eq_policy.pth`
    snapshot exists, p4_only trains its own initial model on the same 20
    demos in _bootstrap()."""
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


def _bootstrap(seed: int, initial_epochs: int, initial_demos: int) -> None:
    """Bring p4_only/ to the SAME starting point baseline_only had at round 0:
    same initial demos, same initial model.

    Demos: the top-level *.json files in baseline_only/demos/ are copied
    verbatim. Round subfolders (baseline's corrective demos collected during
    its DAgger loop) are intentionally NOT copied — p4_only accumulates its
    own per-round prescribed demos under its own demos/.

    Initial model: ordered fallback —
      1. If baseline_only/checkpoints/initial_best_eq_policy.pth exists
         (modern baseline_only runs snapshot it BEFORE the DAgger loop),
         copy it. Same for initial_last_eq_policy.pth. This gives both
         methods byte-identical starting weights.
      2. Otherwise (e.g. an already-completed legacy baseline_only run
         where best_eq_policy.pth is now the converged post-DAgger model),
         we MUST NOT copy that file — it would give P4 a head start that
         has nothing to do with P4. Instead, train p4_only's own initial
         model from scratch on the same 20 demos that were just copied,
         using the same epochs / seed baseline_only used. The result is a
         trained-the-same-way starting point (statistically equivalent;
         not bit-identical because of CUDA non-determinism, but predicts
         very similarly on the heldout)."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # baseline_only/demos/ at top-level holds BOTH the initial BFS demos AND
    # every corrective demo baseline's DAgger loop appended. They are
    # distinguishable by the `source` field in the JSON payload:
    #   - "equivariant_pathway_bfs"     -> initial demos (collect_demos.py)
    #   - "baseline_dagger_correction"  -> DAgger corrections (baseline_dagger._save_corrective_demo)
    # Copying everything would mean p4_only starts with 20 + N demos and
    # would skew every comparison against baseline. So we filter by source
    # and copy ONLY the initial BFS demos. Files we cannot decode or that
    # lack a recognisable source are skipped with a warning rather than
    # silently let through.
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
            f"After filtering, p4_only/demos is empty. baseline_only/demos has no\n"
            f"files with source=='equivariant_pathway_bfs'. Re-run baseline_only/run.sh\n"
            f"so the initial 20 demos are regenerated."
        )

    snap_best = BO_CKPT / "initial_best_eq_policy.pth"
    snap_last = BO_CKPT / "initial_last_eq_policy.pth"
    dst_best = CKPT_DIR / "best_eq_policy.pth"
    dst_last = CKPT_DIR / "last_eq_policy.pth"

    if snap_best.exists():
        _info(f"initial checkpoint: snapshot found at {snap_best}; copying.")
        shutil.copy2(snap_best, dst_best)
        _info(f"  -> {dst_best.name} ({dst_best.stat().st_size} bytes)")
        if snap_last.exists():
            shutil.copy2(snap_last, dst_last)
            _info(f"  -> {dst_last.name} ({dst_last.stat().st_size} bytes)")
        else:
            shutil.copy2(snap_best, dst_last)
            _info(f"  -> {dst_last.name} (=initial_best fallback)")
        return

    _info(
        f"initial checkpoint: snapshot {snap_best.name} NOT found at {BO_CKPT}.\n"
        f"  baseline_only/checkpoints/best_eq_policy.pth would be the post-\n"
        f"  DAgger converged model — copying it would give P4 a head start.\n"
        f"  Falling back to training p4_only's own initial model on the\n"
        f"  {_count_demos()} initial demos for {initial_epochs} epochs (seed="
        f"{seed}). To make future runs byte-identical across methods, re-run\n"
        f"  baseline_only/run.sh — it now writes the initial_* snapshot."
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
        raise RuntimeError(f"p4_only initial training failed (rc={rc})")
    if not dst_best.exists():
        raise RuntimeError(
            f"p4_only initial training finished rc=0 but {dst_best} is missing."
        )
    _info(f"initial checkpoint trained at {dst_best} ({dst_best.stat().st_size} bytes)")


def _rollout(seed: int, layouts_yaml: Path, rollout_dir: Path,
             max_steps: int = 60) -> None:
    """Pure policy rollout on the given layouts. Writes full_output.json
    + per-episode trajectories. Caller decides whether the result feeds
    LLM analysis (correction pool) or just an SR readout (heldout)."""
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


def _analyze(rollout_dir: Path, analysis_dir: Path) -> None:
    from Equivariant_pathway.p4_only.analyze import run as run_p4
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_p4(
        rollout_dir=str(rollout_dir),
        out_dir=str(analysis_dir),
        demo_dir=str(DEMO_DIR),
    )


def _flatten_recommended_layouts(analysis_dir: Path) -> Optional[Path]:
    """Convert <analysis_dir>/p4_only_prescription_report.json's per-cluster
    recommended_layouts into a single layouts_from-compatible JSON."""
    rec_path = analysis_dir / "recommended_layouts.json"
    if rec_path.exists():
        return rec_path
    report_path = analysis_dir / "p4_only_prescription_report.json"
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


# config.yml ships next to this module in source — NOT under the
# overridden run ROOT. Resolve it from the code dir so sweep runs
# inherit the same defaults as a normal run.
CONFIG_PATH = Path(__file__).resolve().parent / "config.yml"


def _load_config() -> Dict:
    if not CONFIG_PATH.exists():
        return {}
    import yaml as _yaml
    with open(CONFIG_PATH, "r") as f:
        return _yaml.safe_load(f) or {}


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "on")


def main():
    global TARGET_SR  # must precede the read in p.add_argument(default=...)
    cfg = _load_config()
    p = argparse.ArgumentParser(description="P4-only equivariant pipeline.")
    p.add_argument("--seed", type=int,
                   default=int(cfg.get("seed", 0)))
    p.add_argument("--round_epochs", type=int,
                   default=int(cfg.get("round_epochs", DEFAULT_ROUND_EPOCHS)))
    p.add_argument("--initial_epochs", type=int,
                   default=int(cfg.get("initial_epochs", 500)),
                   help="Epochs used ONLY when baseline_only's initial_*_eq_policy.pth "
                        "snapshot is missing and p4_only must train its own initial "
                        "model from scratch. Match baseline_only's --initial_epochs.")
    p.add_argument("--initial_demos", type=int,
                   default=int(cfg.get("initial_demos", 20)),
                   help="Number of initial demos cap for the fallback initial training "
                        "(matches baseline_only's --initial_demos).")
    p.add_argument("--max_rounds", type=int,
                   default=int(cfg.get("max_rounds", 50)),
                   help="Hard guard against infinite loops; the loop normally exits "
                        "when heldout success rate >= target_sr.")
    p.add_argument("--max_steps", type=int,
                   default=int(cfg.get("max_steps", 60)),
                   help="Per-episode step cap during the rollout the LLM analyses.")
    p.add_argument("--target_sr", type=float,
                   default=float(cfg.get("target_sr", TARGET_SR)),
                   help="Stop the loop when heldout success rate >= this.")
    p.add_argument("--train_from_scratch", type=_to_bool,
                   default=_to_bool(cfg.get("train_from_scratch", True)),
                   help="true = retrain from RANDOM INIT every round on the cumulative "
                        "demo set (no warm start). Match baseline_only's setting for "
                        "an apples-to-apples comparison.")
    p.add_argument("--force_restart", action="store_true",
                   help="Wipe p4_only/{demos,checkpoints,results} before running. "
                        "baseline_only/ artefacts are NEVER touched.")
    args = p.parse_args()
    TARGET_SR = args.target_sr

    _section("P4-ONLY EQUIVARIANT PIPELINE")
    _info(f"p4_only root: {ROOT}")
    _info(f"baseline_only root (read-only source): {BO_ROOT}")
    _info(f"seed={args.seed}  round_epochs={args.round_epochs}  "
          f"max_rounds={args.max_rounds}")

    _check_baseline_artifacts()

    if args.force_restart:
        _info("--force_restart: wiping p4_only/{demos,checkpoints,results}.")
        for d in (DEMO_DIR, CKPT_DIR, RESULTS_DIR):
            if d.exists():
                shutil.rmtree(d)

    _section("PHASE 1: BOOTSTRAP FROM baseline_only/", char="-")
    _bootstrap(args.seed, args.initial_epochs, args.initial_demos)

    _section("PHASE 2: ROUND 0 — HELDOUT EVAL ON SHARED STARTING MODEL", char="-")
    # Round 0 is a PURE eval on the heldout pool. No analysis, no demo
    # collection. This is the same starting point baseline_only logs.
    round0_eval_dir = RESULTS_DIR / "round_000_heldout_eval"
    _rollout(args.seed, BO_HELDOUT_YAML, round0_eval_dir, max_steps=args.max_steps)
    init_metrics = _read_sr(round0_eval_dir)
    cum_demos = _count_demos()
    _info(f"INITIAL: cum_demos={cum_demos}  "
          f"heldout_sr={init_metrics['success_rate']:.3f} "
          f"({init_metrics['n_successes']}/{init_metrics['n_episodes']})")
    history: List[Dict] = [{
        "round":               0,
        "cum_demos":           cum_demos,
        "heldout_sr":          init_metrics["success_rate"],
        "heldout_n_successes": init_metrics["n_successes"],
        "heldout_n_episodes":  init_metrics["n_episodes"],
        "correction_sr":       None,
        "n_prescribed_layouts": 0,
        "n_new_demos":         0,
    }]
    _persist_curve(history)

    if init_metrics["success_rate"] >= TARGET_SR:
        _info(f"target {TARGET_SR} already reached at round 0; done.")
        _make_chart()
        return

    _section("PHASE 3: P4 LOOP (correction = LLM analysis pool, eval = heldout)", char="-")
    for rnd in range(1, args.max_rounds + 1):
        _section(f"ROUND {rnd}", char="-")
        round_dir = RESULTS_DIR / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Step 1. Rollout on the CORRECTION pool. This is the LLM's input
        # — the failures it analyses come from this set, never from the
        # heldout set. That's how we keep the heldout SR curve clean.
        corr_rollout_dir = round_dir / "correction_rollout"
        _rollout(
            args.seed + rnd * 1000,
            BO_CORRECTION_YAML,
            corr_rollout_dir,
            max_steps=args.max_steps,
        )
        corr_metrics = _read_sr(corr_rollout_dir)
        _info(f"round {rnd} correction-pool rollout sr={corr_metrics['success_rate']:.3f} "
              f"({corr_metrics['n_successes']}/{corr_metrics['n_episodes']})")

        # Step 2. P4 LLM analysis over those correction-pool failures.
        analysis_dir = round_dir / "p4_analysis"
        _analyze(corr_rollout_dir, analysis_dir)

        # Step 3. Flatten the prescription into recommended_layouts.json
        # and BFS-collect the prescribed layouts. Note: the prescribed
        # layouts are NEW synthesised layouts proposed by the LLM, not
        # the failed correction-pool layouts themselves. Either way, no
        # heldout layout ever ends up in the demo set.
        rec_path = _flatten_recommended_layouts(analysis_dir)
        n_new_demos = 0
        n_prescribed = 0
        if rec_path is not None:
            try:
                n_prescribed = int(json.load(open(rec_path)).get("n_layouts", 0) or 0)
            except Exception:
                n_prescribed = 0
            round_demo_dir = DEMO_DIR / f"round_{rnd:03d}"
            n_new_demos = _collect_prescribed(
                rec_path, round_demo_dir, seed=args.seed + 1000 + rnd,
            )
        _info(f"round {rnd}: prescribed_layouts={n_prescribed}  new_demos={n_new_demos}")

        # Step 4. Retrain on (initial demos UNION every prescribed demo so far).
        if n_new_demos > 0:
            _retrain(args.seed, args.round_epochs, args.train_from_scratch)
        else:
            _info(f"round {rnd}: no new demos; skipping retrain.")

        # Step 5. Heldout eval on a separate, never-trained-on layout set.
        # This is the SR curve we plot.
        post_dir = round_dir / "heldout_eval"
        _rollout(args.seed + rnd, BO_HELDOUT_YAML, post_dir, max_steps=args.max_steps)
        post_metrics = _read_sr(post_dir)
        cum_demos = _count_demos()
        history.append({
            "round":               rnd,
            "cum_demos":           cum_demos,
            "heldout_sr":          post_metrics["success_rate"],
            "heldout_n_successes": post_metrics["n_successes"],
            "heldout_n_episodes":  post_metrics["n_episodes"],
            "correction_sr":       corr_metrics["success_rate"],
            "n_prescribed_layouts": n_prescribed,
            "n_new_demos":         n_new_demos,
        })
        _persist_curve(history)
        _info(f"round {rnd}: cum_demos={cum_demos}  "
              f"heldout_sr={post_metrics['success_rate']:.3f}")

        if post_metrics["success_rate"] >= TARGET_SR:
            _info(f"target {TARGET_SR} reached at round {rnd}; stopping.")
            break
        if n_new_demos == 0:
            _info("no new demos this round; loop has stalled, stopping.")
            break

    _section("PHASE 4: CHART", char="-")
    _make_chart()
    _section("DONE")


def _persist_curve(history: List[Dict]) -> None:
    out = {
        "method":          "p4_only",
        "target_sr":       TARGET_SR,
        "demo_dir":        str(DEMO_DIR),
        "checkpoint_dir":  str(CKPT_DIR),
        "heldout_yaml":    str(BO_HELDOUT_YAML),
        "correction_yaml": str(BO_CORRECTION_YAML),
        "training_yaml":   str(BO_TRAIN_YAML),
        "history":         history,
    }
    with open(RESULTS_DIR / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def _make_chart() -> None:
    rc = _run([sys.executable, "-u", "-m", "Equivariant_pathway.p4_only.chart"])
    if rc != 0:
        _info(f"chart generation exited rc={rc}")


if __name__ == "__main__":
    main()
