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
ROOT = PATHWAY_ROOT / "p4_only"
DEMO_DIR = ROOT / "demos"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

BO_ROOT = PATHWAY_ROOT / "baseline_only"
BO_TRAIN_YAML = BO_ROOT / "training_layouts.yaml"
BO_HELDOUT_YAML = BO_ROOT / "heldout_layouts.yaml"
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
    """Fail fast with a clear message if baseline_only/ is missing pieces."""
    missing = []
    for path, label in [
        (BO_TRAIN_YAML,                        "training_layouts.yaml"),
        (BO_HELDOUT_YAML,                      "heldout_layouts.yaml"),
        (BO_DEMOS,                             "demos/"),
        (BO_CKPT / "best_eq_policy.pth",       "checkpoints/best_eq_policy.pth"),
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


def _bootstrap() -> None:
    """Copy initial 20 demos + initial checkpoints from baseline_only/.
    Round subfolders (baseline's corrective demos) are NOT copied — p4_only
    accumulates its own per-round prescribed demos under its own demos/."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for src in sorted(BO_DEMOS.glob("*.json")):
        dst = DEMO_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            n_copied += 1
    n_now = sum(1 for _ in DEMO_DIR.glob("*.json"))
    _info(f"demos: copied {n_copied} initial demos from {BO_DEMOS} "
          f"(top-level total now: {n_now})")

    for name in ("best_eq_policy.pth", "last_eq_policy.pth"):
        src = BO_CKPT / name
        if src.exists():
            shutil.copy2(src, CKPT_DIR / name)
            _info(f"checkpoint: copied {name} ({src.stat().st_size} bytes)")
        elif name == "best_eq_policy.pth":
            raise SystemExit(f"required {src} not found")


def _rollout_for_analysis(seed: int, rollout_dir: Path) -> None:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.rollout_test",
        "--checkpoint", str(CKPT_DIR / "best_eq_policy.pth"),
        "--layouts",    str(BO_HELDOUT_YAML),
        "--out_dir",    str(rollout_dir),
        "--seed",       str(seed),
    ])
    if rc != 0:
        raise RuntimeError(f"rollout for analysis failed (rc={rc})")


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


def _retrain(seed: int, epochs: int) -> None:
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.train",
        "--demo_dir",       str(DEMO_DIR),
        "--checkpoint_dir", str(CKPT_DIR),
        "--epochs",         str(epochs),
        "--seed",           str(seed),
        "--resume",
    ])
    if rc != 0:
        raise RuntimeError(f"retrain failed (rc={rc})")


def main():
    p = argparse.ArgumentParser(description="P4-only equivariant pipeline.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--round_epochs", type=int, default=DEFAULT_ROUND_EPOCHS)
    p.add_argument("--max_rounds", type=int, default=50,
                   help="Hard guard against infinite loops; the loop normally exits "
                        "when heldout success rate >= 0.90.")
    p.add_argument("--force_restart", action="store_true",
                   help="Wipe p4_only/{demos,checkpoints,results} before running. "
                        "baseline_only/ artefacts are NEVER touched.")
    args = p.parse_args()

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
    _bootstrap()

    _section("PHASE 2: ROUND 0 — HELDOUT EVAL ON SHARED STARTING MODEL", char="-")
    round0_dir = RESULTS_DIR / "round_000_rollout"
    _rollout_for_analysis(args.seed, round0_dir)
    init_metrics = _read_sr(round0_dir)
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
        "n_prescribed_layouts": 0,
        "n_new_demos":         0,
    }]
    _persist_curve(history)

    if init_metrics["success_rate"] >= TARGET_SR:
        _info(f"target {TARGET_SR} already reached at round 0; done.")
        _make_chart()
        return

    _section("PHASE 3: P4 LOOP", char="-")
    for rnd in range(1, args.max_rounds + 1):
        _section(f"ROUND {rnd}", char="-")
        round_dir = RESULTS_DIR / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Round 1 reuses round 0's rollout (identical model, same seed
        # would yield the same trajectory anyway). Subsequent rounds
        # produce a fresh rollout against the just-retrained model.
        if rnd == 1:
            rollout_dir = round0_dir
        else:
            rollout_dir = round_dir / "rollout"
            _rollout_for_analysis(args.seed + rnd * 1000, rollout_dir)
            pre_metrics = _read_sr(rollout_dir)
            _info(f"round {rnd} rollout sr={pre_metrics['success_rate']:.3f}")

        analysis_dir = round_dir / "p4_analysis"
        _analyze(rollout_dir, analysis_dir)

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

        if n_new_demos > 0:
            _retrain(args.seed, args.round_epochs)
        else:
            _info(f"round {rnd}: no new demos; skipping retrain.")

        post_dir = round_dir / "heldout_eval"
        _rollout_for_analysis(args.seed + rnd, post_dir)
        post_metrics = _read_sr(post_dir)
        cum_demos = _count_demos()
        history.append({
            "round":               rnd,
            "cum_demos":           cum_demos,
            "heldout_sr":          post_metrics["success_rate"],
            "heldout_n_successes": post_metrics["n_successes"],
            "heldout_n_episodes":  post_metrics["n_episodes"],
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
        "method":         "p4_only",
        "target_sr":      TARGET_SR,
        "demo_dir":       str(DEMO_DIR),
        "checkpoint_dir": str(CKPT_DIR),
        "heldout_yaml":   str(BO_HELDOUT_YAML),
        "training_yaml":  str(BO_TRAIN_YAML),
        "history":        history,
    }
    with open(RESULTS_DIR / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def _make_chart() -> None:
    rc = _run([sys.executable, "-u", "-m", "Equivariant_pathway.p4_only.chart"])
    if rc != 0:
        _info(f"chart generation exited rc={rc}")


if __name__ == "__main__":
    main()
