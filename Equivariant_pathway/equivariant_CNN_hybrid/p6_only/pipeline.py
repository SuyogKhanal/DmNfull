"""Hybrid p6_only pipeline (EquivariantCNNHybridPolicy + P6 LLM = P5 + TKF).

Mirrors equivariant_CNN_hybrid/p5_only/pipeline.py, with:

  1. Different LABEL / report filename (p6_only_hybrid_*).
  2. ``analyze`` is the P6 wrapper, which on top of the per-round
     RAG bank also passes ``tkf.demo_dir = DEMO_DIR``. DEMO_DIR
     contains ONLY BFS demos (initial 20 + per-round LLM-prescribed)
     because the bootstrap source-filter rejects baseline_dagger
     corrections; never any other method's demos; never any other
     pool size's demos. So TKF reads from a completely isolated
     in-method demo set.
  3. RAG bank also lives at ``ROOT / "rag_bank"`` — sibling of P5's
     bank (different ROOT) so they cannot contaminate each other.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HYBRID_ROOT = REPO_ROOT / "Equivariant_pathway" / "equivariant_CNN_hybrid"

_P6_OVERRIDE = os.environ.get("P6_ONLY_ROOT")
_BO_OVERRIDE = os.environ.get("BASELINE_ONLY_ROOT")
ROOT = Path(_P6_OVERRIDE).resolve() if _P6_OVERRIDE else HYBRID_ROOT / "p6_only"
BO_ROOT = Path(_BO_OVERRIDE).resolve() if _BO_OVERRIDE else HYBRID_ROOT / "baseline_only"

DEMO_DIR = ROOT / "demos"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
RAG_BANK_ROOT = ROOT / "rag_bank"

BO_TRAIN_YAML = BO_ROOT / "training_layouts.yaml"
BO_HELDOUT_YAML = BO_ROOT / "heldout_layouts.yaml"
BO_CORRECTION_YAML = BO_ROOT / "correction_layouts.yaml"
BO_DEMOS = BO_ROOT / "demos"
BO_CKPT = BO_ROOT / "checkpoints"

TARGET_SR = 0.90
DEFAULT_ROUND_EPOCHS = 100


def _info(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [p6-hybrid] {msg}", flush=True)


def _section(title, char="="):
    print(f"\n{char*80}\n{title}\n{char*80}", flush=True)


def _run(cmd):
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _count_demos():
    return sum(1 for _ in DEMO_DIR.rglob("*.json")) if DEMO_DIR.exists() else 0


def _check_baseline_artifacts():
    missing = []
    for path, label in [
        (BO_TRAIN_YAML, "training_layouts.yaml"),
        (BO_HELDOUT_YAML, "heldout_layouts.yaml"),
        (BO_CORRECTION_YAML, "correction_layouts.yaml"),
        (BO_DEMOS, "demos/"),
    ]:
        if not path.exists():
            missing.append((label, str(path)))
    n_init = sum(1 for _ in BO_DEMOS.glob("*.json")) if BO_DEMOS.exists() else 0
    if n_init == 0:
        missing.append(("baseline_only top-level *.json demos", str(BO_DEMOS)))
    if missing:
        msg = ["baseline_only/ artefacts missing — run baseline_only first."]
        for label, p in missing:
            msg.append(f"  - {label}: {p}")
        raise SystemExit("\n".join(msg))


def _bootstrap(seed, initial_epochs, initial_demos):
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
        if source == "hybrid_baseline_dagger_correction":
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

    init_best = BO_CKPT / "initial_best_hybrid_policy.pth"
    init_last = BO_CKPT / "initial_last_hybrid_policy.pth"
    dst_best = CKPT_DIR / "best_hybrid_policy.pth"
    dst_last = CKPT_DIR / "last_hybrid_policy.pth"
    if init_best.exists():
        shutil.copy2(init_best, dst_best)
        if init_last.exists():
            shutil.copy2(init_last, dst_last)
        _info(f"copied initial best -> {dst_best}")
    else:
        rc = _run([
            sys.executable, "-u", "-m", "Equivariant_pathway.equivariant_CNN_hybrid.train",
            "--demo_dir", str(DEMO_DIR),
            "--checkpoint_dir", str(CKPT_DIR),
            "--epochs", str(initial_epochs),
            "--seed", str(seed),
            "--max_demos", str(initial_demos),
        ])
        if rc != 0:
            raise RuntimeError(f"hybrid initial training failed (rc={rc})")


def _rollout(seed, layouts_yaml, rollout_dir, max_steps=60):
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.equivariant_CNN_hybrid.rollout_test",
        "--checkpoint", str(CKPT_DIR / "best_hybrid_policy.pth"),
        "--layouts", str(layouts_yaml),
        "--out_dir", str(rollout_dir),
        "--seed", str(seed),
        "--max_steps", str(max_steps),
    ])
    if rc != 0:
        raise RuntimeError(f"rollout failed (rc={rc})")


def _read_sr(rollout_dir):
    full = json.load(open(rollout_dir / "full_output.json"))
    md = full.get("metadata", {}) or {}
    n = int(md.get("n_episodes", 0) or 0)
    ns = int(md.get("n_successes", 0) or 0)
    return {"n_episodes": n, "n_successes": ns,
            "success_rate": ns / n if n else 0.0}


def _analyze(rollout_dir, analysis_dir, rag_bank: Path):
    from Equivariant_pathway.equivariant_CNN_hybrid.p6_only.analyze import run as run_p6
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rag_bank.mkdir(parents=True, exist_ok=True)
    # IMPORTANT: demo_dir is THIS METHOD's DEMO_DIR, not baseline's,
    # not P5's, not another pool size's. Source-filter at bootstrap
    # ensures it only contains BFS demos.
    run_p6(
        rollout_dir=str(rollout_dir),
        out_dir=str(analysis_dir),
        demo_dir=str(DEMO_DIR),
        rag_bank=str(rag_bank),
    )


def _flatten_recommended_layouts(analysis_dir):
    rec_path = analysis_dir / "recommended_layouts.json"
    if rec_path.exists():
        return rec_path
    report_path = analysis_dir / "p6_only_hybrid_prescription_report.json"
    if not report_path.exists():
        return None
    report = json.load(open(report_path))
    flat = []
    for pres in report.get("prescription", {}).get("demonstration_prescriptions", []) or []:
        cluster = pres.get("cluster", "?")
        for li, lay in enumerate(pres.get("recommended_layouts", []) or []):
            flat.append({
                "parent_demo_id": cluster, "layout_index": li,
                "repetition": 1,
                "n_repetitions": int(lay.get("n_repetitions", 1) or 1),
                "start_pos": list(lay.get("start_pos", [])),
                "goal_pos":  list(lay.get("goal_pos", [])),
                "fire_positions": [list(p) for p in lay.get("fire_positions", []) or []],
                "rationale": lay.get("rationale", ""),
            })
    with open(rec_path, "w") as f:
        json.dump({"layouts": flat, "n_layouts": len(flat)}, f, indent=2, default=str)
    return rec_path


def _collect_prescribed(rec_path, round_demo_dir, seed):
    pre = sum(1 for _ in round_demo_dir.rglob("*.json")) if round_demo_dir.exists() else 0
    round_demo_dir.mkdir(parents=True, exist_ok=True)
    rc = _run([
        sys.executable, "-u", "-m", "Equivariant_pathway.collect_demos",
        "--layouts_from", str(rec_path),
        "--demo_dir", str(round_demo_dir),
        "--seed", str(seed),
    ])
    if rc != 0:
        _info(f"collect_demos rc={rc}")
    post = sum(1 for _ in round_demo_dir.rglob("*.json"))
    return post - pre


def _retrain(seed, epochs, train_from_scratch):
    cmd = [
        sys.executable, "-u", "-m", "Equivariant_pathway.equivariant_CNN_hybrid.train",
        "--demo_dir", str(DEMO_DIR),
        "--checkpoint_dir", str(CKPT_DIR),
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    if not train_from_scratch:
        cmd.append("--resume")
    rc = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"retrain failed (rc={rc})")


CONFIG_PATH = Path(__file__).resolve().parent / "config.yml"


def _load_config():
    if not CONFIG_PATH.exists():
        return {}
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def _to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "on")


def main():
    global TARGET_SR
    cfg = _load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 0)))
    p.add_argument("--round_epochs", type=int, default=int(cfg.get("round_epochs", DEFAULT_ROUND_EPOCHS)))
    p.add_argument("--initial_epochs", type=int, default=int(cfg.get("initial_epochs", 500)))
    p.add_argument("--initial_demos", type=int, default=int(cfg.get("initial_demos", 20)))
    p.add_argument("--max_rounds", type=int, default=int(cfg.get("max_rounds", 50)))
    p.add_argument("--max_steps", type=int, default=int(cfg.get("max_steps", 60)))
    p.add_argument("--target_sr", type=float, default=float(cfg.get("target_sr", TARGET_SR)))
    p.add_argument("--train_from_scratch", type=_to_bool, default=_to_bool(cfg.get("train_from_scratch", True)))
    p.add_argument("--force_restart", action="store_true")
    args = p.parse_args()
    TARGET_SR = args.target_sr

    _section("HYBRID P6-ONLY PIPELINE")
    _info(f"p6_only root: {ROOT}  baseline_only ref: {BO_ROOT}")
    _info(f"rag_bank root: {RAG_BANK_ROOT}")
    _info(f"tkf demo_dir : {DEMO_DIR}  (BFS only — corrections filtered out)")
    _check_baseline_artifacts()

    if args.force_restart:
        for d in (DEMO_DIR, CKPT_DIR, RESULTS_DIR, RAG_BANK_ROOT):
            if d.exists():
                shutil.rmtree(d)

    _section("PHASE 1: BOOTSTRAP", char="-")
    _bootstrap(args.seed, args.initial_epochs, args.initial_demos)

    _section("PHASE 2: ROUND 0 — HELDOUT EVAL", char="-")
    round0 = RESULTS_DIR / "round_000_heldout_eval"
    _rollout(args.seed, BO_HELDOUT_YAML, round0, max_steps=args.max_steps)
    init_metrics = _read_sr(round0)
    cum_demos = _count_demos()
    history: List[Dict] = [{
        "round": 0, "cum_demos": cum_demos,
        "heldout_sr": init_metrics["success_rate"],
        "heldout_n_successes": init_metrics["n_successes"],
        "heldout_n_episodes": init_metrics["n_episodes"],
        "correction_sr": None,
        "n_prescribed_layouts": 0, "n_new_demos": 0,
    }]
    _persist_curve(history)
    if init_metrics["success_rate"] >= TARGET_SR:
        _make_chart(); return

    _section("PHASE 3: P6 LOOP", char="-")
    for rnd in range(1, args.max_rounds + 1):
        _section(f"ROUND {rnd}", char="-")
        round_dir = RESULTS_DIR / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        corr_dir = round_dir / "correction_rollout"
        _rollout(args.seed + rnd * 1000, BO_CORRECTION_YAML, corr_dir,
                 max_steps=args.max_steps)
        corr_metrics = _read_sr(corr_dir)
        analysis_dir = round_dir / "p6_analysis"
        # Persistent RAG bank across rounds within this profile. See the
        # matching comment in p5_only/pipeline.py for the rationale —
        # a per-round subdir means round N queries an empty bank because
        # round (N-1)'s writes never reach it.
        rag_bank = RAG_BANK_ROOT
        _analyze(corr_dir, analysis_dir, rag_bank)

        rec_path = _flatten_recommended_layouts(analysis_dir)
        n_new_demos = 0
        n_prescribed = 0
        if rec_path is not None:
            try:
                n_prescribed = int(json.load(open(rec_path)).get("n_layouts", 0) or 0)
            except Exception:
                n_prescribed = 0
            round_demo_dir = DEMO_DIR / f"round_{rnd:03d}"
            n_new_demos = _collect_prescribed(rec_path, round_demo_dir,
                                              seed=args.seed + 1000 + rnd)
        _info(f"round {rnd}: prescribed={n_prescribed} new_demos={n_new_demos}")
        if n_new_demos > 0:
            _retrain(args.seed, args.round_epochs, args.train_from_scratch)

        post_dir = round_dir / "heldout_eval"
        _rollout(args.seed + rnd, BO_HELDOUT_YAML, post_dir, max_steps=args.max_steps)
        post_metrics = _read_sr(post_dir)
        cum_demos = _count_demos()
        history.append({
            "round": rnd, "cum_demos": cum_demos,
            "heldout_sr": post_metrics["success_rate"],
            "heldout_n_successes": post_metrics["n_successes"],
            "heldout_n_episodes": post_metrics["n_episodes"],
            "correction_sr": corr_metrics["success_rate"],
            "n_prescribed_layouts": n_prescribed,
            "n_new_demos": n_new_demos,
        })
        _persist_curve(history)
        if post_metrics["success_rate"] >= TARGET_SR:
            break
        if n_new_demos == 0:
            break

    _section("PHASE 4: CHART", char="-")
    _make_chart()
    _section("DONE")


def _persist_curve(history):
    out = {
        "method": "p6_only_hybrid", "target_sr": TARGET_SR,
        "demo_dir": str(DEMO_DIR), "checkpoint_dir": str(CKPT_DIR),
        "rag_bank_root": str(RAG_BANK_ROOT),
        "tkf_demo_dir": str(DEMO_DIR),
        "heldout_yaml": str(BO_HELDOUT_YAML),
        "correction_yaml": str(BO_CORRECTION_YAML),
        "training_yaml": str(BO_TRAIN_YAML),
        "history": history,
    }
    with open(RESULTS_DIR / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


def _make_chart():
    rc = _run([sys.executable, "-u", "-m",
               "Equivariant_pathway.equivariant_CNN_hybrid.p6_only.chart"])
    if rc != 0:
        _info(f"chart rc={rc}")


if __name__ == "__main__":
    main()
