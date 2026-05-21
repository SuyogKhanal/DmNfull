"""Per-run driver for the baseline_vs_p4_sequential_batch suite.

Invoked by ``run_all.sh`` (once per run, in parallel across all N runs).
For a single ``--run_id`` it:

  1. Triggers the upstream-shared bootstrap (idempotent across runs).
  2. Materializes the per-run correction pool with the in-run
     contamination guard.
  3. Runs the requested methods sequentially in this same process:
       * baseline       (selection.baseline_dagger)
       * p4_sequential  (p4.pipeline_seq)  — built later
       * p4_batch       (p4.pipeline_batch) — built later
  4. Writes results/run_{id}/run_summary.json.

Each method writes into its own ``results/run_{id}/{method}/`` subtree so
they don't interfere with each other.

The ``--smoke`` flag halves everything to a 1-run, 2-budget, 3-round
smoke test against the baseline method only, used to verify orchestration
end-to-end before launching production.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

from ..layouts import contamination as cross_run_contam
from ..layouts.layout_setup import ensure_shared_layouts
from . import bootstrap as bootstrap_mod
from .workspace import RunWorkspace, workspace_for


SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SUITE_ROOT / "config.yaml"


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [run_one] {msg}", flush=True)


def _section(title: str, char: str = "=") -> None:
    print(f"\n{char * 80}\n{title}\n{char * 80}", flush=True)


def _load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _smoke_overrides(cfg: Dict) -> Dict:
    """Apply smoke-test overrides. Keep the schema; just shrink the work."""
    cfg = dict(cfg)
    cfg["n_runs"] = 1
    cfg["budget"] = 2
    cfg["max_rounds"] = 3
    cfg.setdefault("trainer", {})
    cfg["trainer"]["finetune_epochs"] = 3
    cfg["methods"] = ["baseline"]
    return cfg


def _run_baseline(
    ws: RunWorkspace, cfg: Dict, train_yaml: Path, heldout_yaml: Path,
) -> Dict:
    from ..selection import baseline_dagger
    trainer_cfg = cfg.get("trainer", {}) or {}
    # Method-specific finetune-epoch override falls back to the shared
    # trainer.finetune_epochs default, matching swatch.sh's BASELINE_ROUND_EPOCHS.
    baseline_epochs = int(
        cfg.get("baseline_finetune_epochs")
        or trainer_cfg.get("finetune_epochs", 20)
    )
    return baseline_dagger.run(
        run_id=ws.run_id,
        bo_root=ws.method_root("baseline"),
        shared_demo_dir=ws.init_demos,
        shared_ckpt_dir=ws.init_checkpoints,
        train_yaml=train_yaml,
        heldout_yaml=heldout_yaml,
        run_shared_dir=ws.shared,
        correction_n=int(cfg.get("correction_n", 50)),
        budget=int(cfg.get("budget", 15)),
        target_sr=float(cfg.get("target_sr", 0.90)),
        max_rounds=int(cfg.get("max_rounds", 50)),
        max_steps=int(cfg.get("max_steps", 60)),
        seed=int(cfg.get("seed", 0)) + int(ws.run_id),
        finetune_epochs=baseline_epochs,
        lr=float(trainer_cfg.get("lr", 5e-5)),
        batch_size=int(trainer_cfg.get("batch_size", 64)),
        weight_decay=float(trainer_cfg.get("weight_decay", 1e-4)),
        replay_mix=float(trainer_cfg.get("replay_mix", 0.5)),
        replay_mix_floor=float(trainer_cfg.get("replay_mix_floor", 0.2)),
    )


def _run_p4(
    method_name: str,
    ws: RunWorkspace,
    cfg: Dict,
    train_yaml: Path,
    heldout_yaml: Path,
) -> Dict:
    """Dispatch to p4.pipeline_seq or p4.pipeline_batch.

    These modules are imported lazily so the baseline-only smoke path
    doesn't require the upstream LLM pipeline to be importable.
    """
    if method_name == "p4_sequential":
        from ..p4 import pipeline_seq as mod
    elif method_name == "p4_batch":
        from ..p4 import pipeline_batch as mod
    else:
        raise ValueError(f"unknown method: {method_name}")
    # Allow swatch-style ROUND_EPOCHS override to bump the P4 fine-tune
    # epoch count higher than the baseline's. Mutate a copy so the
    # baseline call afterwards (if any) still sees the shared default.
    p4_cfg = dict(cfg)
    p4_epochs = cfg.get("p4_finetune_epochs")
    if p4_epochs is not None:
        trainer_cfg = dict(cfg.get("trainer", {}) or {})
        trainer_cfg["finetune_epochs"] = int(p4_epochs)
        p4_cfg["trainer"] = trainer_cfg
    return mod.run(
        run_id=ws.run_id,
        method_root=ws.method_root(method_name),
        shared_demo_dir=ws.init_demos,
        shared_ckpt_dir=ws.init_checkpoints,
        train_yaml=train_yaml,
        heldout_yaml=heldout_yaml,
        run_shared_dir=ws.shared,
        correction_n=int(cfg.get("correction_n", 50)),
        config=p4_cfg,
    )


def _apply_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Mutate ``cfg`` in-place with any CLI override that was actually
    supplied (i.e. not ``None``). Top-level scalars land at the top;
    trainer knobs land under ``cfg["trainer"]``.
    """
    trainer = dict(cfg.get("trainer", {}) or {})
    overrides_top = {
        "budget": args.budget,
        "target_sr": args.target_sr,
        "correction_n": args.correction_n,
        "heldout_n": args.heldout_n,
        "initial_demos": args.initial_demos,
        "initial_epochs": args.initial_epochs,
        "max_rounds": args.max_rounds,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "corridor_blocking": args.corridor_blocking,
        # Per-method finetune-epoch overrides (read by _run_baseline / _run_p4).
        "baseline_finetune_epochs": args.baseline_finetune_epochs,
        "p4_finetune_epochs": args.p4_finetune_epochs,
    }
    for k, v in overrides_top.items():
        if v is not None:
            cfg[k] = v

    overrides_trainer = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "replay_mix": args.replay_mix,
        "replay_mix_floor": args.replay_mix_floor,
    }
    if args.finetune_epochs is not None:
        # Shared default; per-method --baseline_finetune_epochs /
        # --p4_finetune_epochs above take precedence when set.
        overrides_trainer["finetune_epochs"] = args.finetune_epochs
    for k, v in overrides_trainer.items():
        if v is not None:
            trainer[k] = v
    if trainer:
        cfg["trainer"] = trainer
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=int, required=True)
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument(
        "--methods", type=str, default=None,
        help="Comma-sep subset of {baseline,p4_sequential,p4_batch}. "
             "Default: config.yaml methods.",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="Apply smoke-test overrides: budget=2, max_rounds=3, "
             "epochs=3, methods=[baseline]. Useful for end-to-end verification.",
    )
    # Top-level config overrides (mirror swatch.sh's knob block).
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--target_sr", type=float, default=None)
    ap.add_argument("--correction_n", type=int, default=None)
    ap.add_argument("--heldout_n", type=int, default=None)
    ap.add_argument("--initial_demos", type=int, default=None)
    ap.add_argument("--initial_epochs", type=int, default=None)
    ap.add_argument("--max_rounds", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--corridor_blocking", type=lambda s: str(s).lower() in ("1", "true", "yes", "on"),
        default=None, help="Force corridor blocking on/off (overrides config).",
    )
    # Per-method epoch overrides (mirror swatch.sh ROUND_EPOCHS / BASELINE_ROUND_EPOCHS).
    ap.add_argument("--baseline_finetune_epochs", type=int, default=None,
                    help="Fine-tune epochs for the baseline DAgger round.")
    ap.add_argument("--p4_finetune_epochs", type=int, default=None,
                    help="Fine-tune epochs for each P4 round.")
    # Shared trainer overrides.
    ap.add_argument("--finetune_epochs", type=int, default=None,
                    help="Shared default finetune_epochs (overridden per-method by the two flags above).")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--replay_mix", type=float, default=None)
    ap.add_argument("--replay_mix_floor", type=float, default=None)
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    cfg = _apply_cli_overrides(cfg, args)
    if args.smoke:
        cfg = _smoke_overrides(cfg)

    methods: List[str] = (
        [m.strip() for m in args.methods.split(",") if m.strip()]
        if args.methods
        else list(cfg.get("methods", ["baseline"]) or [])
    )
    cfg["methods"] = methods

    _section(
        f"RUN_ID={args.run_id}  methods={methods}  "
        f"budget={cfg.get('budget')} target_sr={cfg.get('target_sr')}"
    )

    # ---- Phase 1: bootstrap shared assets (idempotent) -------------------
    _section("PHASE 1: BOOTSTRAP", char="-")
    bootstrap_mod.ensure_upstream_bootstrap(
        seed=int(cfg.get("seed", 0)),
        initial_epochs=int(cfg.get("initial_epochs", 500)),
        initial_demos=int(cfg.get("initial_demos", 20)),
        heldout_n=int(cfg.get("heldout_n", 200)),
    )

    # ---- Phase 2: per-run layouts ----------------------------------------
    _section("PHASE 2: LAYOUTS", char="-")
    shared = ensure_shared_layouts(
        initial_demos=int(cfg.get("initial_demos", 20)),
        heldout_n=int(cfg.get("heldout_n", 200)),
    )
    ws = workspace_for(args.run_id)
    bootstrap_mod.mirror_upstream_into_run(
        upstream_demos_dir=shared["train"].parent / "init_demos",
        upstream_ckpt_dir=shared["train"].parent / "init_checkpoints",
        per_run_demos_dir=ws.init_demos,
        per_run_ckpt_dir=ws.init_checkpoints,
    )
    # Correction pool rotates per round inside each method's loop; the
    # orchestrator no longer pre-samples. Clean up the legacy single-pool
    # yaml so it can't be silently picked up by a stale tool reading
    # `shared/correction_layouts.yaml`.
    legacy_pool = ws.shared / "correction_layouts.yaml"
    if legacy_pool.exists():
        try:
            legacy_pool.unlink()
            _info(f"removed legacy single-pool yaml {legacy_pool}")
        except OSError:
            pass
    legacy_report = ws.shared / "layout_setup_report.json"
    if legacy_report.exists():
        try:
            legacy_report.unlink()
        except OSError:
            pass
    _info(f"per-round pools will land under {ws.shared}/round_NNN/")

    # ---- Phase 3: run methods sequentially -------------------------------
    summary: Dict = {
        "run_id": ws.run_id,
        "budget": int(cfg.get("budget", 15)),
        "target_sr": float(cfg.get("target_sr", 0.90)),
        "correction_yaml_dir": str(ws.shared),
        "heldout_yaml": str(shared["heldout"]),
        "training_yaml": str(shared["train"]),
        "correction_n": int(cfg.get("correction_n", 50)),
        "methods": methods,
        "results": {},
    }

    for m in methods:
        _section(f"PHASE 3: {m}", char="-")
        try:
            if m == "baseline":
                result = _run_baseline(ws, cfg, shared["train"], shared["heldout"])
            elif m in ("p4_sequential", "p4_batch"):
                result = _run_p4(m, ws, cfg, shared["train"], shared["heldout"])
            else:
                _info(f"unknown method '{m}', skipping")
                continue
            summary["results"][m] = {
                "stopped_reason": result.get("stopped_reason"),
                "n_history": len(result.get("history", []) or []),
            }
        except Exception as exc:  # noqa: BLE001
            _info(f"method '{m}' FAILED: {exc!r}")
            traceback.print_exc()
            summary["results"][m] = {
                "stopped_reason": "error",
                "error": repr(exc),
            }

    with open(ws.root / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _section(f"RUN_ID={ws.run_id} DONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
