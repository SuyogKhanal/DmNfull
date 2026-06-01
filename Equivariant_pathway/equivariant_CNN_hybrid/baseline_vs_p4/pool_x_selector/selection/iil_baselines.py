"""Five interactive-imitation-learning baselines as demonstration-acquisition
rules, for comparison against P4-LLM on the rotated correction pool.

All five share the per-LAYOUT DAgger loop of ``selection.baseline_dagger.run``
(load this round's pool -> roll out -> score every layout -> pick EXACTLY ONE
-> BFS corrective demo at the first deviation -> replay fine-tune -> held-out
eval -> persist learning_curve.json + ranking.json + prescribed_loss.json).
Only the per-layout *query score* differs between methods:

  * ``safe``     SafeDAgger* — fraction of rollout steps whose action is off
                 the A* expert's optimal set; query when it exceeds ``tau_safe``.
  * ``dropout``  DropoutDAgger — MC-dropout over N stochastic forward passes;
                 query when any step's in-ball sample fraction p_hat < p.
  * ``ensemble`` EnsembleDAgger — M-member action disagreement (doubt) and
                 mean action discrepancy; query when doubt>sigma OR disc>tau.
  * ``thrifty``  ThriftyDAgger — ensemble novelty (doubt) OR success-Q risk
                 (1-Q), with budget-alpha auto-calibrated quantile thresholds.
  * ``stagger``  Stagger — one uniformly-random failure per round (the faithful
                 "one sample per round"; ~ the existing random-rotate control).

Pool sourcing reuses P4's exact per-round layouts via
``ensure_correction_layouts_for_round_filled`` (seed-bump fill for rounds
beyond P4's count). Stop semantics match P4's ROTATE pipeline
(``target_hit`` / ``budget_exhausted`` / ``no_progress`` after
``max_consecutive_empty`` zero-demo rounds / ``max_rounds``) so the baselines
may run more rounds than P4 to reach the demo budget — apples-to-apples on the
shared held-out yardstick.

Upstream helpers (``_rollout_with_loss``, ``_save_corrective_demo``,
``_eval_heldout``, ``_find_deviation_step``, ``_load_model``, ``_count_demos``)
are imported read-only; the ensemble / MC-dropout / success-Q machinery lives
in :mod:`selection.uncertainty` and :mod:`selection.success_q`.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4 import (  # noqa: E402
    baseline_budget as _upstream,
)

from ..layouts.layout_setup import ensure_correction_layouts_for_round_filled
from ..logging_ext.training_log import TrainingLog
from . import success_q, uncertainty

KINDS = ("safe", "dropout", "ensemble", "thrifty", "stagger")
_ENSEMBLE_KINDS = ("ensemble", "thrifty")


def _info(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [iil-baseline] {msg}", flush=True)


def _section(title: str, char: str = "-") -> None:
    print(f"\n{char * 80}\n{title}\n{char * 80}", flush=True)


def _finetune(
    cumulative_demo_dir: Path, new_demo_dir: Optional[Path], ckpt_dir: Path,
    seed: int, epochs: int, lr: float, batch_size: int, weight_decay: float,
    replay_mix: float, replay_mix_floor: float,
) -> int:
    """Single-policy replay fine-tune subprocess (same call as the baselines)."""
    cmd = [
        sys.executable, "-u", "-m",
        "Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4."
        "pool_x_selector.trainer.finetune_replay",
        "--demo_dir", str(cumulative_demo_dir),
        "--checkpoint_dir", str(ckpt_dir),
        "--epochs", str(int(epochs)),
        "--batch_size", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--weight_decay", str(float(weight_decay)),
        "--replay_mix", str(float(replay_mix)),
        "--replay_mix_floor", str(float(replay_mix_floor)),
        "--seed", str(int(seed)),
        "--resume",
    ]
    if new_demo_dir is not None and new_demo_dir.exists():
        cmd += ["--new_demos_dir", str(new_demo_dir)]
    _info("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def _persist_curve(
    method: str, results_dir: Path, history: List[Dict], budget: int,
    target_sr: float, demo_dir: Path, ckpt_dir: Path, run_shared_dir: Path,
    heldout_yaml: Path, run_id: int,
) -> None:
    out = {
        "method": method,
        "run_id": int(run_id),
        "budget": int(budget),
        "target_sr": float(target_sr),
        "demo_dir": str(demo_dir),
        "checkpoint_dir": str(ckpt_dir),
        "correction_yaml_dir": str(run_shared_dir),
        "heldout_yaml": str(heldout_yaml),
        "history": history,
    }
    with open(results_dir / "learning_curve.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _norm(vals: List[float]) -> List[float]:
    """Min-max normalize to [0,1]; flat input -> zeros."""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _quantile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.quantile(np.asarray(vals, dtype=float), float(q)))


def _rollout_pool(kind: str, layouts: List[Dict], ckpt_dir: Path,
                  method_root: Path, hp: Dict, seed: int, rnd: int,
                  max_steps: int, device: torch.device) -> List[Dict]:
    """Roll out every pool layout with the kind-appropriate rollout and return
    one raw record per layout (scores filled in by ``_finalize_scores``).
    """
    recs: List[Dict] = []
    if kind in ("safe", "stagger"):
        model = _upstream._load_model(ckpt_dir / "best_hybrid_policy.pth", device)
    elif kind == "dropout":
        model = uncertainty.load_dropout_model(
            ckpt_dir / "best_hybrid_policy.pth", device, hp["dropout_d"])
    elif kind in _ENSEMBLE_KINDS:
        models = uncertainty.load_ensemble(method_root, hp["M"], device)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    for ep_idx, layout in enumerate(layouts):
        ep_seed = seed + rnd * 1000 + ep_idx
        rec: Dict = {"ep_idx": ep_idx, "layout": layout, "ep_seed": ep_seed}
        if kind in ("safe", "stagger"):
            r = _upstream._rollout_with_loss(model, layout, ep_seed, max_steps, device)
            rec.update(success=r["success"], n_steps=r["n_steps"],
                       policy_loss=r["policy_loss"], policy_loss_sum=r["policy_loss_sum"],
                       actions=r["actions"], trajectory=r["trajectory"])
            rec["discrepancy"] = uncertainty.action_discrepancy(
                layout, r["actions"], r["trajectory"])
        elif kind == "dropout":
            r = uncertainty.mc_dropout_rollout(
                model, layout, ep_seed, max_steps, device, hp["mc_N"])
            du = uncertainty.dropout_uncertainty(
                r["per_step_samples"], r["opt_masks"], hp["p_thresh"])
            rec.update(success=r["success"], n_steps=r["n_steps"],
                       policy_loss=r["policy_loss"], policy_loss_sum=r["policy_loss_sum"],
                       actions=r["actions"], trajectory=r["trajectory"])
            rec["mc_mean_disagreement"] = du["mean_disagreement"]
            rec["mc_frac_low"] = du["frac_low"]
        else:  # ensemble / thrifty
            r = uncertainty.ensemble_rollout(models, layout, ep_seed, max_steps, device)
            rec.update(success=r["success"], n_steps=r["n_steps"],
                       policy_loss=r["policy_loss"], policy_loss_sum=r["policy_loss_sum"],
                       actions=r["actions"], trajectory=r["trajectory"])
            rec["doubt"] = uncertainty.ensemble_doubt(r["per_member_actions"], hp["M"])
            rec["mean_discrepancy"] = uncertainty.ensemble_mean_discrepancy(
                r["per_member_actions"], r["opt_masks"])
            rec["_grid_encs"] = r["grid_encs"]  # kept for thrifty Q scoring (not serialized)
        recs.append(rec)
    return recs


def _finalize_scores(kind: str, recs: List[Dict], method_root: Path, demo_dir: Path,
                     hp: Dict, device: torch.device, rng: random.Random) -> None:
    """Set ``score`` and ``queryable`` on each record in place (kind-specific).
    ``score`` ranks layouts (argmax = pick); ``queryable`` is the method's
    intervention gate (a fallback picks the top-scoring layout when nothing is
    queryable, so a round with any signal is never wasted).
    """
    if kind == "safe":
        for r in recs:
            r["score"] = float(r["discrepancy"])
            r["queryable"] = r["discrepancy"] > hp["tau_safe"]
    elif kind == "dropout":
        for r in recs:
            r["score"] = float(r["mc_mean_disagreement"])
            # Faithful DropoutDAgger: the expert intervenes whenever ANY step's
            # in-ball sample fraction p_hat < p (i.e. frac_low > 0).
            r["queryable"] = r["mc_frac_low"] > 0.0
    elif kind == "ensemble":
        for r in recs:
            r["score"] = float(r["doubt"])
            r["queryable"] = (r["mean_discrepancy"] > hp["tau_ens"]) or \
                             (r["doubt"] > hp["sigma_ens"])
    elif kind == "thrifty":
        # Train the success-Q on cumulative demos + this round's rollouts, then
        # score risk = agg_t (1 - Q(s_t, a_t)) along each layout's trajectory.
        success_q.train_success_q(
            method_root=method_root, demo_dir=demo_dir, round_records=recs,
            gamma=hp["gamma"], epochs=hp["q_epochs"], lr=hp["q_lr"],
            device=device, seed=hp.get("seed", 0))
        qnet = success_q.load_q(method_root, device)
        risks: List[float] = []
        for r in recs:
            if qnet is not None and r.get("_grid_encs"):
                qs = success_q.q_values_for_steps(
                    qnet, r["_grid_encs"], r["actions"], device)
                per_state_risk = [1.0 - q for q in qs]
                if hp["thrifty_risk_agg"] == "max":
                    risk = max(per_state_risk) if per_state_risk else 0.0
                else:
                    risk = float(np.mean(per_state_risk)) if per_state_risk else 0.0
            else:
                risk = 0.0
            r["risk"] = float(risk)
            risks.append(r["risk"])
        novelties = [float(r["doubt"]) for r in recs]
        # Budget alpha_h auto-calibration: (1-alpha_h)-quantile thresholds over
        # the pool's per-layout novelty / risk (the layout-level reduction of
        # the paper's per-state quantiles).
        delta_h = _quantile(novelties, 1.0 - hp["alpha_h"])
        beta_h = _quantile(risks, 1.0 - hp["alpha_h"])
        nov_n = _norm(novelties)
        risk_n = _norm(risks)
        for i, r in enumerate(recs):
            r["novelty"] = novelties[i]
            r["score"] = float(nov_n[i] + risk_n[i])  # combined, normalized
            r["queryable"] = (r["doubt"] > delta_h) or (r["risk"] > beta_h)
    elif kind == "stagger":
        for r in recs:
            # Random score over FAILURES only (successes excluded via score=0):
            # argmax of a per-failure uniform draw == a uniformly random failure.
            r["score"] = (rng.random() if not r["success"] else 0.0)
            r["queryable"] = (not r["success"])
    else:
        raise ValueError(f"unknown kind {kind!r}")
    # Drop heavy temporaries before records are kept around / serialized.
    for r in recs:
        r.pop("_grid_encs", None)


def _pick(recs: List[Dict], rng: random.Random) -> Optional[Dict]:
    """Pick one layout: among queryable records (the method's gate), argmax
    ``score`` with ties broken by (-policy_loss, -n_steps, jitter). If nothing
    is queryable, fall back to any record carrying signal (score>0). Returns
    ``None`` only when no layout is queryable AND none carries signal (e.g. a
    fully-optimal pool / zero failures) -> a genuinely empty round.

    The queryable-but-zero-score case is load-bearing for EnsembleDAgger: in
    early rounds the M members are still identical (doubt==0 everywhere), so a
    failure is queryable only via the discrepancy arm with score==0; we must
    still pick it (by highest loss) so a demo is collected and the members can
    diverge on the next fine-tune. Excluding zero-score records would deadlock
    the ensemble (no pick -> no fine-tune -> members never diverge).
    """
    candidates = [r for r in recs if r.get("queryable") or r["score"] > 0.0]
    if not candidates:
        return None
    queryable = [r for r in candidates if r.get("queryable")]
    pool = queryable if queryable else candidates
    for r in pool:
        r.setdefault("_jitter", rng.random())
    pool_sorted = sorted(
        pool, key=lambda r: (-r["score"], -float(r.get("policy_loss", 0.0)),
                             -int(r.get("n_steps", 0)), r["_jitter"]))
    return pool_sorted[0]


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(
    *,
    run_id: int,
    method_name: str,
    method_kind: str,
    bo_root: Path,
    shared_demo_dir: Path,
    shared_ckpt_dir: Path,
    train_yaml: Path,
    heldout_yaml: Path,
    run_shared_dir: Path,
    correction_n: int,
    budget: int,
    target_sr: float,
    max_rounds: int,
    max_steps: int,
    seed: int,
    heldout_block_yaml: Optional[Path] = None,
    finetune_epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    replay_mix: float,
    replay_mix_floor: float,
    max_consecutive_empty: int = 8,
    # baseline hyperparameters (only the relevant ones are read per kind):
    tau_safe: float = 0.10,
    mc_N: int = 16,
    dropout_d: float = 0.2,
    p_thresh: float = 0.5,
    tau_drop: float = 0.5,
    M: int = 5,
    tau_ens: float = 0.10,
    sigma_ens: float = 0.10,
    ensemble_eval_mode: str = "mean",
    alpha_h: float = 0.10,
    gamma: float = 0.95,
    thrifty_risk_agg: str = "mean",
    q_epochs: int = 50,
    q_lr: float = 1e-3,
) -> Dict:
    if method_kind not in KINDS:
        raise ValueError(f"unknown method_kind {method_kind!r} (expected one of {KINDS})")

    # Pool sampling always blocks against the FULL heldout (avoid contamination);
    # held-out EVAL may use a smaller subset (heldout_yaml) for speed (smoke).
    block_yaml = heldout_block_yaml if heldout_block_yaml is not None else heldout_yaml

    hp = dict(
        tau_safe=tau_safe, mc_N=mc_N, dropout_d=dropout_d, p_thresh=p_thresh,
        tau_drop=tau_drop, M=M, tau_ens=tau_ens, sigma_ens=sigma_ens,
        alpha_h=alpha_h, gamma=gamma, thrifty_risk_agg=thrifty_risk_agg,
        q_epochs=q_epochs, q_lr=q_lr, seed=seed,
    )
    is_ensemble = method_kind in _ENSEMBLE_KINDS

    demo_dir = bo_root / "demos"
    ckpt_dir = bo_root / "checkpoints"
    results_dir = bo_root / "results"
    for d in (demo_dir, ckpt_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Seed cumulative demo pool + initial checkpoint(s) from the shared bootstrap.
    for src in sorted(shared_demo_dir.glob("*.json")):
        dst = demo_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    for name in ("best_hybrid_policy.pth", "last_hybrid_policy.pth"):
        s = shared_ckpt_dir / name
        if s.exists():
            shutil.copy2(s, ckpt_dir / name)
    if is_ensemble:
        uncertainty.init_ensemble(bo_root, shared_ckpt_dir, M)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _eval(tag: str, eval_seed: int) -> Dict:
        if is_ensemble:
            return uncertainty.ensemble_eval_heldout(
                bo_root, M, heldout_yaml, results_dir, tag=tag, seed=eval_seed,
                max_steps=max_steps, device=device, mode=ensemble_eval_mode)
        return _upstream._eval_heldout(
            ckpt_dir, heldout_yaml, results_dir, tag=tag, seed=eval_seed,
            max_steps=max_steps)

    # Round 0 — initial held-out eval.
    initial = _eval("round_000", seed)
    history: List[Dict] = [{
        "round": 0,
        "cum_demos": _upstream._count_demos(demo_dir),
        "extra_demos": 0,
        "budget_remaining": budget,
        "heldout_sr": initial["success_rate"],
        "heldout_n_successes": initial["n_successes"],
        "heldout_n_episodes": initial["n_episodes"],
        "correction_sr": None,
        "n_failures_in_pool": None,
        "n_picked": 0,
        "n_new_demos": 0,
        "picked_score": None,
        "correction_yaml": None,
    }]
    _persist_curve(method_name, results_dir, history, budget, target_sr,
                   demo_dir, ckpt_dir, run_shared_dir, heldout_yaml, run_id)
    tlog = TrainingLog(results_dir / "training_log.csv")
    tlog.write_row(round_idx=0, demos_added_total=0, training_rounds_total=0,
                   heldout_sr=initial["success_rate"], finetune_epochs=0,
                   lr=lr, replay_mix=replay_mix)
    if initial["success_rate"] >= target_sr:
        return {"history": history, "stopped_reason": "initial>=target"}

    extras_saved = 0
    training_rounds_total = 0
    consecutive_empty = 0

    for rnd in range(1, max_rounds + 1):
        remaining = budget - extras_saved
        if remaining <= 0:
            break
        _section(f"{method_name.upper()} ROUND {rnd}  (remaining budget = {remaining})")

        # Pass 0: this round's correction pool — reuse P4's, seed-bump-fill else.
        # Blocks against the full heldout (block_yaml), not the eval subset.
        correction_yaml_rnd = ensure_correction_layouts_for_round_filled(
            run_id=run_id, round_idx=rnd, correction_n=correction_n,
            train_yaml=train_yaml, heldout_yaml=block_yaml,
            run_shared_dir=run_shared_dir,
        )
        layouts = _upstream._load_correction_layouts(correction_yaml_rnd)
        if not layouts:
            raise SystemExit(f"no correction layouts in {correction_yaml_rnd}")

        # Pass 1: roll out the pool + score every layout.
        recs = _rollout_pool(method_kind, layouts, ckpt_dir, bo_root, hp,
                             seed, rnd, max_steps, device)
        n_success = sum(1 for r in recs if r["success"])
        failures = [r for r in recs if not r["success"]]
        correction_sr = n_success / len(recs) if recs else 0.0
        rng = random.Random(0xBADF00D + 1009 * (int(run_id) + 1) + rnd)
        _finalize_scores(method_kind, recs, bo_root, demo_dir, hp, device, rng)
        _info(f"round {rnd}: pool={len(recs)} successes={n_success} "
              f"failures={len(failures)} correction_sr={correction_sr:.3f}")

        round_dir = results_dir / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_demo_dir = demo_dir / f"round_{rnd:03d}"
        round_demo_dir.mkdir(parents=True, exist_ok=True)

        # Pass 2: pick ONE layout and emit its corrective demo.
        picked = _pick(recs, rng)
        saved = 0
        picked_score: Optional[float] = None
        picks_payload: List[Dict] = []
        if picked is not None:
            dev = _upstream._find_deviation_step(
                picked["layout"], picked["actions"], picked["trajectory"])
            if dev < 0:
                dev = 0
            out = _upstream._save_corrective_demo(
                picked["layout"], dev, picked["ep_seed"], picked["actions"],
                demo_dir=round_demo_dir,
                layout_name=picked["layout"].get("name", f"ep{picked['ep_idx']}"),
                suffix=f"correct_r{rnd}_ep{picked['ep_idx']}_dev{dev}",
            )
            if out is not None:
                saved = 1
                picked_score = float(picked["score"])
                picks_payload.append({
                    "layout": picked["layout"],
                    "policy_loss": picked["policy_loss"],
                    "policy_loss_sum": picked["policy_loss_sum"],
                    "n_steps": picked["n_steps"],
                    "ep_seed": picked["ep_seed"],
                    "success": picked["success"],
                })
            else:
                _info(f"round {rnd}: picked ep{picked['ep_idx']} unsavable (no BFS path)")

        # Audit: full per-layout ranking.
        ranked = sorted(recs, key=lambda r: -float(r.get("score", 0.0)))
        with open(round_dir / "ranking.json", "w") as f:
            json.dump([
                {
                    "rank": i + 1,
                    "ep_idx": r["ep_idx"],
                    "score": float(r.get("score", 0.0)),
                    "queryable": bool(r.get("queryable", False)),
                    "policy_loss": r.get("policy_loss"),
                    "policy_loss_sum": r.get("policy_loss_sum"),
                    "n_steps": r.get("n_steps"),
                    "success": r.get("success"),
                    "layout_name": r["layout"].get("name"),
                    "picked": (picked is not None and r["ep_idx"] == picked["ep_idx"] and saved > 0),
                } for i, r in enumerate(ranked)
            ], f, indent=2, default=str)

        # Uniform info-gain proxy (one filename across all methods).
        if picks_payload:
            with open(round_dir / "prescribed_loss.json", "w") as f:
                json.dump({"round": rnd, "method": method_name, "picks": picks_payload},
                          f, indent=2, default=str)

        extras_saved += saved
        _info(f"round {rnd}: picked={1 if picked is not None else 0} saved={saved} "
              f"cum_extras={extras_saved}/{budget}")

        # Pass 3: replay fine-tune (single policy, or every ensemble member).
        if saved > 0:
            if is_ensemble:
                rcs = uncertainty.finetune_ensemble(
                    cumulative_demo_dir=demo_dir, new_demo_dir=round_demo_dir,
                    method_root=bo_root, M=M, base_seed=seed,
                    epochs=finetune_epochs, lr=lr, batch_size=batch_size,
                    weight_decay=weight_decay, replay_mix=replay_mix,
                    replay_mix_floor=replay_mix_floor)
                if any(rc != 0 for rc in rcs):
                    _info(f"round {rnd}: ensemble finetune rcs={rcs} (continuing)")
            else:
                rc = _finetune(
                    cumulative_demo_dir=demo_dir, new_demo_dir=round_demo_dir,
                    ckpt_dir=ckpt_dir, seed=seed, epochs=finetune_epochs, lr=lr,
                    batch_size=batch_size, weight_decay=weight_decay,
                    replay_mix=replay_mix, replay_mix_floor=replay_mix_floor)
                if rc != 0:
                    _info(f"round {rnd}: finetune rc={rc} (continuing)")
            training_rounds_total += 1

        # Pass 4: held-out eval + persist.
        post = _eval(f"round_{rnd:03d}", seed + rnd)
        history.append({
            "round": rnd,
            "cum_demos": _upstream._count_demos(demo_dir),
            "extra_demos": extras_saved,
            "budget_remaining": budget - extras_saved,
            "heldout_sr": post["success_rate"],
            "heldout_n_successes": post["n_successes"],
            "heldout_n_episodes": post["n_episodes"],
            "correction_sr": correction_sr,
            "n_failures_in_pool": len(failures),
            "n_picked": 1 if picked is not None else 0,
            "n_new_demos": saved,
            "picked_score": picked_score,
            "correction_yaml": str(correction_yaml_rnd),
        })
        _persist_curve(method_name, results_dir, history, budget, target_sr,
                       demo_dir, ckpt_dir, run_shared_dir, heldout_yaml, run_id)
        tlog.write_row(round_idx=rnd, demos_added_total=extras_saved,
                       training_rounds_total=training_rounds_total,
                       heldout_sr=post["success_rate"],
                       finetune_epochs=(finetune_epochs if saved > 0 else 0),
                       lr=lr, replay_mix=replay_mix)
        _info(f"round {rnd}: heldout_sr={post['success_rate']:.3f}")

        if post["success_rate"] >= target_sr:
            return {"history": history, "stopped_reason": "target_hit"}
        if extras_saved >= budget:
            return {"history": history, "stopped_reason": "budget_exhausted"}
        if saved == 0:
            # Rotate-pool: a zero-demo round does NOT stop the run (a fresh pool
            # next round may contain failures); cap consecutive empties.
            consecutive_empty += 1
            if consecutive_empty >= max_consecutive_empty:
                return {"history": history, "stopped_reason": "no_progress"}
        else:
            consecutive_empty = 0

    return {"history": history, "stopped_reason": "max_rounds"}
