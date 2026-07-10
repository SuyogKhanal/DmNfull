"""Task + Diff-DAgger hyperparameter presets.

Defaults follow Table IV of the Diff-DAgger paper, adapted to robosuite UR5e:
  V-objective prediction, Td=16 diffusion steps, alpha=0.99, K=2 patience, and
  an uncertainty batch of T*batch_multiplier = 16*32 = 512 noised samples (Nb).
"""
import copy

BASE = dict(
    # horizons
    obs_horizon=2,
    pred_horizon=16,
    act_horizon=8,
    # diffusion policy
    num_diffusion_iters=16,         # Td
    prediction_type="v_prediction",
    scheduler_type="ddim",
    diffusion_step_embed_dim=64,
    down_dims=[64, 128, 256],
    # uncertainty / query
    alpha=0.99,                     # OOD quantile threshold
    patience_window=2,              # K
    patience=2,
    batch_multiplier=32,            # T*batch_multiplier = 512 = Nb
    num_per_batch=1,
    threshold_num_iter=4,           # passes over the dataset to build the CDF
    # training (per round, retrained from scratch)
    lr=1e-4,
    batch_size=256,
    weight_decay=1e-6,
    initial_train_steps=8000,
    round_train_steps=4000,
    # DAgger loop
    num_init_demos=20,              # Ni (fallback; calibrated by the BC sweep in
                                    # 'experiment' mode so round-0 success ~= 50%)
    final_demos=60,                 # Nf (stop when dataset reaches this many trajs)
    interventions_per_round=4,      # Nd
    max_rounds=12,
    max_rollouts_per_round=40,      # safety cap on DAgger episodes per round
    # evaluation: FIXED held-out set (same seeds every round) so the per-round
    # success curve is comparable and low-noise.
    eval_episodes=100,
    eval_seed_base=5_000_000,       # disjoint from collect (<1e6) and dagger (1-2e6)
    # Ni calibration sweep (BC data-scaling to the ~50% regime)
    target_success=0.5,
    sweep_train_steps=4000,
    sweep_eval_episodes=50,
    calib_windows=3000,
    wipe_marker_obs_k=0,            # Wipe: nearest-K remaining-marker obs (0=off)
    # seeds
    seed=0,
    # ── P4-LLM V3 hybrid (only used in mode=p4_hybrid) ──────────────────────
    # Each round SCREENS the policy for failures, clusters them, and the LLM (or a
    # geometric fallback) picks SELECT (correct one) or BRIDGE (prescribe a middle
    # scene). One successful demo/round (nd=1). Seed bands stay disjoint from
    # collect(<1e6)/dagger(1-2e6)/eval(5e6): screen 3e6, bridge 8e6.
    p4=dict(
        screen_episodes=40,          # policy rollouts/round to find failures
        screen_seed_base=3_000_000,
        max_clusters=4,
        analyze_cap=3,               # top-K failures the multi-stage LLM analyses/round
        snap_eps=0.02,               # tight cluster (<= this spread, m) => SELECT else BRIDGE
        infeasible_attempts=4,       # in-round retries (escalate BRIDGE->SELECT)
        bridge_seed_base=8_000_000,
        # ── Eq 9 cluster memory (allocation mechanism, the crown-jewel claim) ──
        memory_gamma=0.6,            # gamma  (recency discount)
        memory_sigma=0.06,           # sigma  (planar-xy kernel width)
        memory_lambda=1.0,           # lambda (0 => cluster memory OFF, Tier-1 #1)
        # ── ablation flags (05_..md); every one defaults to the full-DISTIL value ──
        allocation="distil",         # 'random' => replay a random recorded failure (Stagger control)
        clustering="silhouette",     # 'off'    => prescribe from raw failures, no modes
        k_selection="silhouette",    # 'fixed_k=3'
        decision="llm",              # 'heuristic' => fixed dominant-rep / centroid, no LLM
        vlm=True,                    # False => reasoning LLM sees geometry+root-cause only
        kag=True,                    # False => drop the per-task knowledge graph from the prompt
        bridge=True,                 # False => SELECT-only
        fallback_only=False,         # True  => deterministic nearest-untried every round, no LLM
        near_dominant=True,          # |C|>=|C*|-1 constraint (False => strict dominant only)
        peak_percentile=95,          # descriptor candidate-set percentile knob {90,95,99}
        vlm_frames=3,                # {1,3,5}
        llm_effort="high",           # {low, high}
        yaw_kernel="planar",         # 'yaw_aware' (PushT only)
    ),
)

# Horizons raised for this clean run (02_..md #2 / 03_..md §NEW): Lift/Door were
# under-allocated; with takeover at t_flag (early) + a full-horizon expert budget
# after takeover, the expert can actually finish. expert_max_steps == env_horizon
# (the decoupled, generous post-takeover budget).
TASKS = {
    # num_init_demos: per-task Ni that leaves round-0 HEADROOM (Ni=20 over-provisioned
    # the robots -> Lift saturated at round 0). Fewer init demos => real failures for
    # DISTIL + the allocation ablations to act on.
    "Lift": dict(
        env_horizon=250,             # was 200
        expert_max_steps=250,
        num_init_demos=8,            # was 20 (saturated)
        ni_sweep=[1, 2, 3, 4, 6, 8, 12],
        final_demos=30,
    ),
    "Wipe": dict(
        # stronger setup: a real local dirt map (nearest-K markers) + more time,
        # since the condensed centroid obs left the policy unable to finish the
        # last scattered markers (coverage plateaued ~0.8, success ~0.4).
        env_horizon=500,
        expert_max_steps=500,
        num_init_demos=12,
        ni_sweep=[4, 8, 12, 16, 20, 28],
        final_demos=60,
        wipe_marker_obs_k=12,
    ),
    "Door": dict(
        env_horizon=350,             # was 300
        expert_max_steps=350,
        num_init_demos=4,
        ni_sweep=[2, 4, 6, 8, 12, 16],
        final_demos=40,
    ),
    # GridWorld classifier task (equivariant CNN; multi-label BCE over the BFS mask).
    # env_horizon = rollout/screen step cap; classifier train is cheap so fewer steps.
    "GridWorld": dict(
        env_horizon=60,
        eval_max_steps=50,
        eval_episodes=200,           # frozen held-out layouts
        num_init_demos=20,
        final_demos=40,
        initial_train_steps=2000,
        round_train_steps=1000,
        batch_size=64,
        lr=3e-4,
        weight_decay=1e-4,
        ni_sweep=[5, 10, 20],        # (unused; budget drives the loop)
    ),
}

# Ablation registry (05_..md): name -> {dotted-flag: value}. 'p4.<flag>' targets the
# p4 sub-dict; a bare key targets the top-level cfg. 'full' = all defaults.
ABLATIONS = {
    "full":            {},
    # ── Tier-1 knockouts (submission-critical) ──
    "memory_off":      {"p4.memory_lambda": 0.0},
    "allocation_random": {"p4.allocation": "random"},
    "clustering_off":  {"p4.clustering": "off"},
    "decision_heuristic": {"p4.decision": "heuristic"},
    "vlm_off":         {"p4.vlm": False},
    # ── Tier-2 (drawer) ──
    "kag_off":         {"p4.kag": False},
    "bridge_off":      {"p4.bridge": False},
    "fallback_only":   {"p4.fallback_only": True},
    "near_dominant_off": {"p4.near_dominant": False},
    "fixed_k3":        {"p4.k_selection": "fixed_k=3"},
    "llm_effort_low":  {"p4.llm_effort": "low"},
}

# When-to-query baselines (not DISTIL ablations): routed to distil/baselines.py.
BASELINE_ARMS = {"diffdagger", "safe", "dropout", "ensemble", "thrifty", "stagger"}


ROBOT_TASKS = ("Lift", "Wipe", "Door")


def _apply_ablation(cfg: dict, ablation: str) -> None:
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation '{ablation}'; known: {sorted(ABLATIONS)}")
    for dotted, val in ABLATIONS[ablation].items():
        if dotted.startswith("p4."):
            cfg["p4"][dotted[3:]] = val
        else:
            cfg[dotted] = val


def get_config(task: str, modality: str = "state", ablation: str = "full",
               budget: int = None, smoke: bool = False) -> dict:
    """Resolve one run's config. modality in {state,image}; ablation from ABLATIONS.
    budget counts SUCCESSFUL demos added on top of the bootstrap (09_..md); the loop's
    stop threshold final_demos is set to n_init + budget in run.py once n_init is known."""
    assert task in TASKS, task
    assert modality in ("state", "image"), modality
    if modality == "image":
        raise NotImplementedError(
            f"image modality not built yet (Phase 2): robot tasks need a camera-obs + "
            f"image-encoder path on the diffusion policy; GridWorld needs the RGB CNN head. "
            f"The descriptor stays GEOMETRIC either way (02_..md #4). State modality only for now.")
    cfg = copy.deepcopy(BASE)
    cfg.update(TASKS[task])
    cfg["task"] = task
    cfg["modality"] = modality
    cfg["ablation"] = ablation
    cfg["budget"] = budget
    if ablation not in BASELINE_ARMS:      # baselines are separate arms, not DISTIL flags
        _apply_ablation(cfg, ablation)
    if smoke:
        cfg.update(dict(
            num_diffusion_iters=8,
            batch_multiplier=4,
            threshold_num_iter=1,
            batch_size=32,
            initial_train_steps=40,
            round_train_steps=20,
            num_init_demos=3,
            final_demos=5,
            interventions_per_round=2,
            max_rounds=2,
            max_rollouts_per_round=4,
            eval_episodes=3,
            sweep_train_steps=20,
            sweep_eval_episodes=2,
            ni_sweep=[1, 2],
            calib_windows=200,
        ))
        cfg["env_horizon"] = min(cfg["env_horizon"], 120)
        cfg["expert_max_steps"] = cfg["env_horizon"]
        # shrink the DISTIL screening/retry budget too
        cfg["p4"] = dict(cfg["p4"], screen_episodes=6, infeasible_attempts=2,
                         max_clusters=3, analyze_cap=2)
    return cfg
