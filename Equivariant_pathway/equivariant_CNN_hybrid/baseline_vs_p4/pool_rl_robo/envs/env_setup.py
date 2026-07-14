"""Environment catalog + fork bootstrap for pool_rl_robo (ManiSkill).

This suite is the CONTINUOUS-ACTION-SPACE sibling of pool_x_selector. The
diffusion-policy backbone, the vendored ManiSkill3 simulator, the panda
motion-planner / PushT PPO experts, and the Diff-DAgger query rule all come
from the user's Diff-DAgger fork, wired in read-only at ``external/diff_dagger``
(symlink → /weka/s226137394/diff-dagger). We never edit the fork; we import it.

Importing this module makes ``mani_skill``, the ``diffdagger`` package, and the
``custom_envs`` (PushT-Start-v0) importable by inserting the fork roots onto
``sys.path`` — exactly the bootstrap the fork's own ``run_comparison.py`` does.

The 4 tasks in scope: StackCube-v1, PickCube-v1, PushT-v1, PlugCharger-v1.
Only PushT is wired end-to-end for the first submission (the fork ships a
trained diffusion policy + PPO expert + PushT-Start-v0 reposition env + a Hydra
config for it). The other three carry catalog entries (motion-planner experts
exist in the fork) but their Hydra configs / reposition envs are authored in a
later phase, after PushT validates.
"""
from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---- paths --------------------------------------------------------------
SUITE_ROOT = Path(__file__).resolve().parents[1]          # pool_rl_robo/
# POOL_RESULTS_ROOT redirects the whole results tree (default: suite-local results/).
# Used to write a re-run into its own namespace so published results cannot be
# clobbered; unset => byte-identical behaviour to before.
RESULTS_ROOT = Path(os.environ.get("POOL_RESULTS_ROOT") or (SUITE_ROOT / "results"))
FORK_ROOT = (SUITE_ROOT / "external" / "diff_dagger").resolve()   # → the fork


# The fork's Hydra configs use BARE top-level targets (e.g. `model.RL.PPO`,
# `model.unet1d.ConditionalUnet1D`, `agents.RL_agent.MultipleExperts`,
# `dataset.datasets.TimeSeriesDataset`) that resolve against `diffdagger/<name>`.
# Those dirs are namespace packages (no __init__.py), so a REGULAR same-named
# package elsewhere on sys.path WINS — and the repo root `DmNfull/model` (a
# regular package) shadows the fork's `diffdagger/model`. We bind each fork
# subpackage to its bare name in sys.modules so the bare targets always resolve
# to the fork, regardless of sys.path ordering or DmNfull collisions.
_FORK_BARE_SUBPKGS = ("model", "agents", "dataset", "util",
                      "main_pipeline", "main_analysis")


def bootstrap_fork_path() -> None:
    """Insert the fork roots onto sys.path so `import mani_skill`,
    `from diffdagger... import`, and `import custom_envs.push_t_start` resolve,
    and bind the fork's bare-name subpackages (see comment above).
    Idempotent; mirrors + hardens the fork's run_comparison.py sys.path setup."""
    for p in (FORK_ROOT, FORK_ROOT / "diffdagger", FORK_ROOT / "env_restart"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    dd = FORK_ROOT / "diffdagger"
    for name in _FORK_BARE_SUBPKGS:
        d = dd / name
        if not d.is_dir():
            continue
        cur = sys.modules.get(name)
        cur_path = list(getattr(cur, "__path__", [])) if cur is not None else []
        if str(d) in cur_path:
            continue                      # already bound to the fork dir
        mod = types.ModuleType(name)
        mod.__path__ = [str(d)]           # namespace pkg → fork dir wins
        sys.modules[name] = mod


bootstrap_fork_path()


# ---- env catalog --------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    suite_id: str                 # what config.yaml / CLI / results dirs use
    fork_env_id: str              # the gym id actually instantiated
    hydra_cfg: str                # fork Hydra config (relative to FORK_ROOT)
    expert_kind: str              # "ppo" | "motionplanner"
    reposition_env_id: Optional[str]  # env that honours a prescribed config
    task_description: str         # one line, fed to the VLM / KAG prompts
    wired: bool = False           # True once end-to-end verified for this task


# The fork's diffusion stack + experts run in the panda joint-position family
# (pd_joint_pos for motion planners; rel_joint_pos for the policy). The actual
# control / obs mode is read from the task's Hydra config, not hard-coded here.
TASKS: Dict[str, TaskSpec] = {
    "PushT-v1": TaskSpec(
        suite_id="PushT-v1",
        fork_env_id="PushT-v2",                       # the fork's trained variant
        hydra_cfg="diffdagger/config/sim/pusht_state.yaml",
        expert_kind="ppo",
        reposition_env_id="PushT-Start-v0",           # honours tee_init_xyz/zrot
        task_description=(
            "A Franka Panda (panda_stick end-effector, no gripper) pushes a "
            "T-shaped block across a table to a fixed goal T-pose. Success = the "
            "movable T overlaps the goal T within tolerance."),
        wired=True,
    ),
    "StackCube-v1": TaskSpec(
        suite_id="StackCube-v1",
        fork_env_id="StackCube-v1",
        hydra_cfg="configs/hydra/stackcube_state.yaml",   # suite-local (not fork)
        expert_kind="motionplanner",
        reposition_env_id="StackCube-Start-v0",           # suite-local, honours
                                                          # prescribed cube poses (p4_top3)
        task_description=(
            "A Franka Panda grasps cube A and stacks it on top of cube B. "
            "Success = cube A is resting stably on cube B and the gripper is open."),
    ),
    "PickCube-v1": TaskSpec(
        suite_id="PickCube-v1",
        fork_env_id="PickCube-v1",
        hydra_cfg="diffdagger/config/sim/pickcube_state.yaml",    # authored later
        expert_kind="motionplanner",
        reposition_env_id=None,
        task_description=(
            "A Franka Panda grasps a cube and lifts it to a goal position marked "
            "by a sphere. Success = the cube is held at the goal site."),
    ),
    "PlugCharger-v1": TaskSpec(
        suite_id="PlugCharger-v1",
        fork_env_id="PlugCharger-v1",
        hydra_cfg="configs/hydra/plugcharger_state.yaml",   # suite-local (not fork)
        expert_kind="motionplanner",
        reposition_env_id="PlugCharger-Start-v0",           # suite-local, honours
                                                            # prescribed charger/socket poses (BRIDGE)
        task_description=(
            "A Franka Panda grasps a charger and inserts its prongs into a wall "
            "socket. Contact-rich, tight tolerances. Success = charger fully "
            "inserted into the receptacle."),
    ),
}

ALL_ENVS: List[str] = list(TASKS.keys())


def task_spec(suite_id: str) -> TaskSpec:
    if suite_id not in TASKS:
        raise KeyError(f"unknown suite env id {suite_id!r}; known: {ALL_ENVS}")
    return TASKS[suite_id]


def fork_env_id(suite_id: str) -> str:
    return task_spec(suite_id).fork_env_id


def hydra_cfg_path(suite_id: str) -> Path:
    """Resolve a task's Hydra config. PushT lives in the fork; non-PushT tasks
    (StackCube/…) are authored SUITE-LOCAL (we never edit the fork), so prefer a
    path under SUITE_ROOT and fall back to FORK_ROOT.

    Override hook: if ``$SUITE_HYDRA_CFG`` points to an existing file, it WINS —
    this is how the IMAGE-based run swaps the state config (pusht_state.yaml) for
    pusht_image.yaml (obs_mode=rgb + R3M encoder) for BOTH arms (diff_dagger and
    p4_subtask) without editing the fork or the task catalog. A relative value is
    resolved against SUITE_ROOT."""
    import os
    ov = os.environ.get("SUITE_HYDRA_CFG", "").strip()
    if ov:
        ovp = Path(ov)
        if not ovp.is_absolute():
            ovp = SUITE_ROOT / ovp
        if ovp.is_file():
            return ovp
        print(f"[env_setup] WARNING: SUITE_HYDRA_CFG={ov!r} not found; "
              f"using the default Hydra config for {suite_id}.")
    rel = task_spec(suite_id).hydra_cfg
    suite_p = SUITE_ROOT / rel
    if suite_p.is_file():
        return suite_p
    return FORK_ROOT / rel


def register_envs() -> None:
    """Import ManiSkill task registration + the PushT-Start reposition env.
    Safe to call repeatedly."""
    bootstrap_fork_path()
    import mani_skill.envs  # noqa: F401  registers all tabletop tasks
    try:
        import custom_envs.push_t_start  # noqa: F401  registers PushT-Start-v0
    except Exception as exc:  # pragma: no cover - only PushT needs it
        print(f"[env_setup] note: PushT-Start-v0 not registered ({exc})")
    try:
        from ..p4_subtask import envs as _subtask_envs  # noqa: F401  registers PushT-Subtask-v0
    except Exception as exc:  # pragma: no cover - only p4_subtask needs it
        print(f"[env_setup] note: PushT-Subtask-v0 not registered ({exc})")
    try:
        from . import stackcube_start  # noqa: F401  registers StackCube-Start-v0
    except Exception as exc:  # pragma: no cover
        print(f"[env_setup] note: StackCube-Start-v0 not registered ({exc})")
    try:
        from . import plugcharger_start  # noqa: F401  registers PlugCharger-Start-v0
    except Exception as exc:  # pragma: no cover
        print(f"[env_setup] note: PlugCharger-Start-v0 not registered ({exc})")


def is_success(info: dict) -> bool:
    """Read ManiSkill's per-env success flag from an info dict (tensor or bool)."""
    v = info.get("success", info.get("is_success"))
    if v is None:
        return False
    try:
        return bool(v.item()) if hasattr(v, "item") else bool(v)
    except Exception:
        try:
            return bool(v.reshape(-1)[0])
        except Exception:
            return bool(v)
