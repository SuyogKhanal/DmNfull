"""ManiSkill env construction — thin wrappers over the fork's Hydra setup.

Every method in this suite shares ONE simulator stack from the fork:
  * the policy env (control_mode from the task's Hydra cfg, wrapped),
  * a frozen vectorized held-out eval env (eval_env block),
  * a reposition env that honours an LLM-prescribed config (PushT-Start-v0),
all built exactly the way the fork's run_comparison.py builds them, so the
trained diffusion policy / PPO expert / normalizers line up. We never edit the
fork; we compose its Hydra config and instantiate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from . import env_setup as E

E.bootstrap_fork_path()

import gymnasium as gym  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

_RESOLVERS_REGISTERED = False


def _register_resolvers() -> None:
    """Register the OmegaConf `eval` / `if` resolvers the fork cfg uses
    (run_comparison.py:49-54). Idempotent."""
    global _RESOLVERS_REGISTERED
    if _RESOLVERS_REGISTERED:
        return

    def _cond(condition, true_value, false_value):
        return true_value if eval(condition) else false_value  # noqa: S307

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("if", _cond, replace=True)
    _RESOLVERS_REGISTERED = True


def load_cfg(suite_env_id: str):
    """Compose the fork Hydra config (policy/env/eval_env/dataset/expert) for a
    suite task. Returns a mutable OmegaConf DictConfig."""
    from hydra import compose, initialize_config_dir

    _register_resolvers()
    E.register_envs()
    cfg_path = E.hydra_cfg_path(suite_env_id)
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Hydra config for {suite_env_id} not found at {cfg_path}. "
            f"Only PushT is wired for the first submission; author the others "
            f"(stackcube_state.yaml, ...) after PushT validates.")
    cfg_dir = str(cfg_path.resolve().parent)
    cfg_name = cfg_path.stem
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=cfg_name)
    OmegaConf.set_struct(cfg, False)
    return cfg


def make_policy_env(cfg) -> Any:
    """The interactive policy env (single env), wrapped like the fork."""
    from hydra.utils import instantiate
    from diffdagger.util.maniskill_env import wrap_env

    env = instantiate(cfg.env)
    return wrap_env(env, cfg.obs_keys, cfg.action_space)


def make_eval_env(cfg, num_envs: Optional[int] = None) -> Any:
    """The frozen vectorized held-out evaluator (eval_env block)."""
    from hydra.utils import instantiate
    from diffdagger.util.maniskill_env import wrap_env

    if num_envs is not None:
        cfg.eval_env.num_envs = int(num_envs)
    eval_env = instantiate(cfg.eval_env)
    return wrap_env(eval_env, cfg.obs_keys, cfg.action_space)


def make_reposition_env(cfg, suite_env_id: str) -> Optional[Any]:
    """The env that honours an LLM-prescribed config. PushT → PushT-Start-v0
    (mutates _tee_init_xyz/_tee_init_zrot then reset). Returns None for tasks
    without a reposition env wired yet."""
    from diffdagger.util.maniskill_env import wrap_env

    spec = E.task_spec(suite_env_id)
    if spec.reposition_env_id is None:
        return None
    kw = dict(
        num_envs=1,
        obs_mode=cfg.obs_mode,
        control_mode=cfg.control_mode,
        render_mode=cfg.render_mode,
        max_episode_steps=cfg.max_episode_steps,
        enable_shadow=True,
        sim_backend="gpu",
        robot_init_qpos_noise=0.0,
    )
    # PushT-Start-v0 takes tee_init_* kwargs; StackCube-Start-v0 takes the prescribed
    # cube poses via set_prescription() AFTER make (defaults to random), so pass no
    # task-specific init kwargs here.
    if spec.reposition_env_id == "PushT-Start-v0":
        kw.update(tee_init_xyz=None, tee_init_zrot=None)
    repo = gym.make(spec.reposition_env_id, **kw)
    return wrap_env(repo, cfg.obs_keys, cfg.action_space)


def make_all(suite_env_id: str, *, eval_num_envs: Optional[int] = None,
             need_reposition: bool = True) -> Tuple[Any, Any, Optional[Any], Any]:
    """Build (cfg, policy_env, eval_env, reposition_env)."""
    cfg = load_cfg(suite_env_id)
    env = make_policy_env(cfg)
    eval_env = make_eval_env(cfg, num_envs=eval_num_envs)
    repo = make_reposition_env(cfg, suite_env_id) if need_reposition else None
    return cfg, env, eval_env, repo
