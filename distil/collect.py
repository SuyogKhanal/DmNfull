"""Collect demonstrations from a scripted expert oracle."""
import numpy as np


def rollout_expert(env, expert, *, seed, max_steps):
    """Run the expert to completion; return (state_seq, action_seq, success, len).

    Records (s_t, a_t) pairs: the state observed *before* each expert action.
    """
    s = env.reset(seed=seed)
    expert.reset(env)
    states, actions = [], []
    success, done, t = False, False, 0
    while not done and t < max_steps:
        a = np.asarray(expert.get_action(env, s), dtype=np.float32)
        states.append(np.asarray(s, dtype=np.float32))
        actions.append(a)
        s, r, done, success, info = env.step(a)
        t += 1
    return np.stack(states), np.stack(actions), success, t


def collect_demos(env, expert, num_demos, *, seed_base=0, max_steps=None,
                  require_success=True, log_fn=print):
    """Collect `num_demos` (successful) expert trajectories."""
    max_steps = max_steps or env.horizon
    trajs, seed, tries = [], seed_base, 0
    while len(trajs) < num_demos and tries < num_demos * 8:
        states, actions, success, t = rollout_expert(env, expert, seed=seed, max_steps=max_steps)
        seed += 1
        tries += 1
        if require_success and not success:
            log_fn(f"  [collect] discarded non-success demo (len={t}); retrying")
            continue
        trajs.append({"state": states, "action": actions})
        log_fn(f"  [collect] demo {len(trajs)}/{num_demos}: success={success} len={t}")
    if len(trajs) < num_demos:
        log_fn(f"  [collect] WARNING: only collected {len(trajs)}/{num_demos} demos")
    return trajs
