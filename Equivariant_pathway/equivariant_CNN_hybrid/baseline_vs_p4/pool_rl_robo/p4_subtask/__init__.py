"""p4_subtask — cluster-anchored sub-task-entry P4-LLM for PushT-v1.

This package implements the supervisor's Fix-2 (prescribe the *sub-task entry
point*, not the episode start) plus Fix-1 (cluster → dominant → centroid →
cross-round memory) and Fix-4 (diversity selection) on top of the EXACT P4
architecture (VLM → reasoning analysis → reasoning prescription → infeasibility
loop → feedback loop). It is the improved replacement for the losing `p4_top3`.

All logic here is pure suite-side code; it is INJECTED into the fork's
``LLMGuidedDAggerPipeline`` (set ``pipe._subtask_planner`` + ``pipe._subtask_collect``).
The fork carries only three tiny default-OFF gated hooks; with the planner absent,
the p4_top3 and diff_dagger paths are byte-identical.

Modules
-------
descriptor    : per-failure feature vector + reconstructed t* state from meta.json
clustering    : cluster failures, pick the dominant cluster, coverage centroid
memory        : cross-round prescribed-centroid memory + recency penalty
diversity     : Fix-4 farthest-point candidate selection
subtask_entry : anchor reconstruction + bounded coverage perturbation + ResetSpec
semantic_map  : KAG-grounded semantic direction helpers (prompt-side)
planner       : SubtaskPlanner — the injected object (hooks A/B/C)
telemetry     : exhaustive per-round logging
collect       : collect_subtask_demo (sets tee + agent_qpos, runs PPO expert)
envs          : PushT-Subtask-v0 (honours _agent_init_qpos)
pipeline      : run_p4_subtask_arm (suite arm; injects the planner)
"""
