# 07 — KAG (Knowledge-Augmented Generation) documents per task

Every task carries a **per-task KAG graph** (JSON) that is rendered to text and injected as
`{kag_text}` into the ANALYSIS + DECISION prompts (`06_...md`). It grounds the LLM in the task's
real geometry — the **reliable workspace bounds** (where the scripted/PPO expert actually works),
the node/edge structure (robot→ee→object→goal), the enumerated failure modes, and the
per-failure-mode **reasoning implications** that tell the LLM how to spend a demo. **Copy the
exact KAG JSON used into each run's output folder `kag/`** (golden rule 3; `09_...md`).

> The KAG bounds are also the **feasibility gate** for the infeasibility loop (`02_...md` #5):
> a prescription whose pose falls outside `workspace_constraint` is rejected before any expert
> effort. Keep the KAG bounds and the env reset ranges in sync per task.

## Source graphs (verbatim JSON already on disk — reuse, don't reinvent)
| task | KAG file |
|---|---|
| PushT | `pool_rl_robo/p4/kag/PushT-v1.json` (+ rendered `PushT-v1.kag.txt`) |
| StackCube | `pool_rl_robo/p4/kag/StackCube-v1.json` |
| PickCube / PlugCharger | `pool_rl_robo/p4/kag/{PickCube-v1,PlugCharger-v1}.json` |
| Lift | `diff-dagger-ur5/diffdagger_rs/p4/kag/Lift.json` (cube range `x/y∈[-0.03,0.03] z=0.831`) |
| Door | `diff-dagger-ur5/diffdagger_rs/p4/kag/Door.json` (frame `x[-0.135,-0.108] y[-0.366,-0.340] z=1.10`, yaw `[-1.82,-1.57]`) |
| Wipe | `diff-dagger-ur5/diffdagger_rs/p4/kag/Wipe.json` |
| GridWorld | **build one** — nodes agent/goal/walls/fire; failure modes wrong-direction/hit-fire/timeout; `workspace_constraint` = valid non-wall cells; implications = prescribe a corridor layout |

## Schema (all graphs share it)
`meta` (domain, description, robot, control_mode, action_dim) · `nodes[]` (`id`, `type`, `label`,
`properties`) · `edges[]` (`source`, `target`, `relation`) · `reasoning_implications` (one entry
per failure mode + **`workspace_constraint`** + **`non_emptiness`**). Node types seen:
`Robot, Object, Goal, EndEffector, Observation, Workspace, Controller, SuccessCondition,
FailureMode, Phase`.

## Verbatim example — PushT (`PushT-v1.json`), the primary task
```json
{
  "meta": {
    "schema_version": "1.0", "document_type": "knowledge_augmented_generation",
    "domain": "maniskill_pusht_nonprehensile_manipulation",
    "description": "Franka Panda (panda_stick, no gripper, 7 arm joints) pushes a T-block to a fixed goal T-pose. Grounds P4-LLM failure analysis + prescribed config. World-frame metres; PPO expert reliable only inside the stated tee/tcp ranges.",
    "robot": "Franka Panda (panda_stick)",
    "control_mode": "pd_joint_pos (policy acts in rel_joint_pos = 7 joint deltas)", "action_dim": 7
  },
  "nodes": [
    {"id":"robot","type":"Robot","label":"Panda panda_stick","properties":{"arm_joints":7,"gripper":false,"tcp_link":"panda_hand_tcp"}},
    {"id":"tee","type":"Object","label":"Movable T-block","properties":{"controllable_via":"non-prehensile pushing only","init_xyz_world":"[x,y,z]","init_zrot_rad":"z-rotation of the T"}},
    {"id":"goal","type":"Goal","label":"Fixed goal T-pose","properties":{"goal_offset":[-0.156,-0.1],"goal_z_rot_rad":1.5708,"fixed_per_episode":true}},
    {"id":"tcp","type":"EndEffector","label":"Stick TCP","properties":{"role":"pushes the T; cannot grasp","approach_matters":"must contact the correct T face to push toward the goal"}},
    {"id":"obs","type":"Observation","label":"state_dict (21-d)","properties":{"agent_qpos":"7 arm joint angles (rad)","extra_tcp_pose":"TCP position+quaternion","extra_obj_pose":"T position+quaternion"}},
    {"id":"ws_tee","type":"Workspace","label":"Reliable tee init range","properties":{"x":[-0.20,0.20],"y":[-0.25,0.05],"z":0.021}},
    {"id":"ws_tcp","type":"Workspace","label":"Reliable tcp range","properties":{"x":[-0.35,0.35],"y":[-0.35,0.35],"z":[0.02,0.08]}},
    {"id":"ctrl","type":"Controller","label":"pd_joint_pos / rel_joint_pos","properties":{"policy_action":"7 joint deltas (rel_joint_pos)","expert_action":"PPO → joint_delta_pos (same 7-joint space)"}},
    {"id":"succ","type":"SuccessCondition","label":"T overlaps goal","properties":{"metric":"intersection of movable T and goal T above threshold","info_key":"success"}},
    {"id":"fm_wrong_approach","type":"FailureMode","label":"wrong_approach","properties":{"description":"TCP contacts the wrong face/side of the T, pushing it away from the goal."}},
    {"id":"fm_overshoot","type":"FailureMode","label":"overshoot","properties":{"description":"TCP pushes the T past the goal pose; momentum carries it through."}},
    {"id":"fm_no_contact","type":"FailureMode","label":"no_contact","properties":{"description":"TCP never establishes contact with the T."}},
    {"id":"fm_wrong_orientation","type":"FailureMode","label":"wrong_orientation","properties":{"description":"T's z-rotation not aligned with the goal; pushing translates but does not rotate it correctly."}},
    {"id":"fm_timeout","type":"FailureMode","label":"timeout","properties":{"description":"Horizon reached without success (hesitation / oscillation / slow approach)."}},
    {"id":"ph_pre_contact","type":"Phase","label":"pre_contact","properties":{"description":"approach the T before contact"}},
    {"id":"ph_contact","type":"Phase","label":"contact","properties":{"description":"establish contact on the correct face"}},
    {"id":"ph_push","type":"Phase","label":"push","properties":{"description":"translate the T toward the goal"}},
    {"id":"ph_align","type":"Phase","label":"align","properties":{"description":"fine z-rotation/position alignment with the goal T"}}
  ],
  "edges": [
    {"source":"robot","target":"tcp","relation":"HAS_END_EFFECTOR"},
    {"source":"tcp","target":"tee","relation":"PUSHES"},
    {"source":"tee","target":"goal","relation":"MUST_REACH"},
    {"source":"robot","target":"obs","relation":"OBSERVES"},
    {"source":"robot","target":"ctrl","relation":"CONTROLLED_BY"},
    {"source":"tee","target":"ws_tee","relation":"INIT_WITHIN"},
    {"source":"tcp","target":"ws_tcp","relation":"INIT_WITHIN"},
    {"source":"tee","target":"succ","relation":"SUCCEEDS_WHEN"},
    {"source":"tcp","target":"fm_wrong_approach","relation":"CAUSES_FAILURE"},
    {"source":"tcp","target":"fm_overshoot","relation":"CAUSES_FAILURE"},
    {"source":"tcp","target":"fm_no_contact","relation":"CAUSES_FAILURE"},
    {"source":"tee","target":"fm_wrong_orientation","relation":"CAUSES_FAILURE"},
    {"source":"robot","target":"fm_timeout","relation":"CAUSES_FAILURE"}
  ],
  "reasoning_implications": {
    "wrong_approach":"Prescribe a tee start whose push direction toward the fixed goal is unambiguous; set the tcp start on the push side.",
    "overshoot":"Prescribe the T closer to the goal (shorter push) so the demo teaches a controlled stop at the goal.",
    "no_contact":"Prescribe the tcp start adjacent to the T's push face so the demo emphasises contact establishment.",
    "wrong_orientation":"Prescribe a tee z-rotation differing from the goal so the demo teaches rotate-while-pushing.",
    "timeout":"Prescribe a single direct push corridor toward the goal (no oscillation), poses well inside the reliable workspace.",
    "workspace_constraint":"Every prescribed config MUST keep tee_xyz within x[-0.20,0.20] y[-0.25,0.05] z=0.021 and tcp_xyz within x[-0.35,0.35] y[-0.35,0.35] z[0.02,0.08]; out-of-range poses are dropped and waste the round.",
    "non_emptiness":"A failure is present, so the prescription MUST be concrete and fully-specified (non-empty tee_xyz, tee_zrot, tcp_xyz). Never emit an empty prescription."
  }
}
```
(StackCube adds `cubeA/cubeB`, `action_dim:8` with a gripper command, and a `diffusion loss`
detection node; Lift/Door/Wipe follow the same schema with `grasp_failure/approach_failure/
placement_error/contact_instability/pose_mismatch/timeout` implications. All are on disk — copy,
don't rewrite.)

## Renderer — `format_kag_context(kag) -> str` (verbatim; identical in both p4 packages)
```python
def format_kag_context(kag):
    if not kag: return ""
    meta, nodes, edges = kag.get("meta",{}), kag.get("nodes",[]), kag.get("edges",[])
    impl = kag.get("reasoning_implications", {})
    nl = {n["id"]: n for n in nodes}
    lines = ["=== KAG — TASK KNOWLEDGE GRAPH ===",
             f"Domain: {meta.get('domain','')}",
             f"Description: {meta.get('description','')}", ""]
    by_type = {}
    for n in nodes: by_type.setdefault(n.get("type","Node"), []).append(n)
    for t, ns in by_type.items():
        lines.append(f"[{t}]")
        for n in ns:
            props = ", ".join(f"{k}={v}" for k,v in n.get("properties",{}).items())
            lines.append(f"  - {n.get('label', n['id'])} (id={n['id']}): {props}")
        lines.append("")
    if edges:
        lines.append("[RELATIONS]")
        for e in edges:
            src = nl.get(e.get("source"),{}).get("label", e.get("source"))
            tgt = nl.get(e.get("target"),{}).get("label", e.get("target"))
            lines.append(f"  {src} --[{e.get('relation','RELATED_TO')}]--> {tgt}")
        lines.append("")
    if impl:
        lines.append("[REASONING IMPLICATIONS]")
        for k,v in impl.items(): lines.append(f"  * {k}: {v}")
        lines.append("")
    return "\n".join(lines)
```
Renders to `[Robot]/[Object]/[Goal]/…/[FailureMode]/[Phase]` sections + `[RELATIONS]` +
`[REASONING IMPLICATIONS]`. Source: `pool_rl_robo/p4/kag.py` and
`diff-dagger-ur5/diffdagger_rs/p4/kag.py` (byte-identical logic). Cache the rendered text to
`<task>.kag.txt` and inject THAT as `{kag_text}`.

## Notes / caveats
- The fork's generic `main_analysis/kag_document.txt` (11 KB robot-knowledge doc) is **legacy** —
  every task now renders its own per-task JSON graph with the task's real spawn box + bounds. Use
  the per-task graphs; don't fall back to the generic doc.
- `kag=off` ablation (`05_...md`): drop `{kag_text}` from the ANALYSIS + DECISION prompts and
  disable the KAG feasibility gate (leave only the raw env reset-range clamp). This isolates the
  KAG's contribution to both prescription quality and feasibility.
