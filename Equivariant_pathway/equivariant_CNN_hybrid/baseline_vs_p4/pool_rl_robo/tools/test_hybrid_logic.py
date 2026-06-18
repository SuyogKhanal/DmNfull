"""Offline unit test for the StackCube V3-hybrid decision logic (no GPU/LLM/env).

Run from the DmNfull repo root:
  python -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.tools.test_hybrid_logic

Validates: SELECT/BRIDGE parsing, decide() dispatch (select / bridge / geometric
fallback / no-LLM / retry-escalation), clustering, and the decision addendum —
the pure planner logic, so a GPU smoke only has to confirm the env plumbing.
"""
from __future__ import annotations

import sys
import tempfile

from ..p4.stackcube_hybrid import (
    parse_choice, StackCubeFailureDescriptor, cluster_stackcube,
    CubeLayoutSpec, StackCubeHybridPlanner,
)

_fails = []
_passed = 0


def check(name, cond):
    global _passed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _fails.append(name)
        print(f"  FAIL  {name}")


def mk(eid, ax, ay, bx, by, peak, grasp=0.0, t=50, T=100):
    return StackCubeFailureDescriptor(
        episode_id=eid, seed=eid, peak_loss=peak, t_star=t, T=T,
        cubeA_xyz=[ax, ay, 0.02], cubeA_zrot=0.0,
        cubeB_xyz=[bx, by, 0.02], cubeB_zrot=0.0, grasp=grasp,
        cand={"seed": eid, "exec_actions": [0] * (t + 5), "t_star": t, "n_steps": T,
              "peak_disc": peak})


def main() -> int:
    print("== parse_choice ==")
    check("select tag", parse_choice("SELECT ep7: grasp dominates") == ("select", [7]))
    check("bridge tag", parse_choice("BRIDGE ep3,ep9 middle ground") == ("bridge", [3, 9]))
    check("bridge 3 ids cap", parse_choice("BRIDGE ep1,ep2,ep3,ep4")[1] == [1, 2, 3])
    check("no tag", parse_choice("just place cube A near B") == (None, []))
    check("select no-ep", parse_choice("SELECT 5")[0] == "select")

    print("== clustering ==")
    descs = [mk(0, 0.10, 0.10, -0.05, -0.05, 0.9),
             mk(1, 0.11, 0.09, -0.05, -0.05, 0.8),
             mk(2, -0.10, -0.10, 0.05, 0.05, 0.7, grasp=1.0),
             mk(3, -0.11, -0.09, 0.05, 0.05, 0.6, grasp=1.0),
             mk(4, 0.0, 0.20, 0.0, -0.10, 0.5)]
    cr = cluster_stackcube(descs, max_k=4)
    check("clusters non-empty", len(cr.clusters) >= 2)
    check("dominant set", cr.dominant is not None)
    check("all members covered",
          sorted(i for c in cr.clusters for i in c.member_idxs) == [0, 1, 2, 3, 4])

    work = tempfile.mkdtemp(prefix="hybrid_test_")
    print(f"== decide (work_dir={work}) ==")
    pl = StackCubeHybridPlanner(work_dir=work, cfg={"collect": "hybrid", "snap_eps": 0.03})
    pl.set_round(0, descs)

    add = pl.decision_addendum()
    check("addendum non-empty", bool(add))
    check("addendum mentions SELECT/BRIDGE", "SELECT" in add and "BRIDGE" in add and "ep" in add)

    # explicit SELECT of an existing member
    pres = {"cubeA_xyz": [0.10, 0.10, 0.02], "cubeB_xyz": [-0.05, -0.05, 0.02],
            "rationale": "SELECT ep0: representative grasp failure"}
    s = pl.decide(pres, attempt=0)
    check("SELECT → onpolicy_correction", s.mode == "onpolicy_correction")
    check("SELECT → cited episode_id", s.episode_id == 0 and s.seed == 0)
    check("SELECT → cand passed", s.cand is not None and s.cand.get("seed") == 0)

    # explicit BRIDGE → uses the prescribed layout
    pl.set_round(1, descs)
    pres = {"cubeA_xyz": [0.0, 0.0, 0.02], "cubeB_xyz": [0.08, 0.0, 0.02],
            "rationale": "BRIDGE ep0,ep2: span the two grasp regions"}
    s = pl.decide(pres, attempt=0)
    check("BRIDGE → bridge mode", s.mode == "bridge")
    check("BRIDGE → layout carried", s.layout is not None
          and s.layout["cubeA_xyz"][0] == 0.0)
    check("BRIDGE → cited ids", set(s.cited) == {0, 2})

    # no tag, prescribed cubeA ON a member ⇒ geometric SELECT
    pl.set_round(2, descs)
    s = pl.decide({"cubeA_xyz": [0.10, 0.10, 0.02], "cubeB_xyz": [-0.05, -0.05, 0.02],
                   "rationale": "place A in the open"}, attempt=0)
    check("geometric near-member → select", s.mode == "onpolicy_correction")

    # no tag, prescribed cubeA FAR from any member ⇒ geometric BRIDGE
    pl.set_round(3, descs)
    s = pl.decide({"cubeA_xyz": [0.14, -0.24, 0.02], "cubeB_xyz": [0.0, 0.0, 0.02],
                   "rationale": "place A far"}, attempt=0)
    check("geometric far → bridge", s.mode == "bridge")

    # no LLM (pres None) ⇒ faithful SELECT
    pl.set_round(4, descs)
    s = pl.decide(None, attempt=0)
    check("no-LLM → select", s.mode == "onpolicy_correction")

    # retry (attempt>=1) ⇒ escalated SELECT of an untried member
    pl.set_round(5, descs)
    s0 = pl.decide({"cubeA_xyz": [0.0, 0.0, 0.02], "cubeB_xyz": [0.08, 0.0, 0.02],
                    "rationale": "BRIDGE ep0,ep2"}, attempt=0)
    s1 = pl.decide({"cubeA_xyz": [0.0, 0.0, 0.02], "cubeB_xyz": [0.08, 0.0, 0.02],
                    "rationale": "BRIDGE ep0,ep2"}, attempt=1)
    check("retry → escalated_select", s1.mode == "onpolicy_correction"
          and s1.choice == "escalated_select")

    # note_collect runs + records coverage on success (no exceptions)
    try:
        pl.note_collect(s1, {"success": True, "episode_length": 120,
                             "applied": {"mode": "select"}}, attempt=1)
        pl.note_collect(s0, {"success": True, "episode_length": 130,
                             "applied": {"mode": "bridge"}}, attempt=0)
        check("note_collect ok", True)
    except Exception as exc:
        check(f"note_collect ok ({type(exc).__name__}: {exc})", False)

    print("== feasibility gate (off-table cubes can't be SELECT-corrected) ==")
    ond = mk(10, 0.10, 0.10, -0.05, -0.05, 0.90)             # cubeA on table
    off = mk(11, 0.57, 0.25, 0.0, 0.0, 0.95)
    off.cubeA_xyz = [0.57, 0.25, -0.098]                     # cubeA flung off table
    pl.set_round(7, [ond, off])
    s = pl.decide({"cubeA_xyz": [0.57, 0.25, 0.02], "cubeB_xyz": [0.0, 0.0, 0.02],
                   "rationale": "SELECT ep11"}, attempt=0)
    check("explicit SELECT of off-table member is NOT executed",
          not (s.mode == "onpolicy_correction" and s.episode_id == 11))
    check("→ falls back to the on-table member or BRIDGE",
          (s.mode == "onpolicy_correction" and s.episode_id == 10) or s.mode == "bridge")

    off2 = mk(21, -0.7, -0.6, 0.0, 0.0, 0.8)
    off2.cubeA_xyz = [-0.7, -0.6, -0.05]
    pl.set_round(8, [mk(20, 0.6, 0.5, 0.0, 0.0, 0.9), off2])
    for d in pl._descs:
        d.cubeA_xyz[2] = -0.05
    pl.set_round(8, pl._descs)   # re-cluster with both off-table
    s = pl.decide(None, attempt=0)
    check("all off-table, no-LLM → BRIDGE (no grind)", s.mode == "bridge")

    pl.set_round(9, [mk(10, 0.10, 0.10, -0.05, -0.05, 0.90),
                     mk(11, 0.57, 0.25, 0.0, 0.0, 0.95)])
    pl._descs[1].cubeA_xyz = [0.57, 0.25, -0.098]
    s = pl.decide(None, attempt=0)
    check("mixed, no-LLM → SELECTs the on-table member",
          s.mode == "onpolicy_correction" and s.episode_id == 10)

    # empty round ⇒ inert (no crash), decide falls back to bridge layout
    pl.set_round(6, [])
    s = pl.decide({"cubeA_xyz": [0.0, 0.0, 0.02], "cubeB_xyz": [0.08, 0.0, 0.02],
                   "rationale": "no tag"}, attempt=0)
    check("empty round → no crash", isinstance(s, CubeLayoutSpec))

    print(f"\n{_passed} passed, {len(_fails)} failed"
          + (f" → {_fails}" if _fails else ""))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
