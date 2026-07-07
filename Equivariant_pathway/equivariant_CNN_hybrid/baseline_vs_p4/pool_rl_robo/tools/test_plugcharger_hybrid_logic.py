"""Offline unit test for the PlugCharger V3-hybrid decision logic (no GPU/LLM/env).

Run from the DmNfull repo root:
  python -m Equivariant_pathway.equivariant_CNN_hybrid.baseline_vs_p4.pool_rl_robo.tools.test_plugcharger_hybrid_logic

Validates: SELECT/BRIDGE parsing, decide() dispatch (select / bridge / geometric
fallback / no-LLM / retry-escalation), clustering, the decision addendum, the
charger prescriber's validate, and the BRIDGE-layout → set_prescription kwargs
mapping — the pure planner logic, so a GPU smoke only has to confirm env plumbing.

Mirrors tools/test_hybrid_logic.py (StackCube) using charger/receptacle geometry.
"""
from __future__ import annotations

import sys
import tempfile

from ..p4.plugcharger_hybrid import (
    parse_choice, PlugChargerFailureDescriptor, cluster_plugcharger,
    PlugChargerLayoutSpec, PlugChargerHybridPlanner,
    _charger_validate, charger_prescription_prompt,
    CHARGER_X, CHARGER_Y, CHARGER_Z, RECEP_X, RECEP_Y, RECEP_Z,
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


def mk(eid, cx, cy, rx, ry, peak, grasp=0.0, t=50, T=100):
    """Build a synthetic descriptor. charger at (cx,cy), receptacle at (rx,ry)."""
    return PlugChargerFailureDescriptor(
        episode_id=eid, seed=eid, peak_loss=peak, t_star=t, T=T,
        charger_xyz=[cx, cy, CHARGER_Z], charger_zrot=0.0,
        receptacle_xyz=[rx, ry, RECEP_Z], receptacle_zrot=3.14159, grasp=grasp,
        cand={"seed": eid, "exec_actions": [0] * (t + 5), "t_star": t, "n_steps": T,
              "peak_disc": peak})


def main() -> int:
    print("== parse_choice ==")
    check("select tag", parse_choice("SELECT ep7: grasp dominates") == ("select", [7]))
    check("bridge tag", parse_choice("BRIDGE ep3,ep9 middle ground") == ("bridge", [3, 9]))
    check("bridge 3 ids cap", parse_choice("BRIDGE ep1,ep2,ep3,ep4")[1] == [1, 2, 3])
    check("no tag", parse_choice("just place the charger near the socket") == (None, []))
    check("select no-ep", parse_choice("SELECT 5")[0] == "select")

    print("== clustering ==")
    # Two groups: chargers near top of box / near bottom of box, plus an outlier.
    descs = [mk(0, -0.05, 0.15, 0.05, 0.05, 0.9),
             mk(1, -0.06, 0.14, 0.05, 0.05, 0.8),
             mk(2, -0.05, -0.15, 0.05, -0.05, 0.7, grasp=1.0),
             mk(3, -0.06, -0.14, 0.05, -0.05, 0.6, grasp=1.0),
             mk(4, -0.09, 0.0, 0.02, 0.0, 0.5)]
    cr = cluster_plugcharger(descs, max_k=4)
    check("clusters non-empty", len(cr.clusters) >= 2)
    check("dominant set", cr.dominant is not None)
    check("all members covered",
          sorted(i for c in cr.clusters for i in c.member_idxs) == [0, 1, 2, 3, 4])

    work = tempfile.mkdtemp(prefix="pc_hybrid_test_")
    print(f"== decide (work_dir={work}) ==")
    pl = PlugChargerHybridPlanner(work_dir=work, cfg={"collect": "hybrid", "snap_eps": 0.02})
    pl.set_round(0, descs)

    add = pl.decision_addendum()
    check("addendum non-empty", bool(add))
    check("addendum mentions SELECT/BRIDGE",
          "SELECT" in add and "BRIDGE" in add and "ep" in add)
    check("addendum mentions charger+receptacle",
          "charger" in add and "receptacle" in add)

    # explicit SELECT of an existing member
    pres = {"charger_xyz": [-0.05, 0.15, CHARGER_Z], "receptacle_xyz": [0.05, 0.05, RECEP_Z],
            "rationale": "SELECT ep0: representative grasp failure"}
    s = pl.decide(pres, attempt=0)
    check("SELECT → onpolicy_correction", s.mode == "onpolicy_correction")
    check("SELECT → cited episode_id", s.episode_id == 0 and s.seed == 0)
    check("SELECT → cand passed", s.cand is not None and s.cand.get("seed") == 0)

    # explicit BRIDGE → uses the prescribed layout
    pl.set_round(1, descs)
    pres = {"charger_xyz": [-0.05, 0.0, CHARGER_Z], "receptacle_xyz": [0.05, 0.0, RECEP_Z],
            "rationale": "BRIDGE ep0,ep2: span the two charger regions"}
    s = pl.decide(pres, attempt=0)
    check("BRIDGE → bridge mode", s.mode == "bridge")
    check("BRIDGE → layout carried", s.layout is not None
          and abs(s.layout["charger_xyz"][1] - 0.0) < 1e-9)
    check("BRIDGE → cited ids", set(s.cited) == {0, 2})

    # BRIDGE layout → set_prescription kwargs mapping (the engine calls this)
    kwargs = dict(charger_xyz=s.layout["charger_xyz"],
                  receptacle_xyz=s.layout["receptacle_xyz"],
                  charger_zrot=s.layout.get("charger_zrot"),
                  receptacle_zrot=s.layout.get("receptacle_zrot"))
    check("BRIDGE layout has set_prescription kwargs",
          set(kwargs.keys()) == {"charger_xyz", "receptacle_xyz",
                                 "charger_zrot", "receptacle_zrot"})
    check("BRIDGE layout xyz are 3-vectors",
          len(kwargs["charger_xyz"]) == 3 and len(kwargs["receptacle_xyz"]) == 3)

    # no tag, prescribed charger ON a member ⇒ geometric SELECT
    pl.set_round(2, descs)
    s = pl.decide({"charger_xyz": [-0.05, 0.15, CHARGER_Z],
                   "receptacle_xyz": [0.05, 0.05, RECEP_Z],
                   "rationale": "place the charger near the socket"}, attempt=0)
    check("geometric near-member → select", s.mode == "onpolicy_correction")

    # no tag, prescribed charger FAR from any member ⇒ geometric BRIDGE
    pl.set_round(3, descs)
    s = pl.decide({"charger_xyz": [-0.10, -0.20, CHARGER_Z],
                   "receptacle_xyz": [0.10, 0.10, RECEP_Z],
                   "rationale": "place the charger far"}, attempt=0)
    check("geometric far → bridge", s.mode == "bridge")

    # no LLM (pres None) ⇒ faithful SELECT
    pl.set_round(4, descs)
    s = pl.decide(None, attempt=0)
    check("no-LLM → select", s.mode == "onpolicy_correction")

    # retry (attempt>=1) ⇒ escalated SELECT of an untried member
    pl.set_round(5, descs)
    s0 = pl.decide({"charger_xyz": [-0.05, 0.0, CHARGER_Z],
                    "receptacle_xyz": [0.05, 0.0, RECEP_Z],
                    "rationale": "BRIDGE ep0,ep2"}, attempt=0)
    s1 = pl.decide({"charger_xyz": [-0.05, 0.0, CHARGER_Z],
                    "receptacle_xyz": [0.05, 0.0, RECEP_Z],
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

    print("== feasibility gate (off-workspace chargers can't be SELECT-corrected) ==")
    ond = mk(10, -0.05, 0.15, 0.05, 0.05, 0.90)            # charger in workspace
    off = mk(11, 0.57, 0.25, 0.05, 0.05, 0.95)
    off.charger_xyz = [0.57, 0.25, -0.098]                 # charger flung off table
    pl.set_round(7, [ond, off])
    s = pl.decide({"charger_xyz": [0.57, 0.25, CHARGER_Z],
                   "receptacle_xyz": [0.05, 0.05, RECEP_Z],
                   "rationale": "SELECT ep11"}, attempt=0)
    check("explicit SELECT of off-workspace member is NOT executed",
          not (s.mode == "onpolicy_correction" and s.episode_id == 11))
    check("→ falls back to the in-workspace member or BRIDGE",
          (s.mode == "onpolicy_correction" and s.episode_id == 10) or s.mode == "bridge")

    off2 = mk(21, -0.7, -0.6, 0.05, 0.05, 0.8)
    off2.charger_xyz = [-0.7, -0.6, -0.05]
    pl.set_round(8, [mk(20, 0.6, 0.5, 0.05, 0.05, 0.9), off2])
    for d in pl._descs:
        d.charger_xyz[2] = -0.05
    pl.set_round(8, pl._descs)   # re-cluster with both off-workspace
    s = pl.decide(None, attempt=0)
    check("all off-workspace, no-LLM → BRIDGE (no grind)", s.mode == "bridge")

    pl.set_round(9, [mk(10, -0.05, 0.15, 0.05, 0.05, 0.90),
                     mk(11, 0.57, 0.25, 0.05, 0.05, 0.95)])
    pl._descs[1].charger_xyz = [0.57, 0.25, -0.098]
    s = pl.decide(None, attempt=0)
    check("mixed, no-LLM → SELECTs the in-workspace member",
          s.mode == "onpolicy_correction" and s.episode_id == 10)

    # empty round ⇒ inert (no crash), decide falls back to bridge layout
    pl.set_round(6, [])
    s = pl.decide({"charger_xyz": [-0.05, 0.0, CHARGER_Z],
                   "receptacle_xyz": [0.05, 0.0, RECEP_Z],
                   "rationale": "no tag"}, attempt=0)
    check("empty round → no crash", isinstance(s, PlugChargerLayoutSpec))

    print("== charger prescriber validate ==")
    ok = _charger_validate({"charger_xyz": [-0.05, 0.1, CHARGER_Z],
                            "receptacle_xyz": [0.05, 0.0, RECEP_Z],
                            "charger_zrot": 0.2, "receptacle_zrot": None,
                            "rationale": "x"})
    check("validate keeps charger/receptacle xyz",
          ok["charger_xyz"] == [-0.05, 0.1, CHARGER_Z]
          and ok["receptacle_xyz"] == [0.05, 0.0, RECEP_Z])
    check("validate optf zrot", ok["charger_zrot"] == 0.2 and ok["receptacle_zrot"] is None)
    try:
        _charger_validate({"charger_xyz": [0.0, 0.0]})   # missing receptacle + bad len
        check("validate rejects malformed", False)
    except Exception:
        check("validate rejects malformed", True)

    prompt = charger_prescription_prompt("PlugCharger", "(kag)", "[]", "(none)")
    check("prompt requests charger/receptacle JSON",
          "charger_xyz" in prompt and "receptacle_xyz" in prompt)

    print(f"\n{_passed} passed, {len(_fails)} failed"
          + (f" → {_fails}" if _fails else ""))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
