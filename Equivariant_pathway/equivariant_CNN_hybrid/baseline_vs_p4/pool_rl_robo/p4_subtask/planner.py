"""SubtaskPlanner — the object injected into the fork pipeline.

Ties Fix-1/2/4 together and drives the three default-OFF fork hooks:

  hook A  select_candidates(scored, cap, rnd)
            → build descriptors, cluster, pick the (memory-rotated) target
              cluster, build the anchor, diversity-select the cap-many failures
              the VLM/LLM will analyse.
  hook B  round_context(rnd)
            → text blocks (anchor + centroid + memory + KAG + semantics) appended
              to the reasoning/aggregator prompt addenda.
  hook C  reset_spec_for(cmd, rnd) / note_collect(...)
            → map the LLM's parsed SceneCommand to a sub-task ResetSpec with the
              deterministic escalation perturbed → exact-anchor → fresh-tee, and
              record the collection outcome + memory.

Escalation is keyed off "how many times we re-prescribed within this round": the
first prescription is perturbed (escalation 0); each infeasible/empty re-prescribe
bumps the escalation, so the planner falls back to the safe exact-anchor replay,
then to the old fresh-tee behaviour. With the planner absent the fork is byte-
identical to p4_top3.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .descriptor import load_failure_descriptor, FailureDescriptor
from .clustering import cluster_failures, ClusterResult, Cluster
from .diversity import farthest_point_select
from .memory import CentroidMemory
from .subtask_entry import (
    Anchor, ResetSpec, build_anchor, apply_coverage_perturbation,
    exact_anchor_resetspec, fresh_tee_resetspec, onpolicy_correction_spec,
)
from .telemetry import Telemetry
from . import kag_bounds as KB
from . import semantic_map as SM


class SubtaskPlanner:
    def __init__(self, *, work_dir: str, cfg: Dict[str, Any],
                 kag_json_path: Optional[str] = None):
        self.cfg = dict(cfg or {})
        self.max_k = int(self.cfg.get("max_clusters", 6))
        self.diversity = bool(self.cfg.get("diversity", True))
        self.perturb_max_xy = float(self.cfg.get("perturb_max_xy", 0.06))
        self.perturb_max_theta = float(self.cfg.get("perturb_max_theta", 0.4))
        self.exact_fallback = bool(self.cfg.get("exact_anchor_fallback", True))
        # Collection modes:
        #   "hybrid"   (v3) — the LLM DECIDES each round: SELECT one recorded
        #                failure (→ on-policy correction) or BRIDGE 2-3 failures
        #                with a new middle-ground pose (→ anchored teleport demo).
        #                Supervisor's "LLM recommendation + compression" spec.
        #   "onpolicy" (v2) — always snap to the nearest real failure (pure select).
        #   "teleport" (v1) — always anchored-perturbation reset (ablation).
        self.collect_mode = str(self.cfg.get("collect", "hybrid"))
        self.signal_mode = str(self.cfg.get("signal", "diffusion"))
        # hybrid knobs: a pose within snap_eps of a member counts as SELECTing it;
        # a BRIDGE pose may sit up to bridge_max_* from its nearest cited member.
        self.snap_eps = float(self.cfg.get("snap_eps", 0.02))
        self.bridge_max_xy = float(self.cfg.get("bridge_max_xy", 0.10))
        self.bridge_max_theta = float(self.cfg.get("bridge_max_theta", 0.6))
        # Read by fork hook D: a rollout-SR target hit must be CONFIRMED by the
        # frozen held-out eval before stopping (fixes the 60/60-fluke early stop).
        self.confirm_target_with_heldout = bool(
            self.cfg.get("confirm_target_with_heldout", True))
        self.tele = Telemetry(work_dir)
        self.memory = CentroidMemory(
            str(Path(work_dir) / "telemetry" / "centroid_memory.json"),
            gamma=float(self.cfg.get("memory_gamma", 0.6)),
            sigma=float(self.cfg.get("memory_sigma", 0.06)))
        self.kag_bounds = KB.load_from_kag(kag_json_path) if kag_json_path else KB.as_dict()

        # per-round state
        self._round: Optional[int] = None
        self._escalation: int = 0
        self._descs: List[FailureDescriptor] = []
        self._scored: List[Tuple[float, str, dict]] = []
        self._cr: Optional[ClusterResult] = None
        self._target: Optional[Cluster] = None
        self._anchor: Optional[Anchor] = None
        self._selected_idxs: List[int] = []
        self._ctx: Dict[str, str] = {}
        self._mem_scores: List[dict] = []
        self._tried_seeds: set = set()   # v2: real-failure seeds already attempted this round

    # ── hook A ────────────────────────────────────────────────────────────
    def select_candidates(self, scored: List[Tuple[float, str, dict]],
                          *, cap: int, rnd: int) -> List[Tuple[float, str, dict]]:
        if rnd != self._round:
            self._round = rnd
            self._escalation = 0
            self._compute_round(scored, cap, rnd)
        else:
            self._escalation += 1
            self.tele.event(rnd, "represcribe",
                            {"escalation": self._escalation,
                             "note": "re-prescribe within round (empty/infeasible retry)"})
        # Return the diversity-selected candidates (anchor rep forced first).
        if self._selected_idxs:
            return [self._scored[i] for i in self._selected_idxs]
        # Fallback: plain peak-loss top-cap (planner inert this round).
        s = sorted(scored, key=lambda x: x[0], reverse=True)
        return s[:cap]

    def _compute_round(self, scored, cap, rnd) -> None:
        self.tele.tic("cluster")
        # Build descriptors aligned with a filtered scored list.
        descs: List[FailureDescriptor] = []
        kept: List[Tuple[float, str, dict]] = []
        for (peak, ep_dir, meta) in scored:
            d = load_failure_descriptor(ep_dir, meta)
            if d is not None:
                descs.append(d)
                kept.append((peak, ep_dir, meta))
        self._descs, self._scored = descs, kept
        self._cr = None
        self._target = None
        self._anchor = None
        self._selected_idxs = []
        self._ctx = {}
        self._mem_scores = []
        self._tried_seeds = set()

        self.tele.event(rnd, "failures_in", {
            "n_failures_total": len(scored),
            "n_descriptors": len(descs),
            "descriptors": [d.digest() for d in descs],
            "kag_bounds": self.kag_bounds})

        if not descs:
            self.tele.event(rnd, "no_descriptors",
                            {"note": "no usable failure state frames — planner inert "
                                     "(degrades to p4_top3 fresh-tee)"})
            return

        cr = cluster_failures(descs, max_k=self.max_k)
        self._cr = cr
        # Memory-rotated target cluster (Fix-1 + Suyog's cross-round doubt).
        target, mem_scores = self.memory.select_target(
            cr.clusters, cr.dominant, descs, now=rnd)
        self._target, self._mem_scores = target, mem_scores
        self._anchor = build_anchor(descs[target.representative_idx])

        # Diversity selection (Fix-4): force the anchor rep, then farthest-point.
        if self.diversity:
            sel = farthest_point_select(descs, cap, force_first=target.representative_idx)
        else:
            order = sorted(range(len(descs)), key=lambda i: descs[i].peak_loss, reverse=True)
            if target.representative_idx in order:
                order.remove(target.representative_idx)
            sel = [target.representative_idx] + order
            sel = sel[:max(1, min(cap, len(descs)))]
        self._selected_idxs = sel

        self._ctx = self._build_context(rnd)
        cluster_dump = [{
            "label": c.label, "size": c.size,
            "mean_peak_loss": round(c.mean_peak_loss, 6),
            "centroid_xyz": [round(v, 4) for v in c.centroid_xyz],
            "centroid_theta": round(c.centroid_theta, 4),
            "members_episode_ids": [descs[i].episode_id for i in c.member_idxs],
            "is_target": (c is target),
            "is_dominant": (c is cr.dominant),
        } for c in cr.clusters]
        self.tele.event(rnd, "round_setup", {
            "cluster_method": cr.method,
            "n_clusters": len(cr.clusters),
            "clusters": cluster_dump,
            "dominant_label": cr.dominant.label,
            "target_label": target.label,
            "memory_scores": mem_scores,
            "anchor": self._anchor.digest(),
            "target_centroid": {"xyz": [round(v, 4) for v in target.centroid_xyz],
                                "theta": round(target.centroid_theta, 4)},
            "diversity_selected_episode_ids": [descs[i].episode_id for i in sel],
            "pure_loss_top": [descs[i].episode_id for i in sorted(
                range(len(descs)), key=lambda i: descs[i].peak_loss, reverse=True)[:cap]],
            "cluster_secs": self.tele.toc("cluster"),
        })

    # ── hook B ────────────────────────────────────────────────────────────
    def round_context(self, rnd: int) -> Dict[str, str]:
        return self._ctx or {}

    def _build_context(self, rnd: int) -> Dict[str, str]:
        a, t = self._anchor, self._target
        if a is None or t is None:
            return {}
        covered = [f"({round(e['centroid_xyz'][0],3)},{round(e['centroid_xyz'][1],3)})"
                   for e in self.memory.entries[-8:]]
        sem = "\n".join("    - " + s for s in SM.semantic_lines(a.tee_xyz, t.centroid_xyz))
        if self.collect_mode in ("onpolicy", "hybrid"):
            descs = self._descs
            members = "\n".join(
                f"    - ep{descs[i].episode_id}: tee_xyz="
                f"{[round(v,3) for v in descs[i].tee_xyz]}, "
                f"tee_zrot={round(descs[i].tee_theta,3)} rad, "
                f"peak_loss={round(descs[i].peak_loss,5)}, t*={descs[i].t_star}/{descs[i].T}"
                for i in t.member_idxs[:8])
        if self.collect_mode == "hybrid":
            block = (
                "DEMONSTRATION DECISION CONTEXT (code-computed, authoritative — do "
                "NOT recompute):\n"
                "  You decide HOW the next demonstration is collected. You have TWO "
                "options — choose whichever closes MORE failures per demo:\n"
                "  (A) SELECT — one recorded failure is representative enough: that "
                "exact failed episode is re-run and the expert corrects it on-policy "
                "from the divergence point. Use when the cluster is tight or one "
                "failure clearly dominates.\n"
                "  (B) BRIDGE — no single failure covers the mode: prescribe ONE new "
                "scene pose in the MIDDLE GROUND between 2-3 cited failures (e.g. "
                "failures at (1,1) and (5,5) → bridge near (3,3)); the expert "
                "demonstrates from that pose with the arm starting at the nearest "
                "cited failure's recorded configuration. Use when members are "
                "geometrically spread but share a root cause.\n"
                f"  - Target failure cluster: label {t.label}, {t.size} member(s), "
                f"mean_peak_loss={round(t.mean_peak_loss,5)}.\n"
                f"  - Cluster members (recorded failures):\n{members}\n"
                f"  - Cluster coverage centroid: tee_xyz="
                f"{[round(v,3) for v in t.centroid_xyz]}, "
                f"tee_zrot={round(t.centroid_theta,3)} rad.\n"
                f"  - Grounded geometry:\n{sem}\n"
                f"  - Already-covered regions (recent rounds, prefer NEW coverage): "
                f"{covered or 'none'}.\n"
                "  OUTPUT CONTRACT (hard): emit exactly ONE scene configuration. Its "
                "label field MUST start with either 'SELECT ep<ID>' (and tee_xyz/"
                "tee_zrot = that member's pose) or 'BRIDGE ep<ID>,ep<ID>[,ep<ID>]' "
                "(and tee_xyz/tee_zrot = your middle-ground pose, which must lie "
                f"within {self.bridge_max_xy} m / {self.bridge_max_theta} rad of the "
                "nearest cited member). State WHY in the rationale.\n"
                f"  HARD bounds: tee_x{list(KB.TEE_X)} tee_y{list(KB.TEE_Y)} "
                f"tee_z={KB.TEE_Z}; tcp_x{list(KB.TCP_X)} tcp_y{list(KB.TCP_Y)} "
                f"tcp_z{list(KB.TCP_Z)}."
            )
            return {"reasoning_block": block, "aggregator_block": block}
        if self.collect_mode == "onpolicy":
            block = (
                "ON-POLICY CORRECTION CONTEXT (code-computed, authoritative — do NOT "
                "recompute):\n"
                "  This round, the demonstration will be a Diff-DAgger-style ON-POLICY "
                "expert correction: the chosen REAL failed episode is re-run with the "
                "current policy and the expert takes over from the policy's own "
                "divergence state and finishes the task. You choose WHICH recorded "
                "failure gets corrected — you do NOT invent a new scene.\n"
                f"  - Target failure cluster: label {t.label}, {t.size} member(s), "
                f"mean_peak_loss={round(t.mean_peak_loss,5)}.\n"
                f"  - Cluster members (real recorded failures you may select):\n{members}\n"
                f"  - Cluster coverage centroid: tee_xyz="
                f"{[round(v,3) for v in t.centroid_xyz]}, "
                f"tee_zrot={round(t.centroid_theta,3)} rad.\n"
                f"  - Grounded geometry:\n{sem}\n"
                f"  - Already-corrected regions (recent rounds, prefer NEW coverage): "
                f"{covered or 'none'}.\n"
                "  PRESCRIBE exactly ONE scene configuration: emit the tee_xyz + "
                "tee_zrot of the SINGLE recorded failure (from the member list above) "
                "whose correction closes the most impactful failure mode this round. "
                "Your output is snapped to the nearest recorded failure — poses far "
                "from every member are wasted.\n"
                f"  HARD bounds: tee_x{list(KB.TEE_X)} tee_y{list(KB.TEE_Y)} "
                f"tee_z={KB.TEE_Z}; tcp_x{list(KB.TCP_X)} tcp_y{list(KB.TCP_Y)} "
                f"tcp_z{list(KB.TCP_Z)}."
            )
            return {"reasoning_block": block, "aggregator_block": block}
        block = (
            "SUB-TASK ENTRY CONTEXT (code-computed, authoritative — do NOT recompute):\n"
            f"  This round targets ONE dominant failure mode. The expert will be started "
            f"MID-TASK at a real recorded failure state (the ANCHOR); you only choose WHERE "
            f"to place the movable T so the demo covers this failure cluster.\n"
            f"  - Target failure cluster: label {t.label}, {t.size} member(s), "
            f"mean_peak_loss={round(t.mean_peak_loss,5)}.\n"
            f"  - ANCHOR (a real failure at its high-loss moment t*={a.t_star}/{a.T}): "
            f"tee_xyz={[round(v,3) for v in a.tee_xyz]}, tee_zrot={round(a.tee_theta,3)} rad, "
            f"tcp_xyz={[round(v,3) for v in a.tcp_xyz]} (the arm starts here automatically).\n"
            f"  - Cluster coverage centroid (aim here): "
            f"tee_xyz={[round(v,3) for v in t.centroid_xyz]}, "
            f"tee_zrot={round(t.centroid_theta,3)} rad.\n"
            f"  - Grounded geometry:\n{sem}\n"
            f"  - Already-covered regions (recent rounds, prefer NEW coverage): "
            f"{covered or 'none'}.\n"
            f"  PRESCRIBE exactly ONE scene configuration: a single absolute, in-bounds "
            f"tee_xyz + tee_zrot placed BETWEEN the anchor and the centroid (it will be "
            f"clamped to within {self.perturb_max_xy} m / {self.perturb_max_theta} rad of the "
            f"anchor so the demo stays at the failure sub-task). The robot arm is placed at "
            f"the anchor contact automatically — you do NOT need to re-demonstrate the "
            f"approach.\n"
            f"  HARD bounds: tee_x{list(KB.TEE_X)} tee_y{list(KB.TEE_Y)} tee_z={KB.TEE_Z}; "
            f"tcp_x{list(KB.TCP_X)} tcp_y{list(KB.TCP_Y)} tcp_z{list(KB.TCP_Z)}."
        )
        return {"reasoning_block": block, "aggregator_block": block}

    # ── hook C ────────────────────────────────────────────────────────────
    @staticmethod
    def _pose_dist(d: FailureDescriptor, x: float, y: float, theta: float) -> float:
        """Snap metric: planar distance + a small orientation term (0.1 m/rad)."""
        dth = abs((d.tee_theta - theta + math.pi) % (2.0 * math.pi) - math.pi)
        return math.hypot(d.tee_xyz[0] - x, d.tee_xyz[1] - y) + 0.1 * dth

    def _pick_member_for_correction(self, llm_tee, llm_zrot) -> Optional[FailureDescriptor]:
        """v2 selection-by-pointing: snap the LLM's pose to the nearest UNTRIED real
        failure in the target cluster; escalate to other clusters (size desc,
        peak-loss desc) when the target pool is exhausted. Never invents a state."""
        descs, t = self._descs, self._target
        if not descs:
            return None
        x, y, th = float(llm_tee[0]), float(llm_tee[1]), float(llm_zrot)
        order: List[int] = []
        if t is not None:
            order += sorted(t.member_idxs, key=lambda i: self._pose_dist(descs[i], x, y, th))
        if self._cr is not None:
            others = sorted((c for c in self._cr.clusters if c is not t),
                            key=lambda c: (-c.size, -c.mean_peak_loss))
            for c in others:
                order += sorted(c.member_idxs,
                                key=lambda i: -descs[i].peak_loss)
        for i in order:
            if descs[i].seed not in self._tried_seeds:
                return descs[i]
        # pool exhausted → repeat the overall worst failure (infeasible cap bounds us)
        return max(descs, key=lambda d: d.peak_loss)

    @staticmethod
    def _parse_choice(label: str):
        """Parse the LLM's explicit choice from the scene-config label.
        Returns ("select", [id]) | ("bridge", [ids]) | (None, [])."""
        import re
        s = str(label or "")
        m = re.search(r"\bSELECT\s*[: ]\s*(?:ep\s*)?(\d+)", s, re.IGNORECASE)
        if m:
            return "select", [int(m.group(1))]
        m = re.search(r"\bBRIDGE\s*[: ]\s*((?:(?:ep\s*)?\d+[,\s]*)+)", s, re.IGNORECASE)
        if m:
            ids = [int(x) for x in re.findall(r"\d+", m.group(1))]
            return ("bridge", ids[:3]) if ids else (None, [])
        return None, []

    def _member_by_episode_id(self, eid: int) -> Optional[FailureDescriptor]:
        for d in self._descs:
            if d.episode_id == int(eid):
                return d
        return None

    def _hybrid_spec(self, cmd, llm_tee, llm_zrot, llm_tcp) -> ResetSpec:
        """v3: honor the LLM's per-round SELECT/BRIDGE decision (esc 0); any
        retry escalates to SELECT of the nearest untried real failure (safety
        floor — a bad bridge can never waste the round)."""
        choice, ids = self._parse_choice(getattr(cmd, "label", ""))
        x, y, th = float(llm_tee[0]), float(llm_tee[1]), float(llm_zrot)

        if self._escalation >= 1 or not self._descs:
            member = self._pick_member_for_correction(llm_tee, llm_zrot)
            if member is None:
                return fresh_tee_resetspec(llm_tee, llm_zrot, llm_tcp)
            self._tried_seeds.add(member.seed)
            spec = onpolicy_correction_spec(member)
            spec.provenance["llm_choice"] = choice or "none"
            spec.provenance["escalated_to_select"] = True
            return spec

        # geometric fallback when no explicit tag: very close to a member ⇒ select
        if choice is None:
            nearest = min(self._descs, key=lambda d: self._pose_dist(d, x, y, th))
            choice = ("select" if self._pose_dist(nearest, x, y, th) <= self.snap_eps
                      else "bridge")
            ids = [nearest.episode_id]

        if choice == "select":
            member = (self._member_by_episode_id(ids[0]) if ids else None)
            if member is None or member.seed in self._tried_seeds:
                member = self._pick_member_for_correction(llm_tee, llm_zrot)
            if member is None:
                return fresh_tee_resetspec(llm_tee, llm_zrot, llm_tcp)
            self._tried_seeds.add(member.seed)
            spec = onpolicy_correction_spec(member)
            spec.provenance["llm_choice"] = "select"
            return spec

        # BRIDGE: a new middle-ground pose covering 2-3 cited failures, anchored
        # on the nearest cited member (its recorded arm config + bounded shift —
        # the v1 mechanism, now used ONLY when the LLM judges it worthwhile).
        cited = [m for m in (self._member_by_episode_id(i) for i in ids) if m is not None]
        if not cited:
            cited = self._descs
        anchor_member = min(cited, key=lambda d: self._pose_dist(d, x, y, th))
        spec = apply_coverage_perturbation(
            build_anchor(anchor_member), llm_tee, llm_zrot,
            max_xy=self.bridge_max_xy, max_theta=self.bridge_max_theta)
        spec.mode = "bridge"
        spec.provenance["llm_choice"] = "bridge"
        spec.provenance["cited_episode_ids"] = [m.episode_id for m in cited]
        spec.provenance["anchor_episode_id"] = anchor_member.episode_id
        return spec

    def reset_spec_for(self, cmd, *, rnd: int) -> ResetSpec:
        llm_tee = list(getattr(cmd, "tee_xyz", []) or [0.0, 0.0, KB.TEE_Z])
        llm_zrot = float(getattr(cmd, "tee_zrot", 0.0) or 0.0)
        llm_tcp = list(getattr(cmd, "tcp_xyz", []) or [0.0, 0.0, 0.04])
        esc = self._escalation

        if self.collect_mode == "hybrid":
            spec = self._hybrid_spec(cmd, llm_tee, llm_zrot, llm_tcp)
        elif self.collect_mode == "onpolicy":
            member = self._pick_member_for_correction(llm_tee, llm_zrot)
            if member is not None:
                self._tried_seeds.add(member.seed)
                spec = onpolicy_correction_spec(member)
                spec.provenance["snap_dist"] = round(
                    self._pose_dist(member, llm_tee[0], llm_tee[1], llm_zrot), 4)
            else:
                # no usable failure descriptors at all → legacy fresh-tee
                spec = fresh_tee_resetspec(llm_tee, llm_zrot, llm_tcp)
        else:
            # v1 (teleport) path, kept for ablations
            a = self._anchor
            if a is None:
                spec = fresh_tee_resetspec(llm_tee, llm_zrot, llm_tcp)
            elif esc <= 0:
                spec = apply_coverage_perturbation(
                    a, llm_tee, llm_zrot,
                    max_xy=self.perturb_max_xy, max_theta=self.perturb_max_theta)
            elif esc == 1 and self.exact_fallback:
                spec = exact_anchor_resetspec(a)
            else:
                spec = fresh_tee_resetspec(llm_tee, llm_zrot, llm_tcp)

        self.tele.event(rnd, "reset_spec", {
            "escalation": esc, "collect_mode": self.collect_mode,
            "llm_cmd": {"tee_xyz": [round(float(v), 4) for v in llm_tee[:3]],
                        "tee_zrot": round(llm_zrot, 4),
                        "tcp_xyz": [round(float(v), 4) for v in llm_tcp[:3]],
                        "label": getattr(cmd, "label", "")},
            "spec": spec.digest()})
        return spec

    def note_collect(self, cmd, spec: ResetSpec, result: Dict[str, Any], *, rnd: int) -> None:
        success = bool(result.get("success"))
        tee_t = result.get("target_tee_xyz")
        tee_a = result.get("achieved_tee_xyz")
        tee_err = None
        if tee_t and tee_a:
            tee_err = round(math.dist(tee_t[:2], tee_a[:2]), 5)
        qpos_t = result.get("target_agent_qpos")
        qpos_a = result.get("achieved_agent_qpos")
        qpos_err = None
        if qpos_t and qpos_a:
            n = min(len(qpos_t), len(qpos_a))
            qpos_err = round(math.sqrt(sum((qpos_t[i] - qpos_a[i]) ** 2
                                           for i in range(n))), 5)
        self.tele.event(rnd, "collect_outcome", {
            "mode": spec.mode, "escalation": self._escalation,
            "success": success, "episode_length": result.get("episode_length"),
            "skipped": result.get("skipped"),
            "scene_seed": result.get("scene_seed"),
            "t_star_reroll": result.get("t_star"),
            "reroll_steps": result.get("reroll_steps"),
            "peak_signal_reroll": result.get("peak_signal"),
            "target_tee_xyz": tee_t, "achieved_tee_xyz": tee_a,
            "tee_xy_regression_err": tee_err,
            "agent_qpos_set": qpos_t is not None,
            "agent_qpos_regression_err": qpos_err})

        if success:
            # Record the covered coverage point so future rounds rotate away.
            self.memory.append(rnd, spec.tee_xyz, spec.tee_zrot)
            tgt = self._target
            self.tele.summary_row({
                "round": rnd, "mode": spec.mode, "escalation": self._escalation,
                "target_cluster": (tgt.label if tgt else None),
                "target_size": (tgt.size if tgt else None),
                "covered_tee_xyz": [round(v, 4) for v in spec.tee_xyz],
                "scene_seed": result.get("scene_seed"),
                "t_star_reroll": result.get("t_star"),
                "reroll_steps": result.get("reroll_steps"),
                "tee_xy_regression_err": tee_err,
                "agent_qpos_regression_err": qpos_err,
                "episode_length": result.get("episode_length")})
