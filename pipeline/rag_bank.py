import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_clip_model = None
_clip_processor = None


def _load_clip(model_id: str):
    global _clip_model, _clip_processor
    if _clip_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = _clip_model.to(device).eval()
    return _clip_model, _clip_processor


def _clip_embed(text: str, image_path: Optional[str], model_id: str) -> Optional[np.ndarray]:
    try:
        import torch
        from PIL import Image as PILImage
        model, processor = _load_clip(model_id)
        device = next(model.parameters()).device

        t_enc = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            tv = model.get_text_features(**t_enc)
            tv = torch.nn.functional.normalize(tv, dim=-1)[0].cpu().float().numpy()

        if image_path and os.path.exists(image_path):
            img = PILImage.open(image_path).convert("RGB")
            i_enc = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                iv = model.get_image_features(**i_enc)
                iv = torch.nn.functional.normalize(iv, dim=-1)[0].cpu().float().numpy()
            combined = np.concatenate([tv, iv]).astype("float32")
        else:
            combined = np.concatenate([tv, tv]).astype("float32")

        norm = np.linalg.norm(combined)
        return (combined / (norm + 1e-9)).astype("float32")
    except Exception:
        traceback.print_exc()
        return None


def _normalise_pos(p: Any) -> Optional[Tuple[int, ...]]:
    """Coerce a position-like value into a hashable tuple, or return None."""
    if p is None:
        return None
    try:
        return tuple(int(x) for x in p)
    except Exception:
        return None


def _config_key(start_pos: Any, goal_pos: Any, fire_positions: Any) -> Optional[Tuple]:
    """Build a hashable key from a (start, goal, fires) triple.

    Returns None if all three fields are absent — callers treat that as
    "no usable config to compare against" and fall back to id-only matching.
    Fire positions are sorted so that ordering doesn't affect equality.
    """
    sp = _normalise_pos(start_pos)
    gp = _normalise_pos(goal_pos)
    fp_list = []
    for f in (fire_positions or []):
        nf = _normalise_pos(f)
        if nf is not None:
            fp_list.append(nf)
    fp = tuple(sorted(fp_list))
    if sp is None and gp is None and not fp:
        return None
    return (sp, gp, fp)


def _meta_config_key(m: Dict) -> Optional[Tuple]:
    """Extract a config key from a stored RAG meta entry."""
    return _config_key(
        m.get("start_pos"),
        m.get("goal_pos"),
        m.get("fire_positions"),
    )


class RAGBank:

    def __init__(self, bank_path: str, top_k: int = 3, sim_threshold: float = 0.3, clip_model: str = "openai/clip-vit-large-patch14"):
        self.bank_dir = Path(bank_path)
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self.clip_model = clip_model
        self.index_path = self.bank_dir / "rag_faiss.index"
        self.meta_path  = self.bank_dir / "rag_metadata.json"
        self._index = None
        self._meta: List[Dict] = []
        self._dim: Optional[int] = None
        self._load()

    def _faiss(self):
        import faiss
        return faiss

    def _load(self):
        try:
            faiss = self._faiss()
        except ImportError:
            print("[RAG] faiss-cpu not installed; bank will be in-memory only.")
            return

        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r") as f:
                self._meta = json.load(f)
            self._dim = self._index.d
            print(f"[RAG] Loaded existing bank: {self._index.ntotal} entries, dim={self._dim}")
        else:
            self._index = None
            self._meta = []
            self._dim = None
            print("[RAG] No existing bank found — will initialise on first store.")

    def _ensure_index(self, dim: int):
        faiss = self._faiss()
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
            self._dim = dim
        elif self._index.d != dim:
            print(f"[RAG] Dim mismatch (bank={self._index.d}, new={dim}) — padding/truncating new vector.")

    def _align_vec(self, vec: np.ndarray) -> np.ndarray:
        if self._dim is None:
            return vec
        if vec.shape[0] == self._dim:
            return vec
        if vec.shape[0] < self._dim:
            pad = np.zeros(self._dim - vec.shape[0], dtype=np.float32)
            return np.concatenate([vec, pad]).astype("float32")
        return vec[:self._dim].astype("float32")

    def retrieve(
        self,
        vision_report: str,
        end_frame_path: Optional[str],
        exclude_episode_id: Optional[int] = None,
        exclude_episode_config: Optional[Dict] = None,
    ) -> str:
        """Retrieve top-k similar past failures.

        Self-contamination is prevented by TWO independent filters:

          1. `exclude_episode_id` skips entries whose stored episode_id matches.
             Sufficient within a single ablation-suite run where all profiles
             share rollout indices.

          2. `exclude_episode_config` skips entries whose stored
             (start_pos, goal_pos, fire_positions) matches the current
             episode's config — needed across separate suite invocations,
             where a previous profile run (e.g. p5) may have stored the same
             physical episode under a different run_id but with the SAME
             dynamic config.  Without this, p6's retrieve for ep4 returns
             p5's stored ep4 with sim≈0.95 even though episode_id-only
             filtering passes.

        Both filters are applied; an entry is skipped if EITHER matches.
        """
        if self._index is None or self._index.ntotal == 0 or not self._meta:
            return ""
        qv = _clip_embed(vision_report, end_frame_path, self.clip_model)
        if qv is None:
            return ""
        qv = self._align_vec(qv).reshape(1, -1)
        # Search a wider window so filtering can still leave us with up to
        # top_k useful matches after both filters run.
        k = min(max(self.top_k * 4, self.top_k), self._index.ntotal)
        scores, idxs = self._index.search(qv, k)

        exclude_cfg_key: Optional[Tuple] = None
        if exclude_episode_config:
            exclude_cfg_key = _config_key(
                exclude_episode_config.get("start_pos"),
                exclude_episode_config.get("goal_pos"),
                exclude_episode_config.get("fire_positions"),
            )

        retrieved = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
            if idx < 0:
                continue
            if float(score) < self.sim_threshold:
                continue
            m = self._meta[int(idx)]

            if exclude_episode_id is not None and m.get("episode_id") == exclude_episode_id:
                continue

            if exclude_cfg_key is not None:
                m_key = _meta_config_key(m)
                if m_key is not None and m_key == exclude_cfg_key:
                    continue

            retrieved.append({"rank": rank, "similarity": float(score), **m})
            if len(retrieved) >= self.top_k:
                break

        if not retrieved:
            return ""
        lines = ["=== RAG — SIMILAR PAST FAILURES ==="]
        for r in retrieved:
            lines.append(f"--- rank={r['rank']} sim={r['similarity']:.3f} run={r.get('run_id','?')} ep={r.get('episode_id','?')} ---")
            lines.append(f"  Dynamic config: start={r.get('start_pos','?')} goal={r.get('goal_pos','?')} fires={r.get('fire_positions','?')}")
            lines.append(f"  Summary:        {r.get('summary','?')}")
            lines.append(f"  Root cause:     {r.get('root_cause','?')}")
            lines.append("")
        return "\n".join(lines)

    def store(self, run_id: str, episode: Dict, vision_report: str, prescription: Dict, end_frame_path: Optional[str]):
        qv = _clip_embed(vision_report, end_frame_path, self.clip_model)
        if qv is None:
            print("[RAG] Skip store — embedding failed.")
            return
        self._ensure_index(qv.shape[0])
        qv = self._align_vec(qv).reshape(1, -1)
        self._index.add(qv)

        dyn = episode.get("dynamic_config", {})
        # config_key persisted alongside the entry so future loads can do
        # config-based exclusion without recomputing every time.
        cfg_key = _config_key(
            dyn.get("start_pos"),
            dyn.get("goal_pos"),
            dyn.get("fire_positions"),
        )
        entry = {
            "run_id":         run_id,
            "episode_id":     episode.get("episode_id"),
            "maze_name":      episode.get("maze_name"),
            "seed":           episode.get("seed"),
            "total_steps":    episode.get("total_steps"),
            "total_reward":   episode.get("total_reward"),
            "success":        episode.get("success"),
            "start_pos":      dyn.get("start_pos"),
            "goal_pos":       dyn.get("goal_pos"),
            "fire_positions": dyn.get("fire_positions"),
            "config_key":     list(cfg_key) if cfg_key is not None else None,
            "summary":        prescription.get("summary", ""),
            "root_cause":     prescription.get("root_cause", "pending"),
        }
        self._meta.append(entry)
        self._persist()

    def _persist(self):
        try:
            faiss = self._faiss()
            if self._index is not None:
                faiss.write_index(self._index, str(self.index_path))
            with open(self.meta_path, "w") as f:
                json.dump(self._meta, f, indent=2, default=str)
        except Exception:
            traceback.print_exc()


def save_rag_retrieved(text: str, episode_dir: Path) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)
    out = episode_dir / "rag_retrieved.txt"
    out.write_text(text, encoding="utf-8")
    return out