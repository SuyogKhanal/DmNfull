import json
import shutil
from pathlib import Path
from typing import Optional


class ResponseCache:

    def __init__(self, cache_root, force_rerun: bool = False):
        self._root = Path(cache_root)
        if force_rerun and self._root.exists():
            shutil.rmtree(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        if not force_rerun:
            existing = sum(1 for _ in self._root.rglob("*.json"))
            if existing:
                print(f"[ResponseCache] Loaded existing cache at {self._root} ({existing} entries).")

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    @property
    def root(self) -> Path:
        return self._root

    def episode_scope(self, episode_id: int) -> str:
        return f"episode_{int(episode_id)}"

    def run_scope(self) -> str:
        return "run_level"

    def _path(self, scope: str, key: str) -> Path:
        safe_scope = str(scope).replace("/", "_").replace("\\", "_")
        safe_key   = str(key).replace("/", "_").replace("\\", "_")
        return self._root / safe_scope / f"{safe_key}.json"

    def save(self, scope: str, key: str, data: dict) -> None:
        p = self._path(scope, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, scope: str, key: str) -> Optional[dict]:
        p = self._path(scope, key)
        if p.exists():
            try:
                with open(p, "r") as f:
                    val = json.load(f)
                self._hits += 1
                return val
            except Exception:
                self._misses += 1
                return None
        self._misses += 1
        return None