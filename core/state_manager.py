# core/state_manager.py
import json
from pathlib import Path
from typing import Dict, Any


class StateManager:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

    # ---------- Persona ----------
    def load_persona(self) -> Dict[str, Any]:
        return {
            "immutable": self._read("persona/immutable.json", default={}),
            "stable": self._read("persona/stable.json", default={}),
            "dynamic": self._read("persona/dynamic.json", default={}),
        }

    def save_persona(self, persona: Dict[str, Any]) -> None:
        # Save only stable + dynamic (immutable is not meant to change)
        self._write("persona/stable.json", persona.get("stable", {}))
        self._write("persona/dynamic.json", persona.get("dynamic", {}))

    # ---------- State ----------
    def load_vectors(self) -> Dict[str, Any]:
        return self._read("state/vectors.json", default={})

    def save_vectors(self, vectors: Dict[str, Any]) -> None:
        self._write("state/vectors.json", vectors)

    def load_counters(self) -> Dict[str, Any]:
        return self._read("state/counters.json", default={"current_turn": 0})

    def save_counters(self, counters: Dict[str, Any]) -> None:
        self._write("state/counters.json", counters)

    def get_current_turn(self) -> int:
        counters = self.load_counters()
        return int(counters.get("current_turn", 0))

    # ---------- Stable-trait gating (every N turns) ----------
    def can_update_stable_traits(self, current_turn: int, every_n_turns: int = 20) -> bool:
        """
        Returns True if stable traits are allowed to change now (turn-gated).
        This does NOT perform the update; it just checks gating.
        """
        counters = self.load_counters()
        last = int(counters.get("last_stable_traits_update_turn", 0))
        return (current_turn - last) >= every_n_turns

    def mark_stable_traits_updated(self, current_turn: int) -> None:
        """
        Call this after you actually apply stable trait creation/update.
        """
        counters = self.load_counters()
        counters["last_stable_traits_update_turn"] = int(current_turn)
        self.save_counters(counters)

    # ---------- Helpers ----------
    def _read(self, rel: str, default: Dict[str, Any]) -> Dict[str, Any]:
        path = self.root / rel
        if not path.exists():
            return default
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except Exception:
            return default

    def _write(self, rel: str, data: Dict[str, Any]) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")