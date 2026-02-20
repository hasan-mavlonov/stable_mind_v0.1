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
            "immutable": self._read("persona/immutable.json"),
            "stable": self._read("persona/stable.json"),
            "dynamic": self._read("persona/dynamic.json"),
        }

    def save_persona(self, persona: Dict[str, Any]) -> None:
        self._write("persona/stable.json", persona["stable"])
        self._write("persona/dynamic.json", persona["dynamic"])

    # ---------- State ----------
    def load_vectors(self) -> Dict[str, Any]:
        return self._read("state/vectors.json")

    def save_vectors(self, vectors: Dict[str, Any]) -> None:
        self._write("state/vectors.json", vectors)

    def load_counters(self) -> Dict[str, Any]:
        return self._read("state/counters.json")

    def save_counters(self, counters: Dict[str, Any]) -> None:
        self._write("state/counters.json", counters)

    # ---------- Helpers ----------
    def _read(self, rel: str) -> Dict[str, Any]:
        path = self.root / rel
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, rel: str, data: Dict[str, Any]) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")