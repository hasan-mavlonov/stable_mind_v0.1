# core/state_manager.py
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class StateManager:
    """
    Adds persistence helpers for the perception buffer JSONL.
    This prevents losing buffered perceptions if the process stops.
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

        # Canonical buffer path (you can rename if you already use something else)
        self.perception_buffer_path = self.root / "memory" / "perception_buffer.jsonl"
        self.perception_buffer_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- Persona ----------
    def load_persona(self) -> Dict[str, Any]:
        return {
            "immutable": self._read("persona/immutable.json", default={}),
            "stable": self._read("persona/stable.json", default={}),
            "dynamic": self._read("persona/dynamic.json", default={}),
        }

    def save_persona(self, persona: Dict[str, Any]) -> None:
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

    # ---------- Buffer (JSONL) ----------
    def append_perceptions_to_buffer(self, perceptions: List[Dict[str, Any]], turn: int, session_id: str) -> int:
        """
        Append a batch of normalized perception entities to a persistent JSONL buffer.
        Returns number of items appended.
        """
        if not perceptions:
            return 0

        appended = 0
        with self.perception_buffer_path.open("a", encoding="utf-8") as f:
            for p in perceptions:
                if not isinstance(p, dict):
                    continue
                rec = {
                    "turn": int(turn),
                    "session_id": session_id,
                    "entity_type": p.get("entity_type"),
                    "entity": p.get("entity"),
                    "dimension_values": p.get("dimension_values", {}),
                    "confidence": p.get("confidence", 0.5),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                appended += 1

        return appended

    def read_buffered_perceptions(
        self,
        min_turn_exclusive: int,
        max_turn_inclusive: int,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read buffered perceptions in (min_turn_exclusive, max_turn_inclusive] for a session.
        Returns in the normalized format expected by ConsolidationEngine.
        """
        if not self.perception_buffer_path.exists():
            return []

        out: List[Dict[str, Any]] = []
        min_t = int(min_turn_exclusive)
        max_t = int(max_turn_inclusive)

        with self.perception_buffer_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                try:
                    t = int(rec.get("turn", -1))
                except Exception:
                    continue

                if t <= min_t or t > max_t:
                    continue

                if session_id is not None and rec.get("session_id") != session_id:
                    continue

                # Convert buffer record to consolidation expected format
                et = rec.get("entity_type")
                ent = rec.get("entity")
                dv = rec.get("dimension_values", {})
                conf = rec.get("confidence", 0.5)

                if not isinstance(et, str) or not isinstance(ent, str) or not isinstance(dv, dict):
                    continue

                out.append({
                    "entity_type": et,
                    "entity": ent,
                    "dimension_values": dv,
                    "confidence": float(conf) if isinstance(conf, (int, float, str)) else 0.5
                })

        return out

    def vacuum_buffer_keep_recent(self, keep_turn_greater_than: int, max_lines_keep: int = 5000) -> None:
        """
        Keep buffer bounded:
        - keep only records where turn > keep_turn_greater_than
        - additionally keep at most last max_lines_keep lines after filtering
        """
        if not self.perception_buffer_path.exists():
            return

        keep_t = int(keep_turn_greater_than)

        kept_lines: List[str] = []
        with self.perception_buffer_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    rec = json.loads(line_s)
                    t = int(rec.get("turn", -1))
                except Exception:
                    continue
                if t > keep_t:
                    kept_lines.append(line_s)

        if len(kept_lines) > int(max_lines_keep):
            kept_lines = kept_lines[-int(max_lines_keep):]

        with self.perception_buffer_path.open("w", encoding="utf-8") as f:
            for line_s in kept_lines:
                f.write(line_s + "\n")

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