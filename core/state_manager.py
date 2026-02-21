# core/state_manager.py
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class StateManager:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

        # Buffer file (append-only)
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
        # Save only stable + dynamic (immutable is not meant to change)
        self._write("persona/stable.json", persona.get("stable", {}))
        self._write("persona/dynamic.json", persona.get("dynamic", {}))

    # ---------- State ----------
    def load_vectors(self) -> Dict[str, Any]:
        return self._read("state/vectors.json", default={})

    def save_vectors(self, vectors: Dict[str, Any]) -> None:
        self._write("state/vectors.json", vectors)

    def load_counters(self) -> Dict[str, Any]:
        default = {
            "current_turn": 0,
            "last_rumination_turn": 0,
            "rumination_window_size": 20,
            "turns_since_last_rumination": 0,
            # NEW: buffer commit pointer
            "last_buffer_committed_turn": 0,
        }
        data = self._read("state/counters.json", default=default)

        # Ensure required keys exist even if older file is missing them
        for k, v in default.items():
            data.setdefault(k, v)

        return data

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

    # ---------- Perception buffer (JSONL, crash-safe) ----------
    def append_perceptions_to_buffer(
        self,
        perceptions: List[Dict[str, Any]],
        turn: int,
        session_id: str,
    ) -> int:
        """
        Append normalized perception entities to memory/perception_buffer.jsonl.

        Returns: number of appended items.
        """
        if not perceptions:
            return 0

        n = 0
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
                    "confidence": p.get("confidence"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

        return n

    def read_buffered_perceptions(
        self,
        min_turn_exclusive: int,
        max_turn_inclusive: int,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read buffered perceptions with turn in (min_turn_exclusive, max_turn_inclusive].
        Optionally filter by session_id.
        Returns list in ConsolidationEngine expected format:
        {
          "entity_type": ...,
          "entity": ...,
          "dimension_values": ...,
          "confidence": ...
        }
        """
        out: List[Dict[str, Any]] = []

        if not self.perception_buffer_path.exists():
            return out

        with self.perception_buffer_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                t = int(rec.get("turn", -1))
                if t <= int(min_turn_exclusive) or t > int(max_turn_inclusive):
                    continue

                if session_id is not None and rec.get("session_id") != session_id:
                    continue

                out.append({
                    "entity_type": rec.get("entity_type"),
                    "entity": rec.get("entity"),
                    "dimension_values": rec.get("dimension_values", {}),
                    "confidence": rec.get("confidence", 0.5),
                })

        return out

    def vacuum_buffer_keep_recent(
        self,
        keep_turn_greater_than: int,
        max_lines_keep: int = 5000,
    ) -> None:
        """
        Optional cleanup to prevent buffer file from growing forever.
        Keeps:
          - only records with turn > keep_turn_greater_than
          - and at most last max_lines_keep lines (if still large)
        """
        if not self.perception_buffer_path.exists():
            return

        kept: List[str] = []
        with self.perception_buffer_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    rec = json.loads(line_s)
                except Exception:
                    continue
                t = int(rec.get("turn", -1))
                if t > int(keep_turn_greater_than):
                    kept.append(line_s)

        if len(kept) > max_lines_keep:
            kept = kept[-max_lines_keep:]

        tmp = self.perception_buffer_path.with_suffix(".tmp")
        tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        tmp.replace(self.perception_buffer_path)

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