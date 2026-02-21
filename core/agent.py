# core/agent.py
from __future__ import annotations

from typing import Dict, Any, List

from .state_manager import StateManager
from .perception import PerceptionEngine
from .consolidation import ConsolidationEngine


class Agent:
    def __init__(self, root_dir: str, model: str = "gpt-4.1-mini"):
        self.root_dir = root_dir
        self.state = StateManager(root_dir)

        self.perception = PerceptionEngine(root_dir=root_dir, model=model)
        self.consolidation = ConsolidationEngine()

    # ----------------------------
    # Canonicalization / normalization
    # ----------------------------
    def _canonical_entity_name(self, name: str) -> str:
        if not isinstance(name, str):
            return ""

        s = name.strip().lower()
        s = s.strip(" \t\n\r\"'`.,!?;:()[]{}<>")
        s = " ".join(s.split())

        # Simple plural trimming
        if len(s) > 3 and s.endswith("s") and not s.endswith(("ss", "us", "is")):
            s = s[:-1]

        return s

    def _map_first_person_to_self(self, e: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(e, dict):
            return e

        et = e.get("entity_type")
        name = e.get("entity")

        if not isinstance(name, str):
            return e

        key = name.strip().lower()
        FIRST_PERSON = {"i", "me", "my", "mine", "myself"}

        if et == "person" and key in FIRST_PERSON:
            out = dict(e)
            out["entity_type"] = "self_state"
            out["entity"] = "self"
            return out

        return e

    # ----------------------------
    # Coreference
    # ----------------------------
    def _resolve_coreference(self, entities: List[Dict[str, Any]], wm: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not entities:
            return entities

        last_person = wm.get("last_person_entity")
        last_place = wm.get("last_place_entity")

        PERSON_PRONOUNS = {
            "she", "her", "hers",
            "he", "him", "his",
            "they", "them", "their", "theirs",
            "this person", "that person",
        }
        PLACE_DEIXIS = {"here", "there", "this place", "that place"}

        resolved: List[Dict[str, Any]] = []
        for e in entities:
            if not isinstance(e, dict):
                continue

            et = e.get("entity_type")
            name = e.get("entity")

            if isinstance(name, str):
                key = name.strip().lower()

                if et == "person" and key in PERSON_PRONOUNS and isinstance(last_person, str) and last_person.strip():
                    e = dict(e)
                    e["entity"] = last_person

                if et == "place" and key in PLACE_DEIXIS and isinstance(last_place, str) and last_place.strip():
                    e = dict(e)
                    e["entity"] = last_place

            resolved.append(e)

        return resolved

    def _postprocess_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for e in entities or []:
            if not isinstance(e, dict):
                continue

            e = self._map_first_person_to_self(e)

            et = e.get("entity_type")
            name = e.get("entity")

            if not isinstance(et, str) or not isinstance(name, str):
                continue

            canon = self._canonical_entity_name(name)
            if not canon:
                continue

            ee = dict(e)
            ee["entity"] = canon
            out.append(ee)

        return out

    def step(self, user_message: str, session_id: str = "default") -> Dict[str, Any]:
        persona = self.state.load_persona()
        vectors = self.state.load_vectors()
        counters = self.state.load_counters()

        counters["current_turn"] = int(counters.get("current_turn", 0)) + 1
        turn = int(counters["current_turn"])

        counters["turns_since_last_rumination"] = int(counters.get("turns_since_last_rumination", 0)) + 1
        window = int(counters.get("rumination_window_size", 20))

        persona.setdefault("dynamic", {})
        dyn = persona["dynamic"]
        dyn.setdefault("working_memory", {})
        wm = dyn["working_memory"]

        wm.setdefault("recent_entities", [])
        wm.setdefault("recent_events", [])
        wm.setdefault("open_threads", [])
        wm.setdefault("last_entity_focus", None)
        wm.setdefault("last_person_entity", None)
        wm.setdefault("last_place_entity", None)

        perception_result = self.perception.analyze(
            user_text=user_message,
            session_id=session_id
        )
        entities = perception_result.entities

        entities = self._resolve_coreference(entities, wm)
        entities = self._postprocess_entities(entities)

        appended = 0
        if entities:
            appended = self.state.append_perceptions_to_buffer(
                perceptions=entities,
                turn=turn,
                session_id=session_id,
            )

            wm["last_entity_focus"] = entities[-1]["entity"]

            for e in reversed(entities):
                if e.get("entity_type") == "person":
                    wm["last_person_entity"] = e.get("entity")
                    break

            for e in reversed(entities):
                if e.get("entity_type") == "place":
                    wm["last_place_entity"] = e.get("entity")
                    break

            for e in entities:
                wm["recent_entities"].append({
                    "turn": turn,
                    "entity_type": e.get("entity_type"),
                    "entity": e.get("entity"),
                    "confidence": e.get("confidence"),
                    "dimension_values": e.get("dimension_values", {})
                })

            MAX_RECENT_ENTITIES = 50
            if len(wm["recent_entities"]) > MAX_RECENT_ENTITIES:
                wm["recent_entities"] = wm["recent_entities"][-MAX_RECENT_ENTITIES:]

        did_ruminate = False
        if int(counters["turns_since_last_rumination"]) >= window:
            did_ruminate = True

            last_committed = int(counters.get("last_buffer_committed_turn", 0))

            buffered = self.state.read_buffered_perceptions(
                min_turn_exclusive=last_committed,
                max_turn_inclusive=turn,
                session_id=session_id,
            )

            if buffered:
                self.consolidation.run(
                    persona=persona,
                    perceptions=buffered,
                    turn=turn
                )

            counters["last_buffer_committed_turn"] = turn
            counters["last_rumination_turn"] = turn
            counters["turns_since_last_rumination"] = 0

            self.state.vacuum_buffer_keep_recent(
                keep_turn_greater_than=max(0, turn - 200),
                max_lines_keep=5000
            )

        self.state.save_persona(persona)
        self.state.save_counters(counters)
        self.state.save_vectors(vectors)

        # IMPORTANT FIX:
        # buffer_size should be items after last_committed_turn (not after current last_buffer_committed_turn)
        last_committed_now = int(counters.get("last_buffer_committed_turn", 0))
        uncommitted = self.state.read_buffered_perceptions(
            min_turn_exclusive=last_committed_now,
            max_turn_inclusive=turn,
            session_id=session_id,
        )

        return {
            "turn": turn,
            "perceived_entities": entities,
            "buffer_appended": appended,
            "buffer_size": len(uncommitted),
            "did_ruminate": did_ruminate,
            "beliefs": persona.get("stable", {}).get("beliefs", {})
        }