# core/agent.py
from typing import Dict, Any, List

from .state_manager import StateManager
from .perception import PerceptionEngine
from .consolidation import ConsolidationEngine


class Agent:
    def __init__(self, root_dir: str, model: str = "gpt-4.1-mini"):
        self.root_dir = root_dir
        self.state = StateManager(root_dir)

        self.perception = PerceptionEngine(root_dir=root_dir, model=model)

        # Consolidation will run ONLY during rumination (Option A)
        self.consolidation = ConsolidationEngine()

    # ----------------------------
    # Coreference / normalization
    # ----------------------------
    def _resolve_coreference(self, entities: List[Dict[str, Any]], wm: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rewrite ambiguous references (e.g., "she", "here") into last known concrete entities
        stored in working memory. This prevents stable beliefs from splitting into separate keys.
        """
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

            # Only rewrite if entity is a string
            if isinstance(name, str):
                key = name.strip().lower()

                # Person coreference
                if et == "person" and key in PERSON_PRONOUNS and isinstance(last_person, str) and last_person.strip():
                    e = dict(e)
                    e["entity"] = last_person

                # Place deixis
                if et == "place" and key in PLACE_DEIXIS and isinstance(last_place, str) and last_place.strip():
                    e = dict(e)
                    e["entity"] = last_place

            resolved.append(e)

        return resolved

    def step(self, user_message: str, session_id: str = "default") -> Dict[str, Any]:
        persona = self.state.load_persona()
        vectors = self.state.load_vectors()
        counters = self.state.load_counters()

        # ---- Turn increment
        counters["current_turn"] = int(counters.get("current_turn", 0)) + 1
        turn = int(counters["current_turn"])

        # Track turns since last rumination
        counters["turns_since_last_rumination"] = int(counters.get("turns_since_last_rumination", 0)) + 1
        window = int(counters.get("rumination_window_size", 20))

        # Ensure dynamic structure exists
        persona.setdefault("dynamic", {})
        dyn = persona["dynamic"]
        dyn.setdefault("working_memory", {})
        wm = dyn["working_memory"]

        # WM fields (keep WM light; buffer is now JSONL)
        wm.setdefault("recent_entities", [])
        wm.setdefault("recent_events", [])
        wm.setdefault("open_threads", [])
        wm.setdefault("last_entity_focus", None)
        wm.setdefault("last_person_entity", None)
        wm.setdefault("last_place_entity", None)

        # --- Perception (EVERY TURN)
        perception_result = self.perception.analyze(
            user_text=user_message,
            session_id=session_id
        )
        entities = perception_result.entities

        # --- Coreference resolution BEFORE buffering + BEFORE updating last_* trackers
        entities = self._resolve_coreference(entities, wm)

        # --- Update working memory + append perceptions to JSONL buffer
        appended = 0
        if entities:
            appended = self.state.append_perceptions_to_buffer(
                perceptions=entities,
                turn=turn,
                session_id=session_id,
            )

            # Track last mentions for convenience (post-resolution)
            wm["last_entity_focus"] = entities[-1]["entity"]

            # Update last_person_entity / last_place_entity based on resolved entities
            for e in reversed(entities):
                if e.get("entity_type") == "person":
                    wm["last_person_entity"] = e.get("entity")
                    break

            for e in reversed(entities):
                if e.get("entity_type") == "place":
                    wm["last_place_entity"] = e.get("entity")
                    break

            # Keep a lightweight recent_entities log (bounded)
            for e in entities:
                wm["recent_entities"].append({
                    "turn": turn,
                    "entity_type": e.get("entity_type"),
                    "entity": e.get("entity"),
                    "confidence": e.get("confidence"),
                    "dimension_values": e.get("dimension_values", {})
                })

            # Bound recent_entities size to avoid infinite growth
            MAX_RECENT_ENTITIES = 50
            if len(wm["recent_entities"]) > MAX_RECENT_ENTITIES:
                wm["recent_entities"] = wm["recent_entities"][-MAX_RECENT_ENTITIES:]

        # --- Rumination trigger (ONLY THEN update stable beliefs)
        did_ruminate = False
        if int(counters["turns_since_last_rumination"]) >= window:
            did_ruminate = True

            last_committed = int(counters.get("last_buffer_committed_turn", 0))

            buffered = self.state.read_buffered_perceptions(
                min_turn_exclusive=last_committed,
                max_turn_inclusive=turn,
                session_id=session_id,  # keep stress test isolated
            )

            if buffered:
                self.consolidation.run(
                    persona=persona,
                    perceptions=buffered,
                    turn=turn
                )

            # Advance commit pointer
            counters["last_buffer_committed_turn"] = turn

            # Reset rumination timing
            counters["last_rumination_turn"] = turn
            counters["turns_since_last_rumination"] = 0

            # Optional: vacuum old buffer lines (keeps file small)
            # Keep only turns > (turn - 200) to avoid unbounded growth
            self.state.vacuum_buffer_keep_recent(keep_turn_greater_than=max(0, turn - 200), max_lines_keep=5000)

        # Save everything
        self.state.save_persona(persona)
        self.state.save_counters(counters)
        self.state.save_vectors(vectors)

        # "buffer_size": how many uncommitted items exist right now (estimate by reading range)
        last_committed_now = int(counters.get("last_buffer_committed_turn", 0)) if did_ruminate else int(counters.get("last_buffer_committed_turn", 0))
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