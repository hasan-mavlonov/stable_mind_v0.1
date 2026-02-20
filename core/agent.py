# core/agent.py
from typing import Dict, Any

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

    def step(self, user_message: str, session_id: str = "default") -> Dict[str, Any]:
        persona = self.state.load_persona()
        vectors = self.state.load_vectors()
        counters = self.state.load_counters()

        # ---- Turn increment
        counters["current_turn"] = int(counters.get("current_turn", 0)) + 1
        turn = counters["current_turn"]

        # Track turns since last rumination
        counters["turns_since_last_rumination"] = int(counters.get("turns_since_last_rumination", 0)) + 1
        window = int(counters.get("rumination_window_size", 20))

        # Ensure dynamic structure exists
        persona.setdefault("dynamic", {})
        dyn = persona["dynamic"]
        dyn.setdefault("working_memory", {})
        wm = dyn["working_memory"]
        wm.setdefault("perception_buffer", [])
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

        # --- Update working memory + buffer perceptions
        # Buffer the normalized perception entities (so consolidation uses the same format)
        if entities:
            wm["perception_buffer"].extend(entities)

            # Track last mentions for convenience
            wm["last_entity_focus"] = entities[-1]["entity"]

            # last_person_entity / last_place_entity convenience
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
        if counters["turns_since_last_rumination"] >= window:
            did_ruminate = True

            buffered = wm.get("perception_buffer", [])
            if isinstance(buffered, list) and buffered:
                self.consolidation.run(
                    persona=persona,
                    perceptions=buffered,
                    turn=turn
                )

            # Clear buffer after rumination
            wm["perception_buffer"] = []

            counters["last_rumination_turn"] = turn
            counters["turns_since_last_rumination"] = 0

        # Save everything
        self.state.save_persona(persona)
        self.state.save_counters(counters)
        self.state.save_vectors(vectors)

        return {
            "turn": turn,
            "perceived_entities": entities,
            "buffer_size": len(wm.get("perception_buffer", [])),
            "did_ruminate": did_ruminate,
            "beliefs": persona.get("stable", {}).get("beliefs", {})
        }