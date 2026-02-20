# core/agent.py
from typing import Dict, Any, Optional

from .state_manager import StateManager
from .perception import PerceptionEngine
from .consolidation import ConsolidationEngine


class Agent:
    def __init__(self, root_dir: str, model: str = "gpt-4.1-mini"):
        self.root_dir = root_dir
        self.state = StateManager(root_dir)

        # Perception needs ontology + OpenAI client setup, so pass root_dir + model
        self.perception = PerceptionEngine(root_dir=root_dir, model=model)

        # Consolidation updates stable beliefs immediately (v0.1)
        self.consolidation = ConsolidationEngine()

    def step(self, user_message: str, session_id: str = "default") -> Dict[str, Any]:
        persona = self.state.load_persona()
        vectors = self.state.load_vectors()
        counters = self.state.load_counters()

        counters["current_turn"] = int(counters.get("current_turn", 0)) + 1
        turn = counters["current_turn"]

        # --- Perception
        perception_result = self.perception.analyze(
            user_text=user_message,
            session_id=session_id
        )

        entities = perception_result.entities

        # update last_entity_focus
        if entities:
            persona.setdefault("dynamic", {})
            persona["dynamic"]["last_entity_focus"] = entities[-1]["entity"]

        # --- Consolidation
        self.consolidation.run(
            persona=persona,
            perceptions=entities,
            turn=turn
        )

        # Save
        self.state.save_persona(persona)
        self.state.save_counters(counters)
        self.state.save_vectors(vectors)

        return {
            "turn": turn,
            "perceived_entities": entities,
            "beliefs": persona.get("stable", {}).get("beliefs", {})
        }