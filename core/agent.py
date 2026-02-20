# core/agent.py
from typing import Dict, Any

from .state_manager import StateManager
from .perception import PerceptionEngine
from .consolidation import ConsolidationEngine


_PRONOUNS_PERSON = {"she", "he", "her", "him", "they", "them"}
_PRONOUNS_PLACE = {"there", "here"}


class Agent:
    def __init__(self, root_dir: str, model: str = "gpt-4.1-mini"):
        self.root_dir = root_dir
        self.state = StateManager(root_dir)

        self.perception = PerceptionEngine(root_dir=root_dir, model=model)
        self.consolidation = ConsolidationEngine()

    def step(self, user_message: str, session_id: str = "default") -> Dict[str, Any]:
        persona = self.state.load_persona()
        vectors = self.state.load_vectors()
        counters = self.state.load_counters()

        counters["current_turn"] = int(counters.get("current_turn", 0)) + 1
        turn = counters["current_turn"]

        # Ensure dynamic exists
        persona.setdefault("dynamic", {})
        persona["dynamic"].setdefault("last_person_entity", None)
        persona["dynamic"].setdefault("last_place_entity", None)

        # --- Perception
        perception_result = self.perception.analyze(
            user_text=user_message,
            session_id=session_id
        )
        entities = perception_result.entities

        # --- Canonicalize beliefs namespace (kills "places"/"people" forever)
        self._canonicalize_belief_namespaces(persona)

        # --- Lightweight coreference: pronouns -> last known entity
        entities = self._resolve_pronouns(persona, entities)

        # update last_entity_focus + last_* by type
        if entities:
            persona["dynamic"]["last_entity_focus"] = entities[-1]["entity"]
            for e in entities:
                if e.get("entity_type") == "person":
                    persona["dynamic"]["last_person_entity"] = e.get("entity")
                elif e.get("entity_type") == "place":
                    persona["dynamic"]["last_place_entity"] = e.get("entity")

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

    def _canonicalize_belief_namespaces(self, persona: Dict[str, Any]) -> None:
        """
        If old code created beliefs.places / beliefs.people, merge them into
        beliefs.place / beliefs.person and delete plural keys.
        """
        stable = persona.setdefault("stable", {})
        beliefs = stable.setdefault("beliefs", {})

        plural_to_singular = {
            "places": "place",
            "people": "person",
        }

        for plural, singular in plural_to_singular.items():
            if plural in beliefs and isinstance(beliefs[plural], dict):
                singular_block = beliefs.setdefault(singular, {})
                for entity, dims in beliefs[plural].items():
                    if entity not in singular_block:
                        singular_block[entity] = dims
                del beliefs[plural]

    def _resolve_pronouns(self, persona: Dict[str, Any], entities: list[dict]) -> list[dict]:
        """
        Very simple coreference: map pronoun entities to the last seen entity of that type.
        Prevents belief keys like "she" from appearing.
        """
        last_person = persona.get("dynamic", {}).get("last_person_entity")
        last_place = persona.get("dynamic", {}).get("last_place_entity")

        out = []
        for e in entities:
            et = e.get("entity_type")
            name = (e.get("entity") or "").strip()
            if not name:
                continue

            name_l = name.lower()

            # Person pronouns -> last_person
            if et == "person" and name_l in _PRONOUNS_PERSON and last_person:
                e = dict(e)
                e["entity"] = last_person

            # Place pronouns -> last_place (optional)
            if et == "place" and name_l in _PRONOUNS_PLACE and last_place:
                e = dict(e)
                e["entity"] = last_place

            out.append(e)

        return out