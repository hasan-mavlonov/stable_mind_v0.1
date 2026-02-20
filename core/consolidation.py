# core/consolidation.py
from typing import Dict, Any, List


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ConsolidationEngine:
    def __init__(
        self,
        alpha: float = 0.25,
        conf_gain: float = 0.05,
        min_obs_conf: float = 0.75,
        min_abs_value: float = 0.45,
        min_delta: float = 0.10,
        canonicalize_entities: bool = True,
        max_dims_per_entity: int | None = None,
        migrate_plural_keys: bool = True,
    ):
        self.alpha = alpha
        self.conf_gain = conf_gain
        self.min_obs_conf = min_obs_conf
        self.min_abs_value = min_abs_value
        self.min_delta = min_delta
        self.canonicalize_entities = canonicalize_entities
        self.max_dims_per_entity = max_dims_per_entity
        self.migrate_plural_keys = migrate_plural_keys

    def run(self, persona: Dict[str, Any], perceptions: List[Dict[str, Any]], turn: int) -> None:
        stable = persona.setdefault("stable", {})
        beliefs = stable.setdefault("beliefs", {})

        # --- FIX: migrate old plural namespaces -> singular namespaces ---
        # create_persona.py (or templates) likely created beliefs.places / beliefs.people
        if self.migrate_plural_keys:
            self._migrate_plural_beliefs(beliefs)

        for obs in perceptions:
            entity_type = obs.get("entity_type")
            entity = obs.get("entity")
            dim_values = obs.get("dimension_values") or {}
            obs_conf = float(obs.get("confidence", 0.5))

            if not isinstance(entity_type, str) or not isinstance(entity, str):
                continue
            if not isinstance(dim_values, dict) or not dim_values:
                continue

            # Gate A
            if obs_conf < self.min_obs_conf:
                continue

            entity_key = entity.strip().lower() if self.canonicalize_entities else entity.strip()
            if not entity_key:
                continue

            # Optionally cap dimensions
            items = []
            for dim, value in dim_values.items():
                if not isinstance(dim, str) or not dim.strip():
                    continue
                try:
                    v = float(value)
                except Exception:
                    continue
                items.append((dim.strip(), v))

            if not items:
                continue

            if self.max_dims_per_entity is not None and self.max_dims_per_entity > 0:
                items.sort(key=lambda kv: abs(kv[1]), reverse=True)
                items = items[: self.max_dims_per_entity]

            type_block = beliefs.setdefault(entity_type, {})
            entity_block = type_block.setdefault(entity_key, {})

            for dim, value in items:
                # Gate B
                if abs(value) < self.min_abs_value:
                    continue

                dim_block = entity_block.setdefault(dim, {
                    "mean": 0.0,
                    "confidence": 0.2,
                    "n": 0,
                    "last_updated_turn": turn
                })

                old_mean = float(dim_block.get("mean", 0.0))

                proposed = (1 - self.alpha) * old_mean + self.alpha * value
                proposed = clamp(proposed, -1.0, 1.0)

                # Gate C
                if abs(proposed - old_mean) < self.min_delta and int(dim_block.get("n", 0)) > 0:
                    continue

                dim_block["mean"] = proposed
                dim_block["n"] = int(dim_block.get("n", 0)) + 1
                dim_block["last_updated_turn"] = turn

                dim_block["confidence"] = clamp(
                    float(dim_block.get("confidence", 0.2)) + self.conf_gain * obs_conf,
                    0.0,
                    1.0
                )

    def _migrate_plural_beliefs(self, beliefs: Dict[str, Any]) -> None:
        """
        Merge beliefs['places'] into beliefs['place'], beliefs['people'] into beliefs['person'].
        Then remove plural keys.
        """
        plural_to_singular = {
            "places": "place",
            "people": "person",
        }

        for plural, singular in plural_to_singular.items():
            if plural in beliefs and isinstance(beliefs[plural], dict):
                singular_block = beliefs.setdefault(singular, {})
                # Merge entity dictionaries (do not overwrite existing)
                for entity, dims in beliefs[plural].items():
                    if entity not in singular_block:
                        singular_block[entity] = dims
                # Remove the plural key to stop confusion
                del beliefs[plural]