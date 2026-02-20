# core/consolidation.py
from typing import Dict, Any, List


def clamp(x: float, lo: float, hi: float):
    return max(lo, min(hi, x))


class ConsolidationEngine:
    """
    Updates persona['stable']['beliefs'] using perception results.
    Assumes perception format:

    {
        "entity_type": str,
        "entity": str,
        "dimension_values": {dim: value},
        "confidence": float
    }
    """

    def __init__(self, alpha: float = 0.25, conf_gain: float = 0.05):
        # alpha = smoothing strength
        # conf_gain = confidence increase per supporting observation
        self.alpha = alpha
        self.conf_gain = conf_gain

    def run(self, persona: Dict[str, Any], perceptions: List[Dict[str, Any]], turn: int):

        stable = persona.setdefault("stable", {})
        beliefs = stable.setdefault("beliefs", {})

        for obs in perceptions:
            entity_type = obs["entity_type"]
            entity = obs["entity"]
            dim_values = obs["dimension_values"]
            obs_conf = float(obs.get("confidence", 0.5))

            # ensure entity_type block exists
            type_block = beliefs.setdefault(entity_type, {})

            # ensure entity block exists
            entity_block = type_block.setdefault(entity, {})

            for dim, value in dim_values.items():

                # create dimension belief if not exist
                dim_block = entity_block.setdefault(dim, {
                    "mean": 0.0,              # neutral for [-1,1] scale
                    "confidence": 0.2,        # low starting confidence
                    "n": 0,
                    "last_updated_turn": turn
                })

                old_mean = dim_block["mean"]

                # exponential smoothing
                new_mean = (1 - self.alpha) * old_mean + self.alpha * value
                new_mean = clamp(new_mean, -1.0, 1.0)

                dim_block["mean"] = new_mean
                dim_block["n"] += 1
                dim_block["last_updated_turn"] = turn

                # confidence grows with supporting evidence
                # scaled slightly by observation confidence
                dim_block["confidence"] = clamp(
                    dim_block["confidence"] + self.conf_gain * obs_conf,
                    0.0,
                    1.0
                )