# Copyright 2026 Hasan Mavlonov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from typing import Any, Dict, List


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ConsolidationEngine:
    """
    Updates persona['stable']['beliefs'] using perception results.

    Belief schema per (entity_type, entity, dimension):
    {
        "mean": float in [-1,1],
        "baseline_mean": float in [-1,1],   # last reflection checkpoint anchor
        "confidence": float in [0,1],
        "n": int,
        "last_updated_turn": int,
        "baseline_turn": int               # when baseline_mean was last set
    }

    Perception format:
    {
        "entity_type": str,
        "entity": str,
        "dimension_values": {dim: value},
        "confidence": float
    }
    """

    def __init__(self, alpha: float = 0.25, conf_gain: float = 0.05):
        self.alpha = alpha
        self.conf_gain = conf_gain

    def run(self, persona: Dict[str, Any], perceptions: List[Dict[str, Any]], turn: int) -> None:
        stable = persona.setdefault("stable", {})
        beliefs = stable.setdefault("beliefs", {})

        for obs in perceptions:
            entity_type = obs["entity_type"]
            entity = obs["entity"]
            dim_values = obs.get("dimension_values", {})
            obs_conf = float(obs.get("confidence", 0.5))

            type_block = beliefs.setdefault(entity_type, {})
            entity_block = type_block.setdefault(entity, {})

            for dim, value in dim_values.items():
                dim_block = entity_block.setdefault(
                    dim,
                    {
                        "mean": 0.0,
                        "baseline_mean": 0.0,
                        "confidence": 0.2,
                        "n": 0,
                        "last_updated_turn": int(turn),
                        "baseline_turn": 0,
                    },
                )

                # Backfill baseline fields if older belief records exist
                if "baseline_mean" not in dim_block:
                    dim_block["baseline_mean"] = float(dim_block.get("mean", 0.0))
                if "baseline_turn" not in dim_block:
                    dim_block["baseline_turn"] = int(dim_block.get("last_updated_turn", turn))

                old_mean = float(dim_block.get("mean", 0.0))

                # Exponential smoothing
                new_mean = (1 - self.alpha) * old_mean + self.alpha * float(value)
                new_mean = clamp(new_mean, -1.0, 1.0)

                dim_block["mean"] = new_mean
                dim_block["n"] = int(dim_block.get("n", 0)) + 1
                dim_block["last_updated_turn"] = int(turn)

                # Confidence grows with supporting evidence
                dim_block["confidence"] = clamp(
                    float(dim_block.get("confidence", 0.2)) + self.conf_gain * obs_conf,
                    0.0,
                    1.0,
                )