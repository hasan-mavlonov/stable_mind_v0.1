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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ReflectionEngine:
    """
    Reflection (v1):
    - Runs at rumination checkpoints (Agent calls it after consolidation).
    - Evolves stable traits slowly (stable["trait_vector"]).
    - Baseline rule: per-belief (Rule B).
      Only beliefs that cross drift threshold get baseline_mean updated.

    Minimal mapping (v1):
      stable.trait_vector["trust"]  <-- driven by  stable.beliefs.person.*.warmth
    """

    # User-chosen constants:
    DRIFT_THRESHOLD = 0.25
    CONF_THRESHOLD = 0.70
    MIN_OBS = 10
    TRAIT_STEP_BETA = 0.05
    TRAIT_MAX_STEP = 0.08

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.log_path = self.root / "memory" / "reflection_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, persona: Dict[str, Any], turn: int) -> bool:
        """
        Returns True if reflection performed an update (trait and/or baselines).
        """
        stable = persona.setdefault("stable", {})
        beliefs = stable.get("beliefs", {})
        if not isinstance(beliefs, dict):
            return False

        trait_vector = stable.setdefault("trait_vector", {})
        if "trust" not in trait_vector:
            trait_vector["trust"] = 0.0

        person_block = beliefs.get("person", {})
        if not isinstance(person_block, dict):
            self._append_log(
                {
                    "turn": int(turn),
                    "did_update": False,
                    "reason": "no_person_beliefs",
                }
            )
            return False

        # Gather candidate warmth beliefs
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for person_name, dims in person_block.items():
            if not isinstance(dims, dict):
                continue
            warmth = dims.get("warmth")
            if isinstance(warmth, dict):
                candidates.append((str(person_name), warmth))

        if not candidates:
            self._append_log(
                {
                    "turn": int(turn),
                    "did_update": False,
                    "reason": "no_warmth_beliefs",
                }
            )
            return False

        triggered: List[Dict[str, Any]] = []
        weighted_sum = 0.0
        weight_total = 0.0

        # Evaluate per-belief triggers (Rule B)
        for person_name, dim_block in candidates:
            mean = float(dim_block.get("mean", 0.0))
            baseline_mean = float(dim_block.get("baseline_mean", mean))
            conf = float(dim_block.get("confidence", 0.0))
            n = int(dim_block.get("n", 0) or 0)

            drift = abs(mean - baseline_mean)

            if drift >= self.DRIFT_THRESHOLD and conf >= self.CONF_THRESHOLD and n >= self.MIN_OBS:
                weighted_sum += mean * conf
                weight_total += conf
                triggered.append(
                    {
                        "person": person_name,
                        "dimension": "warmth",
                        "mean": mean,
                        "baseline_mean": baseline_mean,
                        "drift": drift,
                        "confidence": conf,
                        "n": n,
                    }
                )

        if not triggered:
            self._append_log(
                {
                    "turn": int(turn),
                    "did_update": False,
                    "reason": "no_beliefs_triggered",
                    "mapping": {"trait": "trust", "source": "person.*.warmth"},
                }
            )
            return False

        # Aggregate signal in [-1, 1]
        signal = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        signal = clamp(signal, -1.0, 1.0)

        # Update trait (bounded)
        old_trait = float(trait_vector.get("trust", 0.0))
        desired = old_trait + self.TRAIT_STEP_BETA * (signal - old_trait)
        delta = clamp(desired - old_trait, -self.TRAIT_MAX_STEP, self.TRAIT_MAX_STEP)
        new_trait = clamp(old_trait + delta, -1.0, 1.0)

        trait_vector["trust"] = new_trait

        # Update baseline_mean ONLY for triggered beliefs (Rule B)
        for item in triggered:
            p = item["person"]
            dims = person_block.get(p)
            if not isinstance(dims, dict):
                continue
            w = dims.get("warmth")
            if not isinstance(w, dict):
                continue
            w["baseline_mean"] = float(w.get("mean", 0.0))
            w["baseline_turn"] = int(turn)

        self._append_log(
            {
                "turn": int(turn),
                "did_update": True,
                "mapping": {"trait": "trust", "source": "person.*.warmth"},
                "thresholds": {
                    "drift": self.DRIFT_THRESHOLD,
                    "confidence": self.CONF_THRESHOLD,
                    "min_obs": self.MIN_OBS,
                    "beta": self.TRAIT_STEP_BETA,
                    "max_step": self.TRAIT_MAX_STEP,
                },
                "trait_update": {
                    "old": old_trait,
                    "signal": signal,
                    "delta": delta,
                    "new": new_trait,
                },
                "triggered_count": len(triggered),
                "triggered": triggered[:50],
            }
        )

        return True

    def _append_log(self, obj: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")