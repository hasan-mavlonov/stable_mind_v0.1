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
    # Small helpers
    # ----------------------------
    @staticmethod
    def _clamp01(x: float) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.5
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        try:
            v = float(x)
        except Exception:
            v = lo
        return max(lo, min(hi, v))

    @staticmethod
    def _ensure_dynamic_now(dyn: Dict[str, Any]) -> Dict[str, Any]:
        dyn.setdefault("now", {})
        now = dyn["now"]

        # mood vector
        now.setdefault("mood", {})
        mood = now["mood"]
        for k in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
            mood.setdefault(k, 0.5)

        # scalar state
        now.setdefault("last_updated_turn", 0)
        now.setdefault("arousal", 0.5)
        now.setdefault("stress", 0.5)
        now.setdefault("energy", 0.5)
        now.setdefault("confidence", 0.5)

        # context
        now.setdefault("current_topic", None)
        now.setdefault("current_activity", None)
        return now

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

    # ----------------------------
    # Dynamic NOW updater (every turn)
    # ----------------------------
    def _update_dynamic_now(self, dyn: Dict[str, Any], user_message: str, entities: List[Dict[str, Any]], turn: int) -> None:
        """
        Deterministically updates dynamic['now'] each turn using:
        - keyword cues from user_message
        - mild influence from extracted perceptions (valence, emotional_stability, warmth, comfort, etc.)

        This is deliberately lightweight and stable (EMA-like).
        """
        now = self._ensure_dynamic_now(dyn)
        mood = now["mood"]

        text = (user_message or "").lower()

        # --- keyword cues (very small, stable nudges)
        # Each hit adds a delta in [-1, +1] space then we apply small step to [0,1].
        emotion_hits = {
            "joy": [
                "happy", "glad", "joy", "great", "wonderful", "amazing", "excited", "love", "enjoy",
            ],
            "trust": [
                "trust", "safe", "secure", "supported", "reliable",
            ],
            "fear": [
                "afraid", "scared", "fear", "anxious", "worried", "panic",
            ],
            "surprise": [
                "surprised", "shock", "shocked", "unexpected",
            ],
            "sadness": [
                "sad", "down", "depressed", "lonely", "cry", "hurt",
            ],
            "disgust": [
                "disgust", "gross", "nasty", "repuls",  # covers "repulse/repulsed/repulsive"
            ],
            "anger": [
                "angry", "mad", "furious", "rage", "annoyed", "yell", "shout",
            ],
            "anticipation": [
                "hope", "looking forward", "cant wait", "can't wait", "anticipat",
            ],
        }

        # Scalar cues
        stress_up = ["stressed", "overwhelmed", "anxious", "worried", "panic", "tense"]
        stress_down = ["calm", "relaxed", "peaceful", "at ease"]

        energy_up = ["energized", "energetic", "rested"]
        energy_down = ["tired", "exhausted", "sleepy", "drained", "burned out", "burnt out"]

        confidence_up = ["confident", "sure", "proud"]
        confidence_down = ["insecure", "doubt", "uncertain", "ashamed"]

        arousal_up = ["excited", "furious", "panic", "tense", "shocked", "anxious"]
        arousal_down = ["calm", "relaxed", "peaceful", "sleepy"]

        # --- Compute deltas
        mood_delta = {k: 0.0 for k in mood.keys()}
        for emo, keys in emotion_hits.items():
            for kw in keys:
                if kw in text:
                    mood_delta[emo] += 1.0

        def _count_hits(keywords: List[str]) -> float:
            c = 0.0
            for kw in keywords:
                if kw in text:
                    c += 1.0
            return c

        d_stress = _count_hits(stress_up) - _count_hits(stress_down)
        d_energy = _count_hits(energy_up) - _count_hits(energy_down)
        d_conf = _count_hits(confidence_up) - _count_hits(confidence_down)
        d_arousal = _count_hits(arousal_up) - _count_hits(arousal_down)

        # --- Influence from perceptions (small, but useful)
        # Convert [-1,1] perception values into small deltas.
        for e in entities or []:
            if not isinstance(e, dict):
                continue
            et = e.get("entity_type")
            dv = e.get("dimension_values", {}) or {}
            if not isinstance(dv, dict):
                continue

            # concept valence nudges mood
            if et == "concept":
                v = dv.get("valence")
                if isinstance(v, (int, float)):
                    # positive valence -> joy up, sadness down
                    mood_delta["joy"] += float(v) * 0.75
                    mood_delta["sadness"] -= float(v) * 0.50

            # person cues
            if et == "person":
                es = dv.get("emotional_stability")
                w = dv.get("warmth")
                if isinstance(es, (int, float)):
                    # unstable person in context -> stress up a bit
                    d_stress += (-float(es)) * 0.5
                    mood_delta["fear"] += (-float(es)) * 0.25
                if isinstance(w, (int, float)):
                    mood_delta["trust"] += float(w) * 0.4
                    mood_delta["joy"] += float(w) * 0.15

            # place cues
            if et == "place":
                c = dv.get("comfort")
                q = dv.get("quietness")
                if isinstance(c, (int, float)):
                    d_stress -= float(c) * 0.25
                    mood_delta["joy"] += float(c) * 0.2
                if isinstance(q, (int, float)):
                    d_arousal -= float(q) * 0.15  # quiet reduces arousal slightly

            # self_state cues (if your ontology uses these dims)
            if et == "self_state":
                # If you later add these dims to ontology, they will begin to work immediately.
                s = dv.get("stress")
                en = dv.get("energy")
                cf = dv.get("confidence")
                ar = dv.get("arousal")
                if isinstance(s, (int, float)):
                    d_stress += float(s) * 0.75
                if isinstance(en, (int, float)):
                    d_energy += float(en) * 0.75
                if isinstance(cf, (int, float)):
                    d_conf += float(cf) * 0.75
                if isinstance(ar, (int, float)):
                    d_arousal += float(ar) * 0.75

        # --- Apply updates (EMA-ish)
        # Small step sizes so "now" doesn't jump.
        MOOD_STEP = 0.06
        SCALAR_STEP = 0.08

        for k in mood.keys():
            # turn hit-count into signed delta in [-1,1] (cap)
            d = self._clamp(mood_delta.get(k, 0.0), -3.0, 3.0) / 3.0
            mood[k] = self._clamp01(mood[k] + MOOD_STEP * d)

        now["stress"] = self._clamp01(now["stress"] + SCALAR_STEP * self._clamp(d_stress, -2.0, 2.0) / 2.0)
        now["energy"] = self._clamp01(now["energy"] + SCALAR_STEP * self._clamp(d_energy, -2.0, 2.0) / 2.0)
        now["confidence"] = self._clamp01(now["confidence"] + SCALAR_STEP * self._clamp(d_conf, -2.0, 2.0) / 2.0)
        now["arousal"] = self._clamp01(now["arousal"] + SCALAR_STEP * self._clamp(d_arousal, -2.0, 2.0) / 2.0)

        # --- Context fields
        # current_topic: prefer the latest entity mention
        if entities:
            now["current_topic"] = entities[-1].get("entity", now.get("current_topic"))

        # current_activity: if your ontology extracts activity entities, track them
        for e in reversed(entities or []):
            if e.get("entity_type") == "activity":
                now["current_activity"] = e.get("entity")
                break

        now["last_updated_turn"] = int(turn)

    # ----------------------------
    # Main step
    # ----------------------------
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

        # --- Update NOW (EVERY TURN)
        self._update_dynamic_now(dyn=dyn, user_message=user_message, entities=entities, turn=turn)

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
            "beliefs": persona.get("stable", {}).get("beliefs", {}),
            "now": persona.get("dynamic", {}).get("now", {}),
        }