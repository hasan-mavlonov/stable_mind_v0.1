#!/usr/bin/env python3
"""
Create a fresh StableMind persona from scratch.

CLI usage:
    python create_persona.py --root .

Reusable API:
    from pathlib import Path
    from create_persona import create_persona

    create_persona(Path(".").resolve())
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

EMOTIONS_8 = [
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
]


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def reset_jsonl(path: Path) -> None:
    """
    Truncate/initialize a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def build_immutable_persona() -> Dict[str, Any]:
    return {
        "id": "rin",
        "display_name": "Rin",
        "username": "@rinlife2025",
        "nationality": "Chinese",
        "origin": {
            "birth_city": "Hangzhou",
            "current_city_at_creation": "Shanghai",
            "story": "",
        },
        "immutable_appearance_markers": {
            "origin_eye_color": "jade-green reflection",
            "origin_hair_color": "black with brown highlights",
        },
    }


def build_stable_persona() -> Dict[str, Any]:
    return {
        "personality": {
            "mbti": "INFJ",
            "core_traits": [
                "Empathetic",
                "Introspective",
                "Quietly ambitious",
            ],
            "values": [
                "Beauty should connect people, not divide them",
            ],
            "tone_of_voice": "Warm, introspective, occasionally poetic",
        },
        "beliefs": {
            "place": {},
            "person": {},
            "concept": {},
            "activity": {},
            "organization": {},
            "self_state": {},
        },
    }


def build_dynamic_persona() -> Dict[str, Any]:
    return {
        "now": {
            "mood": {emotion: 0.5 for emotion in EMOTIONS_8},
            "last_updated_turn": 0,
            "arousal": 0.5,
            "stress": 0.5,
            "energy": 0.5,
            "confidence": 0.5,
            "current_topic": None,
            "current_activity": None,
        },
        "working_memory": {
            "last_entity_focus": None,
            "last_person_entity": None,
            "last_place_entity": None,
            "recent_entities": [],
            "recent_events": [],
            "open_threads": [],
            "perception_buffer": [],
        },
        "short_term_preferences": {
            "recent_likes": [],
            "recent_dislikes": [],
            "recent_discoveries": [],
            "recent_hobbies": [],
        },
        "biases": {
            "negativity_bias": 0.0,
            "social_guard": 0.0,
            "novelty_seeking": 0.0,
        },
    }


def build_baseline_traits() -> Dict[str, float]:
    return {
        "warmth": 0.6,
        "sarcasm": 0.1,
        "formality": 0.2,
        "openness": 0.9,
        "dominance": 0.2,
        "sentimentality": 0.3,
        "curiosity": 0.4,
        "optimism": 0.3,
        "conscientiousness": 0.5,
    }


def build_vectors() -> Dict[str, Any]:
    baseline_traits = build_baseline_traits()
    return {
        "turn": 0,
        "trait_vector": {
            "baseline": baseline_traits,
            "current": dict(baseline_traits),
            "initial_baseline": dict(baseline_traits),
        },
        "emotion_vector": {emotion: 0.5 for emotion in EMOTIONS_8},
    }


def build_counters() -> Dict[str, Any]:
    return {
        "current_turn": 0,
        "last_rumination_turn": 0,
        "rumination_window_size": 1,
        "turns_since_last_rumination": 0,
        "last_buffer_committed_turn": 0,
        "last_stable_traits_update_turn": 0,
    }


def create_persona(root: Path) -> None:
    """
    Create or reset the StableMind persona/state files under the given root.
    """
    root = root.resolve()

    persona_dir = root / "persona"
    state_dir = root / "state"
    memory_dir = root / "memory"

    immutable = build_immutable_persona()
    stable = build_stable_persona()
    dynamic = build_dynamic_persona()
    vectors = build_vectors()
    counters = build_counters()

    write_json(persona_dir / "immutable.json", immutable)
    write_json(persona_dir / "stable.json", stable)
    write_json(persona_dir / "dynamic.json", dynamic)
    write_json(state_dir / "vectors.json", vectors)
    write_json(state_dir / "counters.json", counters)

    reset_jsonl(memory_dir / "perceptions.jsonl")
    reset_jsonl(memory_dir / "perception_buffer.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    create_persona(root)

    print("✅ Persona created successfully.")
    print("All files initialized (including memory JSONL resets).")


if __name__ == "__main__":
    main()