#!/usr/bin/env python3
"""
Create a fresh StableMind persona from scratch.

Usage:
    python create_persona.py --root .
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any


EMOTIONS_8 = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation"
]


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    persona_dir = root / "persona"
    state_dir = root / "state"

    # -------------------------------
    # 1️⃣ Immutable
    # -------------------------------

    immutable = {
        "id": "rin",
        "display_name": "Rin",
        "username": "@rinlife2025",
        "nationality": "Chinese",
        "origin": {
            "birth_city": "Hangzhou",
            "current_city_at_creation": "Shanghai",
            "story": ""
        },
        "immutable_appearance_markers": {
            "origin_eye_color": "jade-green reflection",
            "origin_hair_color": "black with brown highlights"
        }
    }

    # -------------------------------
    # 2️⃣ Stable Identity
    # -------------------------------

    stable = {
        "personality": {
            "mbti": "INFJ",
            "core_traits": [
                "Empathetic",
                "Introspective",
                "Quietly ambitious"
            ],
            "values": [
                "Beauty should connect people, not divide them"
            ],
            "tone_of_voice": "Warm, introspective, occasionally poetic"
        },
        "beliefs": {
            "places": {},
            "people": {}
        }
    }

    # -------------------------------
    # 3️⃣ Dynamic State
    # -------------------------------

    dynamic = {
        "favorite_spot": None,
        "recent_hobbies": [],
        "current_focus": [],
        "recent_dislikes": [],
        "recent_discoveries": [],
        "last_visited_places": [],
        "last_entity_focus": None
    }

    # -------------------------------
    # 4️⃣ Trait Baseline
    # -------------------------------

    baseline_traits = {
        "warmth": 0.6,
        "sarcasm": 0.1,
        "formality": 0.2,
        "openness": 0.9,
        "dominance": 0.2,
        "sentimentality": 0.3,
        "curiosity": 0.4,
        "optimism": 0.3,
        "conscientiousness": 0.5
    }

    vectors = {
        "turn": 0,
        "trait_vector": {
            "baseline": baseline_traits,
            "current": dict(baseline_traits),
            "initial_baseline": dict(baseline_traits)
        },
        "emotion_vector": {e: 0.5 for e in EMOTIONS_8}
    }

    counters = {
        "current_turn": 0,
        "last_rumination_turn": 0,
        "rumination_window_size": 20,
        "turns_since_last_rumination": 0
    }

    # -------------------------------
    # Write files
    # -------------------------------

    write_json(persona_dir / "immutable.json", immutable)
    write_json(persona_dir / "stable.json", stable)
    write_json(persona_dir / "dynamic.json", dynamic)
    write_json(state_dir / "vectors.json", vectors)
    write_json(state_dir / "counters.json", counters)

    print("✅ Persona created successfully.")
    print("All files initialized.")


if __name__ == "__main__":
    main()