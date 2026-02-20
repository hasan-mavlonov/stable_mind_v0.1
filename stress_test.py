from core.agent import Agent
import json
import random

a = Agent(root_dir=".", model="gpt-4.1")

test_inputs = [
    # place reinforcement
    "I walked in a quiet park. It was peaceful and calm.",
    "The park was silent today.",
    "The park felt calm and relaxing.",
    "The park was noisy because of construction.",
    "The park was extremely loud today.",
    "The park felt peaceful again.",
    "The park was crowded and noisy.",
    "The park was empty and very quiet.",

    # person
    "I talked with my mother today. She was feeling anxious and shouted at me.",
    "My mother seemed calm today.",
    "My mother was supportive and kind.",
    "She was emotionally stable this time.",
    "She yelled again today.",

    # new place
    "The cafe was crowded and loud.",
    "The cafe was bright and clean.",
    "The cafe felt cozy and comfortable.",
    "The cafe was chaotic today.",

    # concept
    "I think success is very important.",
    "Success feels meaningful to me.",
    "Success is overrated sometimes."
]

for i, text in enumerate(test_inputs, start=1):
    out = a.step(text, session_id="stress")
    print("\nTURN:", out["turn"])
    print("INPUT:", text)
    print("PERCEIVED:", json.dumps(out["perceived_entities"], indent=2))
    print("PLACE BELIEFS:", json.dumps(out["beliefs"].get("place", {}), indent=2))
    print("PERSON BELIEFS:", json.dumps(out["beliefs"].get("person", {}), indent=2))
    print("CONCEPT BELIEFS:", json.dumps(out["beliefs"].get("concept", {}), indent=2))