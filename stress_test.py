from core.agent import Agent
import json

a = Agent(root_dir=".", model="gpt-4.1-mini")

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
]

for text in test_inputs:
    out = a.step(text, session_id="stress")
    print("\nTURN:", out["turn"])
    print("INPUT:", text)
    print("PERCEIVED:", json.dumps(out["perceived_entities"], indent=2, ensure_ascii=False))
    print("BUFFER APPENDED:", out["buffer_appended"])
    print("BUFFER SIZE (uncommitted):", out["buffer_size"])
    print("DID RUMINATE:", out["did_ruminate"])
    print("PLACE BELIEFS:", json.dumps(out["beliefs"].get("place", {}), indent=2, ensure_ascii=False))
    print("PERSON BELIEFS:", json.dumps(out["beliefs"].get("person", {}), indent=2, ensure_ascii=False))
    print("CONCEPT BELIEFS:", json.dumps(out["beliefs"].get("concept", {}), indent=2, ensure_ascii=False))