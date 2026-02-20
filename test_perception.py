# test_perception.py
from core.perception import PerceptionEngine

p = PerceptionEngine(root_dir=".", model="gpt-4.1")

r = p.analyze(
    "I walked in a quiet park with my dog. The park was peaceful and calm.",
    session_id="demo",
)

print("TURN:", r.turn)
print("ENTITIES:", r.entities)
print("RAW:", r.raw)