from core.perception import PerceptionEngine

p = PerceptionEngine(root_dir=".", model="gpt-4.1-mini")
r = p.analyze("The park was very quiet and empty.")
print("RAW:", r.raw)
print("ENTITIES:", r.entities)