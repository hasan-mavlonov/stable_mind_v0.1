from core.agent import Agent

a = Agent(root_dir=".", model="gpt-4.1")

out = a.step("I walked in a quiet park. It was peaceful and calm.", session_id="demo")
print(out["turn"])
print(out["perceived_entities"])
print(out["beliefs"].get("place", {}))