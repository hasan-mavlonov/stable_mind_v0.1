from pathlib import Path
import json

from core.llm import DEFAULT_MODEL
from create_persona import create_persona
from talk import answer_user_message
from interactive_test import run_interactive_test_step


def run_create_persona() -> None:
    project_root = Path(__file__).resolve().parent.parent
    create_persona(project_root)


def run_talk(user_text: str) -> str:
    project_root = Path(__file__).resolve().parent.parent
    return answer_user_message(user_text, root=project_root)


def read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_error": f"Invalid JSON in {path.name}"}


def read_jsonl_file(path: Path) -> list[dict]:
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    items = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            items.append({"_raw": line, "_error": "Invalid JSON"})

    return items


def get_persona_data() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    persona_dir = project_root / "persona"

    immutable = read_json_file(persona_dir / "immutable.json")
    stable = read_json_file(persona_dir / "stable.json")
    dynamic = read_json_file(persona_dir / "dynamic.json")

    return {
        "immutable": immutable,
        "stable": stable,
        "dynamic": dynamic,
    }


def get_memory_data() -> dict:
    project_root = Path(__file__).resolve().parent.parent
    memory_dir = project_root / "memory"

    return {
        "perceptions": read_jsonl_file(memory_dir / "perceptions.jsonl"),
        "perception_buffer": read_jsonl_file(memory_dir / "perception_buffer.jsonl"),
        "episodic": read_jsonl_file(memory_dir / "episodic.jsonl"),
        "consolidation_log": read_jsonl_file(memory_dir / "consolidation_log.jsonl"),
        "reflection_log": read_jsonl_file(memory_dir / "reflection_log.jsonl"),
    }


def run_interactive_test(user_text: str) -> dict:
    project_root = Path(__file__).resolve().parent.parent
    return run_interactive_test_step(
        user_text=user_text,
        root=str(project_root),
        model=DEFAULT_MODEL,
        session_id="web",
    )
