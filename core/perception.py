# core/perception.py
from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class PerceptionResult:
    turn: int
    session_id: str
    raw: Dict[str, Any]              # raw parsed function args
    entities: List[Dict[str, Any]]   # normalized list


class PerceptionEngine:
    """
    Uses an LLM to extract *perceptions* from user text:
    - entity_type (must exist in ontology.json)
    - entity (string)
    - dimension_values: only dimensions from ontology.json
    - confidence: 0..1
    """

    TOOL_NAME = "extract_perceptions"

    def __init__(self, root_dir: str, model: str = "gpt-4.1"):
        self.root = Path(root_dir)
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.ontology = self._read_json(self.root / "config" / "ontology.json")
        self.mem_path = self.root / "memory" / "perceptions.jsonl"
        self.state_counters_path = self.root / "state" / "counters.json"

        self.mem_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- public ----------

    def analyze(self, user_text: str, session_id: str = "default") -> PerceptionResult:
        turn = self._next_turn()

        tool_schema = self._build_tool_schema()

        instructions = self._build_instructions()

        # Responses API call (correct tool + tool_choice shapes)
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_text,
            tools=[tool_schema],
            tool_choice={"type": "function", "name": self.TOOL_NAME},  # <-- correct for Responses API
        )

        args = self._extract_function_args(response)
        entities = self._normalize_entities(args)

        # write only after success
        self._append_jsonl({
            "turn": turn,
            "session_id": session_id,
            "user_text": user_text,
            "entities": entities,
        })

        return PerceptionResult(
            turn=turn,
            session_id=session_id,
            raw=args,
            entities=entities,
        )

    # ---------- tool schema ----------

    def _build_tool_schema(self) -> Dict[str, Any]:
        """
        Build function schema for perception extraction.
        Enforces:
          - known entity_type
          - dimension_values required
          - at least one dimension per entity
          - values in [-1, 1]
        """
        entity_types = self.ontology.get("entity_types", {})
        allowed_types = sorted(entity_types.keys())

        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "description": "List of extracted entity perceptions.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_type": {
                                "type": "string",
                                "enum": allowed_types,
                                "description": "Must be one of the known entity types from ontology."
                            },
                            "entity": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Name of the entity mentioned in text."
                            },
                            "dimension_values": {
                                "type": "object",
                                "description": (
                                    "Dictionary of dimension -> numeric value in [-1,1]. "
                                    "Must contain at least one key if entity is returned."
                                ),
                                "minProperties": 1,
                                "additionalProperties": {
                                    "type": "number",
                                    "minimum": -1.0,
                                    "maximum": 1.0
                                }
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence in this extraction."
                            },
                        },
                        "required": [
                            "entity_type",
                            "entity",
                            "dimension_values",
                            "confidence"
                        ],
                        "additionalProperties": False,
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        }

        return {
            "type": "function",
            "name": self.TOOL_NAME,
            "description": (
                "Extract evaluated entities and supported dimension ratings from text. "
                "Only use entity types and dimensions defined in ontology."
            ),
            "parameters": schema,
        }

    def _build_instructions(self) -> str:
        return (
            "You are a perception extractor.\n"
            "Return JSON arguments for the function tool `extract_perceptions`.\n\n"
            "Hard rules:\n"
            "1) Only output entities whose `entity_type` exists in the ontology.\n"
            "2) If you output an entity, you MUST output `dimension_values` with at least ONE dimension.\n"
            "3) Only use dimension keys that exist for that `entity_type` in the ontology.\n"
            "4) Only include a dimension if it is explicitly supported by the text.\n"
            "5) Values must be within [-1, 1]. Use strong values (e.g. 0.6–1.0) only when strongly supported.\n"
            "6) If nothing qualifies, return `entities: []`.\n\n"
            "Example:\n"
            "Text: 'The cafe was loud and crowded.'\n"
            "Output entities: [{entity_type:'place', entity:'cafe', dimension_values:{quietness:-0.8, crowdedness:0.8}, confidence:0.8}]\n"
        )

    # ---------- parsing ----------

    def _extract_function_args(self, response: Any) -> Dict[str, Any]:
        """
        Responses API returns tool calls in response.output.
        We find the function_call with name == TOOL_NAME and parse JSON args.
        """
        tool_calls = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(item)

        for call in tool_calls:
            if getattr(call, "name", None) == self.TOOL_NAME:
                raw_args = getattr(call, "arguments", "{}")
                try:
                    return json.loads(raw_args)
                except json.JSONDecodeError:
                    raise RuntimeError(f"Tool returned non-JSON arguments: {raw_args}")

        # If model didn't call the tool (shouldn't happen because we forced it)
        raise RuntimeError("No function_call returned for extract_perceptions.")

    def _normalize_entities(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enforce: known entity_type, known dimensions for that type, clamp values to [-1, 1].
        Also: ignore unknown entity types (your requirement).
        """
        entity_types = self.ontology.get("entity_types", {})
        out: List[Dict[str, Any]] = []

        entities = args.get("entities", [])
        if not isinstance(entities, list):
            return out

        for e in entities:
            if not isinstance(e, dict):
                continue
            et = e.get("entity_type")
            name = e.get("entity")
            dv = e.get("dimension_values", {})
            conf = e.get("confidence")

            if et not in entity_types:
                continue  # ignore unknown type

            dims_allowed = set((entity_types[et].get("dimensions") or {}).keys())

            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(dv, dict):
                continue

            filtered: Dict[str, float] = {}
            for k, v in dv.items():
                if k not in dims_allowed:
                    continue
                try:
                    vv = float(v)
                except Exception:
                    continue
                # clamp [-1, 1]
                if vv < -1.0:
                    vv = -1.0
                if vv > 1.0:
                    vv = 1.0
                filtered[k] = vv

            # only keep if at least 1 dimension is supported
            if not filtered:
                continue

            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.5
            conf_f = max(0.0, min(1.0, conf_f))

            out.append({
                "entity_type": et,
                "entity": name.strip(),
                "dimension_values": filtered,
                "confidence": conf_f,
            })

        return out

    # ---------- state + io ----------

    def _next_turn(self) -> int:
        counters = self._read_json(self.state_counters_path)
        cur = int(counters.get("current_turn", 0))
        cur += 1
        counters["current_turn"] = cur
        self._write_json(self.state_counters_path, counters)
        return cur

    def _append_jsonl(self, obj: Dict[str, Any]) -> None:
        self.mem_path.parent.mkdir(parents=True, exist_ok=True)
        with self.mem_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)