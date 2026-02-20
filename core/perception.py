# core/perception.py
from __future__ import annotations

import os

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import importlib


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
        openai_module = importlib.import_module("openai")
        self.client = openai_module.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

            "Map text to the most appropriate dimension using semantic understanding.\n"
            "Do not invent entities not mentioned in the text.\n\n"

            "Mapping guidance:\n"
            "- Noise/silence -> quietness.\n"
            "- Number of people -> crowdedness.\n"
            "- Light/dark -> brightness.\n"
            "- Comfort/cozy/stuffy -> comfort.\n"
            "- Beauty/ugliness -> aesthetic.\n"
            "- Anxiety/yelling/calm (person) -> emotional_stability.\n"
            "- Kindness/support -> warmth.\n"
            "- Importance/meaning -> importance.\n"
            "- Overrated/bad/good (concept) -> valence.\n\n"

            "Hard constraints:\n"
            "1) Only use entity types defined in ontology.\n"
            "2) Only use dimensions defined for that entity type.\n"
            "3) If entity is returned, at least one dimension must be present.\n"
            "4) Use values in [-1, 1]. Strong wording -> larger magnitude.\n"
        )

    # ---------- parsing ----------

    def _extract_function_args(self, response: Any) -> Dict[str, Any]:
        """
        Extract tool call arguments from Responses API.
        """
        def _parse_args(raw_args: Any) -> Dict[str, Any]:
            if isinstance(raw_args, dict):
                return raw_args
            if isinstance(raw_args, str):
                try:
                    return json.loads(raw_args)
                except Exception:
                    return {"entities": []}
            return {"entities": []}

        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)

            # Most common Responses API structure for tools
            if item_type in {"function_call", "tool_call"}:
                if getattr(item, "name", None) == self.TOOL_NAME:
                    return _parse_args(getattr(item, "arguments", "{}"))

            # Some SDK versions nest tool calls under item.content
            content = getattr(item, "content", []) or []
            for c in content:
                if getattr(c, "type", None) in {"function_call", "tool_call"}:
                    if getattr(c, "name", None) == self.TOOL_NAME:
                        return _parse_args(getattr(c, "arguments", "{}"))

        # Fallback: maybe model responded as plain text JSON
        try:
            text = response.output_text
            return json.loads(text)
        except Exception:
            return {"entities": []}

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
