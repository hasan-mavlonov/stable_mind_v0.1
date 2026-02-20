# core/perception.py
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import importlib


@dataclass
class PerceptionResult:
    turn: int
    session_id: str
    raw: Dict[str, Any]              # raw parsed function args
    entities: List[Dict[str, Any]]   # normalized list (internal format)


class PerceptionEngine:
    """
    Uses an LLM to extract *perceptions* from user text:
    - entity_type (must exist in ontology.json)
    - entity (string)
    - dimensions: list of {"dimension": str, "value": float in [-1,1]}   (tool-output format)
    - confidence: 0..1

    Internally, we normalize tool-output "dimensions" into:
      dimension_values: Dict[dimension -> float] (filtered to ontology dimensions)
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

        # Debug: confirm ontology loads

    # ---------- public ----------

    def analyze(self, user_text: str, session_id: str = "default") -> PerceptionResult:
        turn = self._next_turn()

        tool_schema = self._build_tool_schema()
        instructions = self._build_instructions()

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_text,
            tools=[tool_schema],
            tool_choice={"type": "function", "name": self.TOOL_NAME},
            temperature=0.2,
        )

        # Debug (keep while developing; remove later if noisy)
        # print("=== RESPONSE TYPE ===", type(response))
        # try:
        #     print("=== RESPONSE DUMP ===")
        #     print(json.dumps(response.model_dump(), indent=2))
        # except Exception as e:
        #     print("model_dump failed:", e)
        #     print("RAW response:", response)
        # print("=== OUTPUT TEXT ===", getattr(response, "output_text", None))
        # print("=== OUTPUT FIELD ===", getattr(response, "output", None))

        args = self._extract_function_args(response)

        # Defensive validation: keep only entities that match the tool-output shape
        args = self._validate_tool_args_shape(args)

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
        IMPORTANT:
        OpenAI tool-parameter JSON schema is a restricted subset.
        - `minProperties` is not permitted
        - open-ended dict schemas via `additionalProperties` can be rejected or cause required mismatch
        So we represent dimension ratings as an array of (dimension, value) objects.
        """
        entity_types = self.ontology.get("entity_types", {})
        allowed_types = sorted(entity_types.keys())

        dim_item_props = {
            "dimension": {"type": "string"},
            "value": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        }

        item_properties = {
            "entity_type": {
                "type": "string",
                "enum": allowed_types,
                "description": "Must be one of the known entity types from ontology."
            },
            "entity": {
                "type": "string",
                "description": "Name of the entity mentioned in text."
            },
            "dimensions": {
                "type": "array",
                "description": (
                    "List of dimension ratings supported by the text. "
                    "If an entity is returned, include at least one item."
                ),
                "items": {
                    "type": "object",
                    "properties": dim_item_props,
                    "required": list(dim_item_props.keys()),
                    "additionalProperties": False,
                },
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in this extraction."
            },
        }

        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "description": "List of extracted entity perceptions.",
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": list(item_properties.keys()),
                        "additionalProperties": False,
                    },
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
            "strict": True,
        }

    def _build_instructions(self) -> str:
        return (
            "You are a perception extractor.\n"
            "You MUST return JSON arguments for the function tool `extract_perceptions`.\n\n"

            "Every returned entity MUST include ALL fields:\n"
            "- entity_type\n"
            "- entity\n"
            "- dimensions (array of {dimension, value}) with at least ONE item\n"
            "- confidence\n\n"

            "If you cannot support ANY dimension from the text, return:\n"
            '{"entities":[]}\n\n'

            "Example:\n"
            'Input: "The park was empty and very quiet."\n'
            "Return:\n"
            '{"entities":[{"entity_type":"place","entity":"park","dimensions":[{"dimension":"quietness","value":0.8},{"dimension":"crowdedness","value":-0.7}],"confidence":0.9}]}\n\n'

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
            "3) If an entity is returned, dimensions MUST contain at least one item.\n"
            "4) Use numeric values in [-1, 1]. Strong wording -> larger magnitude.\n"
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

            if item_type in {"function_call", "tool_call"}:
                if getattr(item, "name", None) == self.TOOL_NAME:
                    return _parse_args(getattr(item, "arguments", "{}"))

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

    def _validate_tool_args_shape(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the parsed args follow the tool-output shape:
        entities: [{entity_type, entity, dimensions(list), confidence}]
        """
        ents = args.get("entities", [])
        if not isinstance(ents, list):
            return {"entities": []}

        cleaned: List[Dict[str, Any]] = []
        for e in ents:
            if not isinstance(e, dict):
                continue
            if "entity_type" not in e or "entity" not in e or "dimensions" not in e or "confidence" not in e:
                continue
            if not isinstance(e.get("dimensions"), list):
                continue
            cleaned.append(e)

        return {"entities": cleaned}

    def _normalize_entities(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize tool-output entities into internal entities:
        {
          "entity_type": et,
          "entity": "...",
          "dimension_values": {dim: value},   # filtered to ontology + clamped
          "confidence": conf
        }
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
            dims_list = e.get("dimensions", [])
            conf = e.get("confidence")

            if et not in entity_types:
                continue

            dims_allowed = set((entity_types[et].get("dimensions") or {}).keys())

            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(dims_list, list):
                continue

            filtered: Dict[str, float] = {}
            for item in dims_list:
                if not isinstance(item, dict):
                    continue
                k = item.get("dimension")
                v = item.get("value")
                if not isinstance(k, str):
                    continue
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

            # require at least one supported dimension
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