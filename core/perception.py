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
    Uses an LLM to extract *perceptions* from user text.

    Tool-output format:
    - entity_type (must exist in ontology.json)
    - entity (string)
    - dimensions: list of {"dimension": str, "value": float in [-1,1]}
    - confidence: 0..1

    Internal normalized format:
    - dimension_values: Dict[dimension -> float] filtered to ontology dimensions
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

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_text,
            tools=[tool_schema],
            tool_choice={"type": "function", "name": self.TOOL_NAME},
            temperature=0.2,
        )

        args = self._extract_function_args(response)
        args = self._validate_tool_args_shape(args)
        entities = self._normalize_entities(args)

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
        OpenAI tool-parameter JSON schema is a restricted subset.
        Use an array of (dimension, value) objects (not open-ended dict).
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
        """
        Key fixes:
        - Make self_state extraction explicit and usable with your current ontology (confidence/stress/energy)
        - Make negative relationship statements map to person.warmth negative (boyfriend hates me etc.)
        - Keep your rule: no valence for neutral factual statements
        """
        return (
            "You are a perception extractor.\n"
            "Return JSON arguments for the function tool `extract_perceptions`.\n\n"

            "Output format:\n"
            '{"entities":[{"entity_type":"...","entity":"...","dimensions":[{"dimension":"...","value":0.0}],"confidence":0.0}]}\n\n'

            "Rules:\n"
            "1) Only use entity types defined in ontology.\n"
            "2) Only use dimensions defined for that entity type.\n"
            "3) If you return an entity, `dimensions` MUST contain at least one item.\n"
            "4) Values must be in [-1, 1]. Stronger wording -> larger magnitude.\n"
            "5) If the text does not support ANY dimension from the ontology, return {\"entities\":[]}.\n\n"

            "IMPORTANT behavior constraints:\n"
            "- Do NOT create entity_type='person' for first-person pronouns (I/me/my).\n"
            "- If the text explicitly describes the speaker's internal state, use entity_type='self_state' with entity='self'.\n"
            "- Do NOT treat animals/objects as entity_type='person'. If unsure, prefer entity_type='concept'.\n"
            "- Do NOT assign concept.valence to purely factual statements without sentiment.\n\n"

            "SELF-STATE mapping (use these when explicitly stated):\n"
            "- \"I am stressed/anxious/overwhelmed/scared/panicking\" -> self_state:self, dimension 'stress' HIGH (positive).\n"
            "- \"I am calm/relaxed\" -> self_state:self, dimension 'stress' LOW (negative).\n"
            "- \"I am tired/exhausted/drained\" OR \"I want to sleep\" -> self_state:self, dimension 'energy' LOW (negative).\n"
            "- \"I am energized/full of energy\" -> self_state:self, dimension 'energy' HIGH (positive).\n"
            "- \"I feel confident\" -> self_state:self, dimension 'confidence' HIGH (positive).\n"
            "- \"I doubt myself / I can't do this\" -> self_state:self, dimension 'confidence' LOW (negative).\n"
            "- \"I feel sad/bad/hurt\" ONLY if your ontology has a matching self_state dimension. If not, DO NOT invent one.\n"
            "  If no mood/valence dimension exists, prefer mapping sadness ONLY when it clearly implies stress (e.g., overwhelmed/heartbroken) -> stress HIGH.\n\n"

            "RELATIONSHIP / PERSON mapping:\n"
            "- \"X is kind/supportive/caring\" -> person:X, warmth HIGH.\n"
            "- \"X is cold/mean/hostile\" -> person:X, warmth LOW.\n"
            "- \"X hates me\" or \"X doesn't like me\" or \"X doesn't love me\" -> person:X, warmth LOW.\n"
            "- \"X broke up with me\" -> person:X, warmth LOW ONLY if the text frames it as rejection/hurt. Otherwise omit.\n\n"

            "Preference mapping (IMPORTANT):\n"
            "- \"I like/love/enjoy X\" -> concept:X, dimension 'valence' positive (and optionally 'importance' if emphasized).\n"
            "- \"I dislike/hate X\" -> concept:X, dimension 'valence' negative.\n\n"

            "Place mapping guidance:\n"
            "- Noise/silence -> quietness.\n"
            "- Number of people -> crowdedness.\n"
            "- Light/dark -> brightness.\n"
            "- Comfort/cozy/stuffy -> comfort.\n"
            "- Beauty/ugliness -> aesthetic.\n\n"

            "Person mapping guidance:\n"
            "- Anxiety/yelling/calm (person) -> emotional_stability.\n"
            "- Kindness/support -> warmth.\n\n"

            "Concept mapping guidance:\n"
            "- Importance/meaning -> importance.\n"
            "- Overrated/bad/good (concept) -> valence.\n\n"

            "Example:\n"
            'Input: \"The park was empty and very quiet.\"\\n'
            "Return:\\n"
            '{"entities":[{"entity_type":"place","entity":"park","dimensions":[{"dimension":"quietness","value":0.8},{"dimension":"crowdedness","value":-0.7}],"confidence":0.9}]}\n'
        )

    # ---------- parsing ----------

    def _extract_function_args(self, response: Any) -> Dict[str, Any]:
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

        try:
            text = response.output_text
            return json.loads(text)
        except Exception:
            return {"entities": []}

    def _validate_tool_args_shape(self, args: Dict[str, Any]) -> Dict[str, Any]:
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

                if vv < -1.0:
                    vv = -1.0
                if vv > 1.0:
                    vv = 1.0

                filtered[k] = vv

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