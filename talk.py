#!/usr/bin/env python3
"""
talk.py — READ-ONLY conversational interface for your persona.

Contract:
- Reads persona/*.json (immutable, stable, dynamic)
- NEVER calls Agent.step() / PerceptionEngine / ConsolidationEngine
- NEVER writes to persona/, memory/, or state/
- Answers only from existing stable beliefs (plus optional dynamic mood coloring)
- If no belief exists for the asked target: returns exactly a short "no belief" response.

CLI usage:
    python talk.py

Reusable API:
    from talk import answer_user_message
    answer = answer_user_message("What do you think about the park?")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import json
from dotenv import load_dotenv

from core.llm import DEFAULT_MODEL, generate_text

load_dotenv()

NO_BELIEF_LINE = "I don’t have a belief about that yet."


# ----------------------------
# Utilities (read-only)
# ----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        value = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, value))


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    current: Any = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_persona_dir(root: Optional[Path] = None) -> Path:
    base_root = root.resolve() if root else get_project_root()
    persona_dir = base_root / "persona"

    if persona_dir.exists():
        return persona_dir

    fallback_dir = Path.cwd() / "persona"
    return fallback_dir


# ----------------------------
# Belief rendering (deterministic facts)
# ----------------------------

@dataclass(frozen=True)
class BeliefFact:
    entity_type: str
    entity: str
    dimension: str
    mean: float
    confidence: float
    n: int


def list_all_belief_keys(stable: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Returns (entity_type, entity, dimension) for everything present.
    """
    output: List[Tuple[str, str, str]] = []
    beliefs = stable.get("beliefs", {})

    if not isinstance(beliefs, dict):
        return output

    for entity_type, entities in beliefs.items():
        if not isinstance(entities, dict):
            continue
        for entity_name, dimensions in entities.items():
            if not isinstance(dimensions, dict):
                continue
            for dimension in dimensions.keys():
                output.append((str(entity_type), str(entity_name), str(dimension)))

    return output


def extract_entity_beliefs(
    stable: Dict[str, Any],
    entity_type: str,
    entity: str,
) -> List[BeliefFact]:
    beliefs = safe_get(stable, "beliefs", entity_type, entity, default={})
    if not isinstance(beliefs, dict):
        return []

    facts: List[BeliefFact] = []
    for dimension, payload in beliefs.items():
        if not isinstance(payload, dict):
            continue

        mean = clamp(payload.get("mean", 0.0), -1.0, 1.0)
        confidence = clamp(payload.get("confidence", 0.0), 0.0, 1.0)
        n = int(payload.get("n", 0) or 0)

        facts.append(
            BeliefFact(
                entity_type=entity_type,
                entity=entity,
                dimension=str(dimension),
                mean=mean,
                confidence=confidence,
                n=n,
            )
        )

    return facts


def qualitative_polarity(mean: float) -> str:
    if mean <= -0.35:
        return "negative"
    if mean >= 0.35:
        return "positive"
    return "neutral"


def intensity_bucket(mean: float) -> str:
    absolute = abs(mean)
    if absolute < 0.2:
        return "weak"
    if absolute < 0.5:
        return "mild"
    if absolute < 0.8:
        return "strong"
    return "very strong"


def belief_facts_to_plain_statements(facts: List[BeliefFact]) -> List[str]:
    """
    Turn facts into minimal, non-creative statements.
    The LLM is later allowed only to rephrase these in the persona's tone.
    """
    lines: List[str] = []

    for fact in sorted(facts, key=lambda item: (item.entity_type, item.entity, item.dimension)):
        polarity = qualitative_polarity(fact.mean)
        intensity = intensity_bucket(fact.mean) if polarity != "neutral" else "weak"

        lines.append(
            f"- {fact.entity_type}:{fact.entity} | {fact.dimension}: {polarity} ({intensity}); "
            f"mean={fact.mean:.3f}; confidence={fact.confidence:.3f}; n={fact.n}"
        )

    return lines


# ----------------------------
# LLM helpers
# ----------------------------

class LLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def classify_target(
        self,
        user_text: str,
        known_keys: List[Tuple[str, str, str]],
    ) -> Optional[Tuple[str, str]]:
        """
        Uses an LLM to pick a (entity_type, entity) target from known belief keys.
        IMPORTANT: It can only select from what already exists.
        If it can't confidently select, return None.
        """
        known_pairs = sorted({(entity_type, entity) for (entity_type, entity, _dim) in known_keys})
        menu = "\n".join([f"- {entity_type}:{entity}" for entity_type, entity in known_pairs[:400]])

        instructions = (
            "You are selecting which known entity the user is asking about.\n"
            "You MUST select exactly one item from the provided list, or reply with NONE.\n"
            "Return ONLY one line in one of these formats:\n"
            "1) etype:entity\n"
            "2) NONE\n"
            "No extra words.\n\n"
            "Selection rules:\n"
            "- If the user question clearly refers to one entity in the list, pick it.\n"
            "- If ambiguous or not present, return NONE.\n"
        )

        text = generate_text(
            f"User: {user_text}\n\nKnown entities:\n{menu}\n",
            model=self.model,
            system=instructions,
            temperature=0.0,
            max_output_tokens=32,
        ).strip()

        if text.upper() == "NONE":
            return None

        if ":" not in text:
            return None

        entity_type, entity = text.split(":", 1)
        entity_type = entity_type.strip()
        entity = entity.strip()

        if (entity_type, entity) not in set(known_pairs):
            return None

        return (entity_type, entity)

    def rewrite_in_tone(
        self,
        tone: str,
        core_traits: List[str],
        facts_lines: List[str],
        user_text: str,
    ) -> str:
        """
        Rephrase ONLY what is in facts_lines, in the given tone.
        No new facts. If facts are insufficient, return the no-belief line.
        """
        instructions = (
            "You are roleplaying as the persona.\n"
            "You are given ONLY factual belief statements.\n"
            "You MUST NOT add new facts, guesses, or assumptions.\n"
            "If the facts do not answer the user, respond exactly:\n"
            "\"I don’t have a belief about that yet.\"\n\n"
            f"Tone: {tone}\n"
            f"Core traits: {', '.join(core_traits) if core_traits else 'N/A'}\n\n"
            "Output requirements:\n"
            "- 1 to 3 short paragraphs max.\n"
            "- Natural conversation.\n"
            "- No bullet lists.\n"
        )

        facts_block = "\n".join(facts_lines)

        output = generate_text(
            f"User asked: {user_text}\n\nAvailable beliefs:\n{facts_block}\n",
            model=self.model,
            system=instructions,
            temperature=0.4,
            max_output_tokens=512,
        ).strip()

        return output or NO_BELIEF_LINE


# ----------------------------
# Persona loading
# ----------------------------

def load_persona_context(root: Optional[Path] = None) -> Dict[str, Any]:
    persona_dir = get_persona_dir(root)

    immutable = read_json(persona_dir / "immutable.json")
    stable = read_json(persona_dir / "stable.json")
    dynamic = read_json(persona_dir / "dynamic.json")

    tone = safe_get(
        stable,
        "personality",
        "tone_of_voice",
        default="Warm, introspective, occasionally poetic",
    )

    core_traits = safe_get(stable, "personality", "core_traits", default=[])
    if not isinstance(core_traits, list):
        core_traits = []

    display_name = safe_get(immutable, "display_name", default="Rin")

    return {
        "immutable": immutable,
        "stable": stable,
        "dynamic": dynamic,
        "tone": tone,
        "core_traits": core_traits,
        "display_name": display_name,
    }


# ----------------------------
# One-turn read-only answer API
# ----------------------------

def answer_user_message(user_text: str, root: Optional[Path] = None) -> str:
    cleaned_user_text = (user_text or "").strip()
    if not cleaned_user_text:
        return NO_BELIEF_LINE

    persona_context = load_persona_context(root)
    stable = persona_context["stable"]
    tone = persona_context["tone"]
    core_traits = persona_context["core_traits"]

    known_keys = list_all_belief_keys(stable)
    if not known_keys:
        return NO_BELIEF_LINE

    llm = LLM(model=os.getenv("TALK_MODEL", DEFAULT_MODEL))

    target = llm.classify_target(cleaned_user_text, known_keys)
    if not target:
        return NO_BELIEF_LINE

    entity_type, entity_name = target
    facts = extract_entity_beliefs(stable, entity_type, entity_name)
    if not facts:
        return NO_BELIEF_LINE

    facts_lines = belief_facts_to_plain_statements(facts)

    answer = llm.rewrite_in_tone(
        tone=tone,
        core_traits=core_traits,
        facts_lines=facts_lines,
        user_text=cleaned_user_text,
    )

    return answer.strip() or NO_BELIEF_LINE


# ----------------------------
# CLI interface
# ----------------------------

def main() -> None:
    persona_context = load_persona_context()
    display_name = persona_context["display_name"]

    print(f"{display_name}: Hi. Ask me anything you want to know about what I already believe.")
    print("(Type :exit to quit)\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_text:
            continue

        if user_text == ":exit":
            print("Bye.")
            return

        answer = answer_user_message(user_text)
        print(f"{display_name}: {answer}\n")


if __name__ == "__main__":
    main()