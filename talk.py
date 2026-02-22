#!/usr/bin/env python3
"""
talk.py — READ-ONLY conversational interface for your persona.

Contract:
- Reads persona/*.json (immutable, stable, dynamic)
- NEVER calls Agent.step() / PerceptionEngine / ConsolidationEngine
- NEVER writes to persona/, memory/, or state/
- Answers only from existing stable beliefs (plus optional dynamic mood coloring)
- If no belief exists for the asked target: returns exactly a short "no belief" response.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import importlib

from dotenv import load_dotenv
load_dotenv()


# ----------------------------
# Utilities (read-only)
# ----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def clamp(x: float, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, v))

def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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
    out: List[Tuple[str, str, str]] = []
    beliefs = stable.get("beliefs", {})
    if not isinstance(beliefs, dict):
        return out

    for etype, entities in beliefs.items():
        if not isinstance(entities, dict):
            continue
        for ename, dims in entities.items():
            if not isinstance(dims, dict):
                continue
            for dim in dims.keys():
                out.append((str(etype), str(ename), str(dim)))
    return out

def extract_entity_beliefs(stable: Dict[str, Any], entity_type: str, entity: str) -> List[BeliefFact]:
    beliefs = safe_get(stable, "beliefs", entity_type, entity, default={})
    if not isinstance(beliefs, dict):
        return []

    facts: List[BeliefFact] = []
    for dim, payload in beliefs.items():
        if not isinstance(payload, dict):
            continue
        mean = clamp(payload.get("mean", 0.0), -1.0, 1.0)
        conf = clamp(payload.get("confidence", 0.0), 0.0, 1.0)
        n = int(payload.get("n", 0) or 0)
        facts.append(BeliefFact(entity_type=entity_type, entity=entity, dimension=str(dim), mean=mean, confidence=conf, n=n))
    return facts

def qualitative_polarity(mean: float) -> str:
    # deterministic, minimal
    if mean <= -0.35:
        return "negative"
    if mean >= 0.35:
        return "positive"
    return "neutral"

def intensity_bucket(mean: float) -> str:
    a = abs(mean)
    if a < 0.2:
        return "weak"
    if a < 0.5:
        return "mild"
    if a < 0.8:
        return "strong"
    return "very strong"

def belief_facts_to_plain_statements(facts: List[BeliefFact]) -> List[str]:
    """
    Turn facts into minimal, non-creative statements.
    The LLM is later allowed only to rephrase these in the persona's tone.
    """
    lines: List[str] = []
    for f in sorted(facts, key=lambda x: (x.entity_type, x.entity, x.dimension)):
        pol = qualitative_polarity(f.mean)
        inten = intensity_bucket(f.mean) if pol != "neutral" else "weak"
        lines.append(
            f"- {f.entity_type}:{f.entity} | {f.dimension}: {pol} ({inten}); "
            f"mean={f.mean:.3f}; confidence={f.confidence:.3f}; n={f.n}"
        )
    return lines


# ----------------------------
# LLM helpers
# ----------------------------

class LLM:
    def __init__(self, model: str = "gpt-4.1-mini"):
        openai_module = importlib.import_module("openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        self.client = openai_module.OpenAI(api_key=api_key)
        self.model = model

    def classify_target(self, user_text: str, known_keys: List[Tuple[str, str, str]]) -> Optional[Tuple[str, str]]:
        """
        Uses an LLM to pick a (entity_type, entity) target from known belief keys.
        IMPORTANT: It can only select from what already exists.
        If it can't confidently select, return None.
        """
        # Build a compact menu of known entities (entity_type, entity)
        pairs = sorted({(et, en) for (et, en, _dim) in known_keys})
        # Keep it bounded; if it grows later, you can add smarter filtering.
        menu = "\n".join([f"- {et}:{en}" for et, en in pairs[:400]])

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

        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=f"User: {user_text}\n\nKnown entities:\n{menu}\n",
            temperature=0.0,
        )

        text = (resp.output_text or "").strip()
        if text.upper() == "NONE":
            return None
        if ":" not in text:
            return None
        et, en = text.split(":", 1)
        et, en = et.strip(), en.strip()
        # validate against known set
        if (et, en) not in set(pairs):
            return None
        return (et, en)

    def rewrite_in_tone(self, tone: str, core_traits: List[str], facts_lines: List[str], user_text: str) -> str:
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
        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=f"User asked: {user_text}\n\nAvailable beliefs:\n{facts_block}\n",
            temperature=0.4,
        )
        out = (resp.output_text or "").strip()
        return out or "I don’t have a belief about that yet."


# ----------------------------
# Read-only conversation loop
# ----------------------------

NO_BELIEF_LINE = "I don’t have a belief about that yet."

def main() -> None:
    root = Path(__file__).resolve().parent

    # Prefer project-local persona folder; fall back to current working directory if needed.
    persona_dir = root / "persona"
    if not persona_dir.exists():
        persona_dir = Path.cwd() / "persona"

    immutable = read_json(persona_dir / "immutable.json")
    stable = read_json(persona_dir / "stable.json")
    dynamic = read_json(persona_dir / "dynamic.json")

    tone = safe_get(stable, "personality", "tone_of_voice", default="Warm, introspective, occasionally poetic")
    core_traits = safe_get(stable, "personality", "core_traits", default=[])
    if not isinstance(core_traits, list):
        core_traits = []

    llm = LLM(model=os.getenv("TALK_MODEL", "gpt-4.1-mini"))

    known_keys = list_all_belief_keys(stable)

    # Small greeting in persona tone, but read-only (no beliefs required)
    name = safe_get(immutable, "display_name", default="Rin")
    print(f"{name}: Hi. Ask me anything you want to know about what I already believe.")
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

        # If there are no beliefs at all, we must not pretend.
        if not known_keys:
            print(f"{name}: {NO_BELIEF_LINE}\n")
            continue

        # Ask LLM to pick a target only from known entities.
        target = llm.classify_target(user_text, known_keys)
        if not target:
            print(f"{name}: {NO_BELIEF_LINE}\n")
            continue

        etype, ename = target
        facts = extract_entity_beliefs(stable, etype, ename)
        if not facts:
            print(f"{name}: {NO_BELIEF_LINE}\n")
            continue

        facts_lines = belief_facts_to_plain_statements(facts)

        # LLM is only allowed to rephrase these facts in-tone.
        answer = llm.rewrite_in_tone(
            tone=tone,
            core_traits=core_traits,
            facts_lines=facts_lines,
            user_text=user_text,
        )

        # Final hard-guard: if somehow it outputs empty, return no-belief
        if not answer.strip():
            answer = NO_BELIEF_LINE

        print(f"{name}: {answer}\n")


if __name__ == "__main__":
    main()