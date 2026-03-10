#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from core.agent import Agent


def run_interactive_test_step(
    user_text: str,
    root: str = ".",
    model: str = "gpt-4.1-mini",
    session_id: str = "cli",
) -> Dict[str, Any]:
    """
    Run one StableMind interactive test step and return structured output.
    """
    cleaned_text = (user_text or "").strip()
    if not cleaned_text:
        return {
            "input": "",
            "turn": None,
            "perceived_entities": [],
            "buffer_size": 0,
            "did_ruminate": False,
            "beliefs": {},
            "error": "Empty input.",
        }

    agent = Agent(root_dir=root, model=model)
    out = agent.step(cleaned_text, session_id=session_id)

    return {
        "input": cleaned_text,
        "turn": out.get("turn"),
        "perceived_entities": out.get("perceived_entities", []),
        "buffer_size": out.get("buffer_size"),
        "did_ruminate": out.get("did_ruminate"),
        "beliefs": out.get("beliefs", {}),
        "raw_output": out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive StableMind test CLI")
    parser.add_argument("--root", type=str, default=".", help="Project root directory")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--session", type=str, default="cli", help="Session id")
    parser.add_argument("--show-beliefs", action="store_true", help="Print beliefs each turn")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("StableMind interactive test")
    print("Type a message and press Enter.")
    print("Commands: /exit, /quit, /help")
    print("Stop anytime: Ctrl+C\n")

    while True:
        try:
            text = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

        if not text:
            continue

        if text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Exiting.")
            return

        if text.lower() in {"/help", "help"}:
            print("Commands:")
            print("  /exit, /quit  - exit the program")
            print("  /help         - show this help")
            print("Flags:")
            print("  --show-beliefs  - print stable beliefs each turn")
            print("  --pretty        - pretty-print JSON output")
            continue

        result = run_interactive_test_step(
            user_text=text,
            root=args.root,
            model=args.model,
            session_id=args.session,
        )

        print(f"\nTURN: {result.get('turn')}")
        print(f"INPUT: {result.get('input')}")

        perceived = result.get("perceived_entities", [])
        if args.pretty:
            print("PERCEIVED:", json.dumps(perceived, indent=2, ensure_ascii=False))
        else:
            print("PERCEIVED:", perceived)

        print(f"BUFFER_SIZE: {result.get('buffer_size')}")
        print(f"DID_RUMINATE: {result.get('did_ruminate')}")

        if args.show_beliefs:
            beliefs = result.get("beliefs", {})
            if args.pretty:
                print("BELIEFS:", json.dumps(beliefs, indent=2, ensure_ascii=False))
            else:
                print("BELIEFS:", beliefs)

        print("")


if __name__ == "__main__":
    main()