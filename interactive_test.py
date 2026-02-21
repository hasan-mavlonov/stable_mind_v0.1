#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from core.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive StableMind test CLI")
    parser.add_argument("--root", type=str, default=".", help="Project root directory")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--session", type=str, default="cli", help="Session id")
    parser.add_argument("--show-beliefs", action="store_true", help="Print beliefs each turn")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    a = Agent(root_dir=args.root, model=args.model)

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

        out = a.step(text, session_id=args.session)

        # Print summary like stress_test
        print(f"\nTURN: {out.get('turn')}")
        print(f"INPUT: {text}")

        perceived = out.get("perceived_entities", [])
        if args.pretty:
            print("PERCEIVED:", json.dumps(perceived, indent=2, ensure_ascii=False))
        else:
            print("PERCEIVED:", perceived)

        print(f"BUFFER_SIZE: {out.get('buffer_size')}")
        print(f"DID_RUMINATE: {out.get('did_ruminate')}")

        if args.show_beliefs:
            beliefs = out.get("beliefs", {})
            if args.pretty:
                print("BELIEFS:", json.dumps(beliefs, indent=2, ensure_ascii=False))
            else:
                print("BELIEFS:", beliefs)

        print("")


if __name__ == "__main__":
    main()