# Copyright 2026 Hasan Mavlonov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""
core/llm.py — MindForm's single entry point to the Gemma / Google AI ecosystem.

MindForm runs entirely on Google's Gemma open models, served through the
Gemini API (Google AI Studio). This module centralises client creation and
text generation so every engine (perception, talk, reflection) shares the
same Gemma configuration.

Environment variables (see .env.example):
    GEMMA_API_KEY          Primary key from https://aistudio.google.com/apikey
    GOOGLE_API_KEY         Accepted fallback
    GOOGLE_AI_STUDIO_KEY   Accepted fallback
    MINDFORM_MODEL         Default Gemma model (default: gemma-3-27b-it)
"""

from __future__ import annotations

import importlib
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("MINDFORM_MODEL", "gemma-3-27b-it")

_API_KEY_VARS = (
    "GEMMA_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_AI_STUDIO_KEY",
    "GOOGLE_GENAI_API_KEY",
)


def resolve_api_key() -> Optional[str]:
    """Return the first configured Gemma / Google AI key, if any."""
    for var in _API_KEY_VARS:
        value = os.getenv(var)
        if value and value.strip():
            return value.strip()
    return None


def get_client():
    """
    Build a Google GenAI client bound to the Gemma ecosystem.

    Raises a clear, actionable error if no key is configured so the
    MindForm UI can surface setup instructions instead of a stack trace.
    """
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "No Gemma API key found. Set GEMMA_API_KEY (or GOOGLE_API_KEY) in "
            "your environment or .env file. Create one at "
            "https://aistudio.google.com/apikey"
        )

    genai = importlib.import_module("google.genai")
    return genai.Client(api_key=api_key)


def generate_text(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.4,
) -> str:
    """
    Generate text with a Gemma model.

    Gemma models exposed through the Gemini API do not take a separate
    system role, so any system guidance is folded into the prompt.
    """
    client = get_client()

    contents = prompt if not system else f"{system}\n\n{prompt}"

    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=contents,
        config={"temperature": temperature},
    )

    return (getattr(response, "text", "") or "").strip()
