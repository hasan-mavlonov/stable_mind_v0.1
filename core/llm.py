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
    MINDFORM_MODEL         Default Google model (default: gemini-2.5-flash-lite)
"""

from __future__ import annotations

import functools
import importlib
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# gemini-2.5-flash-lite is Google's lowest-latency text model. It is the
# fastest sensible default for MindForm's short extraction / rephrasing
# calls; heavier models (gemini-2.5-pro) add several seconds per turn.
DEFAULT_MODEL = os.getenv("MINDFORM_MODEL", "gemma-4-31b-it")

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


@functools.lru_cache(maxsize=1)
def _build_client(api_key: str):
    genai = importlib.import_module("google.genai")
    return genai.Client(api_key=api_key)


def get_client():
    """
    Return a cached Google GenAI client.

    The client is built once per process and reused: rebuilding it on
    every call added avoidable connection-setup latency to each turn.
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

    return _build_client(api_key)


def _thinking_can_be_disabled(model_name: str) -> bool:
    # The Gemini 2.5 Flash family supports thinking_budget=0 to skip the
    # internal "thinking" pass entirely — the single biggest latency win
    # for short, low-creativity calls. gemini-2.5-pro cannot disable it,
    # and Gemma models reject the field, so only opt in for flash models.
    return model_name.lower().startswith("gemini-2.5-flash")


def _response_text(response) -> str:
    # response.text can raise or return None when a candidate has no text
    # parts (e.g. it was truncated before producing any output).
    try:
        text = response.text
    except Exception:
        text = None
    return (text or "").strip()


def _finish_reason(response) -> str:
    try:
        reason = response.candidates[0].finish_reason
    except Exception:
        return ""
    return getattr(reason, "name", str(reason or "")).upper()


def generate_text(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.4,
    max_output_tokens: Optional[int] = None,
) -> str:
    """
    Generate text with a Google model.

    Models exposed through the Gemini API do not take a separate system
    role, so any system guidance is folded into the prompt.

    `max_output_tokens` is treated as a speed optimization, not a hard
    contract: if it (combined with a model that still "thinks", e.g.
    gemini-2.5-pro) truncates the answer to empty, the call is retried
    once without the cap so the feature keeps working.
    """
    client = get_client()

    model_name = model or DEFAULT_MODEL
    contents = prompt if not system else f"{system}\n\n{prompt}"
    thinking_off = _thinking_can_be_disabled(model_name)

    def _call(cap: Optional[int]) -> str:
        config: dict = {"temperature": temperature}
        if cap is not None:
            config["max_output_tokens"] = cap
        if thinking_off:
            config["thinking_config"] = {"thinking_budget": 0}

        try:
            response = client.models.generate_content(
                model=model_name, contents=contents, config=config
            )
        except Exception:
            # If a swapped-in model rejects thinking_config, retry once
            # without it rather than failing the whole turn.
            if "thinking_config" not in config:
                raise
            config.pop("thinking_config", None)
            response = client.models.generate_content(
                model=model_name, contents=contents, config=config
            )

        return _response_text(response), _finish_reason(response)

    text, finish = _call(max_output_tokens)

    # A reasoning model can burn the whole cap on internal thinking and
    # return nothing; lift the cap once so the answer survives.
    if max_output_tokens is not None and (not text or finish == "MAX_TOKENS"):
        text, _ = _call(None)

    return text
