"""Paper prompt construction using the model-owned chat template."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = "You are a clinical assistant. Summarize the note into a Brief Hospital Course."
PROMPT_IDENTIFIER = "surfacelora-base-v1"


def render_prompt(
    tokenizer: Any,
    source_text: str,
    *,
    add_generation_prompt: bool = True,
) -> str:
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must be a non-empty string")
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source_text},
        ],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
