from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    model: str = "llama-3.1-8b-instant"


def load_groq_config(*, secrets: dict[str, object] | None = None) -> GroqConfig:
    secrets = secrets or {}
    api_key = str(secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in Streamlit Secrets.")

    model = str(secrets.get("GROQ_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
    return GroqConfig(api_key=api_key, model=model)


def groq_chat(*, config: GroqConfig, system_prompt: str, user_prompt: str) -> str:
    try:
        from groq import Groq
    except Exception as e:
        raise RuntimeError("Missing dependency: groq") from e

    client = Groq(api_key=config.api_key)
    resp = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    content = (resp.choices[0].message.content or "").strip()
    return content

