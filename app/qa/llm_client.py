"""Groq chat-completions client using the OpenAI-compatible SDK."""

from __future__ import annotations

import logging
from functools import lru_cache

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when Groq is not configured or cannot return a response."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Create and cache the Groq-compatible OpenAI client."""
    if not settings.groq_api_key.strip():
        raise LLMUnavailableError("GROQ_API_KEY is not configured")
    return OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )


def generate(messages: list[dict[str, str]]) -> str:
    """Request a non-streaming JSON response from the configured Groq model."""
    try:
        response = _client().chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMUnavailableError("Groq returned an empty response")
        return content
    except LLMUnavailableError:
        raise
    except Exception as e:
        logger.error(
            f"Groq API Error: {type(e).__name__} - {str(e)}",
            exc_info=True,
        )
        raise LLMUnavailableError(str(e)) from e