"""Pydantic schemas for grounded chat requests and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Scope = Literal["auto", "general", "event_specific"]


class ChatRequest(BaseModel):
    """User question and mandatory rule-scope selection."""

    question: str = Field(min_length=1, max_length=4000)
    scope: Scope = "auto"

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        return value


class Citation(BaseModel):
    """Citation details constructed from an indexed source chunk."""

    page_number: int = Field(ge=1)
    section: str
    content_snippet: str


class ChatResponse(BaseModel):
    """Grounded answer, verified source citations, and searched scope."""

    answer: str
    citations: list[Citation]
    scope_used: Scope