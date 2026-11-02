"""Retrieve-ground-generate flow with server-side citation validation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.config import settings
from app.qa import llm_client
from app.qa.prompt_builder import build_messages
from app.rag.retriever import RetrievedChunk, retrieve
from app.rag.vector_store import get_collection
from app.schemas.chat import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "I don't know based on the provided rulebook."


class IndexUnavailableError(RuntimeError):
    """Raised when no Chroma collection has been indexed."""


def _fallback_with_raw_response(
    request: ChatRequest, raw_response: str
) -> ChatResponse:
    logger.error(f"RAW LLM RESPONSE: {raw_response}")
    return ChatResponse(answer=FALLBACK_ANSWER, citations=[], scope_used=request.scope)


def _source_text(chunk: RetrievedChunk) -> str:
    """Create a concise citation snippet from the retrieved source passage."""
    text = chunk.content.strip()
    if not text:
        return "Source chunk contains a formula image; see the cited page."
    return text[:320]


def _parse_llm_result(raw: str) -> tuple[str, list[str]] | None:
    """Validate the model's strict JSON object and citation ID list shape."""
    try:
        value: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"answer", "citation_ids"}:
        return None
    answer = value["answer"]
    citation_ids = value["citation_ids"]
    if not isinstance(answer, str) or not answer.strip():
        return None
    if not isinstance(citation_ids, list) or any(
        not isinstance(item, str) or not re.fullmatch(r"C\d+", item)
        for item in citation_ids
    ):
        return None
    return answer.strip(), citation_ids


def _make_citations(
    chunks: list[RetrievedChunk], citation_ids: list[str]
) -> list[Citation] | None:
    """Resolve every model citation ID against only the supplied retrieved chunks."""
    chunk_by_id = {f"C{index}": chunk for index, chunk in enumerate(chunks, start=1)}
    if any(identifier not in chunk_by_id for identifier in citation_ids):
        return None
    citations: list[Citation] = []
    seen: set[str] = set()
    for identifier in citation_ids:
        if identifier in seen:
            continue
        seen.add(identifier)
        chunk = chunk_by_id[identifier]
        page_number = chunk.metadata.get("page_number")
        if not isinstance(page_number, int) or page_number < 1:
            return None
        heading = chunk.metadata.get("heading")
        citations.append(
            Citation(
                page_number=page_number,
                section=str(heading) if heading else "Section heading unavailable",
                content_snippet=_source_text(chunk),
            )
        )
    return citations


def answer_question(request: ChatRequest) -> ChatResponse:
    """Run retrieval and grounded generation, resolving citations server-side."""
    collection = get_collection()
    if collection.count() == 0:
        raise IndexUnavailableError("The rulebook index is empty. Run scripts.build_index first.")

    chunks = retrieve(request.question, scope=request.scope, k=settings.retrieval_top_k)
    if not chunks:
        return ChatResponse(answer=FALLBACK_ANSWER, citations=[], scope_used=request.scope)

    raw_result = llm_client.generate(build_messages(request.question, request.scope, chunks))
    parsed = _parse_llm_result(raw_result)
    if parsed is None:
        return _fallback_with_raw_response(request, raw_result)
    answer, citation_ids = parsed
    if answer == FALLBACK_ANSWER:
        return _fallback_with_raw_response(request, raw_result)

    citations = _make_citations(chunks, citation_ids)
    if citations is None or not citations:
        return _fallback_with_raw_response(request, raw_result)
    return ChatResponse(answer=answer, citations=citations, scope_used=request.scope)