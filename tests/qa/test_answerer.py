import json
from types import SimpleNamespace

import pytest

from app.qa import answerer
from app.qa.answerer import FALLBACK_ANSWER, IndexUnavailableError
from app.rag.retriever import RetrievedChunk
from app.schemas.chat import ChatRequest


class NonEmptyCollection:
    def count(self) -> int:
        return 2


def _chunks():
    return [
        RetrievedChunk(
            "chunk-a",
            "Players must be at least 16 years old.",
            {"page_number": 8, "heading": "2.1 Eligibility", "doc_scope": "general"},
            0.1,
        ),
        RetrievedChunk(
            "chunk-b",
            "The minimum age is repeated here.",
            {"page_number": 9, "heading": None, "doc_scope": "general"},
            0.2,
        ),
    ]


def test_answer_maps_citations_from_retrieved_chunks(monkeypatch) -> None:
    monkeypatch.setattr(answerer, "get_collection", lambda: NonEmptyCollection())
    monkeypatch.setattr(answerer, "retrieve", lambda *args, **kwargs: _chunks())
    monkeypatch.setattr(
        answerer.llm_client,
        "generate",
        lambda _: json.dumps({"answer": "The minimum age is 16 [C1].", "citation_ids": ["C1"]}),
    )
    result = answerer.answer_question(ChatRequest(question="age?", scope="general"))
    assert result.answer == "The minimum age is 16 [C1]."
    assert result.citations[0].page_number == 8
    assert result.citations[0].section == "2.1 Eligibility"
    assert result.citations[0].content_snippet == _chunks()[0].content


def test_unknown_llm_citation_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(answerer, "get_collection", lambda: NonEmptyCollection())
    monkeypatch.setattr(answerer, "retrieve", lambda *args, **kwargs: _chunks())
    monkeypatch.setattr(
        answerer.llm_client,
        "generate",
        lambda _: json.dumps({"answer": "Supported [C9].", "citation_ids": ["C9"]}),
    )
    result = answerer.answer_question(ChatRequest(question="age?", scope="general"))
    assert result.answer == FALLBACK_ANSWER
    assert result.citations == []


def test_uncited_claim_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(answerer, "get_collection", lambda: NonEmptyCollection())
    monkeypatch.setattr(answerer, "retrieve", lambda *args, **kwargs: _chunks())
    monkeypatch.setattr(
        answerer.llm_client,
        "generate",
        lambda _: json.dumps({"answer": "The minimum age is 16.", "citation_ids": []}),
    )
    result = answerer.answer_question(ChatRequest(question="age?"))
    assert result.answer == FALLBACK_ANSWER


def test_empty_index_raises_without_retrieval(monkeypatch) -> None:
    class EmptyCollection:
        def count(self) -> int:
            return 0

    monkeypatch.setattr(answerer, "get_collection", lambda: EmptyCollection())
    monkeypatch.setattr(answerer, "retrieve", lambda *args, **kwargs: pytest.fail("must not retrieve"))
    with pytest.raises(IndexUnavailableError):
        answerer.answer_question(ChatRequest(question="age?"))


def test_empty_retrieval_returns_exact_fallback_without_llm(monkeypatch) -> None:
    monkeypatch.setattr(answerer, "get_collection", lambda: NonEmptyCollection())
    monkeypatch.setattr(answerer, "retrieve", lambda *args, **kwargs: [])
    monkeypatch.setattr(answerer.llm_client, "generate", lambda _: pytest.fail("must not call LLM"))
    result = answerer.answer_question(ChatRequest(question="unknown?"))
    assert result.answer == FALLBACK_ANSWER
    assert result.citations == []
