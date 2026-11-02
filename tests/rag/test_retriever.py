from typing import Any

from app.rag import retriever
from app.rag.retriever import RetrievedChunk


class FakeCollection:
    def __init__(self) -> None:
        self.where: dict[str, Any] | None = None

    def count(self) -> int:
        return 6

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.where = kwargs["where"]
        return {
            "ids": [["a", "b", "c", "d", "e", "bad"]],
            "documents": [["one", "two", "three", "four", "five", "illegal"]],
            "metadatas": [[
                {"page_number": 1, "doc_scope": "general", "content_type": "text"},
                {"page_number": 1, "doc_scope": "general", "content_type": "text"},
                {"page_number": 2, "doc_scope": "general", "content_type": "table"},
                {"page_number": 3, "doc_scope": "general", "content_type": "formula_image"},
                {"page_number": 4, "doc_scope": "general", "content_type": "text"},
                {"page_number": 5, "doc_scope": "general", "content_type": "legal_form"},
            ]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.01]],
        }


def test_retrieve_applies_scope_filter_and_excludes_non_rules(monkeypatch) -> None:
    collection = FakeCollection()
    monkeypatch.setattr(retriever, "get_collection", lambda: collection)
    monkeypatch.setattr(retriever, "embed_query", lambda _: [0.0, 1.0])
    monkeypatch.setattr(retriever.settings, "retrieval_max_distance", 0.6)

    results = retriever.retrieve("question", "general", k=4)
    assert [result.chunk_id for result in results] == ["a", "b", "c", "d"]
    assert collection.where == {
        "$and": [
            {"content_type": {"$in": ["text", "table", "formula_image"]}},
            {"doc_scope": "general"},
        ]
    }


def test_retrieve_applies_distance_threshold(monkeypatch) -> None:
    collection = FakeCollection()
    monkeypatch.setattr(retriever, "get_collection", lambda: collection)
    monkeypatch.setattr(retriever, "embed_query", lambda _: [0.0, 1.0])
    monkeypatch.setattr(retriever.settings, "retrieval_max_distance", 0.25)
    results = retriever.retrieve("question", "auto", k=5)
    assert [result.chunk_id for result in results] == ["a", "b"]


def test_select_diverse_results_deduplicates_content_and_blocks() -> None:
    candidates = [
        RetrievedChunk("a", "same", {"page_number": 1}, 0.1),
        RetrievedChunk("b", "same", {"page_number": 2}, 0.2),
        RetrievedChunk("c", "other", {"page_number": 1}, 0.3),
        RetrievedChunk("d", "third", {"page_number": 1}, 0.4),
        RetrievedChunk("e", "fourth", {"page_number": 1}, 0.5),
        RetrievedChunk("f", "fifth", {"page_number": 2}, 0.6),
    ]
    selected = retriever.select_diverse_results(candidates, 5, max_per_page=2)
    assert [item.chunk_id for item in selected] == ["a", "c", "f", "d", "e"]


def test_empty_collection_skips_embedding(monkeypatch) -> None:
    class EmptyCollection:
        def count(self) -> int:
            return 0

    monkeypatch.setattr(retriever, "get_collection", lambda: EmptyCollection())
    monkeypatch.setattr(retriever, "embed_query", lambda _: (_ for _ in ()).throw(AssertionError()))
    assert retriever.retrieve("question", "auto") == []
