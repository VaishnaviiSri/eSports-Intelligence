import json
from pathlib import Path
from typing import Any

import pytest

from app.rag.vector_store import build_where_filter
from scripts.build_index import read_eligible_records


@pytest.mark.parametrize(
    ("scope", "expected_scope"),
    [
        ("general", {"doc_scope": "general"}),
        ("event_specific", {"doc_scope": "event_specific"}),
        ("auto", {"doc_scope": {"$in": ["general", "event_specific"]}}),
    ],
)
def test_where_filter_has_content_allowlist(scope: str, expected_scope: dict[str, Any]) -> None:
    result = build_where_filter(scope)
    assert result == {
        "$and": [
            {"content_type": {"$in": ["text", "table", "formula_image"]}},
            expected_scope,
        ]
    }


def test_unsupported_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported scope"):
        build_where_filter("not_applicable")


def test_index_excludes_front_matter_forms_and_unknown_scopes(tmp_path: Path) -> None:
    source = tmp_path / "chunks.jsonl"
    records = [
        {"chunk_id": "a", "page_number": 1, "content_type": "front_matter", "doc_scope": "not_applicable", "content": "TOC"},
        {"chunk_id": "b", "page_number": 2, "content_type": "legal_form", "doc_scope": "not_applicable", "content": "form"},
        {"chunk_id": "c", "page_number": 3, "content_type": "text", "doc_scope": "unknown", "content": "unknown"},
        {"chunk_id": "d", "page_number": 4, "content_type": "text", "doc_scope": "general", "content": "rule"},
        {"chunk_id": "e", "page_number": 5, "content_type": "table", "doc_scope": "event_specific", "content": [["A", "B"]]},
    ]
    source.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    eligible, stats = read_eligible_records(source)
    assert [record["chunk_id"] for record in eligible] == ["d", "e"]
    assert stats == {"read": 5, "eligible": 2, "excluded": 3, "invalid": 0}
    assert eligible[1]["_embedding_document"] == "Table:\nA | B"
