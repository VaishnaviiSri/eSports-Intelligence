from app.processing.chunker import Chunk
from app.processing.metadata import build_chunk_metadata, extract_document_metadata
from app.processing.structure import DocumentProfile


def test_document_date_is_not_assumed_effective_date() -> None:
    metadata = extract_document_metadata(
        "BGMI Rules Version 1.0.0, October 23, 2026", DocumentProfile(), "rules.pdf"
    )
    assert metadata.version == "1.0.0"
    assert metadata.document_date == "2026-10-23"
    assert metadata.effective_date is None


def test_effective_date_requires_explicit_label() -> None:
    metadata = extract_document_metadata(
        "Version 1.0.0\nEffective Date: 2026-11-01", DocumentProfile(), "rules.pdf"
    )
    assert metadata.effective_date == "2026-11-01"


def test_missing_title_page_metadata_is_reported() -> None:
    metadata = extract_document_metadata("No version or date", DocumentProfile(), "rules.pdf")
    assert metadata.version is None
    assert metadata.document_date is None
    assert metadata.effective_date is None
    assert len(metadata.warnings) == 2


def test_chunk_metadata_separates_scope_and_content_type() -> None:
    document = extract_document_metadata(
        "Version 1.0.0, October 23, 2026", DocumentProfile(), "rules.pdf"
    )
    chunk = Chunk(
        "stable-id", 1, "front_matter", "not_applicable", "Cover text", start_offset=0, end_offset=10
    )
    result = build_chunk_metadata(chunk, document, "rules.pdf", 0)
    assert result["content_type"] == "front_matter"
    assert result["doc_scope"] == "not_applicable"
    assert result["document_date"] == "2026-10-23"
    assert result["effective_date"] is None
    assert result["start_offset"] == 0
    assert result["end_offset"] == 10
    assert set(result) == {
        "document_id", "source_file", "page_number", "chunk_index", "chunk_id",
        "content_type", "doc_scope", "heading", "tournament", "version",
        "document_date", "effective_date", "start_offset", "end_offset",
        "source_bbox", "logical_block_id",
    }
