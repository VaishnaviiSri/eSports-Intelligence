from app.processing.chunker import ChunkingConfig, chunk_page
from app.processing.cleaner import CleanedSegment
from app.processing.extractor import PageRecord


def _page(cleaned_text: str = "") -> PageRecord:
    return PageRecord(1, cleaned_text, cleaned_text, "fixture.pdf")


def test_defaults_and_page_safe_text_splitting() -> None:
    config = ChunkingConfig()
    assert config.max_chars == 1200
    assert config.overlap_chars == 120
    text = "Sentence ends here. " * 100
    page = _page(text)
    segment = CleanedSegment("text", "general", text, 0, len(text))
    chunks = chunk_page(page, [segment], config)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 1200 for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(chunk.start_offset is not None and chunk.end_offset is not None for chunk in chunks)


def test_offsets_are_into_cleaned_text() -> None:
    text = "alpha beta gamma " * 120
    page = _page(text)
    segment = CleanedSegment("text", "general", text, 0, len(text))
    chunks = chunk_page(page, [segment], ChunkingConfig(max_chars=80, overlap_chars=10))
    for chunk in chunks:
        assert page.cleaned_text[chunk.start_offset : chunk.end_offset] == chunk.content


def test_protected_table_is_atomic_and_has_no_text_overlap() -> None:
    text = "ordinary words. " * 100
    rows = [["cell" * 500, "another cell"]]
    page = _page(text + text)
    segments = [
        CleanedSegment("text", "event_specific", text, 0, len(text)),
        CleanedSegment("table", "event_specific", rows, logical_block_id="table-1"),
        CleanedSegment("text", "event_specific", text, len(text), len(text) * 2),
    ]
    chunks = chunk_page(page, segments, ChunkingConfig(max_chars=100, overlap_chars=20))
    table_chunks = [chunk for chunk in chunks if chunk.content_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].content == rows
    assert len(table_chunks[0].content[0][0]) > 100
    after_table = chunks[chunks.index(table_chunks[0]) + 1]
    assert after_table.start_offset == len(text)


def test_formula_stub_and_front_matter_type_are_preserved() -> None:
    page = _page("Cover contents")
    segments = [
        CleanedSegment("front_matter", "not_applicable", "Cover contents", 0, 14),
        CleanedSegment(
            "formula_image",
            "event_specific",
            "Formula image present; see source PDF page 31.",
        ),
    ]
    chunks = chunk_page(page, segments)
    assert chunks[0].content_type == "front_matter"
    assert chunks[0].doc_scope == "not_applicable"
    assert chunks[1].content == "Formula image present; see source PDF page 31."
    assert chunks[1].start_offset is None
