from pathlib import Path

from app.processing.cleaner import clean_page, clean_text, normalize_line_breaks
from app.processing.extractor import extract_pdf_pages
from app.processing.structure import DocumentProfile, analyze_structure


def test_normalizes_line_breaks_and_spacing() -> None:
    assert normalize_line_breaks("One.\r\nTwo.   Three.\rFour.") == "One.\nTwo. Three.\nFour."


def test_repairs_missing_sentence_space() -> None:
    assert clean_text("The rule applies.This is the next sentence.") == (
        "The rule applies. This is the next sentence."
    )


def test_collapses_only_duplicate_heading_lines() -> None:
    result = clean_text("6.4 Disciplinary Action\n6.4 Disciplinary Action\nSame prose\nSame prose")
    assert result == "6.4 Disciplinary Action\nSame prose\nSame prose"


def test_cleaned_offsets_address_cleaned_text(processing_pdf: Path) -> None:
    pages = extract_pdf_pages(processing_pdf)
    structure = analyze_structure(pages)
    page = pages[5]
    segments = clean_page(page, structure.regions_by_page[6], DocumentProfile())
    text_segments = [segment for segment in segments if segment.content_type == "text"]
    assert text_segments
    for segment in text_segments:
        assert page.cleaned_text[segment.start_offset : segment.end_offset] == segment.content


def test_legal_form_tables_are_protected(processing_pdf: Path) -> None:
    pages = extract_pdf_pages(processing_pdf)
    structure = analyze_structure(pages)
    segments = clean_page(pages[31], structure.regions_by_page[32], DocumentProfile())
    assert len(segments) == 1
    assert segments[0].content_type == "legal_form"
    blocks = segments[0].content
    assert isinstance(blocks, list)
    assert any(block["content_type"] == "table" for block in blocks)
