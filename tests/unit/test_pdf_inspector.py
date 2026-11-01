from pathlib import Path

import pytest

from app.ingestion.pdf_inspector import (
    InspectionConfig,
    classify_document,
    classify_page,
    inspect_pdf,
)


@pytest.mark.parametrize(
    ("text", "image_count", "expected"),
    [
        (None, 0, "empty"),
        ("", 1, "potentially_scanned"),
        ("x" * 49, 0, "potentially_scanned"),
        ("x" * 50, 0, "lowtext"),
        ("x" * 99, 0, "lowtext"),
        ("x" * 100, 0, "text_based"),
        ("x" * 50, 0, "text_based"),
    ],
)
def test_classify_page_boundaries(text: str | None, image_count: int, expected: str) -> None:
    config = InspectionConfig()
    if len(text or "") == 50 and expected == "text_based":
        config.low_text_threshold = 50
    assert classify_page(text, image_count, config) == expected


def test_classify_page_custom_thresholds() -> None:
    config = InspectionConfig(scan_threshold=3, low_text_threshold=7)
    assert classify_page("ab", 0, config) == "potentially_scanned"
    assert classify_page("abc", 0, config) == "lowtext"
    assert classify_page("abcdefg", 0, config) == "text_based"


@pytest.mark.parametrize(
    ("classes", "total_chars", "expected"),
    [
        (["text_based", "text_based"], 200, "text_based"),
        (["empty", "empty"], 0, "empty_or_unreadable"),
        (["potentially_scanned", "empty"], 20, "potentially_scanned"),
        (["text_based", "potentially_scanned"], 100, "mixed"),
        (["text_based"] * 81 + ["mixed"] * 19, 8100, "text_based"),
        (["inspection_error", "text_based"], 100, "inspection_error"),
        (["encrypted"], 0, "encrypted"),
    ],
)
def test_classify_document(classes: list[str], total_chars: int, expected: str) -> None:
    assert classify_document(classes, total_chars) == expected


def test_classify_document_blank_and_image_only_pages() -> None:
    assert classify_document(["empty", "potentially_scanned"], 0) == "potentially_scanned"


def test_inspect_text_pdf(pdf_fixtures: Path) -> None:
    result = inspect_pdf(pdf_fixtures / "text_sample.pdf", InspectionConfig())
    assert result["classification"] == "text_based"
    assert result["page_count"] == 2
    assert result["pages_inspected"] == 2
    assert result["extraction"]["total_chars"] > 0


def test_inspect_image_pdf(pdf_fixtures: Path) -> None:
    result = inspect_pdf(pdf_fixtures / "image_only.pdf", InspectionConfig())
    assert result["classification"] == "potentially_scanned"
    assert result["page_count"] == 1
    assert result["pages"][0]["classification"] == "potentially_scanned"
    assert result["pages"][0]["image_count"] == 1
    assert result["pages"][0]["has_large_image"] is True
    assert (
        "Document appears to contain image-only pages with no extractable text. "
        "OCR will likely be required for retrieval."
    ) in result["warnings"]


def test_inspect_blank_pdf(pdf_fixtures: Path) -> None:
    result = inspect_pdf(pdf_fixtures / "blank.pdf", InspectionConfig())
    assert result["classification"] == "empty_or_unreadable"
    assert result["pages"][0]["classification"] == "empty"


def test_corrupt_pdf_returns_error(pdf_fixtures: Path) -> None:
    result = inspect_pdf(pdf_fixtures / "corrupted.pdf", InspectionConfig())
    assert result["classification"] == "inspection_error"
    assert "error" in result


def test_encrypted_pdf_is_not_iterated(pdf_fixtures: Path) -> None:
    result = inspect_pdf(pdf_fixtures / "encrypted.pdf", InspectionConfig())
    assert result["classification"] == "encrypted"
    assert result["encrypted"] is True
    assert "pages" not in result
