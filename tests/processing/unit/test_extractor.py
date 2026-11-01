from pathlib import Path
from typing import Any

import pytest

from app.processing import extractor
from app.processing.extractor import extract_pdf_pages


class FakeTable:
    def __init__(self, columns: int) -> None:
        self.row_count = 9
        self.col_count = columns
        self.bbox = (10, 10, 200, 300)

    def extract(self) -> list[list[str]]:
        return [[f"r{row}c{column}" for column in range(self.col_count)] for row in range(9)]


class FakePage:
    def get_text(self, mode: str, sort: bool = True) -> Any:
        if mode == "dict":
            return {"blocks": []}
        return "synthetic text"

    def find_tables(self, strategy: str = "lines") -> Any:
        columns = 6 if strategy == "lines" else 2
        return type("Finder", (), {"tables": [FakeTable(columns)]})()


class FakeDocument:
    needs_pass = False

    def __enter__(self) -> "FakeDocument":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def __len__(self) -> int:
        return 25

    def load_page(self, _: int) -> FakePage:
        return FakePage()


def test_extract_pdf_pages_preserves_page_numbers_and_text(processing_pdf: Path) -> None:
    pages = extract_pdf_pages(processing_pdf)
    assert len(pages) == 35
    assert pages[0].page_number == 1
    assert "Version 1.0.0" in pages[0].raw_text
    assert pages[0].raw_text == pages[0].cleaned_text


def test_blank_pdf_preserves_blank_page(blank_pdf: Path) -> None:
    pages = extract_pdf_pages(blank_pdf)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].raw_text == ""


def test_page_25_uses_lines_strict_when_default_has_more_than_two_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_document = FakeDocument()
    monkeypatch.setattr(extractor.pymupdf, "open", lambda _: fake_document)
    pages = extract_pdf_pages(tmp_path / "mock.pdf")
    candidate = pages[24].table_regions[0]
    assert candidate.strategy == "lines_strict"
    assert candidate.column_count == 2
    assert len(candidate.rows) == 9


def test_encrypted_pdf_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class EncryptedDocument(FakeDocument):
        needs_pass = True

    monkeypatch.setattr(extractor.pymupdf, "open", lambda _: EncryptedDocument())
    with pytest.raises(ValueError, match="Encrypted PDFs"):
        extract_pdf_pages(tmp_path / "encrypted.pdf")
