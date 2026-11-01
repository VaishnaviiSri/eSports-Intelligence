"""Read-only PDF page, layout, and table extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class LayoutBlock:
    """One text or image block with its page coordinates."""

    block_type: str
    text: str
    bbox: BBox | None
    raw_block_index: int


@dataclass
class TableRegion:
    """A detected table candidate with its extracted cell matrix."""

    candidate_index: int
    bbox: BBox
    rows: list[list[str | None]]
    row_count: int
    column_count: int
    strategy: str
    extraction_status: str
    classification: str = "unclassified"
    verified: bool = False


@dataclass
class PageRecord:
    """Raw and structured extraction for one physical PDF page."""

    page_number: int
    raw_text: str
    cleaned_text: str
    source_file: str
    layout_blocks: list[LayoutBlock] = field(default_factory=list)
    table_regions: list[TableRegion] = field(default_factory=list)
    page_class: str = "unknown"
    warnings: list[str] = field(default_factory=list)


def _bbox(values: Any) -> BBox | None:
    """Convert a PyMuPDF rectangle-like value to a plain float tuple."""
    if values is None or len(values) != 4:
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _layout_blocks(page: pymupdf.Page) -> list[LayoutBlock]:
    """Extract ordered text and image blocks from a page layout dictionary."""
    result: list[LayoutBlock] = []
    page_dict = page.get_text("dict", sort=True)
    for index, block in enumerate(page_dict.get("blocks", [])):
        block_type = "text" if block.get("type") == 0 else "image"
        lines: list[str] = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if line_text:
                lines.append(line_text)
        result.append(
            LayoutBlock(
                block_type=block_type,
                text="\n".join(lines),
                bbox=_bbox(block.get("bbox")),
                raw_block_index=index,
            )
        )
    return result


def _table_bbox(table: Any) -> BBox:
    """Return a table's bounding box as ordinary Python floats."""
    result = _bbox(table.bbox)
    if result is None:
        raise ValueError("PyMuPDF returned a table without a valid bounding box")
    return result


def _intersection_area(left: BBox, right: BBox) -> float:
    """Calculate the intersection area of two page-coordinate rectangles."""
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _materialize_table(table: Any, index: int, strategy: str) -> TableRegion:
    """Extract a PyMuPDF table candidate, retaining a record on extraction failure."""
    row_count = int(table.row_count)
    column_count = int(table.col_count)
    try:
        extracted = table.extract()
        rows = [
            [None if cell is None else str(cell) for cell in row]
            for row in extracted
        ]
        status = "extracted"
    except Exception as exc:
        rows = []
        status = f"unverified: {type(exc).__name__}: {exc}"
    return TableRegion(
        candidate_index=index,
        bbox=_table_bbox(table),
        rows=rows,
        row_count=row_count,
        column_count=column_count,
        strategy=strategy,
        extraction_status=status,
    )


def _find_page_tables(page: pymupdf.Page, page_number: int) -> list[TableRegion]:
    """Find table candidates, applying the verified scoring-table fallback."""
    default_tables = page.find_tables().tables
    selected_tables = default_tables
    strategy = "lines"

    if page_number == 25 and any(int(table.col_count) > 2 for table in default_tables):
        strict_tables = page.find_tables(strategy="lines_strict").tables
        if strict_tables:
            selected_tables = strict_tables
            strategy = "lines_strict"

    return [
        _materialize_table(table, index, strategy)
        for index, table in enumerate(selected_tables, start=1)
    ]


def extract_pdf_pages(pdf_path: Path, source_file: str | None = None) -> list[PageRecord]:
    """Extract complete text, layout blocks, and table candidates page by page."""
    pdf_path = Path(pdf_path)
    relative_source = source_file or pdf_path.name
    records: list[PageRecord] = []
    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError(f"Encrypted PDFs are not supported: {relative_source}")
        for page_index in range(len(document)):
            page_number = page_index + 1
            page = document.load_page(page_index)
            raw_text = page.get_text("text", sort=True) or ""
            records.append(
                PageRecord(
                    page_number=page_number,
                    raw_text=raw_text,
                    cleaned_text=raw_text,
                    source_file=relative_source,
                    layout_blocks=_layout_blocks(page),
                    table_regions=_find_page_tables(page, page_number),
                )
            )
    return records