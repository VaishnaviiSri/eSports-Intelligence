"""Conservative text cleanup and protected page-content segmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.processing.extractor import BBox, PageRecord, TableRegion
from app.processing.structure import DocumentProfile, ScopeRegion


@dataclass
class CleanedSegment:
    """One ordered page segment; text offsets address PageRecord.cleaned_text."""

    content_type: str
    doc_scope: str
    content: str | list[dict[str, Any]] | list[list[str | None]]
    start_offset: int | None = None
    end_offset: int | None = None
    source_bbox: BBox | None = None
    heading: str | None = None
    logical_block_id: str | None = None


def normalize_line_breaks(text: str) -> str:
    """Normalize line endings and horizontal spacing without joining lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def collapse_duplicate_heading_lines(text: str) -> str:
    """Collapse only adjacent identical numbered or all-caps heading lines."""
    heading_pattern = re.compile(
        r"^(?:\d{1,2}(?:\.\d+)*\.?\s+\S.*|[A-Z][A-Z0-9 &’'()/:-]{4,})$"
    )
    kept: list[str] = []
    for line in text.split("\n"):
        if (
            kept
            and line.strip()
            and kept[-1].strip() == line.strip()
            and heading_pattern.match(line.strip())
        ):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_text(text: str) -> str:
    """Apply deterministic whitespace and sentence-boundary normalization."""
    text = normalize_line_breaks(text)
    text = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", text)
    text = collapse_duplicate_heading_lines(text)
    return text


def _intersects(left: BBox | None, right: BBox) -> bool:
    """Return whether two page-coordinate boxes have positive overlap."""
    if left is None:
        return False
    return (
        min(left[2], right[2]) > max(left[0], right[0])
        and min(left[3], right[3]) > max(left[1], right[1])
    )


def _effective_scope(regions: list[ScopeRegion]) -> str:
    """Select the last recognized scope on a page for its protected regions."""
    for region in reversed(regions):
        if region.doc_scope != "unknown":
            return region.doc_scope
    return "unknown"


def _add_text_segment(
    text: str,
    scope: str,
    content_type: str,
    heading: str | None,
    segments: list[CleanedSegment],
    text_parts: list[str],
    offset: int,
) -> int:
    """Clean and append an ordinary text segment with cleaned-text offsets."""
    cleaned = clean_text(text)
    if not cleaned:
        return offset
    separator = "\n" if text_parts else ""
    if separator:
        text_parts.append(separator)
        offset += len(separator)
    start = offset
    text_parts.append(cleaned)
    offset += len(cleaned)
    segments.append(
        CleanedSegment(
            content_type,
            scope,
            cleaned,
            start_offset=start,
            end_offset=offset,
            heading=heading,
        )
    )
    return offset


def clean_page(
    page: PageRecord,
    regions: list[ScopeRegion],
    profile: DocumentProfile,
) -> list[CleanedSegment]:
    """Build ordered, page-local cleaned segments and set ``cleaned_text``."""
    segments: list[CleanedSegment] = []
    text_parts: list[str] = []
    offset = 0
    scope = _effective_scope(regions)
    protected_tables = [
        table
        for table in page.table_regions
        if table.classification in {"rulebook_table", "legal_form_grid"}
    ]

    if page.page_number in profile.legal_form_pages:
        blocks: list[tuple[float, dict[str, Any]]] = []
        for block in page.layout_blocks:
            if block.block_type == "text" and block.text.strip() and not any(
                _intersects(block.bbox, table.bbox)
                for table in protected_tables
                if table.classification == "legal_form_grid"
            ):
                top = block.bbox[1] if block.bbox else float(block.raw_block_index)
                blocks.append((top, {"content_type": "text", "text": clean_text(block.text)}))
        for table in protected_tables:
            if table.classification != "legal_form_grid":
                continue
            blocks.append(
                (
                    table.bbox[1],
                    {
                        "content_type": "table", "rows": table.rows,
                        "source_bbox": list(table.bbox),
                    },
                )
            )
        ordered_blocks = [item for _, item in sorted(blocks, key=lambda pair: pair[0])]
        segments.append(
            CleanedSegment(
                "legal_form", "not_applicable", ordered_blocks,
                source_bbox=None, logical_block_id=f"{page.source_file}:p{page.page_number}:form",
            )
        )
    elif any(table.classification == "rulebook_table" for table in protected_tables):
        protected_tables = [
            table for table in protected_tables if table.classification == "rulebook_table"
        ]
        sortable: list[tuple[float, int, str, Any]] = []
        for block in page.layout_blocks:
            if block.block_type != "text" or not block.text.strip():
                continue
            if any(_intersects(block.bbox, table.bbox) for table in protected_tables):
                continue
            top = block.bbox[1] if block.bbox else float(block.raw_block_index)
            sortable.append((top, block.raw_block_index, "text", block))
        for index, table in enumerate(protected_tables):
            sortable.append((table.bbox[1], 100000 + index, "table", table))
        for _, _, kind, item in sorted(sortable, key=lambda entry: (entry[0], entry[1])):
            if kind == "text":
                offset = _add_text_segment(
                    item.text, scope, "text", None, segments, text_parts, offset
                )
            else:
                segments.append(
                    CleanedSegment(
                        "table", scope, item.rows,
                        source_bbox=item.bbox,
                        heading=None,
                        logical_block_id=(
                            f"{page.source_file}:p{page.page_number}:table{item.candidate_index}"
                        ),
                    )
                )
    else:
        for region in regions:
            raw_region = page.raw_text[region.start_offset : region.end_offset]
            offset = _add_text_segment(
                raw_region,
                region.doc_scope,
                region.content_type,
                region.heading,
                segments,
                text_parts,
                offset,
            )

    page.cleaned_text = "".join(text_parts)
    if page.page_number in profile.formula_pages:
        segments.append(
            CleanedSegment(
                "formula_image",
                scope,
                f"Formula image present; see source PDF page {page.page_number}.",
                source_bbox=None,
                logical_block_id=f"{page.source_file}:p{page.page_number}:formula",
            )
        )
    return segments