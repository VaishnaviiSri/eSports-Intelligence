"""Deterministic page-local text chunking and protected chunk creation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any

from app.processing.cleaner import CleanedSegment
from app.processing.extractor import PageRecord


@dataclass(frozen=True)
class ChunkingConfig:
    """Maximum text chunk size and same-segment overlap in characters."""

    max_chars: int = 1200
    overlap_chars: int = 120


@dataclass
class Chunk:
    """One output chunk whose content belongs to exactly one physical page."""

    chunk_id: str
    page_number: int
    content_type: str
    doc_scope: str
    content: str | list[dict[str, Any]] | list[list[str | None]]
    heading: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    source_bbox: tuple[float, float, float, float] | None = None
    logical_block_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _choose_cut(text: str, start: int, max_chars: int) -> int:
    """Choose a paragraph, sentence, or word boundary before the hard limit."""
    limit = min(start + max_chars, len(text))
    if limit == len(text):
        return limit
    minimum = start + max(1, max_chars // 2)

    paragraph_candidates = [match.end() for match in re.finditer(r"\n\s*\n", text[start:limit])]
    paragraph_cuts = [start + cut for cut in paragraph_candidates if start + cut >= minimum]
    if paragraph_cuts:
        return paragraph_cuts[-1]

    sentence_cuts = [
        start + match.end()
        for match in re.finditer(r"[.!?](?:[\"'’)]*)?(?=\s|$)", text[start:limit])
        if start + match.end() >= minimum
    ]
    if sentence_cuts:
        return sentence_cuts[-1]

    whitespace_cuts = [
        start + match.start()
        for match in re.finditer(r"\s+", text[start:limit])
        if start + match.start() >= minimum
    ]
    if whitespace_cuts:
        return whitespace_cuts[-1]
    return limit


def _text_slices(text: str, max_chars: int, overlap_chars: int) -> list[tuple[str, int, int]]:
    """Split one ordinary text segment while retaining offsets and bounded overlap."""
    if not text:
        return []
    slices: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = _choose_cut(text, start, max_chars)
        if end <= start:
            end = min(start + max_chars, len(text))
        slices.append((text[start:end], start, end))
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap_chars)
        if (
            next_start < len(text)
            and next_start > 0
            and text[next_start].isalnum()
            and text[next_start - 1].isalnum()
        ):
            forward = next_start
            while forward < end and forward - next_start <= 32 and not text[forward].isspace():
                forward += 1
            if forward < end and text[forward].isspace():
                next_start = forward + 1
            else:
                backward = next_start
                while backward > start and next_start - backward <= 32 and not text[backward - 1].isspace():
                    backward -= 1
                if backward > start and text[backward - 1].isspace():
                    next_start = backward
        start = next_start
    return slices


def _stable_document_id(source_file: str) -> str:
    """Derive a deterministic short ID from the source path."""
    return sha256(source_file.encode("utf-8")).hexdigest()[:12]


def chunk_page(
    page: PageRecord,
    segments: list[CleanedSegment],
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Create ordered chunks from cleaned page segments without crossing pages."""
    config = config or ChunkingConfig()
    if config.max_chars < 1:
        raise ValueError("max_chars must be positive")
    if config.overlap_chars < 0 or config.overlap_chars >= config.max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    document_id = _stable_document_id(page.source_file)
    chunks: list[Chunk] = []
    chunk_index = 0
    protected_types = {"table", "formula_image", "legal_form"}

    for segment in segments:
        if segment.content_type in protected_types:
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-p{page.page_number:04d}-c{chunk_index:04d}",
                    page_number=page.page_number,
                    content_type=segment.content_type,
                    doc_scope=segment.doc_scope,
                    content=segment.content,
                    heading=segment.heading,
                    source_bbox=segment.source_bbox,
                    logical_block_id=segment.logical_block_id,
                )
            )
            chunk_index += 1
            continue

        if not isinstance(segment.content, str):
            raise TypeError(f"Text segment content must be str, got {type(segment.content).__name__}")
        for piece, relative_start, relative_end in _text_slices(
            segment.content, config.max_chars, config.overlap_chars
        ):
            absolute_start = (
                segment.start_offset + relative_start
                if segment.start_offset is not None
                else None
            )
            absolute_end = (
                segment.start_offset + relative_end
                if segment.start_offset is not None
                else None
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-p{page.page_number:04d}-c{chunk_index:04d}",
                    page_number=page.page_number,
                    content_type=segment.content_type,
                    doc_scope=segment.doc_scope,
                    content=piece,
                    heading=segment.heading,
                    start_offset=absolute_start,
                    end_offset=absolute_end,
                )
            )
            chunk_index += 1
    return chunks


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Return a JSON-ready mapping while preserving structured cell content."""
    return asdict(chunk)