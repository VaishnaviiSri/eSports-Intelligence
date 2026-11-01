"""Document and chunk metadata extraction and assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any

from app.processing.chunker import Chunk
from app.processing.structure import DocumentProfile


@dataclass
class DocumentMetadata:
    """Metadata derived from a title page and explicit document profile."""

    document_id: str
    tournament: str | None
    version: str | None
    document_date: str | None
    effective_date: str | None
    warnings: list[str] = field(default_factory=list)


_VERSION_RE = re.compile(r"\bVersion\s*:?[ \t]*(\d+(?:\.\d+){1,3})\b", re.IGNORECASE)
_MONTH_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_EFFECTIVE_LABEL_RE = re.compile(
    r"\b(?:effective\s+date|effective\s+from|in\s+force\s+from)\b\s*[:\-]?\s*"
    r"([^\n.;]{4,40})",
    re.IGNORECASE,
)


def _normalize_date(value: str) -> str | None:
    """Normalize a supported textual or ISO calendar date."""
    iso_match = _ISO_DATE_RE.search(value)
    if iso_match:
        try:
            parsed = datetime.strptime(iso_match.group(0), "%Y-%m-%d").date()
            return parsed.isoformat()
        except ValueError:
            return None
    month_match = _MONTH_DATE_RE.search(value)
    if month_match:
        try:
            parsed = datetime.strptime(month_match.group(0), "%B %d, %Y").date()
            return parsed.isoformat()
        except ValueError:
            try:
                parsed = datetime.strptime(
                    f"{month_match.group(1)} {month_match.group(2)} {month_match.group(3)}",
                    "%B %d %Y",
                ).date()
                return parsed.isoformat()
            except ValueError:
                return None
    return None


def extract_document_metadata(
    title_page_text: str,
    profile: DocumentProfile,
    source_file: str,
) -> DocumentMetadata:
    """Extract version, document date, and only explicitly stated effective date."""
    normalized_text = title_page_text.replace("\r\n", "\n").replace("\r", "\n")
    version_match = _VERSION_RE.search(normalized_text)
    version = version_match.group(1) if version_match else None

    document_date: str | None = None
    if version_match:
        version_line_end = normalized_text.find("\n", version_match.start())
        if version_line_end < 0:
            version_line_end = len(normalized_text)
        document_date = _normalize_date(normalized_text[version_match.end() : version_line_end])
    if document_date is None:
        for line in normalized_text.splitlines():
            if re.search(r"\b(?:document\s+date|date)\b\s*[:\-]", line, re.IGNORECASE):
                document_date = _normalize_date(line)
                if document_date:
                    break

    effective_date: str | None = None
    effective_match = _EFFECTIVE_LABEL_RE.search(normalized_text)
    if effective_match:
        effective_date = _normalize_date(effective_match.group(1))

    warnings: list[str] = []
    if version is None:
        warnings.append("Document version was not found on the title page.")
    if document_date is None:
        warnings.append("Document date was not found or could not be parsed on the title page.")
    if effective_match and effective_date is None:
        warnings.append("Explicit effective-date label was found but its date could not be parsed.")

    normalized_source = PurePosixPath(source_file).as_posix()
    document_id = sha256(normalized_source.encode("utf-8")).hexdigest()[:12]
    return DocumentMetadata(
        document_id=document_id,
        tournament=profile.tournament,
        version=version,
        document_date=document_date,
        effective_date=effective_date,
        warnings=warnings,
    )


def build_chunk_metadata(
    chunk: Chunk,
    document: DocumentMetadata,
    source_file: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Assemble the stable metadata mapping attached to a serialized chunk."""
    bbox = list(chunk.source_bbox) if chunk.source_bbox is not None else None
    return {
        "document_id": document.document_id,
        "source_file": PurePosixPath(source_file).as_posix(),
        "page_number": chunk.page_number,
        "chunk_index": chunk_index,
        "chunk_id": chunk.chunk_id,
        "content_type": chunk.content_type,
        "doc_scope": chunk.doc_scope,
        "heading": chunk.heading,
        "tournament": document.tournament if chunk.doc_scope == "event_specific" else None,
        "version": document.version,
        "document_date": document.document_date,
        "effective_date": document.effective_date,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "source_bbox": bbox,
        "logical_block_id": chunk.logical_block_id,
    }