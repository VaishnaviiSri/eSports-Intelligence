"""Detect document headings, scope regions, and protected content."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.processing.extractor import PageRecord, TableRegion


@dataclass(frozen=True)
class DocumentProfile:
    """Known, reviewable layout facts for a specific source document."""

    tournament: str = "BMIC 2026"
    formula_pages: frozenset[int] = frozenset({31})
    rulebook_table_pages: frozenset[int] = frozenset({23, 25, 27, 30})
    legal_form_pages: frozenset[int] = frozenset({32, 33, 34, 35})
    expected_front_matter_pages: tuple[int, int] = (1, 5)


@dataclass(frozen=True)
class HeadingMatch:
    """A heading recognized at a raw-text page offset."""

    page_number: int
    text: str
    start_offset: int
    end_offset: int
    section_number: str | None = None
    kind: str = "numbered"


@dataclass(frozen=True)
class ScopeRegion:
    """A typed, scoped interval within one physical page."""

    page_number: int
    start_offset: int
    end_offset: int
    content_type: str
    doc_scope: str
    heading: str | None = None


@dataclass
class StructureResult:
    """Structural annotations for a complete document."""

    headings: list[HeadingMatch] = field(default_factory=list)
    regions_by_page: dict[int, list[ScopeRegion]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


_NUMBERED_HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d+)*\.?)\s+([^\n]{2,120}?)\s*$")
_APPENDIX_HEADING = re.compile(r"^\s*APPENDIX\s+([12])\s*[-:–]\s*([^\n]+?)\s*$", re.IGNORECASE)
_TOC_LEADER = re.compile(r"(?:\.{3,}|…{2,}|\s\.{2,}\s*\d+\s*$)")


def _lines_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split text into lines while retaining half-open raw-text offsets."""
    result: list[tuple[str, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        result.append((line.rstrip("\r\n"), offset, end))
        offset = end
    if not result and text == "":
        return []
    if text and (not result or offset < len(text)):
        result.append((text[offset:], offset, len(text)))
    return result


def detect_headings(page: PageRecord) -> list[HeadingMatch]:
    """Detect numbered and major headings without accepting TOC leaders."""
    matches: list[HeadingMatch] = []
    for line, start, end in _lines_with_offsets(page.raw_text):
        stripped = line.strip()
        if not stripped or _TOC_LEADER.search(stripped):
            continue
        upper = stripped.upper()
        if upper in {"GENERAL TERMS", "COMPETITION-SPECIFIC RULES"}:
            matches.append(
                HeadingMatch(page.page_number, stripped, start, end, kind="major")
            )
            continue
        appendix_match = _APPENDIX_HEADING.match(stripped)
        if appendix_match:
            matches.append(
                HeadingMatch(
                    page.page_number,
                    stripped,
                    start,
                    end,
                    section_number=f"APPENDIX {appendix_match.group(1)}",
                    kind="appendix",
                )
            )
            continue
        numbered_match = _NUMBERED_HEADING.match(stripped)
        if numbered_match:
            section_number = numbered_match.group(1).rstrip(".")
            matches.append(
                HeadingMatch(
                    page.page_number,
                    stripped,
                    start,
                    end,
                    section_number=section_number,
                )
            )
    return matches


def _unique_heading(
    headings: list[HeadingMatch], value: str, kind: str
) -> HeadingMatch | None:
    """Return a unique exact major heading, or None when absent/ambiguous."""
    candidates = [
        heading
        for heading in headings
        if heading.kind == kind and heading.text.strip().upper() == value
    ]
    return candidates[0] if len(candidates) == 1 else None


def _has_section_heading(
    pages: list[PageRecord], page_number: int, section_prefix: str
) -> bool:
    """Confirm a nearby numbered body section on the candidate heading page."""
    page = next((item for item in pages if item.page_number == page_number), None)
    if page is None:
        return False
    for line, _, _ in _lines_with_offsets(page.raw_text):
        match = _NUMBERED_HEADING.match(line.strip())
        if match and match.group(1).rstrip(".").split(".", 1)[0] == section_prefix:
            return True
    return False


def _appendix_starts(headings: list[HeadingMatch]) -> list[HeadingMatch]:
    """Return appendix heading matches in source order."""
    return sorted(
        (heading for heading in headings if heading.kind == "appendix"),
        key=lambda heading: (heading.page_number, heading.start_offset),
    )


def _page_scope_intervals(
    page: PageRecord,
    headings: list[HeadingMatch],
    general_start: HeadingMatch | None,
    competition_start: HeadingMatch | None,
    uncertain_start: HeadingMatch | None,
    appendix_starts: list[HeadingMatch],
    profile: DocumentProfile,
    scope_boundary_valid: bool,
) -> list[ScopeRegion]:
    """Build ordered, page-local content/scope regions."""
    page_end = len(page.raw_text)
    if page.page_number in profile.legal_form_pages:
        return [
            ScopeRegion(
                page.page_number,
                0,
                page_end,
                "legal_form",
                "not_applicable",
                next(
                    (item.text for item in appendix_starts if item.page_number <= page.page_number),
                    None,
                ),
            )
        ]

    front_start, front_end = profile.expected_front_matter_pages
    if general_start is not None and page.page_number < general_start.page_number:
        return [ScopeRegion(page.page_number, 0, page_end, "front_matter", "not_applicable")]
    if general_start is None and front_start <= page.page_number <= front_end:
        return [ScopeRegion(page.page_number, 0, page_end, "front_matter", "not_applicable")]

    scope = "unknown"
    active_heading: str | None = None
    page_events: list[tuple[int, str, str | None]] = []
    if general_start is not None and general_start.page_number == page.page_number:
        page_events.append((general_start.start_offset, "general", general_start.text))
    if (
        competition_start is not None
        and scope_boundary_valid
        and competition_start.page_number == page.page_number
    ):
        page_events.append(
            (competition_start.start_offset, "event_specific", competition_start.text)
        )
    if uncertain_start is not None and uncertain_start.page_number == page.page_number:
        page_events.append((uncertain_start.start_offset, "unknown", uncertain_start.text))
    page_events.extend(
        (item.start_offset if item.page_number == page.page_number else 0, "not_applicable", item.text)
        for item in appendix_starts
        if item.page_number == page.page_number
    )
    page_events.sort(key=lambda event: event[0])

    if general_start is not None and general_start.page_number < page.page_number:
        scope = "general"
        active_heading = general_start.text
    if (
        competition_start is not None
        and scope_boundary_valid
        and competition_start.page_number < page.page_number
    ):
        scope = "event_specific"
        active_heading = competition_start.text
    if uncertain_start is not None and uncertain_start.page_number < page.page_number:
        scope = "unknown"
        active_heading = uncertain_start.text
    if any(item.page_number < page.page_number for item in appendix_starts):
        scope = "not_applicable"
        active_heading = next(
            item.text for item in reversed(appendix_starts) if item.page_number < page.page_number
        )

    regions: list[ScopeRegion] = []
    cursor = 0
    for event_offset, event_scope, heading_text in page_events:
        boundary = min(max(event_offset, cursor), page_end)
        if boundary > cursor:
            regions.append(
                ScopeRegion(page.page_number, cursor, boundary, "text", scope, active_heading)
            )
        cursor = boundary
        scope = event_scope
        active_heading = heading_text
    if cursor < page_end or not regions:
        regions.append(ScopeRegion(page.page_number, cursor, page_end, "text", scope, active_heading))
    return regions


def analyze_structure(
    pages: list[PageRecord], profile: DocumentProfile | None = None
) -> StructureResult:
    """Annotate document front matter, scopes, appendices, and table candidates."""
    profile = profile or DocumentProfile()
    headings = [heading for page in pages for heading in detect_headings(page)]
    general_start = _unique_heading(headings, "GENERAL TERMS", "major")
    competition_start = _unique_heading(headings, "COMPETITION-SPECIFIC RULES", "major")
    warnings: list[str] = []

    if general_start is None:
        warnings.append("GENERAL TERMS body boundary is missing or ambiguous; scope is unknown.")
    elif not _has_section_heading(pages, general_start.page_number, "1"):
        warnings.append("GENERAL TERMS heading is not corroborated by Section 1; scope is unknown.")
        general_start = None

    competition_valid = competition_start is not None
    if competition_start is None:
        warnings.append(
            "COMPETITION-SPECIFIC RULES body boundary is missing or ambiguous; affected scope is unknown."
        )
    elif not _has_section_heading(pages, competition_start.page_number, "13"):
        competition_valid = False
        warnings.append(
            "COMPETITION-SPECIFIC RULES heading is not corroborated by Section 13; affected scope is unknown."
        )

    uncertain_start = None
    if not competition_valid:
        section_thirteen = sorted(
            (
                heading
                for heading in headings
                if heading.section_number == "13"
                and (general_start is None or heading.page_number >= general_start.page_number)
            ),
            key=lambda heading: (heading.page_number, heading.start_offset),
        )
        uncertain_start = section_thirteen[0] if section_thirteen else None

    appendices = _appendix_starts(headings)
    if not appendices:
        warnings.append("No appendix boundary was identified; legal-form scope is unverified.")

    regions_by_page: dict[int, list[ScopeRegion]] = {}
    for page in pages:
        for candidate in page.table_regions:
            candidate.classification = classify_table_candidate(
                page.page_number, candidate, profile
            )
            candidate.verified = (
                candidate.extraction_status == "extracted"
                and len(candidate.rows) == candidate.row_count
            )
        regions = _page_scope_intervals(
            page,
            headings,
            general_start,
            competition_start,
            uncertain_start,
            appendices,
            profile,
            competition_valid,
        )
        page.page_class = regions[0].content_type if regions else "unknown"
        page.warnings.extend(
            warning for warning in warnings if "scope" in warning.lower() or "boundary" in warning.lower()
        )
        regions_by_page[page.page_number] = regions

    return StructureResult(headings, regions_by_page, warnings)


def classify_table_candidate(
    page_number: int,
    candidate: TableRegion,
    profile: DocumentProfile,
) -> str:
    """Classify a table candidate using the verified document profile."""
    if page_number in profile.legal_form_pages:
        return "legal_form_grid"
    if page_number in profile.rulebook_table_pages:
        return "rulebook_table"
    return "false_positive"