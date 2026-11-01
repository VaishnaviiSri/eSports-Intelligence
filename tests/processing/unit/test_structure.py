from pathlib import Path

from app.processing.extractor import PageRecord, extract_pdf_pages
from app.processing.structure import DocumentProfile, analyze_structure, classify_table_candidate


def test_front_matter_ends_at_body_general_terms(processing_pdf: Path) -> None:
    pages = extract_pdf_pages(processing_pdf)
    result = analyze_structure(pages)
    for number in range(1, 6):
        region = result.regions_by_page[number][0]
        assert region.content_type == "front_matter"
        assert region.doc_scope == "not_applicable"
    assert result.regions_by_page[6][-1].doc_scope == "general"


def test_scopes_and_appendices(processing_pdf: Path) -> None:
    pages = extract_pdf_pages(processing_pdf)
    result = analyze_structure(pages)
    assert result.regions_by_page[23][0].doc_scope == "general"
    assert result.regions_by_page[24][-1].doc_scope == "event_specific"
    for number in range(32, 36):
        region = result.regions_by_page[number][0]
        assert region.content_type == "legal_form"
        assert region.doc_scope == "not_applicable"


def test_missing_competition_boundary_marks_section_thirteen_unknown() -> None:
    pages = [
        PageRecord(
            page_number=6,
            raw_text="GENERAL TERMS\n1. General opening\n",
            cleaned_text="GENERAL TERMS\n1. General opening\n",
            source_file="fixture.pdf",
        ),
        PageRecord(
            page_number=7,
            raw_text="13. Competition section\nUnknown scope content.\n",
            cleaned_text="13. Competition section\nUnknown scope content.\n",
            source_file="fixture.pdf",
        ),
    ]
    result = analyze_structure(pages)
    assert result.regions_by_page[6][-1].doc_scope == "general"
    assert result.regions_by_page[7][-1].doc_scope == "unknown"
    assert any("boundary is missing" in warning for warning in result.warnings)


def test_toc_entry_does_not_count_as_scope_heading() -> None:
    pages = [
        PageRecord(
            page_number=2,
            raw_text=(
                "TABLE OF CONTENTS\nGENERAL TERMS........................6\n"
                "COMPETITION-SPECIFIC RULES........................24\n"
            ),
            cleaned_text="",
            source_file="fixture.pdf",
        )
    ]
    result = analyze_structure(pages)
    assert result.regions_by_page[2][0].content_type == "front_matter"
    assert any("body boundary is missing" in warning for warning in result.warnings)


def test_document_profile_classifies_table_candidates(processing_pdf: Path) -> None:
    pages = extract_pdf_pages(processing_pdf)
    analyze_structure(pages, DocumentProfile())
    assert pages[22].table_regions[0].classification == "rulebook_table"
    assert pages[31].table_regions[0].classification == "legal_form_grid"
    assert classify_table_candidate(20, pages[22].table_regions[0], DocumentProfile()) == "false_positive"
