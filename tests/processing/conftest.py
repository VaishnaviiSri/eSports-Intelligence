from pathlib import Path

import pymupdf
import pytest


def _draw_grid(page: pymupdf.Page, rows: int, columns: int, x: float, y: float, width: float, height: float) -> None:
    cell_width = width / columns
    cell_height = height / rows
    for row in range(rows + 1):
        y_pos = y + row * cell_height
        page.draw_line((x, y_pos), (x + width, y_pos), color=(0, 0, 0), width=0.5)
    for column in range(columns + 1):
        x_pos = x + column * cell_width
        page.draw_line((x_pos, y), (x_pos, y + height), color=(0, 0, 0), width=0.5)
    for row in range(rows):
        for column in range(columns):
            value = f"R{row + 1}C{column + 1}"
            page.insert_text(
                (x + column * cell_width + 3, y + row * cell_height + min(12, cell_height - 2)),
                value,
                fontsize=7,
            )


@pytest.fixture(scope="session")
def processing_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a deterministic 35-page structural fixture outside the repository."""
    directory = tmp_path_factory.mktemp("processing-pdf")
    pdf_path = directory / "synthetic_bmic_rules.pdf"
    document = pymupdf.open()
    for page_number in range(1, 36):
        page = document.new_page(width=612, height=792)
        if page_number == 1:
            page.insert_text((72, 72), "BGMI Rules Version 1.0.0, October 23, 2026")
        elif 2 <= page_number <= 5:
            page.insert_text((72, 72), "TABLE OF CONTENTS")
            page.insert_text((72, 100), "GENERAL TERMS................................................6")
            page.insert_text((72, 125), "COMPETITION-SPECIFIC RULES.................................24")
        elif page_number == 6:
            page.insert_text((72, 72), "GENERAL TERMS")
            page.insert_text((72, 110), "1. Acceptance of General Rules")
            page.insert_text((72, 140), "General rules text for this fixture.")
        elif page_number == 23:
            page.insert_text((72, 72), "12. Changelog")
            _draw_grid(page, 2, 3, 72, 150, 468, 50)
        elif page_number == 24:
            page.insert_text((72, 72), "COMPETITION-SPECIFIC RULES")
            page.insert_text((72, 110), "13. Overview")
            page.insert_text((72, 140), "BMIC event-specific rules text.")
        elif page_number == 25:
            page.insert_text((72, 72), "14. Scoring System")
            _draw_grid(page, 9, 2, 180, 130, 250, 220)
            page.insert_text((72, 400), "Scoring notes remain ordinary text.")
        elif page_number == 27:
            page.insert_text((72, 72), "15.2 Blue Zone Settings")
            _draw_grid(page, 11, 4, 150, 130, 300, 240)
        elif page_number == 30:
            page.insert_text((72, 72), "17. Prizes")
            _draw_grid(page, 5, 2, 150, 130, 300, 170)
            _draw_grid(page, 3, 2, 150, 340, 300, 80)
        elif page_number == 31:
            page.insert_text((72, 72), "Tournament MVP formula is presented as an image.")
        elif page_number == 32:
            page.insert_text((72, 72), "APPENDIX 1 - TEAM MEMBER CERTIFICATE AND ACCEPTANCE FORM")
            _draw_grid(page, 4, 3, 72, 150, 468, 170)
        elif page_number == 33:
            page.insert_text((72, 72), "Appendix 1 continuation form.")
            _draw_grid(page, 2, 3, 72, 150, 468, 120)
        elif page_number == 34:
            page.insert_text((72, 72), "APPENDIX 2 - PARENT OR GUARDIAN CONSENT")
            _draw_grid(page, 6, 2, 72, 150, 468, 240)
        elif page_number == 35:
            page.insert_text((72, 72), "Appendix 2 continuation.")
        else:
            page.insert_text((72, 72), f"Section text on page {page_number}.")
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    """Create a one-page blank PDF fixture."""
    pdf_path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    return pdf_path
