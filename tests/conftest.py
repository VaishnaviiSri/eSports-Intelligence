from pathlib import Path

import pymupdf
import pytest


@pytest.fixture(scope="session")
def pdf_fixtures(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create representative PDFs in pytest-managed temporary storage."""
    fixture_dir = tmp_path_factory.mktemp("pdf_fixtures")

    text_document = pymupdf.open()
    for page_number in range(2):
        page = text_document.new_page()
        sample = (
            f"BGMI esports rulebook sample page {page_number + 1}. "
            "This page contains enough readable text to classify as text based. "
            "Teams must follow the published tournament rules and match procedures. "
        )
        page.insert_text((72, 72), sample * 3)
    text_document.save(fixture_dir / "text_sample.pdf")
    text_document.close()

    image_document = pymupdf.open()
    image_page = image_document.new_page()
    samples = bytes((70, 150, 90)) * 100 * 100
    raster = pymupdf.Pixmap(pymupdf.csRGB, 100, 100, samples, False)
    image_page.insert_image(image_page.rect, pixmap=raster)
    image_document.save(fixture_dir / "image_only.pdf")
    image_document.close()

    blank_document = pymupdf.open()
    blank_document.new_page()
    blank_document.save(fixture_dir / "blank.pdf")
    blank_document.close()

    (fixture_dir / "corrupted.pdf").write_bytes(b"not a valid PDF".ljust(100, b"!"))

    encrypted_document = pymupdf.open()
    encrypted_document.new_page().insert_text((72, 72), "Password-protected fixture")
    encrypted_document.save(
        fixture_dir / "encrypted.pdf",
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    encrypted_document.close()

    return fixture_dir
