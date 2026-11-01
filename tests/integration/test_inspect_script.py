import json
from pathlib import Path

from scripts.inspect_pdfs import main


TOP_LEVEL_KEYS = {
    "schema_version",
    "input_dir",
    "config",
    "summary",
    "documents",
    "errors",
}
DOCUMENT_KEYS = {
    "relative_path",
    "size_bytes",
    "page_count",
    "pages_inspected",
    "encrypted",
    "pdf_metadata",
    "extraction",
    "classification",
    "warnings",
}
PAGE_KEYS = {
    "page_number",
    "char_count",
    "word_count",
    "image_count",
    "has_large_image",
    "classification",
    "sample_text",
}


def test_report_schema_and_deterministic_output(pdf_fixtures: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "nested" / "report.json"
    args = ["--input-dir", str(pdf_fixtures), "--output", str(report_path)]

    assert main(args) == 0
    first_output = report_path.read_bytes()
    assert main(args) == 0
    assert report_path.read_bytes() == first_output

    report = json.loads(first_output)
    assert set(report) == TOP_LEVEL_KEYS
    assert set(report["config"]) == {
        "scan_threshold",
        "low_text_threshold",
        "max_sample_chars",
        "max_pages",
    }
    assert set(report["summary"]) == {
        "total_pdfs",
        "by_classification",
        "total_pages",
        "total_extracted_chars",
    }
    assert report["summary"]["total_pdfs"] == 5
    assert report["documents"] == sorted(
        report["documents"], key=lambda document: document["relative_path"]
    )
    for document in report["documents"]:
        assert DOCUMENT_KEYS <= set(document)
        assert {"total_chars", "total_words", "avg_chars_per_page"} == set(
            document["extraction"]
        )
        for page in document.get("pages", []):
            assert PAGE_KEYS <= set(page)


def test_missing_input_dir_returns_one(tmp_path: Path) -> None:
    assert main(["--input-dir", str(tmp_path / "missing"), "--output", str(tmp_path / "out.json")]) == 1
