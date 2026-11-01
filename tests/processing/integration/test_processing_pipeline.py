import json
from pathlib import Path

from app.processing.pipeline import ProcessingConfig, process_pdf
from app.processing.structure import DocumentProfile


def test_pipeline_produces_page_safe_deterministic_jsonl(
    processing_pdf: Path, tmp_path: Path
) -> None:
    output_path = tmp_path / "chunks.jsonl"
    profile = DocumentProfile()
    config = ProcessingConfig()

    first = process_pdf(processing_pdf, output_path, processing_pdf.parent, profile, config)
    first_bytes = output_path.read_bytes()
    second = process_pdf(processing_pdf, output_path, processing_pdf.parent, profile, config)
    second_bytes = output_path.read_bytes()

    assert first_bytes == second_bytes
    assert first.chunk_count == second.chunk_count
    assert first.page_count == 35

    records = [json.loads(line) for line in first_bytes.decode("utf-8").splitlines()]
    assert len(records) == first.chunk_count
    assert all(record["page_number"] == record["metadata"]["page_number"] for record in records)
    assert all(record["content_type"] == record["metadata"]["content_type"] for record in records)
    assert all(record["doc_scope"] == record["metadata"]["doc_scope"] for record in records)

    front_matter = [record for record in records if record["page_number"] <= 5]
    assert front_matter
    assert all(record["content_type"] == "front_matter" for record in front_matter)
    assert all(record["doc_scope"] == "not_applicable" for record in front_matter)

    general_chunks = [record for record in records if record["page_number"] == 23]
    event_chunks = [record for record in records if record["page_number"] == 24]
    assert any(record["doc_scope"] == "general" for record in general_chunks)
    assert any(record["doc_scope"] == "event_specific" for record in event_chunks)

    formula_chunks = [record for record in records if record["content_type"] == "formula_image"]
    assert len(formula_chunks) == 1
    assert formula_chunks[0]["page_number"] == 31
    assert "Formula image present; see source PDF page 31." == formula_chunks[0]["content"]

    forms = [record for record in records if record["content_type"] == "legal_form"]
    assert {record["page_number"] for record in forms} == {32, 33, 34, 35}
    assert all(record["doc_scope"] == "not_applicable" for record in forms)

    tables = [record for record in records if record["content_type"] == "table"]
    assert len(tables) == 5
    assert all(isinstance(record["content"], list) for record in tables)

    for record in records:
        start = record["metadata"]["start_offset"]
        end = record["metadata"]["end_offset"]
        if start is not None:
            assert end is not None and end >= start
