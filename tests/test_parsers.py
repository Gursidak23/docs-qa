from __future__ import annotations

import pymupdf

from docsqa.ingest.parsers.registry import detect_source_type, parse_document


def test_detect_source_type_by_extension_and_content_type() -> None:
    assert detect_source_type("guide.pdf") == "pdf"
    assert detect_source_type("guide.md") == "md"
    assert detect_source_type("guide.html") == "html"
    assert detect_source_type("guide.txt") == "text"
    assert detect_source_type("guide.unknown") == "text"
    assert detect_source_type("noext", "application/pdf") == "pdf"


def test_parse_markdown_keeps_headings() -> None:
    md = b"# Title\n\nIntro text here.\n\n## Section\n\nMore detailed text."
    doc = parse_document(md, "md", uri="t.md")
    headings = [b.text for b in doc.blocks if b.kind == "heading"]
    assert headings == ["Title", "Section"]
    assert doc.title == "Title"
    assert any(b.kind == "text" and "Intro" in b.text for b in doc.blocks)


def test_parse_html_extracts_structure_and_drops_scripts() -> None:
    html = (
        b"<html><head><title>Doc</title></head><body>"
        b"<h1>Heading</h1><p>Para one.</p><script>evil()</script>"
        b"</body></html>"
    )
    doc = parse_document(html, "html", uri="t.html")
    assert doc.title == "Doc"
    assert any(b.kind == "heading" and b.text == "Heading" for b in doc.blocks)
    assert any(b.kind == "text" and "Para one." in b.text for b in doc.blocks)
    assert all("evil()" not in b.text for b in doc.blocks)


def test_parse_text_splits_paragraphs() -> None:
    doc = parse_document(b"First para.\n\nSecond para.", "text", uri="t.txt")
    assert len(doc.blocks) == 2


def test_parse_pdf_roundtrip_with_page_numbers() -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Hello PDF world.")
    data = pdf.tobytes()
    pdf.close()

    parsed = parse_document(data, "pdf", uri="t.pdf")
    assert parsed.source_type == "pdf"
    assert any("Hello PDF" in b.text for b in parsed.blocks)
    assert parsed.blocks[0].page_no == 1
